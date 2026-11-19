"""SARR-Net: Sensor-Aware Relation Routing for multispectral camouflaged object detection."""
import json
import pickle
import types
from typing import Dict, Iterator, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fgsd import ForegroundGuidedSpatialDecoder
from .layers import EPS
from .smt import smt_t
from .sre import SensorAwareRelationEncoder
from .tsbr import ThinStructureBoundaryRefiner

SMT_T_STAGE_DIMS = (64, 128, 256, 512)


def load_band_stats(path: str) -> Tuple[List[float], List[float]]:
    """Per-band p1/p99 percentiles (inputs scaled to [0, 1]) from a JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        stats = json.load(f)
    return [float(v) for v in stats['p1']], [float(v) for v in stats['p99']]


class SARRNet(nn.Module):
    """SARR-Net with an SMT-T backbone on eight-band input.

    ``forward`` returns four mask logits at the input resolution: the final
    prediction, the last query-decoder prediction, the mid-level prediction and the
    coarse prediction. With ``return_aux=True`` it also returns the auxiliary
    outputs used by the training objective.
    """

    def __init__(self, p1: Sequence[float], p99: Sequence[float], token_dim: int = 96):
        super().__init__()
        self.register_buffer('p1', torch.tensor(p1, dtype=torch.float32).view(1, 8, 1, 1), persistent=True)
        self.register_buffer('p99', torch.tensor(p99, dtype=torch.float32).view(1, 8, 1, 1), persistent=True)
        self.backbone = smt_t(in_chans=8)
        self.sre = SensorAwareRelationEncoder(token_dim=token_dim, stage_dims=SMT_T_STAGE_DIMS)
        self.fgsd = ForegroundGuidedSpatialDecoder(token_dim=token_dim, stage_dims=SMT_T_STAGE_DIMS, hidden_dim=64, relation_ch=32)
        self.tsbr = ThinStructureBoundaryRefiner(feat_ch=self.fgsd.hidden_dim, relation_ch=32, token_dim=token_dim)
        self.deep_head = nn.Conv2d(SMT_T_STAGE_DIMS[3], 1, kernel_size=3, padding=1)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Per-band percentile normalization, Eq. (1)."""
        x = torch.clamp(x, self.p1, self.p99)
        x = (x - self.p1) / (self.p99 - self.p1 + EPS)
        return torch.clamp(x, 0.0, 1.0)

    @staticmethod
    def _resize(x: torch.Tensor, size) -> torch.Tensor:
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

    @staticmethod
    def _entropy(weights: torch.Tensor) -> torch.Tensor:
        """Mean entropy of the six slot weights of a relation family."""
        return (-(weights.clamp_min(EPS) * weights.clamp_min(EPS).log()).sum(dim=1)).mean()

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        target_size = x.shape[-2:]
        x = self.normalize(x)
        feats = self.backbone.forward_features(x)
        (f1, f2, f3, f4), tokens, family, routed, stage3_logit = self.sre(x, feats)
        local_feat, query_final, query_aux, mid, coarse, boundary, mixed = self.fgsd(f1, f2, f3, f4, tokens, routed, family)
        gate_delta, mask_delta, boundary_refined, small_prior, thin_prior = self.tsbr(
            x, local_feat, boundary, query_aux, tokens, family, mixed
        )
        deep = self.deep_head(f4)
        final = query_final + gate_delta + mask_delta

        # Routing statistics for the training objective, computed per replica so that
        # DataParallel reduces them exactly as during training of the paper runs.
        transition_gate_entropy = self._entropy(family['transition_gates'])
        longgap_gate_entropy = self._entropy(family['longgap_gates'])
        w_t = routed['token32_transition_weight'].clamp_min(EPS)
        w_l = routed['token32_longgap_weight'].clamp_min(EPS)
        token32_entropy = (-(w_t * w_t.log() + w_l * w_l.log())).mean()

        y1 = self._resize(final, target_size)
        y2 = self._resize(query_aux, target_size)
        y3 = self._resize(mid, target_size)
        y4 = self._resize(coarse + F.interpolate(deep, size=coarse.shape[-2:], mode='bilinear', align_corners=False), target_size)
        if not return_aux:
            return y1, y2, y3, y4
        aux = {
            'boundary': boundary_refined,
            'relation_conf_logit': mixed['relation_conf_logit_p4'],
            'relation_dis_logit': mixed['relation_dis_logit_p4'],
            'stage3_logit': stage3_logit,
            'small_prior': small_prior,
            'thin_prior': thin_prior,
            'roi_logit': mixed['roi_logit'],
            'mix_logit_p8': mixed['mix_logit_p8'],
            'transition_gate_entropy': transition_gate_entropy,
            'longgap_gate_entropy': longgap_gate_entropy,
            'token32_entropy': token32_entropy,
        }
        return y1, y2, y3, y4, aux

    def backbone_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.backbone.parameters()

    def other_parameters(self) -> Iterator[nn.Parameter]:
        for name, param in self.named_parameters():
            if not name.startswith('backbone.'):
                yield param


def _expand_first_conv_weight(weight: torch.Tensor, target_in_chans: int) -> torch.Tensor:
    """Resample an RGB stem kernel to ``target_in_chans`` input channels.

    Trilinear interpolation along the channel axis, a [0.25, 0.5, 0.25] smoothing
    along the channels and a rescaling by in/out channels.
    """
    in_chans = weight.shape[1]
    if in_chans == target_in_chans:
        return weight
    w = weight.unsqueeze(1)
    w = F.interpolate(w, size=(target_in_chans, weight.shape[2], weight.shape[3]), mode='trilinear', align_corners=True)
    kernel = torch.tensor([0.25, 0.5, 0.25], device=w.device, dtype=w.dtype).view(1, 1, 3, 1, 1)
    w = F.pad(w, (0, 0, 0, 0, 1, 1), mode='replicate')
    w = F.conv3d(w, kernel)
    scale = float(in_chans) / float(target_in_chans)
    return (w.squeeze(1) * scale).contiguous()


class _Placeholder(dict):
    """Stands in for pickled objects whose class cannot be imported (e.g. a training config)."""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def __setstate__(self, state):
        pass


class _LenientUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ImportError, AttributeError):
            return _Placeholder


_lenient_pickle = types.SimpleNamespace(__name__='lenient_pickle', Unpickler=_LenientUnpickler, load=pickle.load)


def _load_pretrained(path: str):
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except Exception:
        # Release files that also pickle training state (optimizer, config objects).
        return torch.load(path, map_location='cpu', weights_only=False, pickle_module=_lenient_pickle)


def load_backbone_weights(model: SARRNet, path: str) -> Dict[str, List[str]]:
    """Load ImageNet SMT-T weights (official ``smt_tiny.pth``) into the eight-band backbone.

    The three-channel stem kernel is resampled to eight channels; every other
    backbone tensor is copied unchanged.
    """
    checkpoint = _load_pretrained(path)
    state = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
    state = dict(state)
    first_key = 'patch_embed1.conv.0.weight'
    if first_key in state and state[first_key].shape[1] != 8:
        state[first_key] = _expand_first_conv_weight(state[first_key], 8)
    current = model.backbone.state_dict()
    filtered, skipped = {}, []
    for key, value in state.items():
        if key not in current:
            continue
        if current[key].shape != value.shape:
            skipped.append(key)
            continue
        filtered[key] = value
    missing, unexpected = model.backbone.load_state_dict(filtered, strict=False)
    return {'missing': list(missing), 'unexpected': list(unexpected), 'skipped': skipped}
