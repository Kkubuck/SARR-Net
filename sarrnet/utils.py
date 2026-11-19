"""Config loading, model building and the prediction readout shared by the scripts."""
import os
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from .models import SARRNet, load_band_stats


def load_config(path: str) -> Dict:
    """Read a YAML config; ``model.band_stats`` is resolved relative to the config file."""
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    stats = cfg['model']['band_stats']
    if not os.path.isabs(stats):
        cfg['model']['band_stats'] = os.path.join(os.path.dirname(os.path.abspath(path)), stats)
    return cfg


def build_model(cfg: Dict, checkpoint: Optional[str] = None) -> SARRNet:
    """SARR-Net from a config, optionally with weights saved by train.py."""
    p1, p99 = load_band_stats(cfg['model']['band_stats'])
    model = SARRNet(p1, p99, token_dim=int(cfg['model']['token_dim']))
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        state = state['model'] if isinstance(state, dict) and 'model' in state else state
        model.load_state_dict(state, strict=True)
    return model


def readout(logits: Sequence[torch.Tensor], plan: Dict[str, int], out_size: Sequence[int]) -> np.ndarray:
    """Paper readout of the four (1, 1, H, W) logits of one image.

    Sum of the four logits, crop of the padding, bilinear resize to ``out_size``,
    sigmoid and min-max normalization to [0, 1].
    """
    res = logits[0] + logits[1] + logits[2] + logits[3]
    top, left = plan['pad_top'], plan['pad_left']
    res = res[..., top:top + plan['new_h'], left:left + plan['new_w']]
    res = F.interpolate(res, size=tuple(out_size), mode='bilinear', align_corners=False)
    res = torch.sigmoid(res).cpu().numpy().squeeze()
    return (res - res.min()) / (res.max() - res.min() + 1e-8)


@torch.no_grad()
def predict_mask(model: SARRNet, image: torch.Tensor, plan: Dict[str, int], out_size: Sequence[int]) -> np.ndarray:
    """Normalized prediction in [0, 1] at ``out_size`` for one padded (1, 8, H, W) cube."""
    return readout(model(image), plan, out_size)
