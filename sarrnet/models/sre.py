"""Sensor-Aware Relation Encoder (SRE), Sec. III-B, Eqs. (3)-(9).

The eight MCOD bands are grouped into five sensor roles (g1 = B1, g2 = B2-B4,
g3 = B5, g4 = B6, g5 = B7-B8). The SRE

* encodes each role with a small stem and injects the role features into the
  first two backbone stages through a role gate and gated residual fusion;
* builds one token per band from band statistics and sensor metadata
  (center wavelength and FWHM);
* computes two relation families, transition (neighboring red-edge/NIR bands)
  and long-gap (visible-to-NIR), each with six sensor-indexed pair slots, gated
  slot weights, relation maps, confidence/disagreement heads and relation tokens;
* routes the two families per scale and injects band and relation tokens into
  stages 3 and 4.
"""
import math
from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import EPS, ConvGNAct, ResidualBlock, gn_groups, safe_log_ratio, safe_normalized_difference

# MCOD sensor metadata in nm: band center wavelengths and full widths at half maximum.
MCOD_CWL = torch.tensor([422.5, 487.5, 550.0, 602.5, 660.0, 725.0, 785.0, 887.5], dtype=torch.float32)
MCOD_FWHM = torch.tensor([55.0, 65.0, 50.0, 45.0, 60.0, 40.0, 70.0, 125.0], dtype=torch.float32)

# Sensor roles as zero-based band indices.
BAND_ROLES = {'g1': [0], 'g2': [1, 2, 3], 'g3': [4], 'g4': [5], 'g5': [6, 7]}

# Pair slots of the two relation families; integers are bands, strings are roles.
TRANSITION_PAIRS = ((4, 5), (4, 6), (5, 6), (5, 7), ('g3', 'g4'), ('g4', 'g5'))
LONGGAP_PAIRS = ((0, 6), (0, 7), (3, 6), (3, 7), ('g1', 'g5'), ('g2', 'g5'))


def _norm_meta(values: torch.Tensor) -> torch.Tensor:
    return (values - values.min()) / (values.max() - values.min() + EPS)


def _band_interval(idx: int) -> Tuple[float, float, float, float]:
    c = float(MCOD_CWL[idx].item())
    f = float(MCOD_FWHM[idx].item())
    return c, f, c - 0.5 * f, c + 0.5 * f


def _role_interval(role: str) -> Tuple[float, float, float, float]:
    idxs = BAND_ROLES[role]
    c = float(MCOD_CWL[idxs].mean().item())
    l = min(float((MCOD_CWL[i] - 0.5 * MCOD_FWHM[i]).item()) for i in idxs)
    r = max(float((MCOD_CWL[i] + 0.5 * MCOD_FWHM[i]).item()) for i in idxs)
    f = max(r - l, 1.0)
    return c, f, l, r


def _endpoint_interval(spec) -> Tuple[float, float, float, float]:
    return _band_interval(spec) if isinstance(spec, int) else _role_interval(spec)


def _pair_features(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]):
    """Sensor metadata of one pair slot: positions, widths, distance, overlap and gap."""
    c1, f1, l1, r1 = a
    c2, f2, l2, r2 = b
    span = float(MCOD_CWL.max().item() - MCOD_CWL.min().item())
    delta = abs(c1 - c2) / (span + EPS)
    overlap = max(0.0, min(r1, r2) - max(l1, l2))
    overlap_ratio = overlap / max(f1, f2, 1.0)
    gap = max(0.0, max(l1, l2) - min(r1, r2)) / (span + EPS)
    return [
        (c1 - float(MCOD_CWL.min().item())) / (span + EPS),
        (c2 - float(MCOD_CWL.min().item())) / (span + EPS),
        f1 / float(MCOD_FWHM.max().item()),
        f2 / float(MCOD_FWHM.max().item()),
        delta,
        overlap_ratio,
        gap,
    ]


def build_pair_meta(pairs: Sequence) -> torch.Tensor:
    feats = [_pair_features(_endpoint_interval(left), _endpoint_interval(right)) for left, right in pairs]
    return torch.tensor(feats, dtype=torch.float32)


def endpoint_map(x: torch.Tensor, spec) -> torch.Tensor:
    """One band, or the mean of a sensor role, as a (B, 1, H, W) map."""
    if isinstance(spec, int):
        return x[:, spec:spec + 1]
    idxs = BAND_ROLES[spec]
    out = x[:, idxs[0]:idxs[-1] + 1]
    return out if len(idxs) == 1 else out.mean(dim=1, keepdim=True)


def role_context(x: torch.Tensor) -> torch.Tensor:
    """Image-level mean of each sensor role, (B, 5)."""
    groups = [
        x[:, 0:1].mean(dim=(1, 2, 3)),
        x[:, 1:4].mean(dim=(1, 2, 3)),
        x[:, 4:5].mean(dim=(1, 2, 3)),
        x[:, 5:6].mean(dim=(1, 2, 3)),
        x[:, 6:8].mean(dim=(1, 2, 3)),
    ]
    return torch.stack(groups, dim=1)


def _token_stats(tokens: torch.Tensor) -> torch.Tensor:
    return torch.stack([
        tokens.mean(dim=(1, 2)),
        tokens.flatten(1).std(dim=1, unbiased=False),
        tokens.norm(dim=-1).mean(dim=1),
        tokens.abs().amax(dim=(1, 2)),
    ], dim=1)


class BandTokenEncoder(nn.Module):
    """One token per band from band statistics and sensor metadata."""

    def __init__(self, token_dim: int = 96):
        super().__init__()
        self.register_buffer('cwl', _norm_meta(MCOD_CWL).view(1, 8), persistent=False)
        self.register_buffer('fwhm', (MCOD_FWHM / (MCOD_FWHM.max() + EPS)).view(1, 8), persistent=False)
        hidden = max(64, token_dim)
        self.proj = nn.Sequential(
            nn.Linear(5, hidden, bias=False),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, token_dim, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(2, 3))
        std = x.flatten(2).std(dim=2, unbiased=False)
        mx = x.flatten(2).amax(dim=2)
        cwl = self.cwl.expand(x.size(0), -1)
        fwhm = self.fwhm.expand(x.size(0), -1)
        token_in = torch.stack([mean, std, mx, cwl, fwhm], dim=-1)
        return self.proj(token_in)


class RoleGroupEncoder(nn.Module):
    def __init__(self, in_ch: int, p4_ch: int, p8_ch: int):
        super().__init__()
        self.p4 = nn.Sequential(
            ConvGNAct(in_ch, max(16, p4_ch // 2), 3, 2),
            ConvGNAct(max(16, p4_ch // 2), p4_ch, 3, 2),
            ResidualBlock(p4_ch),
        )
        self.p8 = nn.Sequential(
            ConvGNAct(p4_ch, p8_ch, 3, 2),
            ResidualBlock(p8_ch),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        p4 = self.p4(x)
        p8 = self.p8(p4)
        return p4, p8


class BandRoleStem(nn.Module):
    """Role-wise encoders for the five sensor roles, fused at 1/4 and 1/8 scale."""

    def __init__(self):
        super().__init__()
        self.g1 = RoleGroupEncoder(1, 16, 32)
        self.g2 = RoleGroupEncoder(3, 16, 32)
        self.g3 = RoleGroupEncoder(1, 16, 32)
        self.g4 = RoleGroupEncoder(1, 16, 32)
        self.g5 = RoleGroupEncoder(2, 16, 32)
        self.fuse4 = nn.Sequential(ConvGNAct(80, 64, 1, 1, 0), ResidualBlock(64))
        self.fuse8 = nn.Sequential(ConvGNAct(160, 128, 1, 1, 0), ResidualBlock(128))

    def forward(self, x: torch.Tensor, size_p4: Tuple[int, int], size_p8: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        g1_p4, g1_p8 = self.g1(x[:, 0:1])
        g2_p4, g2_p8 = self.g2(x[:, 1:4])
        g3_p4, g3_p8 = self.g3(x[:, 4:5])
        g4_p4, g4_p8 = self.g4(x[:, 5:6])
        g5_p4, g5_p8 = self.g5(x[:, 6:8])
        role_p4 = self.fuse4(torch.cat([g1_p4, g2_p4, g3_p4, g4_p4, g5_p4], dim=1))
        role_p8 = self.fuse8(torch.cat([g1_p8, g2_p8, g3_p8, g4_p8, g5_p8], dim=1))
        if role_p4.shape[-2:] != size_p4:
            role_p4 = F.interpolate(role_p4, size=size_p4, mode='bilinear', align_corners=False)
        if role_p8.shape[-2:] != size_p8:
            role_p8 = F.interpolate(role_p8, size=size_p8, mode='bilinear', align_corners=False)
        return role_p4, role_p8


class BandRoleGate(nn.Module):
    """Image-level gates in [0.5, 1.5] for the role features at 1/4 and 1/8 scale."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(19, 64, bias=False),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 2, bias=True),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = x.mean(dim=(2, 3))
        std = x.flatten(2).std(dim=2, unbiased=False)
        nd56 = safe_normalized_difference(x[:, 4:5], x[:, 5:6]).mean(dim=(2, 3))
        nd67 = safe_normalized_difference(x[:, 5:6], x[:, 6:7]).mean(dim=(2, 3))
        nd68 = safe_normalized_difference(x[:, 5:6], x[:, 7:8]).mean(dim=(2, 3))
        feat = torch.cat([mean, std, nd56, nd67, nd68], dim=1)
        gates = 0.5 + torch.sigmoid(self.proj(feat))
        return gates[:, 0], gates[:, 1]


class GatedResidualFusion(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.GroupNorm(gn_groups(channels), channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        self.scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, base: torch.Tensor, side: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate(torch.cat([base, side], dim=1)))
        return base + self.scale * gate * side


class BandTokenCrossAttention(nn.Module):
    """Cross-attention from backbone features to a token set, added as a scaled residual."""

    def __init__(self, feat_ch: int, token_dim: int = 96, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.dim_head = max(16, feat_ch // heads)
        inner = self.heads * self.dim_head
        self.feat_norm = nn.LayerNorm(feat_ch)
        self.token_norm = nn.LayerNorm(token_dim)
        self.to_q = nn.Linear(feat_ch, inner, bias=False)
        self.to_k = nn.Linear(token_dim, inner, bias=False)
        self.to_v = nn.Linear(token_dim, inner, bias=False)
        self.proj = nn.Linear(inner, feat_ch, bias=True)
        self.local = nn.Sequential(
            nn.Conv2d(feat_ch, feat_ch, kernel_size=3, padding=1, groups=max(1, feat_ch // 32), bias=False),
            nn.GELU(),
            nn.Conv2d(feat_ch, feat_ch, kernel_size=1, bias=True),
        )
        nn.init.zeros_(self.local[-1].weight)
        nn.init.zeros_(self.local[-1].bias)
        self.scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, feat: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        b, c, h, w = feat.shape
        feat_flat = feat.flatten(2).transpose(1, 2)
        q = self.to_q(self.feat_norm(feat_flat)).view(b, h * w, self.heads, self.dim_head).transpose(1, 2)
        tok = self.token_norm(tokens)
        k = self.to_k(tok).view(b, tok.shape[1], self.heads, self.dim_head).permute(0, 2, 1, 3)
        v = self.to_v(tok).view(b, tok.shape[1], self.heads, self.dim_head).permute(0, 2, 1, 3)
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dim_head), dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(b, h * w, self.heads * self.dim_head)
        out = self.proj(out).transpose(1, 2).reshape(b, c, h, w)
        out = out + self.local(feat)
        return feat + self.scale * out


def _gate_mlp(in_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, 32, bias=False),
        nn.LayerNorm(32),
        nn.GELU(),
        nn.Linear(32, 1, bias=True),
    )


class RelationFamilyEncoder(nn.Module):
    """One relation family: gated pair slots, relation maps, confidence/disagreement and tokens.

    Each slot holds the normalized difference and log-ratio of a band (or role) pair.
    Slot gates combine a static term from the pair metadata, a dynamic term from the
    slot statistics and a context term from the role means; a temperature-scaled
    softmax over the six slots gives the slot weights.
    """

    def __init__(self, pairs: Sequence, token_dim: int = 96, map_ch: int = 32):
        super().__init__()
        self.pairs = tuple(pairs)
        self.num_pairs = len(self.pairs)
        self.affine = nn.Conv2d(self.num_pairs * 2, self.num_pairs, kernel_size=1, groups=self.num_pairs, bias=True)
        self.depthwise = nn.Conv2d(self.num_pairs, self.num_pairs, kernel_size=3, padding=1, groups=self.num_pairs, bias=False)
        self.refine = nn.Sequential(ConvGNAct(self.num_pairs, map_ch, 1, 1, 0), ResidualBlock(map_ch))
        self.map_p4 = nn.Sequential(ConvGNAct(map_ch, map_ch, 3, 1), ResidualBlock(map_ch))
        self.map_p8 = nn.Sequential(ConvGNAct(map_ch, map_ch, 3, 2), ResidualBlock(map_ch))
        self.conf_p4 = nn.Sequential(ConvGNAct(map_ch, map_ch, 3, 1), nn.Conv2d(map_ch, 1, 1))
        self.conf_p8 = nn.Sequential(ConvGNAct(map_ch, map_ch, 3, 1), nn.Conv2d(map_ch, 1, 1))
        self.token16 = nn.Sequential(nn.Linear(3, token_dim), nn.LayerNorm(token_dim), nn.GELU(), nn.Linear(token_dim, token_dim))
        self.token32 = nn.Sequential(nn.Linear(3, token_dim), nn.LayerNorm(token_dim), nn.GELU(), nn.Linear(token_dim, token_dim))
        self.register_buffer('pair_meta', build_pair_meta(self.pairs), persistent=False)
        meta_dim = int(self.pair_meta.shape[1])
        # Not used in the forward pass; kept for the parameter and checkpoint layout of the paper model.
        self.gate = _gate_mlp(meta_dim)
        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        self.static_gate = _gate_mlp(meta_dim)
        self.dynamic_gate = _gate_mlp(meta_dim + 3)
        self.dynamic_scale = nn.Parameter(torch.tensor(0.5))
        self.context_gate = _gate_mlp(meta_dim + 3 + 5)
        self.context_scale = nn.Parameter(torch.tensor(0.35))

    def _gated_slots(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pair_maps = []
        for left, right in self.pairs:
            a = endpoint_map(x, left)
            b = endpoint_map(x, right)
            pair_maps.append(torch.cat([safe_normalized_difference(a, b), safe_log_ratio(a, b)], dim=1))
        raw = torch.stack(pair_maps, dim=1)
        bsz, pairs, two, h, w = raw.shape
        stats = torch.stack([
            raw.mean(dim=(2, 3, 4)),
            raw.flatten(2).std(dim=2, unbiased=False),
            raw.abs().flatten(2).amax(dim=2),
        ], dim=-1)
        meta = self.pair_meta.to(device=x.device, dtype=x.dtype).unsqueeze(0).expand(bsz, -1, -1)
        static_logits = self.static_gate(meta).squeeze(-1)
        dynamic_logits = self.dynamic_gate(torch.cat([meta, stats], dim=-1)).squeeze(-1)
        logits = static_logits + self.dynamic_scale * dynamic_logits
        group_ctx = role_context(x).unsqueeze(1).expand(-1, pairs, -1)
        ctx_logits = self.context_gate(torch.cat([meta, stats, group_ctx], dim=-1)).squeeze(-1)
        logits = logits + self.context_scale * ctx_logits
        temp = F.softplus(self.logit_scale) + 0.5
        gates = torch.softmax(logits / temp, dim=1)
        raw = raw * gates.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return raw.view(bsz, pairs * two, h, w), gates

    @staticmethod
    def _tokenize(rel: torch.Tensor, head: nn.Module, size: Tuple[int, int]) -> torch.Tensor:
        rel_s = F.interpolate(rel, size=size, mode='bilinear', align_corners=False)
        stats = torch.stack([
            rel_s.mean(dim=(2, 3)),
            rel_s.flatten(2).std(dim=2, unbiased=False),
            rel_s.abs().flatten(2).amax(dim=2),
        ], dim=-1)
        return head(stats)

    def forward(self, x: torch.Tensor, size_p4, size_p8, size_p16, size_p32) -> Dict[str, torch.Tensor]:
        raw, gates = self._gated_slots(x)
        calibrated = self.affine(raw)
        calibrated = calibrated + self.depthwise(calibrated)
        refined = self.refine(calibrated)
        map_p4 = F.interpolate(self.map_p4(refined), size=size_p4, mode='bilinear', align_corners=False)
        map_p8 = F.interpolate(self.map_p8(refined), size=size_p8, mode='bilinear', align_corners=False)
        dis = calibrated.std(dim=1, keepdim=True)
        dis_logit_p4 = F.interpolate(dis, size=size_p4, mode='bilinear', align_corners=False)
        dis_logit_p8 = F.interpolate(dis, size=size_p8, mode='bilinear', align_corners=False)
        conf_logit_p4 = F.interpolate(self.conf_p4(map_p4), size=size_p4, mode='bilinear', align_corners=False)
        conf_logit_p8 = F.interpolate(self.conf_p8(map_p8), size=size_p8, mode='bilinear', align_corners=False)
        conf_p4 = torch.sigmoid(conf_logit_p4)
        conf_p8 = torch.sigmoid(conf_logit_p8)
        tok16 = self._tokenize(calibrated, self.token16, size_p16)
        tok32 = self._tokenize(calibrated, self.token32, size_p32)
        return {
            'maps_p4': map_p4,
            'maps_p8': map_p8,
            'conf_p4': conf_p4,
            'conf_p8': conf_p8,
            'dis_p4': torch.sigmoid(dis_logit_p4),
            'dis_p8': torch.sigmoid(dis_logit_p8),
            'conf_logit_p4': conf_logit_p4,
            'conf_logit_p8': conf_logit_p8,
            'dis_logit_p4': dis_logit_p4,
            'dis_logit_p8': dis_logit_p8,
            'tokens_p16': tok16,
            'tokens_p32': tok32,
            'gates': gates,
        }


def _routing_head(in_dim: int) -> nn.Sequential:
    head = nn.Sequential(
        nn.Linear(in_dim, 32, bias=False),
        nn.LayerNorm(32),
        nn.GELU(),
        nn.Linear(32, 2, bias=True),
    )
    nn.init.zeros_(head[-1].weight)
    nn.init.zeros_(head[-1].bias)
    return head


class RelationRouter(nn.Module):
    """Scale-specific routing between the transition and long-gap families.

    The 1/8 relation maps and the 1/16 relation tokens are routed from family
    statistics only; the 1/32 relation tokens are also conditioned on the band tokens.
    """

    def __init__(self):
        super().__init__()
        # map_p4_base / map_p4_head are not used in the forward pass; they are kept for
        # the parameter and checkpoint layout of the paper model.
        self.map_p4_base = nn.Parameter(torch.tensor([1.5, 0.0], dtype=torch.float32))
        self.map_p8_base = nn.Parameter(torch.tensor([1.25, 0.0], dtype=torch.float32))
        self.token16_base = nn.Parameter(torch.tensor([1.25, 0.0], dtype=torch.float32))
        self.token32_base = nn.Parameter(torch.tensor([0.0, 1.5], dtype=torch.float32))
        self.delta_scale = nn.Parameter(torch.tensor(0.5))
        self.map_p4_head = _routing_head(8)
        self.map_p8_head = _routing_head(8)
        self.token16_head = _routing_head(8)
        self.token32_head = _routing_head(12)

    @staticmethod
    def _map_stats(maps: torch.Tensor, conf: torch.Tensor, dis: torch.Tensor) -> torch.Tensor:
        return torch.stack([
            maps.mean(dim=(1, 2, 3)),
            maps.flatten(1).std(dim=1, unbiased=False),
            conf.mean(dim=(1, 2, 3)),
            dis.mean(dim=(1, 2, 3)),
        ], dim=1)

    def _route(self, left: torch.Tensor, right: torch.Tensor, base_logits: torch.Tensor, head: nn.Module, context=None) -> torch.Tensor:
        inp = torch.cat([left, right], dim=1) if context is None else torch.cat([left, right, context], dim=1)
        delta = head(inp)
        logits = base_logits.view(1, 2) + self.delta_scale * delta
        return torch.softmax(logits, dim=1)

    @staticmethod
    def _mix_maps(a: torch.Tensor, b: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return weights[:, 0].view(-1, 1, 1, 1) * a + weights[:, 1].view(-1, 1, 1, 1) * b

    @staticmethod
    def _mix_tokens(a: torch.Tensor, b: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return weights[:, 0].view(-1, 1, 1) * a + weights[:, 1].view(-1, 1, 1) * b

    def forward(self, family: Dict[str, torch.Tensor], band_tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        stats_t_p8 = self._map_stats(family['transition_maps_p8'], family['transition_conf_p8'], family['transition_dis_p8'])
        stats_l_p8 = self._map_stats(family['longgap_maps_p8'], family['longgap_conf_p8'], family['longgap_dis_p8'])
        stats_t_t16 = _token_stats(family['transition_tokens_p16'])
        stats_l_t16 = _token_stats(family['longgap_tokens_p16'])
        stats_t_t32 = _token_stats(family['transition_tokens_p32'])
        stats_l_t32 = _token_stats(family['longgap_tokens_p32'])
        band_ctx = _token_stats(band_tokens)

        w_p8 = self._route(stats_t_p8, stats_l_p8, self.map_p8_base, self.map_p8_head)
        w_t16 = self._route(stats_t_t16, stats_l_t16, self.token16_base, self.token16_head)
        w_t32 = self._route(stats_t_t32, stats_l_t32, self.token32_base, self.token32_head, band_ctx)

        return {
            'relation_maps_p8': self._mix_maps(family['transition_maps_p8'], family['longgap_maps_p8'], w_p8),
            'relation_conf_p8': self._mix_maps(family['transition_conf_p8'], family['longgap_conf_p8'], w_p8),
            'relation_dis_p8': self._mix_maps(family['transition_dis_p8'], family['longgap_dis_p8'], w_p8),
            'relation_tokens_p16': self._mix_tokens(family['transition_tokens_p16'], family['longgap_tokens_p16'], w_t16),
            'relation_tokens_p32': self._mix_tokens(family['transition_tokens_p32'], family['longgap_tokens_p32'], w_t32),
            'token32_transition_weight': w_t32[:, 0],
            'token32_longgap_weight': w_t32[:, 1],
        }


class RelationLocalEnhancer(nn.Module):
    """Adds the routed 1/8 relation maps to the stage-3 features as a scaled residual."""

    def __init__(self, feat_ch: int, relation_ch: int = 32):
        super().__init__()
        self.rel_proj = nn.Sequential(
            ConvGNAct(relation_ch + 2, feat_ch, kernel_size=1, stride=1, padding=0),
            ResidualBlock(feat_ch),
        )
        self.fuse = nn.Sequential(
            ConvGNAct(feat_ch * 2, feat_ch, kernel_size=1, stride=1, padding=0),
            ResidualBlock(feat_ch),
            nn.Conv2d(feat_ch, feat_ch, kernel_size=1, bias=True),
        )
        nn.init.zeros_(self.fuse[-1].weight)
        nn.init.zeros_(self.fuse[-1].bias)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, feat: torch.Tensor, relation: Dict[str, torch.Tensor], size: Tuple[int, int]) -> torch.Tensor:
        rel = torch.cat([
            F.interpolate(relation['relation_maps_p8'], size=size, mode='bilinear', align_corners=False),
            F.interpolate(relation['relation_conf_p8'], size=size, mode='bilinear', align_corners=False),
            F.interpolate(relation['relation_dis_p8'], size=size, mode='bilinear', align_corners=False),
        ], dim=1)
        rel_feat = self.rel_proj(rel)
        delta = self.fuse(torch.cat([feat, rel_feat], dim=1))
        return feat + self.scale * delta


class SensorAwareRelationEncoder(nn.Module):
    """SRE (Sec. III-B): band roles, band tokens and the two routed relation families.

    ``forward`` takes the normalized cube and the four backbone stages and returns the
    relation-enhanced stages, the merged token set (8 band tokens and the routed 1/16
    and 1/32 relation tokens), both families, the routed relation and the stage-3
    auxiliary logit.
    """

    def __init__(self, token_dim: int = 96, stage_dims: Sequence[int] = (64, 128, 256, 512)):
        super().__init__()
        self.token_dim = int(token_dim)
        d1, d2, d3, d4 = [int(v) for v in stage_dims]
        self.band_tokens = BandTokenEncoder(token_dim=self.token_dim)
        self.role_stem = BandRoleStem()
        self.role_fuse_p4 = GatedResidualFusion(d1)
        self.role_fuse_p8 = GatedResidualFusion(d2)
        self.band_attn3 = BandTokenCrossAttention(d3, token_dim=self.token_dim, heads=4)
        self.band_attn4 = BandTokenCrossAttention(d4, token_dim=self.token_dim, heads=8)
        self.relation_attn3 = BandTokenCrossAttention(d3, token_dim=self.token_dim, heads=4)
        self.relation_attn4 = BandTokenCrossAttention(d4, token_dim=self.token_dim, heads=8)
        self.local3 = RelationLocalEnhancer(d3, relation_ch=32)
        # Auxiliary stage-3 mask head (training objective only).
        aux_ch = max(64, d3 // 4)
        self.stage3_head = nn.Sequential(
            ConvGNAct(d3, aux_ch, kernel_size=3, stride=1),
            nn.Conv2d(aux_ch, 1, kernel_size=3, padding=1),
        )
        self.role_gate = BandRoleGate()
        self.transition = RelationFamilyEncoder(TRANSITION_PAIRS, token_dim=self.token_dim, map_ch=32)
        self.longgap = RelationFamilyEncoder(LONGGAP_PAIRS, token_dim=self.token_dim, map_ch=32)
        self.router = RelationRouter()
        # Not used in the forward pass; kept for the parameter and checkpoint layout of the paper model.
        self.keep_ratio_head = nn.Sequential(
            nn.Linear(8, 32, bias=False),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, 2, bias=True),
        )

    def forward(self, x: torch.Tensor, feats: Sequence[torch.Tensor]):
        f1, f2, f3, f4 = feats
        role_p4, role_p8 = self.role_stem(x, f1.shape[-2:], f2.shape[-2:])
        gate_p4, gate_p8 = self.role_gate(x)
        role_p4 = role_p4 * gate_p4.view(-1, 1, 1, 1)
        role_p8 = role_p8 * gate_p8.view(-1, 1, 1, 1)
        f1 = self.role_fuse_p4(f1, role_p4)
        f2 = self.role_fuse_p8(f2, role_p8)
        band_tokens = self.band_tokens(x)

        sizes = (f1.shape[-2:], f2.shape[-2:], f3.shape[-2:], f4.shape[-2:])
        family = {}
        for prefix, encoder in (('transition', self.transition), ('longgap', self.longgap)):
            for key, value in encoder(x, *sizes).items():
                family[f'{prefix}_{key}'] = value
        routed = self.router(family, band_tokens)

        f3_band = self.band_attn3(f3, band_tokens)
        f4_band = self.band_attn4(f4, band_tokens)
        f3_local = self.local3(f3_band, routed, f3.shape[-2:])
        f3 = self.relation_attn3(f3_local, routed['relation_tokens_p16'])
        f4 = self.relation_attn4(f4_band, routed['relation_tokens_p32'])
        stage3_logit = self.stage3_head(f3)
        tokens = torch.cat([band_tokens, routed['relation_tokens_p16'], routed['relation_tokens_p32']], dim=1)
        return (f1, f2, f3, f4), tokens, family, routed, stage3_logit
