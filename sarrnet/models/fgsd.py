"""Foreground-Guided Spatial Decoder (FGSD), Sec. III-C, Eqs. (10)-(16).

A top-down pyramid decoder with a query refinement stage. On the 1/8 decoder
features, the FGSD predicts an ROI prior and a per-pixel mixing between the
transition and long-gap relation families (boundary vs. interior context), uses
the ROI prior to modulate the high-resolution features, and lets the queries
attend to a compact memory selected by the ROI prior, the query masks, the
boundary map, the relation confidence and the mixing certainty. The queries and
the dense heads are driven by the stable (routed) relation tokens.
"""
from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import EPS, ConvGNAct, ResidualBlock, summarize_tokens


def prob_to_logit(prob: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    prob = torch.clamp(prob, eps, 1.0 - eps)
    return torch.log(prob) - torch.log1p(-prob)


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 3):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * max(num_layers - 1, 0) + [out_dim]
        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx != len(self.layers) - 1:
                x = F.gelu(x)
        return x


class ForegroundMemoryGate(nn.Module):
    """Parameters of an ROI head that is not used in the forward pass.

    Kept for the parameter and checkpoint layout of the paper model.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.context = nn.Sequential(
            ConvGNAct(hidden_dim + 4, hidden_dim, 3, 1),
            ResidualBlock(hidden_dim),
        )
        self.roi_head = nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1)
        self.keep_head = nn.Sequential(
            nn.Linear(6, 32, bias=False),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, 1, bias=True),
        )


class ROIFamilyMixer(nn.Module):
    """ROI prior, boundary/interior family mixing and the memory keep ratio at 1/8 scale."""

    def __init__(self, feat_ch: int, relation_ch: int = 32):
        super().__init__()
        self.roi_context = nn.Sequential(
            ConvGNAct(feat_ch + 4, feat_ch, 3, 1),
            ResidualBlock(feat_ch),
        )
        self.roi_head = nn.Conv2d(feat_ch, 1, kernel_size=3, padding=1)
        self.transition_proj = ConvGNAct(relation_ch, 16, kernel_size=1, stride=1, padding=0)
        self.longgap_proj = ConvGNAct(relation_ch, 16, kernel_size=1, stride=1, padding=0)
        self.mix_context = nn.Sequential(
            ConvGNAct(feat_ch + 16 + 16 + 8, feat_ch, 3, 1),
            ResidualBlock(feat_ch),
            ConvGNAct(feat_ch, 32, 3, 1),
        )
        self.mix_head = nn.Conv2d(32, 2, kernel_size=1)
        self.keep_head = nn.Sequential(
            nn.Linear(8, 32, bias=False),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, 1, bias=True),
        )

    @staticmethod
    def _blend(left: torch.Tensor, right: torch.Tensor, w_left: torch.Tensor, w_right: torch.Tensor) -> torch.Tensor:
        return left * w_left + right * w_right

    def forward(self, p2_base: torch.Tensor, coarse: torch.Tensor, boundary: torch.Tensor, family: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        size_p8 = p2_base.shape[-2:]
        size_p4 = family['transition_maps_p4'].shape[-2:]
        coarse_up = F.interpolate(torch.sigmoid(coarse), size=size_p8, mode='bilinear', align_corners=False)
        boundary_prob = torch.sigmoid(boundary)

        t_map_p8 = F.interpolate(family['transition_maps_p8'], size=size_p8, mode='bilinear', align_corners=False)
        l_map_p8 = F.interpolate(family['longgap_maps_p8'], size=size_p8, mode='bilinear', align_corners=False)
        t_conf_p8 = F.interpolate(family['transition_conf_p8'], size=size_p8, mode='bilinear', align_corners=False)
        l_conf_p8 = F.interpolate(family['longgap_conf_p8'], size=size_p8, mode='bilinear', align_corners=False)
        t_dis_p8 = F.interpolate(family['transition_dis_p8'], size=size_p8, mode='bilinear', align_corners=False)
        l_dis_p8 = F.interpolate(family['longgap_dis_p8'], size=size_p8, mode='bilinear', align_corners=False)

        # ROI prior.
        avg_conf = 0.5 * (t_conf_p8 + l_conf_p8)
        avg_dis = 0.5 * (t_dis_p8 + l_dis_p8)
        roi_feat = self.roi_context(torch.cat([p2_base, coarse_up, boundary_prob, avg_conf, avg_dis], dim=1))
        roi_logit = self.roi_head(roi_feat)
        roi_prob = torch.sigmoid(roi_logit)

        # Per-pixel mixing of the two relation families.
        conf_gap = torch.abs(t_conf_p8 - l_conf_p8)
        dis_gap = torch.abs(t_dis_p8 - l_dis_p8)
        mix_feat = self.mix_context(torch.cat([
            p2_base,
            self.transition_proj(t_map_p8),
            self.longgap_proj(l_map_p8),
            roi_prob,
            boundary_prob,
            t_conf_p8,
            l_conf_p8,
            t_dis_p8,
            l_dis_p8,
            conf_gap,
            dis_gap,
        ], dim=1))
        mix_logit_p8 = self.mix_head(mix_feat)
        mix_prob_p8 = torch.softmax(mix_logit_p8, dim=1)
        mix_transition_p8 = mix_prob_p8[:, 0:1]
        mix_longgap_p8 = mix_prob_p8[:, 1:2]
        mix_transition_p4 = F.interpolate(mix_transition_p8, size=size_p4, mode='bilinear', align_corners=False)
        mix_longgap_p4 = F.interpolate(mix_longgap_p8, size=size_p4, mode='bilinear', align_corners=False)

        # Fraction of memory positions kept by the memory selection.
        keep_stats = torch.stack([
            roi_prob.mean(dim=(2, 3)).squeeze(1),
            roi_prob.flatten(2).std(dim=2, unbiased=False).squeeze(1),
            boundary_prob.mean(dim=(2, 3)).squeeze(1),
            coarse_up.mean(dim=(2, 3)).squeeze(1),
            conf_gap.mean(dim=(2, 3)).squeeze(1),
            dis_gap.mean(dim=(2, 3)).squeeze(1),
            mix_transition_p8.mean(dim=(2, 3)).squeeze(1),
            mix_longgap_p8.mean(dim=(2, 3)).squeeze(1),
        ], dim=1)
        keep_stats = torch.nan_to_num(keep_stats, nan=0.0, posinf=1.0, neginf=-1.0)
        roi_keep_ratio = 0.08 + 0.14 * torch.sigmoid(self.keep_head(keep_stats)).squeeze(1)
        roi_keep_ratio = torch.nan_to_num(roi_keep_ratio, nan=0.15, posinf=0.22, neginf=0.08).clamp(0.08, 0.22)

        return {
            'roi_logit': roi_logit,
            'roi_prob': roi_prob,
            'roi_keep_ratio': roi_keep_ratio,
            'mix_logit_p8': mix_logit_p8,
            'mix_transition_p8': mix_transition_p8,
            'mix_longgap_p8': mix_longgap_p8,
            'mix_transition_p4': mix_transition_p4,
            'mix_longgap_p4': mix_longgap_p4,
            'mix_certainty_p8': torch.abs(mix_transition_p8 - mix_longgap_p8),
            'relation_maps_p4': self._blend(family['transition_maps_p4'], family['longgap_maps_p4'], mix_transition_p4, mix_longgap_p4),
            'relation_conf_p4': self._blend(family['transition_conf_p4'], family['longgap_conf_p4'], mix_transition_p4, mix_longgap_p4),
            'relation_conf_p8': self._blend(t_conf_p8, l_conf_p8, mix_transition_p8, mix_longgap_p8),
            'relation_dis_p4': self._blend(family['transition_dis_p4'], family['longgap_dis_p4'], mix_transition_p4, mix_longgap_p4),
            'relation_conf_logit_p4': self._blend(
                family['transition_conf_logit_p4'], family['longgap_conf_logit_p4'], mix_transition_p4, mix_longgap_p4
            ),
            'relation_dis_logit_p4': self._blend(
                family['transition_dis_logit_p4'], family['longgap_dis_logit_p4'], mix_transition_p4, mix_longgap_p4
            ),
        }


class ForegroundGuidedSpatialDecoder(nn.Module):
    """FGSD (Sec. III-C).

    Returns the high-resolution decoder feature, the query mask logit, the last
    auxiliary query logit, the mid-level logit, the coarse logit, the boundary
    logit and the mixed relation (ROI prior, family mixing and blended relation maps).
    """

    def __init__(
        self,
        token_dim: int = 96,
        stage_dims: Sequence[int] = (64, 128, 256, 512),
        hidden_dim: int = 64,
        relation_ch: int = 32,
        num_queries: int = 4,
        num_layers: int = 3,
        num_heads: int = 4,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_queries = int(num_queries)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        d1, d2, d3, d4 = [int(v) for v in stage_dims]
        self.lat4 = ConvGNAct(d4, hidden_dim, kernel_size=1, stride=1, padding=0)
        self.lat3 = ConvGNAct(d3, hidden_dim, kernel_size=1, stride=1, padding=0)
        self.lat2 = ConvGNAct(d2, hidden_dim, kernel_size=1, stride=1, padding=0)
        self.lat1 = ConvGNAct(d1, hidden_dim, kernel_size=1, stride=1, padding=0)
        self.rel8 = ConvGNAct(relation_ch, hidden_dim, kernel_size=1, stride=1, padding=0)
        self.rel4 = ConvGNAct(relation_ch, hidden_dim, kernel_size=1, stride=1, padding=0)
        self.token4 = nn.Linear(token_dim, hidden_dim, bias=False)
        self.token1 = nn.Linear(token_dim, hidden_dim, bias=False)
        self.query_token = nn.Linear(token_dim, hidden_dim, bias=False)
        self.fuse3 = nn.Sequential(
            ConvGNAct(hidden_dim * 2, hidden_dim, kernel_size=1, stride=1, padding=0),
            ConvGNAct(hidden_dim, hidden_dim, kernel_size=3, stride=1),
        )
        self.fuse2 = nn.Sequential(
            ConvGNAct(hidden_dim * 3, hidden_dim, kernel_size=1, stride=1, padding=0),
            ConvGNAct(hidden_dim, hidden_dim, kernel_size=3, stride=1),
        )
        self.fuse1 = nn.Sequential(
            ConvGNAct(hidden_dim * 4 + 1, hidden_dim, kernel_size=1, stride=1, padding=0),
            ConvGNAct(hidden_dim, hidden_dim, kernel_size=3, stride=1),
        )
        self.coarse_pred = nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1)
        self.mid_pred = nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1)
        self.boundary_pred = nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1)
        self.seed_pred = nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1)
        self.mask_feat = nn.Sequential(ConvGNAct(hidden_dim, hidden_dim, 3, 1), ConvGNAct(hidden_dim, hidden_dim, 3, 1))
        self.query_content = nn.Embedding(num_queries, hidden_dim)
        self.query_pos = nn.Embedding(num_queries, hidden_dim)
        self.level_embed = nn.Embedding(3, hidden_dim)
        self.cross_attn = nn.ModuleList([nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True) for _ in range(num_layers)])
        self.self_attn = nn.ModuleList([nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True) for _ in range(num_layers)])
        self.query_norm1 = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.query_norm2 = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.query_norm3 = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.mem_norm = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.mask_context = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(num_layers)])
        self.ffn = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(), nn.Linear(hidden_dim * 4, hidden_dim))
            for _ in range(num_layers)
        ])
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.mask_embed = MLPHead(hidden_dim, hidden_dim, hidden_dim, num_layers=3)
        self.class_embed = nn.Linear(hidden_dim, 2)
        self.final_pred = nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1)
        self.memory_gate = ForegroundMemoryGate(hidden_dim)
        self.gate_floor = nn.Parameter(torch.tensor(0.20))
        self.refine = nn.Sequential(
            ConvGNAct(hidden_dim * 3 + 7, hidden_dim, kernel_size=1, stride=1, padding=0),
            ConvGNAct(hidden_dim, hidden_dim, kernel_size=3, stride=1),
        )
        self.roi_mixer = ROIFamilyMixer(hidden_dim, relation_ch=relation_ch)

    def _weighted_object_logit(self, query_feat: torch.Tensor, mask_feat: torch.Tensor, seed: torch.Tensor):
        """Object-score weighted fusion of the query masks."""
        qn = self.out_norm(query_feat)
        mask_logits = torch.einsum('bqc,bchw->bqhw', self.mask_embed(qn), mask_feat) + seed
        class_logits = self.class_embed(qn)
        obj_score = torch.softmax(class_logits, dim=-1)[..., 1]
        denom = obj_score.sum(dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1).clamp_min(EPS)
        obj_prob = torch.sigmoid(mask_logits)
        fused_prob = (obj_score.unsqueeze(-1).unsqueeze(-1) * obj_prob).sum(dim=1, keepdim=True) / denom
        fusion_weight = obj_score / obj_score.sum(dim=1, keepdim=True).clamp_min(EPS)
        return prob_to_logit(fused_prob), mask_logits, fusion_weight

    def _select_memory(
        self,
        mem_feat: torch.Tensor,
        prev_masks: torch.Tensor,
        boundary: torch.Tensor,
        roi_prob: torch.Tensor,
        rel_conf: torch.Tensor,
        mix_cert: torch.Tensor,
        keep_ratio: torch.Tensor,
    ) -> torch.Tensor:
        """Memory selection: gate the memory features and keep the top-k positions."""
        size = mem_feat.shape[-2:]
        roi = torch.nan_to_num(F.interpolate(roi_prob, size=size, mode='bilinear', align_corners=False), nan=0.5, posinf=1.0, neginf=0.0)
        prev = torch.nan_to_num(F.interpolate(torch.sigmoid(prev_masks).mean(dim=1, keepdim=True), size=size, mode='bilinear', align_corners=False), nan=0.5, posinf=1.0, neginf=0.0)
        boundary_prob = torch.nan_to_num(F.interpolate(torch.sigmoid(boundary), size=size, mode='bilinear', align_corners=False), nan=0.5, posinf=1.0, neginf=0.0)
        conf = torch.nan_to_num(F.interpolate(rel_conf, size=size, mode='bilinear', align_corners=False), nan=0.0, posinf=1.0, neginf=0.0)
        cert = torch.nan_to_num(F.interpolate(mix_cert, size=size, mode='bilinear', align_corners=False), nan=0.0, posinf=1.0, neginf=0.0)
        score = 0.40 * roi + 0.25 * prev + 0.15 * boundary_prob + 0.10 * conf + 0.10 * cert
        score = torch.nan_to_num(score, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        floor = torch.sigmoid(self.gate_floor)
        gate = floor + (1.0 - floor) * score
        gate = torch.nan_to_num(gate, nan=float(floor.detach().item()), posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        gated_feat = mem_feat * gate
        seq = gated_feat.flatten(2).transpose(1, 2)
        score_flat = torch.nan_to_num(score.flatten(2).squeeze(1), nan=0.0, posinf=1.0, neginf=0.0)
        seq_len = score_flat.shape[1]
        keep_scalar = torch.nan_to_num(keep_ratio.float().mean(), nan=0.15, posinf=0.22, neginf=0.08).clamp(0.08, 0.22)
        k = min(seq_len, max(32, int(round(float(keep_scalar.item()) * float(seq_len)))))
        idx = torch.topk(score_flat, k=k, dim=1).indices
        return seq.gather(1, idx.unsqueeze(-1).expand(-1, -1, seq.shape[-1]))

    def forward(
        self,
        f1: torch.Tensor,
        f2: torch.Tensor,
        f3: torch.Tensor,
        f4: torch.Tensor,
        tokens: torch.Tensor,
        routed: Dict[str, torch.Tensor],
        family: Dict[str, torch.Tensor],
    ):
        # Stable token path: the decoder and the queries use the routed relation tokens.
        stable_token = summarize_tokens(tokens)
        rel8_stable = self.rel8(routed['relation_maps_p8'])
        t4 = self.token4(stable_token).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, f4.shape[-2], f4.shape[-1])
        p4 = self.lat4(f4) + t4
        coarse = self.coarse_pred(p4)
        p3 = self.fuse3(torch.cat([self.lat3(f3), F.interpolate(p4, size=f3.shape[-2:], mode='bilinear', align_corners=False)], dim=1))
        p2_base = self.fuse2(torch.cat([self.lat2(f2), F.interpolate(p3, size=f2.shape[-2:], mode='bilinear', align_corners=False), rel8_stable], dim=1))
        mid_dense = self.mid_pred(p2_base)
        boundary = self.boundary_pred(p2_base)

        mixed = self.roi_mixer(p2_base, coarse, boundary, family)
        rel4 = self.rel4(mixed['relation_maps_p4'])
        roi_prob = mixed['roi_prob']
        roi_keep_ratio = mixed['roi_keep_ratio']
        roi_gate_p2 = 0.25 + 0.75 * roi_prob
        p2 = p2_base * roi_gate_p2
        roi_up = F.interpolate(roi_prob, size=f1.shape[-2:], mode='bilinear', align_corners=False)
        roi_gate_p1 = 0.25 + 0.75 * roi_up
        t1 = self.token1(stable_token).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, f1.shape[-2], f1.shape[-1])
        p1_seed = (self.lat1(f1) + t1) * roi_gate_p1
        p1 = self.fuse1(torch.cat([
            p1_seed,
            F.interpolate(p2, size=f1.shape[-2:], mode='bilinear', align_corners=False),
            t1,
            F.interpolate(coarse, size=f1.shape[-2:], mode='bilinear', align_corners=False),
            rel4 * roi_gate_p1,
        ], dim=1))
        mask_feat = self.mask_feat(p1)
        seed = self.seed_pred(mask_feat)

        bsz = f1.shape[0]
        query = self.query_content.weight.unsqueeze(0).expand(bsz, -1, -1) + self.query_token(stable_token).unsqueeze(1)
        query_pos = self.query_pos.weight.unsqueeze(0).expand(bsz, -1, -1)
        memory_feats = [p4, p3, p2]
        query_logit, prev_masks, fusion_weight = self._weighted_object_logit(query, mask_feat, seed)
        aux_logits = [query_logit]
        for idx in range(self.num_layers):
            level = idx % len(memory_feats)
            mem_feat = memory_feats[level]
            mem_seq = self._select_memory(
                mem_feat,
                prev_masks,
                boundary,
                roi_prob,
                mixed['relation_conf_p8'],
                mixed['mix_certainty_p8'],
                roi_keep_ratio,
            )
            prev_prob = F.interpolate(torch.sigmoid(prev_masks), size=mem_feat.shape[-2:], mode='bilinear', align_corners=False)
            pooled = torch.einsum('bqhw,bchw->bqc', prev_prob, mem_feat)
            pooled = pooled / prev_prob.flatten(2).sum(dim=-1, keepdim=True).clamp_min(1.0)
            mem_in = self.mem_norm[idx](mem_seq + self.level_embed.weight[level].view(1, 1, -1))
            q_in = self.query_norm1[idx](query + query_pos + self.mask_context[idx](pooled))
            cross_out, _ = self.cross_attn[idx](q_in, mem_in, mem_in, need_weights=False)
            query = query + cross_out
            self_in = self.query_norm2[idx](query + query_pos)
            self_out, _ = self.self_attn[idx](self_in, self_in, self_in, need_weights=False)
            query = query + self_out
            query = query + self.ffn[idx](self.query_norm3[idx](query))
            query_logit, prev_masks, fusion_weight = self._weighted_object_logit(query, mask_feat, seed)
            aux_logits.append(query_logit)

        query_ctx = torch.einsum('bq,bqc->bc', fusion_weight, self.out_norm(query))
        query_ctx = query_ctx / fusion_weight.sum(dim=1, keepdim=True).clamp_min(EPS)
        query_map = query_ctx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, mask_feat.shape[-2], mask_feat.shape[-1])
        boundary_up = F.interpolate(boundary, size=mask_feat.shape[-2:], mode='bilinear', align_corners=False)
        conf_up = F.interpolate(mixed['relation_conf_p4'], size=mask_feat.shape[-2:], mode='bilinear', align_corners=False)
        dis_up = F.interpolate(mixed['relation_dis_p4'], size=mask_feat.shape[-2:], mode='bilinear', align_corners=False)
        mix_t_up = F.interpolate(mixed['mix_transition_p4'], size=mask_feat.shape[-2:], mode='bilinear', align_corners=False)
        mix_l_up = F.interpolate(mixed['mix_longgap_p4'], size=mask_feat.shape[-2:], mode='bilinear', align_corners=False)
        local_feat = self.refine(torch.cat([
            mask_feat,
            query_map,
            rel4 * (0.25 + 0.75 * roi_up),
            torch.sigmoid(seed),
            torch.sigmoid(boundary_up),
            conf_up,
            dis_up,
            roi_up,
            mix_t_up,
            mix_l_up,
        ], dim=1))
        final = query_logit + self.final_pred(local_feat)
        mid = 0.5 * F.interpolate(mid_dense, size=mask_feat.shape[-2:], mode='bilinear', align_corners=False) + 0.5 * aux_logits[0]
        query_aux = aux_logits[-1]
        return local_feat, final, query_aux, mid, coarse, boundary, mixed
