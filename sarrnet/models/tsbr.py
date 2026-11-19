"""Thin-Structure Boundary Refiner (TSBR), Sec. III-D, Eqs. (17)-(18).

The TSBR refines the 1/4-scale decoder feature with three branches that reuse the
relation evidence: a transition branch (red-edge bands, their normalized
differences and the transition-family confidence), a region branch (long-gap
relation maps and the ROI prior) and a thin-structure branch. It outputs a
residual mask logit, a refined boundary logit and small-object / thin-structure
priors. A foreground-background suppression gate turns the refined feature into a
second residual mask logit.
"""
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import ConvGNAct, ResidualBlock, safe_normalized_difference, summarize_tokens


class ForegroundBackgroundSuppressionGate(nn.Module):
    """Amplifies foreground and boundary evidence, suppresses background, predicts a residual logit."""

    def __init__(self, token_dim: int = 96):
        super().__init__()
        self.token_proj = nn.Linear(token_dim, 64, bias=False)
        self.bg_gate = nn.Sequential(
            ConvGNAct(64 + 64 + 1 + 1, 64, kernel_size=1, stride=1),
            nn.Conv2d(64, 64, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.refine = nn.Sequential(
            ConvGNAct(64, 64, kernel_size=3, stride=1),
            ConvGNAct(64, 32, kernel_size=3, stride=1),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )

    def forward(self, feat: torch.Tensor, region: torch.Tensor, boundary: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        token = self.token_proj(summarize_tokens(tokens)).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, feat.shape[-2], feat.shape[-1])
        region_prob = torch.sigmoid(region)
        boundary_prob = torch.sigmoid(boundary)
        gate = self.bg_gate(torch.cat([feat, token, region_prob, boundary_prob], dim=1))
        feat = feat * (1.0 + region_prob + 0.5 * boundary_prob) * (1.0 - 0.5 * gate)
        return self.refine(feat)


class ThinStructureBoundaryRefiner(nn.Module):
    """TSBR (Sec. III-D) followed by the foreground-background suppression gate."""

    def __init__(self, feat_ch: int = 64, relation_ch: int = 32, token_dim: int = 96):
        super().__init__()
        self.transition_branch = nn.Sequential(
            ConvGNAct(11, 32, kernel_size=3, stride=1),
            ResidualBlock(32),
            ConvGNAct(32, 32, kernel_size=3, stride=1),
        )
        self.region_branch = nn.Sequential(
            ConvGNAct(relation_ch + 4, 32, kernel_size=1, stride=1, padding=0),
            ResidualBlock(32),
        )
        self.feat_scale = nn.Parameter(torch.tensor(0.1))
        self.mask_scale = nn.Parameter(torch.tensor(0.1))
        self.thin_branch = nn.Sequential(
            ConvGNAct(8, 32, kernel_size=3, stride=1),
            ResidualBlock(32),
            ConvGNAct(32, 32, kernel_size=3, stride=1),
        )
        self.thin_head = nn.Sequential(
            ConvGNAct(64, 32, kernel_size=3, stride=1),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )
        self.feat_fuse = nn.Sequential(
            ConvGNAct(feat_ch + 32 + 32 + 32 + 3, feat_ch, kernel_size=1, stride=1, padding=0),
            ResidualBlock(feat_ch),
            nn.Conv2d(feat_ch, feat_ch, kernel_size=1, bias=True),
        )
        self.boundary_head = nn.Sequential(
            ConvGNAct(96, 32, kernel_size=3, stride=1),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )
        self.small_head = nn.Sequential(
            ConvGNAct(96, 32, kernel_size=3, stride=1),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )
        self.mask_head = nn.Sequential(
            ConvGNAct(feat_ch + 32 + 32 + 32 + 3, feat_ch, kernel_size=1, stride=1, padding=0),
            ConvGNAct(feat_ch, 32, kernel_size=3, stride=1),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )
        self.suppression_gate = ForegroundBackgroundSuppressionGate(token_dim)

    def forward(
        self,
        x: torch.Tensor,
        local_feat: torch.Tensor,
        boundary: torch.Tensor,
        query_aux: torch.Tensor,
        tokens: torch.Tensor,
        family: Dict[str, torch.Tensor],
        mixed: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns the gate residual logit, the TSBR residual logit, the refined
        boundary logit and the small-object and thin-structure prior logits."""
        size = local_feat.shape[-2:]
        boundary_up = F.interpolate(boundary, size=size, mode='bilinear', align_corners=False)
        b5 = F.interpolate(x[:, 4:5], size=size, mode='bilinear', align_corners=False)
        b6 = F.interpolate(x[:, 5:6], size=size, mode='bilinear', align_corners=False)
        b7 = F.interpolate(x[:, 6:7], size=size, mode='bilinear', align_corners=False)
        b8 = F.interpolate(x[:, 7:8], size=size, mode='bilinear', align_corners=False)
        nd56 = safe_normalized_difference(b5, b6)
        nd67 = safe_normalized_difference(b6, b7)
        nd68 = safe_normalized_difference(b6, b8)
        boundary_prob = torch.sigmoid(boundary_up)
        roi_up = F.interpolate(mixed['roi_prob'], size=size, mode='bilinear', align_corners=False)
        mix_cert = F.interpolate(mixed['mix_certainty_p8'], size=size, mode='bilinear', align_corners=False)

        # Transition branch.
        t_conf = F.interpolate(family['transition_conf_p4'], size=size, mode='bilinear', align_corners=False)
        t_dis = F.interpolate(family['transition_dis_p4'], size=size, mode='bilinear', align_corners=False)
        mix_t = F.interpolate(mixed['mix_transition_p4'], size=size, mode='bilinear', align_corners=False)
        transition_feat = self.transition_branch(torch.cat([b5, b6, b7, b8, nd56, nd67, nd68, boundary_prob, t_conf, t_dis, mix_t], dim=1))

        # Region branch.
        long_map = F.interpolate(family['longgap_maps_p4'], size=size, mode='bilinear', align_corners=False)
        long_conf = F.interpolate(family['longgap_conf_p4'], size=size, mode='bilinear', align_corners=False)
        long_dis = F.interpolate(family['longgap_dis_p4'], size=size, mode='bilinear', align_corners=False)
        mix_l = F.interpolate(mixed['mix_longgap_p4'], size=size, mode='bilinear', align_corners=False)
        region_feat = self.region_branch(torch.cat([long_map, long_conf, long_dis, mix_l, roi_up], dim=1))

        # Thin-structure branch.
        thin_feat = self.thin_branch(torch.cat([nd56, nd67, nd68, boundary_prob, roi_up, mix_cert, t_conf, mix_t], dim=1))
        thin_prior = self.thin_head(torch.cat([thin_feat, transition_feat], dim=1))
        thin_prob = torch.sigmoid(thin_prior)

        feat_delta = self.feat_fuse(torch.cat([local_feat, transition_feat, region_feat, thin_feat, boundary_prob, roi_up, thin_prob], dim=1))
        local_refined = local_feat + self.feat_scale * feat_delta
        branch_feat = torch.cat([transition_feat, region_feat, thin_feat], dim=1)
        boundary_refined = boundary_up + self.boundary_head(branch_feat)
        small_prior = self.small_head(branch_feat)
        mask_delta = self.mask_scale * self.mask_head(torch.cat([
            local_refined,
            transition_feat,
            region_feat,
            thin_feat,
            torch.sigmoid(boundary_refined),
            roi_up,
            thin_prob,
        ], dim=1))

        gate_delta = self.suppression_gate(local_refined, query_aux, boundary_refined, tokens)
        return gate_delta, mask_delta, boundary_refined, small_prior, thin_prior
