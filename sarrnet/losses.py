"""Training objective of SARR-Net, Eq. (19).

All terms are averaged over the valid (unpadded) pixels. The objective is the sum of

* the task term: BCE + IoU on the four mask logits plus 0.25 x (BCE + IoU) on the
  refined boundary logit;
* spectral-relation terms: the relation confidence and disagreement of the
  boundary/interior-mixed relation (1/4 scale) are supervised by a hardness target
  and a disagreement target built from the detached mask predictions and the mask
  boundary, and the relation-enhanced stage-3 feature by an auxiliary BCE + IoU
  mask loss;
* spatial terms: the ROI prior (dilated mask), the boundary/interior family mixing
  (soft cross-entropy), the thin-structure prior and the small-object prior;
* a routing term that keeps the slot-gate entropies of both relation families and
  the entropy of the 1/32 token routing inside fixed bands.

The weights are constant over training.
"""
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage

EPS = 1e-6


def _odd(kernel_size: int) -> int:
    kernel_size = int(kernel_size)
    return kernel_size + 1 if kernel_size % 2 == 0 else kernel_size


def _mask_denominator(mask: torch.Tensor, channels: int = 1) -> torch.Tensor:
    return torch.clamp(mask.sum() * float(max(channels, 1)), min=1.0)


def masked_bce_with_logits(pred: torch.Tensor, target: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    if valid_mask is None:
        return loss.mean()
    loss = loss * valid_mask
    return loss.sum() / _mask_denominator(valid_mask, pred.shape[1])


def masked_iou_loss(pred: torch.Tensor, mask: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    pred = torch.sigmoid(pred)
    if valid_mask is not None:
        pred = pred * valid_mask
        mask = mask * valid_mask
    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    iou = 1.0 - (inter + 1.0) / (union - inter + 1.0)
    return iou.mean()


def masked_soft_ce(logits: torch.Tensor, target: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    loss = -(target * F.log_softmax(logits, dim=1)).sum(dim=1, keepdim=True)
    if valid_mask is None:
        return loss.mean()
    loss = loss * valid_mask
    return loss.sum() / _mask_denominator(valid_mask, 1)


def mask_to_boundary(mask: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    pad = kernel_size // 2
    dil = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=pad)
    ero = -F.max_pool2d(-mask, kernel_size=kernel_size, stride=1, padding=pad)
    return torch.clamp(dil - ero, 0.0, 1.0)


def small_object_map(mask: torch.Tensor, area_ratio: float = 0.02) -> torch.Tensor:
    """Per connected component of ``mask > 0.5``: max(0, 1 - area / (area_ratio * H * W))."""
    b, _, h, w = mask.shape
    area_thr = max(float(h * w) * float(area_ratio), 1.0)
    out_np = np.zeros((b, 1, h, w), dtype=np.float32)
    mask_np = (mask.detach().cpu().numpy() > 0.5).astype(np.uint8)
    for bi in range(b):
        labeled, num = ndimage.label(mask_np[bi, 0])
        if num <= 0:
            continue
        areas = np.bincount(labeled.ravel(), minlength=num + 1).astype(np.float64)
        weights = np.maximum(0.0, 1.0 - areas / area_thr)
        weights[0] = 0.0
        out_np[bi, 0] = weights[labeled]
    return torch.from_numpy(out_np).to(device=mask.device, dtype=mask.dtype)


class SARRNetLoss(nn.Module):
    def __init__(
        self,
        boundary_weight: float = 0.25,
        relation_conf_weight: float = 0.10,
        relation_dis_weight: float = 0.03,
        stage3_weight: float = 0.15,
        small_prior_weight: float = 0.02,
        routing_weight: float = 0.02,
        roi_weight: float = 0.05,
        mix_weight: float = 0.04,
        thin_weight: float = 0.03,
        hardness_weights: Sequence[float] = (0.3, 0.2, 0.15, 0.15, 0.1),
        roi_margin_kernel: int = 17,
        mix_boundary_kernel: int = 5,
        gate_entropy_range: Sequence[float] = (1.05, 1.45),
        token_entropy_range: Sequence[float] = (0.35, 0.62),
    ):
        super().__init__()
        self.boundary_weight = float(boundary_weight)
        self.relation_conf_weight = float(relation_conf_weight)
        self.relation_dis_weight = float(relation_dis_weight)
        self.stage3_weight = float(stage3_weight)
        self.small_prior_weight = float(small_prior_weight)
        self.routing_weight = float(routing_weight)
        self.roi_weight = float(roi_weight)
        self.mix_weight = float(mix_weight)
        self.thin_weight = float(thin_weight)
        # Weights of the prediction error, GT boundary, scale disagreement,
        # uncertainty and false-negative maps in the hardness target.
        self.hardness_weights = [float(v) for v in hardness_weights]
        self.roi_margin_kernel = _odd(roi_margin_kernel)
        self.mix_boundary_kernel = _odd(mix_boundary_kernel)
        self.gate_entropy_range = [float(v) for v in gate_entropy_range]
        self.token_entropy_range = [float(v) for v in token_entropy_range]

    @classmethod
    def from_config(cls, cfg: Dict) -> 'SARRNetLoss':
        return cls(**cfg)

    def forward(self, logits: Sequence[torch.Tensor], aux: Dict[str, torch.Tensor], gts: torch.Tensor,
                valid_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        s1, s2, s3, s4 = logits
        size = gts.shape[-2:]

        # Task term.
        seg = (
            masked_bce_with_logits(s1, gts, valid_mask) + masked_iou_loss(s1, gts, valid_mask)
            + masked_bce_with_logits(s2, gts, valid_mask) + masked_iou_loss(s2, gts, valid_mask)
            + masked_bce_with_logits(s3, gts, valid_mask) + masked_iou_loss(s3, gts, valid_mask)
            + masked_bce_with_logits(s4, gts, valid_mask) + masked_iou_loss(s4, gts, valid_mask)
        )
        boundary_gt = mask_to_boundary(gts)
        boundary_pred = F.interpolate(aux['boundary'], size=size, mode='bilinear', align_corners=False)
        boundary_loss = masked_bce_with_logits(boundary_pred, boundary_gt, valid_mask) + masked_iou_loss(boundary_pred, boundary_gt, valid_mask)
        seg = seg + self.boundary_weight * boundary_loss

        # Targets from the detached predictions.
        probs = [
            torch.sigmoid(F.interpolate(s.detach(), size=size, mode='bilinear', align_corners=False))
            for s in (s1, s2, s3, s4)
        ]
        stacked = torch.stack(probs, dim=1)
        mean_prob = stacked.mean(dim=1)
        scale_dis = stacked.std(dim=1, unbiased=False)
        uncert = 4.0 * mean_prob * (1.0 - mean_prob)
        coarse_err = torch.abs(mean_prob - gts)
        fn_boost = torch.clamp(gts - mean_prob, 0.0, 1.0)
        w_err, w_bd, w_ctx, w_unc, w_fn = self.hardness_weights
        hard_gt = torch.clamp(
            w_err * coarse_err + w_bd * boundary_gt + w_ctx * scale_dis + w_unc * uncert + w_fn * fn_boost,
            0.0,
            1.0,
        )
        dis_gt = torch.clamp(0.50 * scale_dis + 0.25 * uncert + 0.25 * boundary_gt, 0.0, 1.0)

        # Spectral-relation terms.
        conf_logit = F.interpolate(aux['relation_conf_logit'], size=size, mode='bilinear', align_corners=False)
        relation_conf_loss = masked_bce_with_logits(conf_logit, hard_gt, valid_mask)
        dis_logit = F.interpolate(aux['relation_dis_logit'], size=size, mode='bilinear', align_corners=False)
        relation_dis_loss = masked_bce_with_logits(dis_logit, dis_gt, valid_mask)
        stage3_logit = aux['stage3_logit']
        gt16 = F.interpolate(gts, size=stage3_logit.shape[-2:], mode='area')
        valid16 = F.interpolate(valid_mask, size=stage3_logit.shape[-2:], mode='nearest') if valid_mask is not None else None
        stage3_loss = masked_bce_with_logits(stage3_logit, gt16, valid16) + masked_iou_loss(stage3_logit, gt16, valid16)

        # Spatial terms.
        small_target = small_object_map(gts)
        small_prior = F.interpolate(aux['small_prior'], size=size, mode='bilinear', align_corners=False)
        small_prior_loss = masked_bce_with_logits(small_prior, small_target, valid_mask)

        k = self.roi_margin_kernel
        roi_target = F.max_pool2d(gts, kernel_size=k, stride=1, padding=k // 2)
        roi_logit = F.interpolate(aux['roi_logit'], size=size, mode='bilinear', align_corners=False)
        roi_loss = masked_bce_with_logits(roi_logit, roi_target, valid_mask)

        mix_logit = aux['mix_logit_p8']
        mix_size = mix_logit.shape[-2:]
        k = self.mix_boundary_kernel
        interior_gt = -F.max_pool2d(-gts, kernel_size=k, stride=1, padding=k // 2)
        interior_gt = torch.clamp(interior_gt, 0.0, 1.0)
        boundary_obj = torch.clamp(gts - interior_gt, 0.0, 1.0)
        boundary_small = F.interpolate(boundary_obj, size=mix_size, mode='area')
        interior_small = F.interpolate(interior_gt, size=mix_size, mode='area')
        target_mass = boundary_small + interior_small
        mix_target = torch.cat([
            boundary_small / target_mass.clamp_min(EPS),
            interior_small / target_mass.clamp_min(EPS),
        ], dim=1)
        region_mask = (target_mass > 0.01).float()
        if valid_mask is not None:
            region_mask = region_mask * F.interpolate(valid_mask, size=mix_size, mode='nearest')
        mix_loss = masked_soft_ce(mix_logit, mix_target, region_mask)

        thin_target = torch.clamp(
            F.max_pool2d(boundary_gt, kernel_size=5, stride=1, padding=2)
            + F.max_pool2d(small_target, kernel_size=3, stride=1, padding=1),
            0.0,
            1.0,
        )
        thin_prior = F.interpolate(aux['thin_prior'], size=size, mode='bilinear', align_corners=False)
        thin_loss = masked_bce_with_logits(thin_prior, thin_target, valid_mask)

        # Routing term. Under DataParallel each entry holds one value per replica.
        g_min, g_max = self.gate_entropy_range
        t_min, t_max = self.token_entropy_range
        trans_ent = aux['transition_gate_entropy'].float().mean()
        long_ent = aux['longgap_gate_entropy'].float().mean()
        token_ent = aux['token32_entropy'].float().mean()
        routing_loss = (
            F.relu(g_min - trans_ent) + F.relu(trans_ent - g_max)
            + F.relu(g_min - long_ent) + F.relu(long_ent - g_max)
            + F.relu(t_min - token_ent) + F.relu(token_ent - t_max)
        ).mean()

        loss = (
            seg
            + self.relation_conf_weight * relation_conf_loss
            + self.relation_dis_weight * relation_dis_loss
            + self.stage3_weight * stage3_loss
            + self.small_prior_weight * small_prior_loss
            + self.routing_weight * routing_loss
            + self.roi_weight * roi_loss
            + self.mix_weight * mix_loss
            + self.thin_weight * thin_loss
        )
        return {
            'loss': loss,
            'seg': seg,
            'boundary': boundary_loss,
            'relation_conf': relation_conf_loss,
            'relation_dis': relation_dis_loss,
            'stage3': stage3_loss,
            'small_prior': small_prior_loss,
            'routing': routing_loss,
            'roi': roi_loss,
            'mix': mix_loss,
            'thin': thin_loss,
        }
