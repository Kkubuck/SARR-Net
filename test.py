"""Evaluate SARR-Net on the MCOD test split.

Readout of the paper: sum of the four returned logits, crop of the padding,
bilinear resize to the original mask size, sigmoid and per-image min-max
normalization.

    python test.py --config configs/sarrnet_smt_t.yaml --data_root data/MCOD \
        --checkpoint runs/sarrnet_seed266409173/sarrnet_final.pth --output results/sarrnet --save_masks
"""
import argparse
import json
import os

import numpy as np
import torch
from PIL import Image

from sarrnet.data import MCODTestDataset
from sarrnet.evaluation import MCODEvaluator
from sarrnet.utils import build_model, load_config, predict_mask


DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs', 'sarrnet_smt_t.yaml')


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate SARR-Net on the MCOD test split.')
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data_root', default=None, help='MCOD root with TestDataset/{Mat,GT}; overrides data.root')
    parser.add_argument('--output', required=True, help='directory for metrics.json (and masks/)')
    parser.add_argument('--save_masks', action='store_true', help='write the normalized predictions as 8-bit PNG files')
    parser.add_argument('--device', default=None, help='cuda or cpu (default: cuda if available)')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    dcfg = cfg['data']
    data_root = args.data_root or dcfg['root']
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    model = build_model(cfg, args.checkpoint).to(device).eval()
    dataset = MCODTestDataset(
        os.path.join(data_root, dcfg['test_split']),
        short_side=int(dcfg['short_side']),
        size_divisor=int(dcfg['size_divisor']),
        max_size=int(dcfg['max_size']),
    )
    os.makedirs(args.output, exist_ok=True)
    mask_dir = os.path.join(args.output, 'masks')
    if args.save_masks:
        os.makedirs(mask_dir, exist_ok=True)

    evaluator = MCODEvaluator()
    for idx in range(len(dataset)):
        image, gt, name, plan = dataset[idx]
        pred = predict_mask(model, image.unsqueeze(0).to(device), plan, gt.shape)
        evaluator.step(pred * 255.0, gt * 255.0)
        if args.save_masks:
            Image.fromarray(np.round(pred * 255.0).astype(np.uint8)).save(os.path.join(mask_dir, name + '.png'))

    metrics = dict(evaluator.results(), num_images=len(dataset), checkpoint=os.path.abspath(args.checkpoint))
    with open(os.path.join(args.output, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    print(' '.join(f'{k}={metrics[k]:.4f}' for k in ('S_m', 'E_adp', 'F_adp', 'F_avg', 'MAE')))


if __name__ == '__main__':
    main()
