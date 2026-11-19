"""Predict masks for a folder of eight-band .mat cubes with a trained SARR-Net.

Uses the same readout as test.py and writes one 8-bit PNG per cube at the cube's
original resolution.

    python tools/predict.py --checkpoint runs/sarrnet_seed266409173/sarrnet_final.pth \
        --input data/MCOD/TestDataset/Mat --output predictions
"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sarrnet.data import cube_to_tensor, read_cube  # noqa: E402
from sarrnet.data.mcod import GT_EXTS  # noqa: E402
from sarrnet.utils import build_model, load_config, predict_mask  # noqa: E402


def parse_args():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description='Predict SARR-Net masks for a folder of .mat cubes.')
    parser.add_argument('--config', default=os.path.join(repo, 'configs', 'sarrnet_smt_t.yaml'))
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--input', required=True, help='folder of .mat cubes')
    parser.add_argument('--output', required=True, help='folder for the PNG masks')
    parser.add_argument('--gt_dir', default=None,
                        help='optional mask folder; when given, each cube is oriented to its mask as in test.py')
    parser.add_argument('--device', default=None, help='cuda or cpu (default: cuda if available)')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    dcfg = cfg['data']
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    model = build_model(cfg, args.checkpoint).to(device).eval()
    os.makedirs(args.output, exist_ok=True)

    names = sorted(f for f in os.listdir(args.input) if f.lower().endswith('.mat'))
    for fname in names:
        stem = os.path.splitext(fname)[0]
        gt_shape = None
        if args.gt_dir is not None:
            for ext in GT_EXTS:
                gt_path = os.path.join(args.gt_dir, stem + ext)
                if os.path.isfile(gt_path):
                    with Image.open(gt_path) as gt_img:
                        gt_shape = (gt_img.height, gt_img.width)
                    break
        cube = read_cube(os.path.join(args.input, fname), gt_shape=gt_shape)
        image, plan = cube_to_tensor(cube, int(dcfg['short_side']), int(dcfg['size_divisor']), int(dcfg['max_size']))
        pred = predict_mask(model, image.unsqueeze(0).to(device), plan, cube.shape[1:])
        Image.fromarray(np.round(pred * 255.0).astype(np.uint8)).save(os.path.join(args.output, stem + '.png'))
        print(stem)


if __name__ == '__main__':
    main()
