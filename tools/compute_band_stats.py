"""Per-band p1/p99 percentiles of the training cubes (normalization of Eq. (1)).

Accumulates a 256-bin histogram per band over all training cubes and takes the
first bin whose cumulative count reaches 1% and 99% of the pixels; values are
divided by 255.

    python tools/compute_band_stats.py --mat_dir data/MCOD/TrainDataset/Mat --output configs/mcod_band_stats.json
"""
import argparse
import json
import os

import h5py
import numpy as np


def read_cube_uint8(path):
    with h5py.File(path, 'r') as f:
        key = 'gray' if 'gray' in f else list(f.keys())[0]
        arr = np.array(f[key])
    if arr.ndim == 2:
        arr = arr[None, :, :]
    elif arr.ndim == 3 and arr.shape[0] not in (1, 3, 8) and arr.shape[-1] in (1, 3, 8):
        arr = arr.transpose(2, 0, 1)
    if arr.shape[0] != 8:
        raise RuntimeError(f'Expected an 8-band cube, got {arr.shape} from {path}')
    return np.clip(arr, 0, 255).astype(np.uint8)


def percentile_from_hist(hist, q):
    total = int(hist.sum())
    if total <= 0:
        return 0
    cdf = np.cumsum(hist)
    return int(np.searchsorted(cdf, total * float(q), side='left'))


def compute_band_stats(mat_dir):
    files = sorted(os.path.join(mat_dir, n) for n in os.listdir(mat_dir) if n.lower().endswith('.mat'))
    if not files:
        raise RuntimeError(f'No .mat files found under {mat_dir}')
    hist = np.zeros((8, 256), dtype=np.int64)
    for path in files:
        arr = read_cube_uint8(path)
        for ch in range(8):
            hist[ch] += np.bincount(arr[ch].reshape(-1), minlength=256)
    p1, p99 = [], []
    for ch in range(8):
        lo = percentile_from_hist(hist[ch], 0.01)
        hi = percentile_from_hist(hist[ch], 0.99)
        if hi <= lo:
            hi = min(255, lo + 1)
        p1.append(float(lo) / 255.0)
        p99.append(float(hi) / 255.0)
    return {'p1': p1, 'p99': p99}


def main():
    parser = argparse.ArgumentParser(description='Compute per-band p1/p99 of the MCOD training cubes.')
    parser.add_argument('--mat_dir', required=True, help='folder with the training .mat cubes')
    parser.add_argument('--output', required=True, help='output JSON file')
    args = parser.parse_args()
    stats = compute_band_stats(args.mat_dir)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats))


if __name__ == '__main__':
    main()
