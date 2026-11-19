"""MCOD data pipeline: eight-band .mat cubes (MATLAB v7.3) and binary masks.

Each split directory holds ``Mat/<id>.mat`` and ``GT/<id>.<ext>``. Images are
resized so that the shorter side is ``short_side`` (the longer side is capped at
``max_size``) and zero-padded at the bottom/right to a multiple of
``size_divisor``. The valid extent is returned so that losses and predictions
ignore the padding.
"""
import os
import random
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils import data

GT_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')


def _list_files(root: str, suffixes) -> List[str]:
    return sorted([os.path.join(root, f) for f in os.listdir(root) if f.lower().endswith(suffixes)])


def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def match_pairs(mat_dir: str, gt_dir: str) -> List[Tuple[str, str]]:
    """(cube, mask) path pairs with the same file stem, sorted by stem."""
    gt_map = {_stem(p): p for p in _list_files(gt_dir, GT_EXTS)}
    mat_map = {_stem(p): p for p in _list_files(mat_dir, ('.mat',))}
    common = sorted(set(mat_map.keys()) & set(gt_map.keys()))
    if not common:
        raise RuntimeError(f'No matched .mat/GT pairs under {mat_dir} and {gt_dir}')
    return [(mat_map[k], gt_map[k]) for k in common]


def read_gt(path: str) -> np.ndarray:
    with open(path, 'rb') as f:
        gt = Image.open(f).convert('L')
    return np.array(gt, dtype=np.uint8)


def read_cube(path: str, gt_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Read an eight-band cube as a (C, H, W) uint8 array.

    With ``gt_shape`` the spatial axes are matched to the mask. Without it the
    MATLAB v7.3 layout is assumed, which stores the cube column-major so that the
    HDF5 dataset reads as (C, W, H).
    """
    with h5py.File(path, 'r') as f:
        key = 'gray' if 'gray' in f else list(f.keys())[0]
        arr = np.array(f[key])

    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    elif arr.ndim == 3:
        if arr.shape[0] in (1, 3, 4, 8):
            pass
        elif arr.shape[-1] in (1, 3, 4, 8):
            arr = arr.transpose(2, 0, 1)
        else:
            raise RuntimeError(f'Unsupported cube shape {arr.shape} at {path}')
    else:
        raise RuntimeError(f'Unsupported cube ndim {arr.ndim} at {path}')

    if gt_shape is None:
        arr = arr.transpose(0, 2, 1)
    else:
        h, w = gt_shape
        ah, aw = arr.shape[1], arr.shape[2]
        if (ah, aw) != (h, w):
            if (aw, ah) == (h, w):
                arr = arr.transpose(0, 2, 1)
            else:
                raise RuntimeError(f'Cube/GT size mismatch: {arr.shape[1:]} vs {(h, w)} at {path}')

    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _round_up(value: int, divisor: int) -> int:
    return int((value + divisor - 1) // divisor * divisor)


def resize_plan(h: int, w: int, short_side: int = 512, size_divisor: int = 32, max_size: Optional[int] = 2048) -> Dict[str, int]:
    """Short-side resize with bottom/right padding to a multiple of ``size_divisor``."""
    short = max(1, min(h, w))
    long_side = max(h, w)
    scale = float(short_side) / float(short)
    if max_size is not None and int(round(long_side * scale)) > int(max_size):
        scale = float(max_size) / float(long_side)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    pad_h = _round_up(new_h, size_divisor) - new_h
    pad_w = _round_up(new_w, size_divisor) - new_w
    return {'new_h': new_h, 'new_w': new_w, 'pad_top': 0, 'pad_bottom': pad_h, 'pad_left': 0, 'pad_right': pad_w}


def apply_resize_plan(tensor: torch.Tensor, plan: Dict[str, int], pad_value: float = 0.0) -> torch.Tensor:
    """Bilinear resize of a (C, H, W) tensor to the planned size, then zero padding."""
    tensor = F.interpolate(tensor.unsqueeze(0), size=(plan['new_h'], plan['new_w']), mode='bilinear', align_corners=False).squeeze(0)
    return F.pad(
        tensor,
        (plan['pad_left'], plan['pad_right'], plan['pad_top'], plan['pad_bottom']),
        mode='constant',
        value=pad_value,
    )


def valid_mask(plan: Dict[str, int]) -> torch.Tensor:
    out_h = plan['new_h'] + plan['pad_top'] + plan['pad_bottom']
    out_w = plan['new_w'] + plan['pad_left'] + plan['pad_right']
    mask = torch.zeros(1, out_h, out_w, dtype=torch.float32)
    top = plan['pad_top']
    left = plan['pad_left']
    mask[:, top:top + plan['new_h'], left:left + plan['new_w']] = 1.0
    return mask


def cube_to_tensor(cube: np.ndarray, short_side: int = 512, size_divisor: int = 32, max_size: Optional[int] = 2048):
    """uint8 (C, H, W) cube -> resized and padded float tensor in [0, 1] and its resize plan."""
    image_t = torch.from_numpy(np.ascontiguousarray(cube)).float() / 255.0
    plan = resize_plan(image_t.shape[-2], image_t.shape[-1], short_side, size_divisor, max_size)
    return apply_resize_plan(image_t, plan, pad_value=0.0), plan


# Training augmentation (applied to uint8 arrays before resizing).

def random_flip(image: np.ndarray, gt: np.ndarray):
    if random.randint(0, 1) == 1:
        image = image[:, :, ::-1]
        gt = gt[:, ::-1]
    return image, gt


def random_crop(image: np.ndarray, gt: np.ndarray, border: int = 30):
    _, h, w = image.shape
    crop_w = np.random.randint(max(w - border, 1), w + 1)
    crop_h = np.random.randint(max(h - border, 1), h + 1)
    x1 = (w - crop_w) >> 1
    y1 = (h - crop_h) >> 1
    x2 = x1 + crop_w
    y2 = y1 + crop_h
    return image[:, y1:y2, x1:x2], gt[y1:y2, x1:x2]


def random_rotation(image: np.ndarray, gt: np.ndarray):
    if random.random() <= 0.8:
        return image, gt
    angle = int(np.random.randint(-15, 15))
    channels = []
    for c in range(image.shape[0]):
        ch = Image.fromarray(image[c])
        ch = ch.rotate(angle, Image.BICUBIC)
        channels.append(np.array(ch, dtype=np.uint8))
    image = np.stack(channels, axis=0)
    gt = np.array(Image.fromarray(gt).rotate(angle, Image.BICUBIC), dtype=np.uint8)
    return image, gt


def random_pepper(gt: np.ndarray) -> np.ndarray:
    gt = gt.copy()
    noise_num = int(0.0015 * gt.shape[0] * gt.shape[1])
    for _ in range(noise_num):
        rand_x = random.randint(0, gt.shape[0] - 1)
        rand_y = random.randint(0, gt.shape[1] - 1)
        gt[rand_x, rand_y] = 0 if random.randint(0, 1) == 0 else 255
    return gt


class MCODTrainDataset(data.Dataset):
    """Training split: augmented cube, soft mask and valid-extent mask, each (C, H, W)."""

    def __init__(self, split_dir: str, short_side: int = 512, size_divisor: int = 32, max_size: Optional[int] = 2048):
        self.short_side = int(short_side)
        self.size_divisor = int(size_divisor)
        self.max_size = max_size
        self.pairs = match_pairs(os.path.join(split_dir, 'Mat'), os.path.join(split_dir, 'GT'))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        mat_path, gt_path = self.pairs[index]
        gt = read_gt(gt_path)
        image = read_cube(mat_path, gt_shape=gt.shape)

        image, gt = random_flip(image, gt)
        image, gt = random_crop(image, gt)
        image, gt = random_rotation(image, gt)
        gt = random_pepper(gt)

        image_t = torch.from_numpy(np.ascontiguousarray(image)).float() / 255.0
        gt_t = torch.from_numpy(np.ascontiguousarray(gt)).float().unsqueeze(0) / 255.0
        plan = resize_plan(image_t.shape[-2], image_t.shape[-1], self.short_side, self.size_divisor, self.max_size)
        image_t = apply_resize_plan(image_t, plan, pad_value=0.0)
        gt_t = apply_resize_plan(gt_t, plan, pad_value=0.0)
        return image_t, gt_t, valid_mask(plan)


def _pad_to_max(tensors, pad_value: float = 0.0) -> torch.Tensor:
    max_h = max(int(t.shape[-2]) for t in tensors)
    max_w = max(int(t.shape[-1]) for t in tensors)
    padded = []
    for t in tensors:
        pad_h = max_h - int(t.shape[-2])
        pad_w = max_w - int(t.shape[-1])
        padded.append(F.pad(t, (0, pad_w, 0, pad_h), mode='constant', value=pad_value))
    return torch.stack(padded, dim=0)


def pad_collate(batch):
    """Zero-pad images, masks and valid masks of a batch to a common size."""
    images, gts, valids = zip(*batch)
    return _pad_to_max(images), _pad_to_max(gts), _pad_to_max(valids)


def build_train_loader(split_dir: str, batch_size: int, short_side: int = 512, size_divisor: int = 32,
                       max_size: Optional[int] = 2048, num_workers: int = 8) -> data.DataLoader:
    dataset = MCODTrainDataset(split_dir, short_side=short_side, size_divisor=size_divisor, max_size=max_size)
    return data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=pad_collate,
    )


class MCODTestDataset(data.Dataset):
    """Test split: (cube (C, H, W), mask normalized to [0, 1] at the original size, id, resize plan)."""

    def __init__(self, split_dir: str, short_side: int = 512, size_divisor: int = 32, max_size: Optional[int] = 2048):
        self.short_side = int(short_side)
        self.size_divisor = int(size_divisor)
        self.max_size = max_size
        self.pairs = match_pairs(os.path.join(split_dir, 'Mat'), os.path.join(split_dir, 'GT'))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        mat_path, gt_path = self.pairs[index]
        gt = read_gt(gt_path)
        image = read_cube(mat_path, gt_shape=gt.shape)
        image_t, plan = cube_to_tensor(image, self.short_side, self.size_divisor, self.max_size)
        gt = gt.astype(np.float32)
        gt = gt / (gt.max() + 1e-8)
        return image_t, gt, _stem(gt_path), plan
