"""Train SARR-Net on the MCOD training split.

Paper protocol: DataParallel over all visible GPUs (three in the paper runs) with a
global batch of 8, 20 epochs, AdamW, element-wise gradient clamping and no use of
the test split during training.

    python train.py --config configs/sarrnet_smt_t.yaml --data_root data/MCOD \
        --pretrained pretrained/smt_tiny.pth --output runs/sarrnet_seed266409173
"""
import argparse
import json
import logging
import os
import random
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from sarrnet.data import build_train_loader
from sarrnet.losses import SARRNetLoss
from sarrnet.models import load_backbone_weights
from sarrnet.utils import build_model, load_config


DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs', 'sarrnet_smt_t.yaml')


def parse_args():
    parser = argparse.ArgumentParser(description='Train SARR-Net on MCOD.')
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument('--data_root', default=None, help='MCOD root with TrainDataset/{Mat,GT}; overrides data.root')
    parser.add_argument('--pretrained', required=True, help='ImageNet SMT-T weights (smt_tiny.pth)')
    parser.add_argument('--output', required=True, help='output directory for logs and checkpoints')
    parser.add_argument('--seed', type=int, default=None, help='overrides train.seed')
    parser.add_argument('--epochs', type=int, default=None, help='overrides train.epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='overrides train.batch_size')
    parser.add_argument('--num_workers', type=int, default=None, help='overrides train.num_workers')
    parser.add_argument('--device', default=None, help='cuda or cpu (default: cuda if available)')
    parser.add_argument('--save_every_epoch', action='store_true', help='also keep epoch_XXX.pth after every epoch')
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def unwrap(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def clip_gradient(optimizer, grad_clip):
    """Clamp every gradient element to [-grad_clip, grad_clip]."""
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def setup_logger(output_dir):
    logger = logging.getLogger('sarrnet')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    for handler in (logging.FileHandler(os.path.join(output_dir, 'train.log')), logging.StreamHandler()):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def main():
    args = parse_args()
    cfg = load_config(args.config)
    tcfg = cfg['train']
    seed = int(args.seed if args.seed is not None else tcfg['seed'])
    epochs = int(args.epochs if args.epochs is not None else tcfg['epochs'])
    batch_size = int(args.batch_size if args.batch_size is not None else tcfg['batch_size'])
    num_workers = int(args.num_workers if args.num_workers is not None else tcfg['num_workers'])
    data_root = args.data_root or cfg['data']['root']

    os.makedirs(args.output, exist_ok=True)
    logger = setup_logger(args.output)
    run_cfg = dict(cfg, run={'seed': seed, 'epochs': epochs, 'batch_size': batch_size, 'num_workers': num_workers,
                             'data_root': data_root, 'pretrained': args.pretrained})
    with open(os.path.join(args.output, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(run_cfg, f, indent=2)

    set_seed(seed)
    cudnn.benchmark = True
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))

    model = build_model(cfg)
    report = load_backbone_weights(model, args.pretrained)
    logger.info('Loaded %s (missing %d, unexpected %d, shape-mismatched %d)', args.pretrained,
                len(report['missing']), len(report['unexpected']), len(report['skipped']))
    model = model.to(device)
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        logger.info('DataParallel over %d GPUs', torch.cuda.device_count())
    logger.info('Parameters: %d', sum(p.numel() for p in unwrap(model).parameters()))

    optimizer = torch.optim.AdamW(
        [
            {'params': list(unwrap(model).backbone_parameters()), 'lr': float(tcfg['lr_backbone'])},
            {'params': list(unwrap(model).other_parameters()), 'lr': float(tcfg['lr_other'])},
        ],
        weight_decay=float(tcfg['weight_decay']),
    )
    criterion = SARRNetLoss.from_config(cfg['loss'])
    dcfg = cfg['data']
    loader = build_train_loader(
        os.path.join(data_root, dcfg['train_split']),
        batch_size=batch_size,
        short_side=int(dcfg['short_side']),
        size_divisor=int(dcfg['size_divisor']),
        max_size=int(dcfg['max_size']),
        num_workers=num_workers,
    )
    total_steps = len(loader)
    logger.info('Training images: %d, steps per epoch: %d', len(loader.dataset), total_steps)

    for epoch in range(1, epochs + 1):
        start = time.time()
        decay = float(tcfg['lr_decay_rate']) ** (epoch // int(tcfg['lr_decay_epoch']))
        optimizer.param_groups[0]['lr'] = float(tcfg['lr_backbone']) * decay
        optimizer.param_groups[1]['lr'] = float(tcfg['lr_other']) * decay
        model.train()
        sums = {}
        for step, (images, gts, valid) in enumerate(loader, start=1):
            images = images.to(device)
            gts = gts.to(device)
            valid = valid.to(device)
            optimizer.zero_grad()
            *logits, aux = model(images, return_aux=True)
            losses = criterion(logits, aux, gts, valid)
            losses['loss'].backward()
            clip_gradient(optimizer, float(tcfg['grad_clip']))
            optimizer.step()
            for key, value in losses.items():
                sums[key] = sums.get(key, 0.0) + float(value.detach().item())
            if step == 1 or step % 100 == 0 or step == total_steps:
                logger.info('epoch %d/%d step %d/%d %s', epoch, epochs, step, total_steps,
                            ' '.join(f'{k}={float(v.detach().item()):.4f}' for k, v in losses.items()))
        logger.info('epoch %d/%d done in %.0fs, mean %s', epoch, epochs, time.time() - start,
                    ' '.join(f'{k}={v / step:.4f}' for k, v in sums.items()))
        checkpoint = {'model': unwrap(model).state_dict(), 'epoch': epoch, 'seed': seed}
        if args.save_every_epoch:
            torch.save(checkpoint, os.path.join(args.output, f'epoch_{epoch:03d}.pth'))
        if epoch == epochs:
            torch.save(checkpoint, os.path.join(args.output, 'sarrnet_final.pth'))
    logger.info('Saved %s', os.path.join(args.output, 'sarrnet_final.pth'))


if __name__ == '__main__':
    main()
