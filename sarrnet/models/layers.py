"""Building blocks shared by the SARR-Net modules."""
from typing import Optional

import torch
import torch.nn as nn

EPS = 1e-6


def gn_groups(channels: int) -> int:
    return max(1, min(8, channels // 8))


def safe_normalized_difference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a - b) / (a + b + EPS)


def safe_log_ratio(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.clamp(a, min=EPS)) - torch.log(torch.clamp(b, min=EPS))


def summarize_tokens(tokens: torch.Tensor) -> torch.Tensor:
    """Mean over the token axis of a (B, N, C) token set."""
    return tokens.mean(dim=1)


class ConvGNAct(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1, padding: Optional[int] = None):
        if padding is None:
            padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False),
            nn.GroupNorm(gn_groups(out_ch), out_ch),
            nn.GELU(),
        )


class ResidualBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvGNAct(ch, ch, 3, 1),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.GroupNorm(gn_groups(ch), ch),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))
