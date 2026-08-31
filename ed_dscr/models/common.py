from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvBNAct(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 stride: int = 1, dilation: int = 1, act: bool = True):
        padding = dilation * (kernel_size // 2)
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding,
                      dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        if act:
            layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)


class DepthwiseSeparableConv(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        pad = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_ch, in_ch, kernel_size, padding=pad,
                      groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


def spatial_gradient(x: torch.Tensor) -> torch.Tensor:
    """Forward finite differences, padded back to the input shape."""
    dx = F.pad(x[..., 1:] - x[..., :-1], (0, 1, 0, 0))
    dy = F.pad(x[..., 1:, :] - x[..., :-1, :], (0, 0, 0, 1))
    return torch.cat([dx, dy], dim=1)


def sobel_magnitude(x: torch.Tensor) -> torch.Tensor:
    channels = x.shape[1]
    kx = x.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    ky = x.new_tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    kx = kx.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    ky = ky.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    gx = F.conv2d(x, kx, padding=1, groups=channels)
    gy = F.conv2d(x, ky, padding=1, groups=channels)
    return (gx.square() + gy.square() + 1e-12).sqrt().mean(1, keepdim=True)

