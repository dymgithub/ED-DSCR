from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .common import ConvBNAct, sobel_magnitude


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(1, keepdim=True)
        maximum = x.amax(1, keepdim=True)
        return x * torch.sigmoid(self.conv(torch.cat([avg, maximum], 1)))


class DRDB(nn.Module):
    """Five-layer dense residual dilated block from Sec. 3.2."""
    def __init__(self, channels: int, growth: int | None = None):
        super().__init__()
        growth = growth or max(channels // 4, 8)
        self.layers = nn.ModuleList([
            ConvBNAct(channels + i * growth, growth, dilation=2)
            for i in range(5)
        ])
        self.aggregate = nn.Conv2d(channels + 5 * growth, channels, 1)
        self.channel_attention = ChannelAttention(channels)
        self.spatial_attention = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        for layer in self.layers:
            features.append(layer(torch.cat(features, 1)))
        y = self.aggregate(torch.cat(features, 1)) + x
        return self.spatial_attention(self.channel_attention(y))


class DDG(nn.Module):
    """RGB-to-normal U-Net with difference-constrained calibration."""
    def __init__(self, base_channels: int = 32):
        super().__init__()
        c = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        self.stem = ConvBNAct(3, c[0])
        self.encoders = nn.ModuleList([DRDB(v) for v in c])
        self.down = nn.ModuleList([ConvBNAct(c[i], c[i + 1], stride=2) for i in range(3)])
        self.up = nn.ModuleList([
            nn.ConvTranspose2d(c[i + 1], c[i], 2, stride=2) for i in reversed(range(3))
        ])
        self.decoders = nn.ModuleList([DRDB(c[i] * 2) for i in reversed(range(3))])
        self.reduce = nn.ModuleList([nn.Conv2d(c[i] * 2, c[i], 1) for i in reversed(range(3))])
        self.normal_head = nn.Conv2d(c[0], 3, 1)
        self.calibration = nn.Conv2d(2, 1, 3, padding=1)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(image)
        skips = []
        for i, block in enumerate(self.encoders):
            x = block(x)
            skips.append(x)
            if i < len(self.down):
                x = self.down[i](x)
        for up, block, reduce, skip in zip(self.up, self.decoders, self.reduce, reversed(skips[:-1])):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = reduce(block(torch.cat([x, skip], 1)))
        normal = torch.tanh(self.normal_head(x))
        edge = sobel_magnitude(normal)
        norm_offset = (torch.linalg.vector_norm(normal, dim=1, keepdim=True) - 1.0).abs()
        gate = torch.sigmoid(self.calibration(torch.cat([edge, norm_offset], 1)))
        return normal * gate, normal

