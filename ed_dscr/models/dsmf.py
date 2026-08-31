from __future__ import annotations

import torch
from torch import nn

from .common import DepthwiseSeparableConv


class DSMF(nn.Module):
    """Depth-Semantic Multi-Path Fusion, Eqs. (12)-(17)."""
    def __init__(self, channels: int = 256):
        super().__init__()
        pair_channels = channels * 2
        self.semantic_gate = DepthwiseSeparableConv(pair_channels, channels, 3)
        self.geometry_gate = DepthwiseSeparableConv(pair_channels, channels, 3)
        self.branches = nn.ModuleList([
            DepthwiseSeparableConv(pair_channels, channels, kernel) for kernel in (3, 5, 7)
        ])
        self.mix = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, semantic: torch.Tensor, geometry: torch.Tensor) -> torch.Tensor:
        pair = torch.cat([semantic, geometry], 1)
        semantic_gated = semantic * torch.sigmoid(self.semantic_gate(pair))
        geometry_gated = geometry * torch.sigmoid(self.geometry_gate(torch.cat([geometry, semantic], 1)))
        mixed_input = torch.cat([semantic_gated, geometry_gated], 1)
        mixed = self.mix(torch.cat([branch(mixed_input) for branch in self.branches], 1))
        return semantic + geometry + mixed
