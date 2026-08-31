from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .common import ConvBNAct


class TinyPyramidBackbone(nn.Module):
    """Small offline-test backbone; not the paper's reported backbone."""
    def __init__(self, channels: tuple[int, ...] = (32, 64, 128, 256)):
        super().__init__()
        stages = []
        in_ch = 3
        for out_ch in channels:
            stages.append(nn.Sequential(ConvBNAct(in_ch, out_ch, stride=2), ConvBNAct(out_ch, out_ch)))
            in_ch = out_ch
        self.stages = nn.ModuleList(stages)
        self.out_channels = list(channels)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        outputs = []
        for stage in self.stages:
            x = stage(x)
            outputs.append(x)
        return outputs


class SegFormerB5Backbone(nn.Module):
    """Hugging Face MiT-B5 feature extractor initialized from ImageNet."""
    def __init__(self, pretrained: bool = True):
        super().__init__()
        try:
            from transformers import SegformerConfig, SegformerModel
        except ImportError as exc:
            raise ImportError("Install transformers to use SegFormer-B5") from exc
        model_id = "nvidia/mit-b5"
        if pretrained:
            self.model = SegformerModel.from_pretrained(model_id)
        else:
            self.model = SegformerModel(SegformerConfig(
                depths=[3, 6, 40, 3],
                hidden_sizes=[64, 128, 320, 512],
                num_attention_heads=[1, 2, 5, 8],
                decoder_hidden_size=768,
            ))
        self.out_channels = list(self.model.config.hidden_sizes)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        result = self.model(pixel_values=x, output_hidden_states=True, return_dict=True)
        return list(result.hidden_states)


class MultiScaleSemanticProjector(nn.Module):
    def __init__(self, in_channels: list[int], channels: int = 256):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Sequential(nn.Conv2d(value, channels, 1, bias=False), nn.BatchNorm2d(channels), nn.ReLU(inplace=True))
            for value in in_channels
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * len(in_channels), channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        size = features[0].shape[-2:]
        projected = [
            F.interpolate(layer(feature), size=size, mode="bilinear", align_corners=False)
            if feature.shape[-2:] != size else layer(feature)
            for feature, layer in zip(features, self.projections)
        ]
        return self.fuse(torch.cat(projected, 1))


def build_backbone(name: str, pretrained: bool = True) -> nn.Module:
    if name == "segformer_b5":
        return SegFormerB5Backbone(pretrained=pretrained)
    if name == "tiny":
        return TinyPyramidBackbone()
    raise ValueError(f"Unknown backbone: {name}")
