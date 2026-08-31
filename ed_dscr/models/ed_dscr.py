from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .backbone import MultiScaleSemanticProjector, build_backbone
from .ddg import DDG
from .dsmf import DSMF
from .edar import EDAR


class EDDSCR(nn.Module):
    """Paper-aligned ED-DSCR model.

    EDAR is invoked only when both training targets are supplied. RGB-only
    inference follows Backbone -> DDG -> DSMF -> Segmentation head.
    """
    def __init__(self, num_classes: int, backbone: str = "segformer_b5",
                 channels: int = 256, edar_iterations: int = 3,
                 pretrained: bool = True, ignore_index: int = 255,
                 ddg_base_channels: int = 32):
        super().__init__()
        self.ignore_index = ignore_index
        self.backbone = build_backbone(backbone, pretrained=pretrained)
        self.semantic_projector = MultiScaleSemanticProjector(self.backbone.out_channels, channels)
        self.ddg = DDG(base_channels=ddg_base_channels)
        self.geometry_projector = nn.Sequential(
            nn.Conv2d(3, channels, 1, bias=False), nn.BatchNorm2d(channels), nn.ReLU(inplace=True)
        )
        self.dsmf = DSMF(channels)
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(channels, num_classes, 1),
        )
        self.edar = EDAR(channels, num_classes, edar_iterations)

    def forward(self, image: torch.Tensor, target: torch.Tensor | None = None,
                normal_ref: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        semantic = self.semantic_projector(self.backbone(image))
        calibrated_normal, raw_normal = self.ddg(image)
        normal_at_feature_scale = F.interpolate(
            calibrated_normal, size=semantic.shape[-2:], mode="bilinear", align_corners=False
        )
        geometry = self.geometry_projector(normal_at_feature_scale)
        fused = self.dsmf(semantic, geometry)
        logits = self.segmentation_head(fused)
        logits = F.interpolate(logits, size=image.shape[-2:], mode="bilinear", align_corners=False)
        output = {
            "logits": logits,
            "normal": calibrated_normal,
            "normal_raw": raw_normal,
            "semantic_features": semantic,
            "fused_features": fused,
        }
        if self.training and target is not None and normal_ref is not None:
            auxiliary = self.edar(semantic, geometry, target, normal_ref, self.ignore_index)
            output["edar_loss"] = auxiliary["loss"]
            output["entropy_discrepancies"] = auxiliary["entropy_discrepancies"]
        return output

