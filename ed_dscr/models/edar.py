from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .common import ConvBNAct, spatial_gradient


def normalized_entropy(probability: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    classes = probability.shape[1]
    return -(probability * (probability + eps).log()).sum(1, keepdim=True) / math.log(classes)


class EDAR(nn.Module):
    """Training-only Entropy Discrepancy-Guided Adaptive Reasoning."""
    def __init__(self, channels: int, num_classes: int, iterations: int = 3):
        super().__init__()
        self.iterations = iterations
        self.semantic_response = nn.Conv2d(channels, num_classes, 1)
        self.geometry_response = nn.Conv2d(channels, num_classes, 1)
        self.geometry_decoder = nn.Conv2d(channels, 3, 1)
        self.semantic_gate = nn.Sequential(nn.Conv2d(2, 1, 3, padding=1), nn.Sigmoid())
        self.geometry_gate = nn.Sequential(nn.Conv2d(2, 1, 3, padding=1), nn.Sigmoid())
        self.depth_to_semantic = ConvBNAct(channels * 2, channels)
        self.semantic_to_depth = ConvBNAct(channels * 2, channels)

    def forward(self, semantic: torch.Tensor, geometry: torch.Tensor,
                target: torch.Tensor, normal_ref: torch.Tensor,
                ignore_index: int = 255) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        state_s, state_d = semantic, geometry
        losses_s, losses_d, entropy_maps = [], [], []
        target_small = F.interpolate(
            target[:, None].float(), size=state_s.shape[-2:], mode="nearest"
        ).squeeze(1).long()
        normal_small = F.interpolate(normal_ref, size=state_d.shape[-2:], mode="bilinear", align_corners=False)
        valid = target_small.ne(ignore_index)

        for _ in range(self.iterations):
            logits_s = self.semantic_response(state_s)
            logits_g = self.geometry_response(state_d)
            prob_s = logits_s.softmax(1)
            prob_g = logits_g.softmax(1)
            discrepancy = (normalized_entropy(prob_s) - normalized_entropy(prob_g)).abs()

            safe_target = target_small.masked_fill(~valid, 0)
            semantic_error = -prob_s.gather(1, safe_target[:, None]).clamp_min(1e-8).log()
            semantic_error = semantic_error * valid[:, None]
            normal_pred = self.geometry_decoder(state_d)
            geometry_error = (normal_pred - normal_small).abs().mean(1, keepdim=True)

            gate_s = self.semantic_gate(torch.cat([discrepancy, semantic_error], 1))
            gate_d = self.geometry_gate(torch.cat([discrepancy, geometry_error], 1))
            pair = torch.cat([state_s, state_d], 1)
            state_s = state_s + gate_s * self.depth_to_semantic(pair)
            state_d = state_d + gate_d * self.semantic_to_depth(pair)

            logits_s = self.semantic_response(state_s)
            prob_s = logits_s.softmax(1)
            one_hot = F.one_hot(safe_target, logits_s.shape[1]).permute(0, 3, 1, 2).float()
            boundary_valid = valid[:, None].expand_as(prob_s)
            boundary = (spatial_gradient(prob_s) - spatial_gradient(one_hot)).abs()
            boundary_mask = torch.cat([boundary_valid, boundary_valid], 1)
            boundary_loss = boundary[boundary_mask].mean() if boundary_mask.any() else boundary.sum() * 0
            losses_s.append(F.cross_entropy(logits_s, target_small, ignore_index=ignore_index) + 0.5 * boundary_loss)

            normal_pred = self.geometry_decoder(state_d)
            normal_l1 = (normal_pred - normal_small).abs().mean()
            normal_grad = (spatial_gradient(normal_pred) - spatial_gradient(normal_small)).abs().mean()
            losses_d.append(normal_l1 + 0.5 * normal_grad)
            entropy_maps.append(discrepancy)

        return {
            "loss": torch.stack(losses_s).sum() + torch.stack(losses_d).sum(),
            "semantic_losses": losses_s,
            "geometry_losses": losses_d,
            "entropy_discrepancies": entropy_maps,
        }

