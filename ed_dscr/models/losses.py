from __future__ import annotations

import torch
from torch.nn import functional as F


def ddg_loss(normal_pred: torch.Tensor, normal_ref: torch.Tensor,
             lambda_angle: float = 1.0, lambda_norm: float = 0.5) -> torch.Tensor:
    pred_unit = F.normalize(normal_pred, dim=1, eps=1e-6)
    ref_unit = F.normalize(normal_ref, dim=1, eps=1e-6)
    angular = 1.0 - (pred_unit * ref_unit).sum(1)
    norm_offset = (torch.linalg.vector_norm(normal_pred, dim=1) - 1.0).abs()
    return lambda_angle * angular.mean() + lambda_norm * norm_offset.mean()


def total_loss(outputs: dict[str, torch.Tensor], target: torch.Tensor,
               normal_ref: torch.Tensor | None, lambda_ddg: float = 1.0,
               lambda_edar: float = 0.5, ignore_index: int = 255) -> dict[str, torch.Tensor]:
    segmentation = F.cross_entropy(outputs["logits"], target, ignore_index=ignore_index)
    zero = segmentation.new_zeros(())
    geometry = zero if normal_ref is None else ddg_loss(outputs["normal_raw"], normal_ref)
    edar = outputs.get("edar_loss", zero)
    total = segmentation + lambda_ddg * geometry + lambda_edar * edar
    return {"loss": total, "seg": segmentation, "ddg": geometry, "edar": edar}

