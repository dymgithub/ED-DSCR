from __future__ import annotations

import torch


class SegmentationMetrics:
    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion = torch.zeros(num_classes, num_classes, dtype=torch.float64)

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        prediction, target = prediction.cpu().reshape(-1), target.cpu().reshape(-1)
        valid = target.ne(self.ignore_index) & target.ge(0) & target.lt(self.num_classes)
        bins = target[valid] * self.num_classes + prediction[valid]
        self.confusion += torch.bincount(bins, minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)

    def compute(self) -> dict[str, float | list[float]]:
        tp = self.confusion.diag()
        gt = self.confusion.sum(1)
        pred = self.confusion.sum(0)
        union = gt + pred - tp
        iou = tp / union.clamp_min(1)
        f1 = 2 * tp / (gt + pred).clamp_min(1)
        oa = tp.sum() / self.confusion.sum().clamp_min(1)
        return {"oa": oa.item(), "miou": iou.mean().item(), "mean_f1": f1.mean().item(),
                "iou_per_class": iou.tolist(), "f1_per_class": f1.tolist()}

