from __future__ import annotations

import torch
from torch.nn import functional as F


@torch.no_grad()
def sliding_window_logits(model, image: torch.Tensor, crop_size: int = 256,
                          stride: int = 170) -> torch.Tensor:
    """Overlap-averaged whole-image inference without test-time augmentation."""
    batch, _, height, width = image.shape
    if batch != 1:
        raise ValueError("Sliding-window inference currently expects batch size 1")
    ys = list(range(0, max(height - crop_size, 0) + 1, stride))
    xs = list(range(0, max(width - crop_size, 0) + 1, stride))
    if not ys or ys[-1] != max(height - crop_size, 0): ys.append(max(height - crop_size, 0))
    if not xs or xs[-1] != max(width - crop_size, 0): xs.append(max(width - crop_size, 0))
    logits_sum = count = None
    for y in ys:
        for x in xs:
            crop = image[..., y:y + crop_size, x:x + crop_size]
            crop_h, crop_w = crop.shape[-2:]
            if crop_h < crop_size or crop_w < crop_size:
                crop = F.pad(crop, (0, crop_size - crop_w, 0, crop_size - crop_h))
            logits = model(crop)["logits"][..., :crop_h, :crop_w]
            if logits_sum is None:
                logits_sum = logits.new_zeros((1, logits.shape[1], height, width))
                count = logits.new_zeros((1, 1, height, width))
            logits_sum[..., y:y + crop_h, x:x + crop_w] += logits
            count[..., y:y + crop_h, x:x + crop_w] += 1
    return logits_sum / count.clamp_min(1)
