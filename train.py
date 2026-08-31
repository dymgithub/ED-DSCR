from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ed_dscr.data import RemoteSensingDataset
from ed_dscr.models.ed_dscr import EDDSCR
from ed_dscr.models.losses import total_loss
from ed_dscr.utils.config import load_config
from ed_dscr.utils.seed import seed_everything


def learning_rate(step: int, base: float, minimum: float, warmup: int, total: int) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    progress = (step - warmup) / max(total - warmup, 1)
    return minimum + 0.5 * (base - minimum) * (1 + math.cos(math.pi * progress))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = RemoteSensingDataset(cfg["data"]["root"], "train", cfg["data"]["crop_size"], True, True,
                                   cfg["data"].get("ignore_index", 255))
    loader = DataLoader(dataset, batch_size=cfg["train"]["batch_size"], shuffle=True,
                        num_workers=cfg["train"].get("workers", 4), pin_memory=True, drop_last=True)
    model_cfg = dict(cfg["model"])
    if args.resume:
        model_cfg["pretrained"] = False
    model = EDDSCR(num_classes=cfg["data"]["num_classes"], **model_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"],
                                 betas=tuple(cfg["train"]["betas"]), weight_decay=cfg["train"]["weight_decay"])
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["train"].get("amp", True) and device.type == "cuda")
    output_dir = Path(cfg["train"].get("output_dir", "outputs/default")); output_dir.mkdir(parents=True, exist_ok=True)
    start_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"]); optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = checkpoint["step"] + 1

    iterator = iter(loader)
    model.train()
    progress = tqdm(range(start_step, cfg["train"]["iterations"]), initial=start_step,
                    total=cfg["train"]["iterations"])
    for step in progress:
        try: batch = next(iterator)
        except StopIteration: iterator = iter(loader); batch = next(iterator)
        lr = learning_rate(step, cfg["train"]["lr"], cfg["train"]["min_lr"],
                           cfg["train"]["warmup_iterations"], cfg["train"]["iterations"])
        for group in optimizer.param_groups: group["lr"] = lr
        image, target, normal = (batch[key].to(device, non_blocking=True) for key in ("image", "mask", "normal"))
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            outputs = model(image, target, normal)
            losses = total_loss(outputs, target, normal, cfg["loss"]["lambda_ddg"],
                                cfg["loss"]["lambda_edar"], cfg["data"].get("ignore_index", 255))
        scaler.scale(losses["loss"]).backward(); scaler.step(optimizer); scaler.update()
        progress.set_postfix(loss=f"{losses['loss'].item():.3f}", lr=f"{lr:.2e}")
        if (step + 1) % cfg["train"].get("checkpoint_interval", 5000) == 0:
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "step": step, "config": cfg}, output_dir / f"step_{step + 1}.pth")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "step": cfg["train"]["iterations"] - 1, "config": cfg}, output_dir / "last.pth")


if __name__ == "__main__":
    main()
