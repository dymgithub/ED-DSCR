from __future__ import annotations

import argparse
import json

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ed_dscr.data import RemoteSensingDataset
from ed_dscr.engine.inference import sliding_window_logits
from ed_dscr.engine.metrics import SegmentationMetrics
from ed_dscr.models.ed_dscr import EDDSCR
from ed_dscr.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True); parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args(); cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = dict(cfg["model"]); model_cfg["pretrained"] = False
    model = EDDSCR(num_classes=cfg["data"]["num_classes"], **model_cfg).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu")["model"]); model.eval()
    dataset = RemoteSensingDataset(cfg["data"]["root"], "val", cfg["data"]["crop_size"], False, False,
                                   cfg["data"].get("ignore_index", 255))
    loader = DataLoader(dataset, batch_size=1, num_workers=cfg["train"].get("workers", 4))
    metrics = SegmentationMetrics(cfg["data"]["num_classes"], cfg["data"].get("ignore_index", 255))
    for batch in tqdm(loader):
        logits = sliding_window_logits(model, batch["image"].to(device), cfg["data"]["crop_size"],
                                       cfg["data"].get("stride", 170))
        metrics.update(logits.argmax(1), batch["mask"])
    print(json.dumps(metrics.compute(), indent=2))


if __name__ == "__main__": main()
