from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ed_dscr.data.dataset import IMAGENET_MEAN, IMAGENET_STD
from ed_dscr.engine.inference import sliding_window_logits
from ed_dscr.models.ed_dscr import EDDSCR
from ed_dscr.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True); parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args(); cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = dict(cfg["model"]); model_cfg["pretrained"] = False
    model = EDDSCR(num_classes=cfg["data"]["num_classes"], **model_cfg).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu")["model"]); model.eval()
    rgb = np.asarray(Image.open(args.image).convert("RGB"), dtype=np.float32) / 255.0
    image = torch.from_numpy(rgb).permute(2, 0, 1); image = ((image - IMAGENET_MEAN) / IMAGENET_STD)[None].to(device)
    logits = sliding_window_logits(model, image, cfg["data"]["crop_size"], cfg["data"].get("stride", 170))
    label = logits.argmax(1)[0].byte().cpu().numpy()
    output = Image.fromarray(label, mode="L")
    palette = sum(cfg["data"]["palette"], []); output.putpalette(palette + [0] * (768 - len(palette)))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True); output.save(args.output)


if __name__ == "__main__": main()
