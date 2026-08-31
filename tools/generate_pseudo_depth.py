"""Offline pseudo-depth generation for LoveDA.

Uses the official Depth Anything V2 repository via torch.hub. Pin a reviewed
revision in production; generated depth never enters ED-DSCR at inference.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--input-dir", required=True); parser.add_argument("--output-dir", required=True)
    parser.add_argument("--encoder", default="vitl", choices=("vits", "vitb", "vitl", "vitg"))
    args = parser.parse_args(); target = Path(args.output_dir); target.mkdir(parents=True, exist_ok=True)
    model = torch.hub.load("DepthAnything/Depth-Anything-V2", "DepthAnythingV2", encoder=args.encoder, pretrained=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model = model.to(device).eval()
    paths = [p for p in Path(args.input_dir).iterdir() if p.suffix.lower() in {".png", ".jpg", ".tif", ".tiff"}]
    for path in tqdm(sorted(paths)):
        image = cv2.imread(str(path)); depth = model.infer_image(image).astype(np.float32)
        low, high = float(depth.min()), float(depth.max()); depth = (depth - low) / max(high - low, 1e-6)
        np.save(target / f"{path.stem}.npy", depth)


if __name__ == "__main__": main()

