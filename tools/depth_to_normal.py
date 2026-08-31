from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def depth_to_normal(depth: np.ndarray, normalize_depth: bool = False) -> np.ndarray:
    depth = depth.astype(np.float32)
    if normalize_depth:
        low, high = float(np.nanmin(depth)), float(np.nanmax(depth))
        depth = (depth - low) / max(high - low, 1e-6)
    du = np.zeros_like(depth); dv = np.zeros_like(depth)
    du[:, 1:-1] = depth[:, 2:] - depth[:, :-2]
    dv[1:-1, :] = depth[2:, :] - depth[:-2, :]
    normal = np.stack([-du, -dv, np.ones_like(depth)], axis=0)
    normal /= np.maximum(np.linalg.norm(normal, axis=0, keepdims=True), 1e-6)
    return normal.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True); parser.add_argument("--output-dir", required=True)
    parser.add_argument("--normalize-per-image", action="store_true",
                        help="Required for LoveDA pseudo-depth; normally off for metric DSMs")
    args = parser.parse_args(); source, target = Path(args.input_dir), Path(args.output_dir)
    target.mkdir(parents=True, exist_ok=True)
    for path in sorted(p for p in source.iterdir() if p.suffix.lower() in {".png", ".tif", ".tiff", ".npy"}):
        depth = np.load(path) if path.suffix.lower() == ".npy" else np.asarray(Image.open(path), dtype=np.float32)
        if depth.ndim == 3: depth = depth[..., 0]
        np.save(target / f"{path.stem}.npy", depth_to_normal(depth, args.normalize_per_image))


if __name__ == "__main__": main()

