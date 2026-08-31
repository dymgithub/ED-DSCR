from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


PALETTE = [
    [0, 0, 0],
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255],
    [255, 255, 0],
    [255, 255, 255],
]


def make_sample(index: int, size: int, num_classes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y, x = np.mgrid[:size, :size]
    mask = ((x // 16) + (y // 16) + index) % num_classes
    rgb = np.stack(
        [
            (x * 3 + index * 17) % 256,
            (y * 5 + index * 29) % 256,
            ((x + y) * 2 + index * 11) % 256,
        ],
        axis=-1,
    ).astype(np.uint8)

    depth = (
        0.5 * np.sin((x + index) / 11.0)
        + 0.5 * np.cos((y - index) / 13.0)
        + mask.astype(np.float32) / max(num_classes - 1, 1)
    ).astype(np.float32)
    normal = depth_to_normal(depth)
    return rgb, mask.astype(np.uint8), depth, normal


def depth_to_normal(depth: np.ndarray) -> np.ndarray:
    du = np.zeros_like(depth, dtype=np.float32)
    dv = np.zeros_like(depth, dtype=np.float32)
    du[:, 1:-1] = depth[:, 2:] - depth[:, :-2]
    dv[1:-1, :] = depth[2:, :] - depth[:-2, :]
    normal = np.stack([-du, -dv, np.ones_like(depth, dtype=np.float32)], axis=0)
    normal /= np.maximum(np.linalg.norm(normal, axis=0, keepdims=True), 1e-6)
    return normal.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a tiny synthetic dataset for smoke tests.")
    parser.add_argument("--output", default="data/smoke")
    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--num-classes", type=int, default=6)
    args = parser.parse_args()

    root = Path(args.output)
    for name in ("images", "masks", "depths", "normals", "splits"):
        (root / name).mkdir(parents=True, exist_ok=True)

    stems = []
    for index in range(args.num_samples):
        stem = f"sample_{index:03d}"
        stems.append(stem)
        rgb, mask, depth, normal = make_sample(index, args.image_size, args.num_classes)
        Image.fromarray(rgb).save(root / "images" / f"{stem}.png")
        label = Image.fromarray(mask, mode="L")
        flat_palette = sum(PALETTE, [])
        label.putpalette(flat_palette + [0] * (768 - len(flat_palette)))
        label.save(root / "masks" / f"{stem}.png")
        np.save(root / "depths" / f"{stem}.npy", depth)
        np.save(root / "normals" / f"{stem}.npy", normal)

    split = max(1, int(round(args.num_samples * 0.67)))
    (root / "splits" / "train.txt").write_text("\n".join(stems[:split]) + "\n", encoding="utf-8")
    (root / "splits" / "val.txt").write_text("\n".join(stems[split:] or stems[-1:]) + "\n", encoding="utf-8")
    print(f"Wrote smoke dataset to {root}")


if __name__ == "__main__":
    main()
