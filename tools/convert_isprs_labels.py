from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


# Official ISPRS order used by common MMSegmentation preparations.
PALETTE = [(255, 255, 255), (0, 0, 255), (0, 255, 255),
           (0, 255, 0), (255, 255, 0), (255, 0, 0)]


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--input-dir", required=True); parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(); source, target = Path(args.input_dir), Path(args.output_dir); target.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.glob("*.tif")) + sorted(source.glob("*.png")):
        rgb = np.asarray(Image.open(path).convert("RGB")); label = np.full(rgb.shape[:2], 255, np.uint8)
        for index, color in enumerate(PALETTE): label[np.all(rgb == color, axis=-1)] = index
        Image.fromarray(label).save(target / f"{path.stem}.png")


if __name__ == "__main__": main()

