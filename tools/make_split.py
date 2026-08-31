from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--image-dir", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args(); stems = sorted({p.stem for p in Path(args.image_dir).iterdir() if p.is_file()})
    Path(args.output).parent.mkdir(parents=True, exist_ok=True); Path(args.output).write_text("\n".join(stems) + "\n")


if __name__ == "__main__": main()
