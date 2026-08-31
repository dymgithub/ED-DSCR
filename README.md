# ED-DSCR Reproduction

Independent PyTorch implementation of **Entropy Discrepancy-Guided Depth-Spatial Correlation Reasoning for Remote Sensing Semantic Segmentation**.

This repository is a research reproduction, not the authors' official code. It follows the architecture equations, loss terms, and training constants stated in the manuscript. Places where the paper does not uniquely specify an engineering choice are documented in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Highlights

- RGB-only inference: no DSM, pseudo-depth, normals, or ground-truth residuals are required at test time.
- Training-time depth supervision: DSMs or offline pseudo-depth maps are converted to reference normals.
- Implemented modules: DDG, DSMF, EDAR, SegFormer-B5 backbone wrapper, sliding-window evaluation, preprocessing tools, and smoke tests.
- Reproducibility notes: dataset layout, fixed hyperparameters, underspecified choices, and paper-to-code mapping are included.

## Repository Layout

```text
ED-DSCR-reproduction/
|-- configs/                  # Vaihingen, Potsdam, LoveDA, and smoke-test YAML files
|-- ed_dscr/
|   |-- data/                 # RGB/mask/normal dataset
|   |-- engine/               # Sliding-window inference and metrics
|   |-- models/               # Backbone, DDG, DSMF, EDAR, and losses
|   `-- utils/
|-- tools/                    # Dataset conversion, depth/normal generation, smoke data
|-- tests/                    # Unit tests for tensor contracts and preprocessing
|-- train.py
|-- evaluate.py
|-- predict.py
|-- PAPER_IMPLEMENTATION_MAP.md
`-- REPRODUCIBILITY.md
```

The repository intentionally excludes datasets, checkpoints, experiment logs, and generated predictions. Those paths are ignored by Git.

## Installation

The paper reports PyTorch 1.12.0 and an RTX 4090. Newer PyTorch versions should also work, but strict paper comparison should pin the complete environment.

```bash
git clone https://github.com/dymgithub/ED-DSCR-reproduction.git
cd ED-DSCR-reproduction
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
```

For development and tests:

```bash
pip install -e ".[dev]"
```

For LoveDA pseudo-depth generation:

```bash
pip install -e ".[depth]"
```

Install the CUDA-enabled PyTorch wheel that matches your machine from the official PyTorch instructions before long training runs.

## Quick Smoke Test

Use the synthetic smoke dataset to verify the training/evaluation/prediction pipeline before preparing real data:

```bash
python tools/create_smoke_dataset.py --output data/smoke --num-samples 6
python train.py --config configs/smoke_test.yaml
python evaluate.py --config configs/smoke_test.yaml --checkpoint outputs/smoke/last.pth
python predict.py --config configs/smoke_test.yaml --checkpoint outputs/smoke/last.pth \
  --image data/smoke/images/sample_000.png --output outputs/smoke/pred_sample_000.png
```

The smoke configuration uses a deliberately small backbone and two training iterations. It only checks that the implementation is wired correctly; it is not a paper result.

## Dataset Preparation

Datasets are not redistributed. Prepare each dataset with this layout:

```text
data/vaihingen/
|-- images/<sample>.(png|jpg|tif|tiff)
|-- masks/<sample>.png          # one-channel class indices
|-- depths/<sample>.(tif|npy)   # DSM or offline pseudo-depth for training preparation
|-- normals/<sample>.npy        # float32 array, shape 3xHxW or HxWx3
`-- splits/
    |-- train.txt
    `-- val.txt
```

For ISPRS color labels and DSMs:

```bash
python tools/convert_isprs_labels.py --input-dir RAW_LABELS --output-dir data/vaihingen/masks
python tools/depth_to_normal.py --input-dir data/vaihingen/depths --output-dir data/vaihingen/normals
```

If you already have a directory of images and need a simple stem list:

```bash
python tools/make_split.py --image-dir data/vaihingen/images --output data/vaihingen/splits/train.txt
```

For LoveDA, generate pseudo-depth offline, then convert it to normals with per-image normalization:

```bash
python tools/generate_pseudo_depth.py --input-dir data/loveda/images --output-dir data/loveda/depths
python tools/depth_to_normal.py --input-dir data/loveda/depths --output-dir data/loveda/normals --normalize-per-image
```

Use the official dataset splits for reported comparisons. Do not create train/validation partitions indiscriminately with `make_split.py`.

## Training

```bash
python train.py --config configs/vaihingen.yaml
```

Resume from a checkpoint:

```bash
python train.py --config configs/vaihingen.yaml --resume outputs/vaihingen/last.pth
```

The default experiment configs use the manuscript constants: 256x256 crops, batch size 8, Adam, 1,500 warm-up iterations, cosine decay, and 80,000 training iterations.

## Evaluation And Prediction

```bash
python evaluate.py --config configs/vaihingen.yaml --checkpoint outputs/vaihingen/last.pth
```

```bash
python predict.py --config configs/vaihingen.yaml --checkpoint outputs/vaihingen/last.pth \
  --image path/to/image.tif --output outputs/example.png
```

Evaluation and prediction call the RGB-only inference graph. EDAR and ground-truth-derived discrepancy signals are used only during training.

## Tests

```bash
pytest -q
python -m compileall -q ed_dscr tools train.py evaluate.py predict.py
```

## Citation

Please cite the ED-DSCR manuscript if this reproduction supports your work. Replace the BibTeX entry below once official bibliographic metadata is available.

```bibtex
@misc{deng2026eddscr,
  title  = {Entropy Discrepancy-Guided Depth-Spatial Correlation Reasoning for Remote Sensing Semantic Segmentation},
  author = {Deng, Yiming and Liu, Hongning and Sun, Hui and Wang, Guanglu and Liu, Xinyue},
  year   = {2026}
}
```

## License

This independent implementation is released under the MIT License. Dataset licenses, pretrained SegFormer weights, and Depth Anything weights retain their original terms.
