# Reproducibility notes

This repository is an independent implementation reconstructed from the supplied manuscript. It is not a dump of the authors' unpublished source code and does not include reported checkpoints.

## Fixed by the manuscript

| Item | Value |
|---|---|
| Semantic backbone | ImageNet-pretrained SegFormer-B5 |
| Latent width | 256 |
| Input crop | 256 x 256 |
| Batch size | 8 |
| Optimizer | Adam, betas=(0.9, 0.999), weight decay=1e-4 |
| Initial/minimum LR | 6e-5 / 1e-6 |
| Schedule | 1,500-step linear warm-up + cosine annealing |
| Training length | 80,000 iterations |
| EDAR iterations | 3 |
| DDG loss weights | angular=1.0, norm-offset=0.5 |
| EDAR semantic boundary weight | 0.5 |
| EDAR geometry weights | value=1.0, gradient=0.5 |
| Overall weights | lambda_DDG=1.0, lambda_EDAR=0.5 |
| Augmentation | random crop, horizontal/vertical flip, 90-degree rotations |
| Inference | sliding window, no TTA, RGB only; EDAR removed |

## Engineering choices not uniquely specified

- DRDB growth width, encoder stage widths, attention reduction ratio, and exact down/up-sampling operators.
- Exact convolution widths inside EDAR gate and correction heads.
- Sliding-window overlap/stride. The default 170 gives overlap for a 256 crop and can be changed in YAML.
- Random seed, AMP use, checkpoint cadence, data-loader worker count, and file naming.
- The manuscript says "Depth Anything" while the tooling defaults to the official Depth Anything V2 hub interface. Pin the exact model/revision used for a strict LoveDA comparison.
- Hugging Face `nvidia/mit-b5` is used to supply the ImageNet-pretrained MiT-B5 weights. A strict MMSegmentation comparison may require matching the precise preprocessing and checkpoint lineage.

These choices are configuration-visible so that later author clarification can be incorporated without changing the public API.

## Expected dataset counts from the manuscript

- Vaihingen: 344 prepared training samples, 398 prepared validation samples.
- Potsdam: 3,456 prepared training samples, 2,016 prepared validation samples.
- LoveDA official split: 2,522 train and 1,669 validation images.

## Claims this package does not make

Passing unit tests validates tensor contracts and preprocessing mathematics; it does not reproduce the paper's reported mIoU. Matching the tables additionally requires the same raw data, partitions, preprocessing, pretrained weights, software stack, hardware-sensitive settings, and training runs.

