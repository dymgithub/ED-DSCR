# Paper-to-code map

| Manuscript section / equations | Implementation |
|---|---|
| Depth or DSM to reference normals, Eqs. (1)-(5), (34) | `tools/generate_pseudo_depth.py`, `tools/depth_to_normal.py` |
| DDG RGB-to-normal branch, Eq. (6) | `ed_dscr/models/ddg.py::DDG` |
| Angular and norm-offset loss, Eq. (7) | `ed_dscr/models/losses.py::ddg_loss` |
| Sobel/norm calibration, Eqs. (8)-(11) | `ed_dscr/models/ddg.py::DDG.forward` |
| DSMF gates, Eqs. (12)-(14) | `ed_dscr/models/dsmf.py::DSMF.forward` |
| 3/5/7 multi-path mixing, Eqs. (15)-(17) | `ed_dscr/models/dsmf.py::DSMF` |
| EDAR state and response projection, Eqs. (18)-(21) | `ed_dscr/models/edar.py::EDAR` |
| Normalized entropy discrepancy, Eqs. (22)-(23) | `ed_dscr/models/edar.py::normalized_entropy` |
| Task-specific error maps and dual gates, Eqs. (24)-(25) | `ed_dscr/models/edar.py::EDAR.forward` |
| Bidirectional correction, Eqs. (26)-(27) | `ed_dscr/models/edar.py::EDAR.forward` |
| Iterative auxiliary losses, Eqs. (28)-(31) | `ed_dscr/models/edar.py::EDAR.forward` |
| Overall loss, Eqs. (32)-(33) | `ed_dscr/models/losses.py::total_loss` |
| SegFormer-B5 + C=256 | `ed_dscr/models/backbone.py`, dataset YAML files |
| RGB-only test graph | `ed_dscr/models/ed_dscr.py::EDDSCR.forward` |
| Sliding-window evaluation, no TTA | `ed_dscr/engine/inference.py`, `evaluate.py` |

The manuscript figure uses schematic labels `Pred_depth` and `GT_depth`; the code uses three-channel predicted and reference normal representations, matching the method text and equations.
