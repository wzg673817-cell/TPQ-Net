# TPQ-Net

Official code and experiment-ready multi-SNR LFM I/Q dataset for:

**TPQ-Net: A Trajectory-Pyramid Quality-Adaptive Metric Network for Low-SNR Radar Specific Emitter Identification**

## Overview

Radar specific emitter identification (SEI) aims to distinguish individual emitters
of the same type from subtle hardware-induced fingerprints embedded in received
signals. Under low signal-to-noise ratio (SNR) conditions, these fine-grained
amplitude and phase deviations are easily overwhelmed by noise, while conventional
raw-I/Q models and generic image backbones often struggle to preserve both robust
trajectory structure and inter-class separability.

**TPQ-Net** addresses this problem through a unified trajectory-representation and
quality-adaptive metric-learning framework. First, the proposed
**Delay-Conjugate Tri-density Trajectory Tensor (DCT-3T)** converts each complex LFM
I/Q sample into a three-channel 2D tensor that jointly describes global trajectory
density, local segment density, and temporally ordered segment density. This makes
weak hardware-induced differences more spatially observable than in the original
one-dimensional waveform.

The resulting DCT-3T tensor is processed by an LKA-enhanced
**Deep Multi-Scale Trajectory Pyramid Extractor (DMS-TPE)**, which models local,
mid-scale, and global trajectory patterns through parallel dilated convolutions and
large-kernel attention. **Hierarchical Residual EMA Calibration (HREC)** then
adaptively refines Stage 3 and Stage 4 features, while
**Cross-Layer Trajectory Scale Fusion (CTSF)** combines mid-level fingerprint
details with high-level semantic representations. At the classification stage, the
**Quality-Gated Batch-Normalization Neck (QGBN Neck)** balances raw and normalized
embeddings, and the **Quality-Adaptive Cosine Margin Head (QACM Head)** adjusts the
classification margin according to sample quality.

This repository provides the paper-aligned implementation of DCT-3T and TPQ-Net,
together with the exact experiment-ready LFM I/Q datasets used under clean, 30, 20,
15, 10, 5, and 0 dB conditions. On the six-class LFM dataset, TPQ-Net achieves an
average recognition accuracy of **88.07%** across all seven signal conditions and
**76.72%** over the 10, 5, and 0 dB low-SNR conditions.

## Repository structure

```text
TPQ-Net/
├── README.md
├── LICENSE
├── .gitignore
├── 01_build_dct3.py
├── 02_train_tpqnet.py
├── data/
│   ├── README.md
│   └── split_622.npz
├── generated/                 # created locally by 01_build_dct3.py
│   └── dct3/
└── outputs/
```
## Dataset protocol

The dataset contains six emitter classes and 1,000 samples per emitter, for 6,000
samples at every signal condition.

For each emitter, the first 600 samples are used for training, the next 200 samples
for validation, and the final 200 samples for testing. This fixed 6:2:2 split yields
3,600 training samples, 1,200 validation samples, and 1,200 test samples in total.

## Environment

```text
pytorch:2.3.0-cuda12.1-python3.10-ubuntu22.04-v09
```
## Build DCT-3T

The default command builds DCT-3T for the clean dataset:

```bash
python 01_build_dct3.py
```

For another SNR condition:

```bash
python 01_build_dct3.py \
  --in_npz ./data/LFM_SEI_0dB_iq.npz \
  --split_npz ./data/split_622.npz \
  --out_dir ./generated/dct3
```

The paper configuration is:

```text
delay tau = 2
grid size = 128
displacement stride = 1
```

The coordinate range is estimated using the training split only.

## Train TPQ-Net

Clean condition:

```bash
python 02_train_tpqnet.py
```

Example for 0 dB:

```bash
python 02_train_tpqnet.py \
  --data_npz ./generated/dct3/LFM_SEI_0dB_iq__dct3_tau2_grid128_disp1.npz \
  --split_npz ./data/split_622.npz \
  --out_dir ./outputs/0dB
```

### Model configuration

| Parameter | Value |
|---|---:|
| Front-end stem channels | 16 |
| DMS-TPE stage channels | [64, 128, 192, 256] |
| DMS-TPE stage depths | [1, 2, 2, 2] |
| Dilated branch rates | [1, 2, 4] |
| Embedding dimension | 256 |
| Dropout | 0.10 |
| QGBN hidden dimension | 64 |
| QGBN gate initialization bias | -2.0 |
| QGBN gate upper bound | 0.75 |
| QACM scaling factor | 24.0 |
| QACM base margin | 0.12 |
| QACM adaptive strength | 0.15 |
| QACM margin range | [0.04, 0.20] |
| QACM quality clipping bound | 1.0 |
| QACM quality-statistics EMA | 0.99 |
| HREC groups | 8 |
| HREC gamma initialization | 0.0 |
| HREC gamma upper bound (Stage 3 / Stage 4) | 0.25 / 0.50 |

### Training configuration

| Hyperparameter | Value |
|---|---:|
| Random seed | 42 |
| Optimizer | AdamW |
| Learning rate | 1 × 10^-3 |
| Weight decay | 1 × 10^-4 |
| Batch size | 32 |
| Training epochs | 60 |
| Learning-rate schedule | Cosine annealing |
| Gradient clipping | 5.0 |
| Data normalization | Per-channel standard-deviation normalization |
| Loss function | Cross-entropy |
| Best-checkpoint criterion | Validation balanced accuracy |
## Reported performance

| SNR (dB) | Accuracy | Precision | F1 Score | Kappa |
|---|---:|---:|---:|---:|
| Clean | 0.9900 | 0.9902 | 0.9900 | 0.9880 |
| 30 | 0.9833 | 0.9835 | 0.9833 | 0.9800 |
| 20 | 0.9592 | 0.9593 | 0.9592 | 0.9510 |
| 15 | 0.9308 | 0.9313 | 0.9307 | 0.9170 |
| 10 | 0.8758 | 0.8806 | 0.8804 | 0.8570 |
| 5 | 0.7842 | 0.7813 | 0.7860 | 0.7450 |
| 0 | 0.6417 | 0.6429 | 0.6385 | 0.5670 |
| **Average** | **0.8807** | **0.8813** | **0.8812** | **0.8579** |

Small numerical differences may occur across GPU models, CUDA/cuDNN versions, and
PyTorch versions despite deterministic settings.
## Data integrity

Use the hashes in `data/SHA256SUMS.txt` to verify the downloaded files. A complete
consistency check is documented in `VALIDATION_REPORT.md`.

## Citation

The BibTeX entry will be added after the paper is formally published. Until then,
please cite the manuscript title shown above.

## License

The code is released under the MIT License. The dataset is intended for academic
research use; see `DATA_USE_NOTICE.md`.
