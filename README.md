# TPQ-Net

Official code and experiment-ready multi-SNR LFM I/Q dataset for:

**TPQ-Net: A Trajectory-Pyramid Quality-Adaptive Metric Network for Low-SNR Radar Specific Emitter Identification**

## Overview

TPQ-Net contains the following paper-aligned components:

- **DCT-3T**: Delay-Conjugate Tri-density Trajectory Tensor
- **DMS-TPE**: Deep Multi-Scale Trajectory Pyramid Extractor
- **LKA**: Large-Kernel Attention
- **HREC**: Hierarchical Residual EMA Calibration
- **CTSF**: Cross-Layer Trajectory Scale Fusion
- **QGBN Neck**: Quality-Gated Batch-Normalization Neck
- **QACM Head**: Quality-Adaptive Cosine Margin Head

The released data are the exact effective-window I/Q samples used in the reported
clean, 30, 20, 15, 10, 5, and 0 dB experiments.

## Repository structure

```text
TPQ-Net/
├── README.md
├── requirements.txt
├── .gitignore
├── .gitattributes
├── 01_build_dct3.py
├── 02_train_tpqnet.py
├── VALIDATION_REPORT.md
├── data/
│   ├── README.md
│   ├── LFM_SEI_clean_iq.npz
│   ├── LFM_SEI_30dB_iq.npz
│   ├── LFM_SEI_20dB_iq.npz
│   ├── LFM_SEI_15dB_iq.npz
│   ├── LFM_SEI_10dB_iq.npz
│   ├── LFM_SEI_5dB_iq.npz
│   ├── LFM_SEI_0dB_iq.npz
│   ├── split_622.npz
│   ├── dataset_manifest.json
│   ├── dataset_manifest.csv
│   └── SHA256SUMS.txt
├── generated/
│   └── dct3/
└── outputs/
```

## Dataset protocol

The dataset contains six emitter classes and 1,000 samples per emitter, for 6,000
samples at every signal condition.

For the six noisy conditions, the data generation order was:

```text
complete selected clean pulse
→ define SNR using the complete-pulse mean power
→ add complex AWGN to the complete pulse
→ extract the predefined effective signal window
→ zero-pad variable-length windows
→ save the experiment-ready I/Q dataset
```

Therefore, the target SNR is defined on the complete pulse. The SNR measured again
inside the extracted effective window is higher because signal energy is concentrated
within the effective region.

## Environment

Python 3.10 or later is recommended.

Install the PyTorch build that matches the local CUDA environment, then run:

```bash
pip install -r requirements.txt
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

Default training settings:

```text
optimizer = AdamW
learning rate = 1e-3
weight decay = 1e-4
batch size = 32
epochs = 60
gradient clipping = 5.0
normalization = per-channel standard-deviation normalization
```

## Reported accuracy

| Condition | Accuracy |
|---|---:|
| Clean | 99.00% |
| 30 dB | 98.33% |
| 20 dB | 95.92% |
| 15 dB | 93.08% |
| 10 dB | 87.58% |
| 5 dB | 78.42% |
| 0 dB | 64.17% |
| Average | 88.07% |

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
