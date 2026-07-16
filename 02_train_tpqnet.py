# -*- coding: utf-8 -*-
"""
Train and evaluate TPQ-Net for low-SNR radar specific emitter identification.

The implementation follows the module names used in the paper:

    DCT-3T input
      -> DMS-TPE backbone with Large-Kernel Attention (LKA)
      -> Hierarchical Residual EMA Calibration (HREC) on Stages 3 and 4
      -> Cross-Layer Trajectory Scale Fusion (CTSF)
      -> Quality-Gated Batch-Normalization Neck (QGBN Neck)
      -> Quality-Adaptive Cosine Margin Head (QACM Head)
      -> emitter identity logits

Default paper configuration
---------------------------
- DMS-TPE channels: [64, 128, 192, 256]
- DMS-TPE depths: [1, 2, 2, 2]
- dilation rates: [1, 2, 4]
- HREC bounds: Stage 3 = 0.25, Stage 4 = 0.50
- QGBN gate upper bound: 0.75
- QACM scale/base margin/range: 24.0 / 0.12 / [0.04, 0.20]
- optimizer: AdamW, learning rate 1e-3, weight decay 1e-4
- epochs/batch size: 60 / 32

Run this script from the repository root.
"""

from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONHASHSEED", "42")

import argparse
import json
import random
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# =====================================================================================
# Default protocol
# =====================================================================================

DEFAULT_DATA_NPZ = "./generated/dct3/LFM_SEI_clean_iq__dct3_tau2_grid128_disp1.npz"
DEFAULT_SPLIT_NPZ = "./data/split_622.npz"
DEFAULT_OUT_DIR = "./outputs/clean"


@dataclass
class RunConfig:
    data_npz: str
    split_npz: str
    out_dir: str

    seed: int
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    num_workers: int

    use_amp: bool
    max_grad_norm: float
    save_every: int
    label_smoothing: float

    frontend_stem_channels: int
    stage_channels: Tuple[int, ...]
    stage_depths: Tuple[int, ...]
    branch_dilations: Tuple[int, ...]
    dropout: float
    fc_dim: int
    neck_type: str
    qgbn_hidden: int
    qgbn_init_bias: float
    qgbn_gate_max: float
    normalization: str

    use_hrec: bool
    hrec_groups: int
    hrec_gamma_init: float
    hrec_stage3_gamma_max: float
    hrec_stage4_gamma_max: float

    qacm_scale: float
    qacm_base_margin: float
    qacm_strength: float
    qacm_quality_ema: float
    qacm_quality_clip: float
    qacm_min_margin: float
    qacm_max_margin: float

    # Paper modules
    use_lka: bool
    use_ctsf: bool


# =====================================================================================
# Utility
# =====================================================================================

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch, "set_float32_matmul_precision"):
        try:
            torch.set_float32_matmul_precision("highest")
        except Exception:
            pass
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def save_json(obj: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, cls=NpEncoder)


def torch_load_compat(path, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def make_grad_scaler(use_amp: bool, device: torch.device):
    enabled = bool(use_amp and device.type == "cuda")
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    if hasattr(torch, "cuda") and hasattr(torch.cuda.amp, "GradScaler"):
        return torch.cuda.amp.GradScaler(enabled=enabled)
    return None


def amp_autocast_cuda(enabled: bool):
    if not enabled:
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        try:
            return torch.amp.autocast(device_type="cuda", enabled=True)
        except TypeError:
            return torch.amp.autocast("cuda", enabled=True)
    if hasattr(torch, "cuda") and hasattr(torch.cuda.amp, "autocast"):
        return torch.cuda.amp.autocast(enabled=True)
    return nullcontext()


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute the metrics reported in the TPQ-Net paper without extra dependencies."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    classes = np.unique(np.concatenate([y_true, y_pred]))

    precisions: List[float] = []
    recalls: List[float] = []
    f1_scores: List[float] = []
    for cls in classes:
        tp = float(np.sum((y_true == cls) & (y_pred == cls)))
        fp = float(np.sum((y_true != cls) & (y_pred == cls)))
        fn = float(np.sum((y_true == cls) & (y_pred != cls)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    accuracy = float(np.mean(y_true == y_pred))
    balanced_accuracy = float(np.mean(recalls)) if recalls else 0.0

    # Cohen's kappa.
    n = float(len(y_true))
    if n > 0:
        true_counts = np.asarray([np.sum(y_true == cls) for cls in classes], dtype=np.float64)
        pred_counts = np.asarray([np.sum(y_pred == cls) for cls in classes], dtype=np.float64)
        expected_agreement = float(np.sum(true_counts * pred_counts) / (n * n))
        kappa = (
            (accuracy - expected_agreement) / (1.0 - expected_agreement)
            if expected_agreement < 1.0 else 0.0
        )
    else:
        kappa = 0.0

    return {
        "acc": accuracy,
        "bAcc": balanced_accuracy,
        "macro_precision": float(np.mean(precisions)) if precisions else 0.0,
        "macro_f1": float(np.mean(f1_scores)) if f1_scores else 0.0,
        "kappa": float(kappa),
    }


def compute_metrics_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    predictions = torch.argmax(logits, dim=1).cpu().numpy()
    labels = targets.cpu().numpy()
    return compute_classification_metrics(labels, predictions)


# =====================================================================================
# Data
# =====================================================================================

def detect_xy_keys(npz_obj: np.lib.npyio.NpzFile) -> Tuple[str, str]:
    keys = set(npz_obj.files)
    x_key = None
    for cand in ["X", "x", "signals", "data"]:
        if cand in keys:
            x_key = cand
            break
    if x_key is None:
        raise KeyError(f"未找到 X/x/signals/data 键，当前 keys={sorted(keys)}")
    y_key = None
    for cand in ["y", "labels", "label"]:
        if cand in keys:
            y_key = cand
            break
    if y_key is None:
        raise KeyError(f"未找到 y/labels/label 键，当前 keys={sorted(keys)}")
    return x_key, y_key


def detect_split_keys(npz_obj: np.lib.npyio.NpzFile) -> Tuple[str, str, str]:
    keys = set(npz_obj.files)
    train_key = next((k for k in ["idx_train", "train_idx"] if k in keys), None)
    val_key   = next((k for k in ["idx_val",   "val_idx"]   if k in keys), None)
    test_key  = next((k for k in ["idx_test",  "test_idx"]  if k in keys), None)
    if train_key is None or val_key is None or test_key is None:
        raise KeyError(f"split 文件缺少 train/val/test 索引键，当前 keys={sorted(keys)}")
    return train_key, val_key, test_key


@dataclass
class LoadedData:
    x: np.ndarray
    y: np.ndarray
    data_key: str
    label_key: str
    extra_keys: List[str]


def load_data_npz(data_npz: str) -> LoadedData:
    obj = np.load(data_npz, allow_pickle=True)
    x_key, y_key = detect_xy_keys(obj)
    x = obj[x_key].astype(np.float32, copy=False)
    y = obj[y_key].astype(np.int64, copy=False)
    if x.ndim != 4:
        raise ValueError(f"X 应为 [N,C,H,W]，当前 shape={x.shape}")
    if x.shape[1] != 3:
        raise ValueError(f"DCT-3T requires three input channels, got C={x.shape[1]}")
    extra_keys = sorted(list(set(obj.files) - {x_key, y_key}))
    return LoadedData(x=x, y=y, data_key=x_key, label_key=y_key, extra_keys=extra_keys)


def load_locked_splits(split_npz: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    obj = np.load(split_npz, allow_pickle=True)
    train_key, val_key, test_key = detect_split_keys(obj)
    return (
        obj[train_key].astype(np.int64, copy=False),
        obj[val_key].astype(np.int64, copy=False),
        obj[test_key].astype(np.int64, copy=False),
    )


def validate_split_indices(
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    idx_test: np.ndarray,
    n_total: int,
) -> None:
    for split_name, idx in [("train", idx_train), ("val", idx_val), ("test", idx_test)]:
        if idx.ndim != 1:
            raise RuntimeError(f"[SPLIT] {split_name} 索引必须是一维，当前 shape={idx.shape}")
        if idx.size == 0:
            raise RuntimeError(f"[SPLIT] {split_name} 为空")
        if np.any(idx < 0) or np.any(idx >= n_total):
            raise RuntimeError(
                f"[SPLIT] {split_name} 存在越界索引：min={idx.min()}, max={idx.max()}, n_total={n_total}"
            )
    train_set = set(idx_train.tolist())
    val_set   = set(idx_val.tolist())
    test_set  = set(idx_test.tolist())
    if (train_set & val_set) or (train_set & test_set) or (val_set & test_set):
        raise RuntimeError("[SPLIT] train/val/test 索引存在重叠，请检查 split 文件。")


def compute_train_channel_stats(x_train: np.ndarray, eps: float = 1e-12) -> Dict[str, np.ndarray]:
    mean = x_train.mean(axis=(0, 2, 3)).astype(np.float32, copy=False)
    std  = x_train.std(axis=(0, 2, 3)).astype(np.float32, copy=False)
    std  = np.maximum(std, eps)
    return {"mean": mean, "std": std}


def apply_channel_normalization_np(
    x: np.ndarray,
    stats: Dict[str, np.ndarray],
    mode: str = "std",
) -> np.ndarray:
    if mode == "none":
        return x.astype(np.float32, copy=False)
    x = x.astype(np.float32, copy=False)
    mean = stats["mean"][None, :, None, None]
    std  = stats["std"][None, :, None, None]
    if mode == "std":
        return (x / std).astype(np.float32, copy=False)
    if mode == "zscore":
        return ((x - mean) / std).astype(np.float32, copy=False)
    raise ValueError(f"不支持的 normalization={mode}，仅支持 none/std/zscore")


class NumpyTensorDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


def make_worker_init_fn(seed: int):
    def _seed_worker(worker_id: int) -> None:
        worker_seed = int(seed) + int(worker_id)
        random.seed(worker_seed)
        np.random.seed(worker_seed % (2**32 - 1))
        torch.manual_seed(worker_seed)
    return _seed_worker


def build_dataloaders(
    x: np.ndarray,
    y: np.ndarray,
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    idx_test: np.ndarray,
    batch_size: int,
    num_workers: int,
    normalization: str,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, np.ndarray]]:
    x_train = x[idx_train]
    x_val   = x[idx_val]
    x_test  = x[idx_test]

    stats       = compute_train_channel_stats(x_train)
    x_train_n   = apply_channel_normalization_np(x_train, stats, mode=normalization)
    x_val_n     = apply_channel_normalization_np(x_val,   stats, mode=normalization)
    x_test_n    = apply_channel_normalization_np(x_test,  stats, mode=normalization)

    ds_train = NumpyTensorDataset(x_train_n, y[idx_train])
    ds_val   = NumpyTensorDataset(x_val_n,   y[idx_val])
    ds_test  = NumpyTensorDataset(x_test_n,  y[idx_test])

    train_generator = torch.Generator()
    train_generator.manual_seed(int(seed))

    common = dict(
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
        worker_init_fn=make_worker_init_fn(int(seed)) if num_workers > 0 else None,
    )
    dl_train = DataLoader(
        ds_train, batch_size=batch_size, shuffle=True,
        drop_last=False, generator=train_generator, **common,
    )
    dl_val  = DataLoader(ds_val,  batch_size=batch_size, shuffle=False, drop_last=False, **common)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False, drop_last=False, **common)
    return dl_train, dl_val, dl_test, stats


# =====================================================================================
# Model — new components
# =====================================================================================

# ──────────────────────────────────────────────────────────────────────────────────────
# ① LKA — Large Kernel Attention (Guo et al., VAN, NeurIPS 2022)
# ──────────────────────────────────────────────────────────────────────────────────────

class LargeKernelAttention(nn.Module):
    """
    Large-Kernel Attention used inside each DMS-TPE block.

    Attention path:
        depthwise 5x5
        -> depthwise 7x7 with dilation 3
        -> pointwise 1x1
        -> element-wise modulation
        -> zero-initialized 1x1 projection
        -> residual addition

    The dilated 7x7 operator has an effective kernel size of 19x19. Because it
    follows a 5x5 depthwise convolution, the theoretical sequential receptive
    field is 23x23, which covers the 16x16 Stage-3/Stage-4 feature maps.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        # 大核深度可分离注意力生成器
        self.dw   = nn.Conv2d(channels, channels, kernel_size=5, padding=2,
                              groups=channels, bias=False)
        self.dw_d = nn.Conv2d(channels, channels, kernel_size=7, padding=9,
                              dilation=3, groups=channels, bias=False)
        self.pw   = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        # 输出投影：zero-init → 训练起点严格等于恒等映射
        self.proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

    def reset_to_near_identity(self) -> None:
        """Zero-init 输出投影：epoch 0 时 proj(x * attn) = 0，输出 = x（恒等）。"""
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 注意力图：[B, C, H, W]，与输入等维，联合通道+空间加权
        attn = self.pw(self.dw_d(self.dw(x)))
        # 外部残差：epoch 0 proj 输出=0，x+0=x（恒等）；之后逐步学习
        return x + self.proj(x * attn)


# =====================================================================================
# Model — backbone components (from original, with LKA added)
# =====================================================================================

class ConvBNAct2D(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        dilation: int = 1,
        activation: bool = True,
    ) -> None:
        super().__init__()
        pad = ((kernel_size - 1) // 2) * dilation
        self.conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=pad,
            dilation=dilation,
            groups=groups,
            bias=False,
        )
        self.bn  = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True) if activation else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DMSTPEBlock(nn.Module):
    """
    多尺度空洞卷积残差块 + LKA 大核注意力。

    结构：
      3 路并行 (1×1 降维 → 3×3 空洞卷积) → concat → 1×1 融合
      → LKA（联合空间-通道大核注意力）→ dropout → + shortcut → ReLU

    LKA 位置：fuse 之后、dropout/残差之前，对已融合的多尺度特征做大感受野校准。
    use_lka=False 时退化为原始无注意力 block（兼容消融实验）。
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        branch_dilations: Sequence[int] = (1, 2, 3),
        dropout: float = 0.0,
        use_lka: bool = True,
    ) -> None:
        super().__init__()
        if len(branch_dilations) < 2:
            raise ValueError("branch_dilations 至少应包含两个尺度")

        n_branch = len(branch_dilations)
        base = out_ch // n_branch
        branch_dims = [base] * n_branch
        branch_dims[-1] += out_ch - sum(branch_dims)

        self.branches = nn.ModuleList([
            nn.Sequential(
                ConvBNAct2D(in_ch, bch, kernel_size=1),
                ConvBNAct2D(bch, bch, kernel_size=3, dilation=d),
            )
            for bch, d in zip(branch_dims, branch_dilations)
        ])

        self.fuse = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        # LKA 大核注意力（zero-init 输出投影，epoch 0 恒等映射）
        self.lka = LargeKernelAttention(out_ch) if use_lka else nn.Identity()

        self.shortcut = (
            nn.Identity()
            if in_ch == out_ch
            else nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        )
        self.act     = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        feats = [branch(x) for branch in self.branches]
        out = torch.cat(feats, dim=1)
        out = self.fuse(out)
        out = self.lka(out)       # ← LKA 大核注意力（VAN NeurIPS 2022）
        out = self.dropout(out)
        out = out + residual
        out = self.act(out)
        return out


class DMSTPEStage(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        depth: int,
        branch_dilations: Sequence[int],
        dropout: float,
        downsample: bool = True,
        use_lka: bool = True,
    ) -> None:
        super().__init__()
        blocks = []
        cur_in = in_ch
        for _ in range(depth):
            blocks.append(
                DMSTPEBlock(
                    in_ch=cur_in,
                    out_ch=out_ch,
                    branch_dilations=branch_dilations,
                    dropout=dropout,
                    use_lka=use_lka,
                )
            )
            cur_in = out_ch
        self.blocks     = nn.Sequential(*blocks)
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2) if downsample else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        x = self.downsample(x)
        return x


class DMSTPEBackbone(nn.Module):
    def __init__(
        self,
        in_channels: int,
        stage_channels: Sequence[int] = (64, 128, 192, 256),
        stage_depths: Sequence[int] = (1, 1, 1, 1),
        branch_dilations: Sequence[int] = (1, 2, 3),
        dropout: float = 0.1,
        use_lka: bool = True,
    ) -> None:
        super().__init__()
        if len(stage_channels) != len(stage_depths):
            raise ValueError("stage_channels 与 stage_depths 长度必须一致")

        self.stem = nn.Sequential(
            ConvBNAct2D(in_channels, stage_channels[0], kernel_size=7),
            ConvBNAct2D(stage_channels[0], stage_channels[0], kernel_size=5),
        )

        stages = []
        cur_in = stage_channels[0]
        for i, (out_ch, depth) in enumerate(zip(stage_channels, stage_depths)):
            stages.append(
                DMSTPEStage(
                    in_ch=cur_in,
                    out_ch=out_ch,
                    depth=depth,
                    branch_dilations=branch_dilations,
                    dropout=dropout if i >= 1 else 0.0,
                    downsample=(i != len(stage_channels) - 1),
                    use_lka=use_lka,
                )
            )
            cur_in = out_ch

        self.stages      = nn.Sequential(*stages)
        self.out_channels = cur_in

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        return x


# =====================================================================================
# Model — classifier heads
# =====================================================================================

class QACMHead(nn.Module):
    """Quality-Adaptive Cosine Margin Head (QACM Head).

    The raw embedding norm is used as a sample-quality proxy. Running quality
    statistics are updated by EMA and mapped to a bounded sample-specific CosFace
    margin, matching Eqs. (11)-(14) in the paper.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        scale: float = 24.0,
        base_margin: float = 0.12,
        qacm_strength: float = 0.15,
        q_ema: float = 0.99,
        q_clip: float = 1.0,
        min_margin: float = 0.04,
        max_margin: float = 0.20,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.scale              = float(scale)
        self.base_margin        = float(base_margin)
        self.qacm_strength  = float(qacm_strength)
        self.q_ema              = float(q_ema)
        self.q_clip             = float(q_clip)
        self.min_margin         = float(min_margin)
        self.max_margin         = float(max_margin)
        self.register_buffer("q_mean",        torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("q_std",         torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("q_initialized", torch.tensor(False, dtype=torch.bool))

    @torch.no_grad()
    def _update_quality_stats(self, q: torch.Tensor) -> None:
        batch_mean = q.mean()
        batch_std  = q.std(unbiased=False).clamp_min(1e-6)
        if not bool(self.q_initialized.item()):
            self.q_mean.copy_(batch_mean)
            self.q_std.copy_(batch_std)
            self.q_initialized.copy_(torch.tensor(True, device=self.q_initialized.device))
        else:
            self.q_mean.mul_(self.q_ema).add_(batch_mean * (1.0 - self.q_ema))
            self.q_std.mul_(self.q_ema).add_(batch_std  * (1.0 - self.q_ema))

    def _compute_adaptive_margin(self, feat: torch.Tensor) -> torch.Tensor:
        q = feat.detach().norm(p=2, dim=1)
        if self.training:
            self._update_quality_stats(q)
            q_mean = q.mean()
            q_std  = q.std(unbiased=False).clamp_min(1e-6)
        else:
            q_mean = self.q_mean.detach()
            q_std  = self.q_std.detach().clamp_min(1e-6)
        q_norm = ((q - q_mean) / q_std).clamp(-self.q_clip, self.q_clip)
        margin = self.base_margin * (1.0 + self.qacm_strength * q_norm)
        return margin.clamp(self.min_margin, self.max_margin).to(device=feat.device, dtype=feat.dtype)

    def forward(
        self,
        feat: torch.Tensor,
        targets: torch.Tensor | None = None,
        margin_feat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        cosine = F.linear(F.normalize(feat, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
        if targets is None:
            return cosine * self.scale
        quality_feat = feat if margin_feat is None else margin_feat
        margin = self._compute_adaptive_margin(quality_feat)
        logits = cosine.clone()
        batch_idx = torch.arange(logits.size(0), device=logits.device)
        logits[batch_idx, targets] = logits[batch_idx, targets] - margin
        return logits * self.scale


# =====================================================================================
# Quality-Gated Batch-Normalization Neck (QGBN Neck)
# =====================================================================================

class QGBNNeck(nn.Module):
    """Quality-Gated Batch-Normalization Neck (QGBN Neck).

    z = z_raw + g * (BN(z_raw) - z_raw), where
    g = qgbn_gate_max * sigmoid(MLP(z_raw)).
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int = 64,
        qgbn_init_bias: float = -2.0,
        qgbn_gate_max: float = 0.75,
    ) -> None:
        super().__init__()
        self.dim        = int(dim)
        hidden_dim      = int(hidden_dim) if int(hidden_dim) > 0 else max(16, int(dim) // 4)
        self.qgbn_gate_max   = float(qgbn_gate_max)
        if self.qgbn_gate_max <= 0.0 or self.qgbn_gate_max > 1.0:
            raise ValueError(f"qgbn_gate_max must be in (0, 1], got {self.qgbn_gate_max}")
        self.bn = nn.BatchNorm1d(dim)
        self.gate_mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim),
        )
        self.reset_gate(float(qgbn_init_bias))

    def reset_gate(self, qgbn_init_bias: float = -2.0) -> None:
        last = self.gate_mlp[-1]
        nn.init.zeros_(last.weight)
        nn.init.constant_(last.bias, float(qgbn_init_bias))

    def forward(self, raw_feat: torch.Tensor) -> torch.Tensor:
        bn_feat = self.bn(raw_feat)
        gate    = self.qgbn_gate_max * torch.sigmoid(self.gate_mlp(raw_feat))
        return raw_feat + gate * (bn_feat - raw_feat)


# =====================================================================================
# HREC internal EMA operator
# =====================================================================================

class EMAAttention2D(nn.Module):
    """Efficient Multi-Scale Attention (EMA) used inside HREC."""

    def __init__(self, channels: int, groups: int = 8) -> None:
        super().__init__()
        channels = int(channels)
        groups   = int(groups)
        groups   = min(groups, channels)
        while channels % groups != 0 and groups > 1:
            groups -= 1

        self.channels      = channels
        self.groups        = groups
        self.group_channels = channels // groups

        self.softmax  = nn.Softmax(dim=-1)
        self.agp      = nn.AdaptiveAvgPool2d((1, 1))
        self.gn       = nn.GroupNorm(self.group_channels, self.group_channels)
        self.conv1x1  = nn.Conv2d(self.group_channels, self.group_channels, kernel_size=1)
        self.conv3x3  = nn.Conv2d(self.group_channels, self.group_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if c != self.channels:
            raise ValueError(f"EMA 输入通道不匹配：expected={self.channels}, got={c}")

        gx  = x.reshape(b * self.groups, self.group_channels, h, w)
        x_h = gx.mean(dim=3, keepdim=True)
        x_w = gx.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)

        hw       = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        att_h, att_w = torch.split(hw, [h, w], dim=2)
        att_w    = att_w.permute(0, 1, 3, 2)

        x1 = self.gn(gx * att_h.sigmoid() * att_w.sigmoid())
        x2 = self.conv3x3(gx)

        q1 = self.softmax(self.agp(x1).reshape(b * self.groups, 1, self.group_channels))
        k1 = x2.reshape(b * self.groups, self.group_channels, h * w)
        q2 = self.softmax(self.agp(x2).reshape(b * self.groups, 1, self.group_channels))
        k2 = x1.reshape(b * self.groups, self.group_channels, h * w)

        weights = (torch.matmul(q1, k1) + torch.matmul(q2, k2)).reshape(b * self.groups, 1, h, w)
        return (gx * weights.sigmoid()).reshape(b, c, h, w)


class HRECBlock(nn.Module):
    """One Hierarchical Residual EMA Calibration (HREC) branch.

    F_out = F + alpha * (EMA(F) - F),
    alpha = gamma_max * tanh(gamma), with gamma initialized to zero.
    """

    def __init__(
        self,
        channels: int,
        groups: int = 8,
        gamma_init: float = 0.0,
        gamma_max: float = 0.50,
    ) -> None:
        super().__init__()
        self.ema       = EMAAttention2D(channels=channels, groups=groups)
        self.gamma     = nn.Parameter(torch.tensor(float(gamma_init)))
        self.gamma_max = float(gamma_max)

    def effective_alpha(self) -> torch.Tensor:
        return torch.tanh(self.gamma) * self.gamma_max

    def effective_alpha_value(self) -> float:
        with torch.no_grad():
            return float(self.effective_alpha().detach().cpu().item())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.ema(x)
        alpha = self.effective_alpha().to(dtype=x.dtype, device=x.device)
        return x + alpha * (y - x)


# =====================================================================================
# Model — final classifier: TPQ-Net
# =====================================================================================

class TPQNet(nn.Module):
    """TPQ-Net as described in the manuscript.

    DCT-3T -> DMS-TPE+LKA -> HREC -> CTSF -> QGBN Neck -> QACM Head.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        frontend_stem_channels: int = 16,
        stage_channels: Sequence[int] = (64, 128, 192, 256),
        stage_depths: Sequence[int] = (1, 2, 2, 2),
        branch_dilations: Sequence[int] = (1, 2, 4),
        dropout: float = 0.1,
        fc_dim: int = 256,
        neck_type: str = "qgbn",
        qgbn_hidden: int = 64,
        qgbn_init_bias: float = -2.0,
        qgbn_gate_max: float = 0.75,
        qacm_scale: float = 24.0,
        qacm_base_margin: float = 0.12,
        qacm_strength: float = 0.15,
        qacm_quality_ema: float = 0.99,
        qacm_quality_clip: float = 1.0,
        qacm_min_margin: float = 0.04,
        qacm_max_margin: float = 0.20,
        use_hrec: bool = True,
        hrec_groups: int = 8,
        hrec_gamma_init: float = 0.0,
        hrec_stage3_gamma_max: float = 0.25,
        hrec_stage4_gamma_max: float = 0.50,
        # ── 最终结构开关（默认与最终模型一致）──────────────────────────────
        use_lka: bool = True,
        use_ctsf: bool = True,
        **unused_kwargs: object,
    ) -> None:
        super().__init__()

        # Front-end stem: 1x1 channel projection without spatial smoothing.
        self.front_stem = nn.Sequential(
            nn.Conv2d(in_channels, frontend_stem_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(frontend_stem_channels),
            nn.ReLU(inplace=True),
        )

        # DMS-TPE backbone with depths [1,2,2,2], dilations [1,2,4], and LKA.
        self.backbone = DMSTPEBackbone(
            in_channels=frontend_stem_channels,
            stage_channels=stage_channels,
            stage_depths=stage_depths,
            branch_dilations=branch_dilations,
            dropout=dropout,
            use_lka=bool(use_lka),
        )

        if len(stage_channels) < 4:
            raise ValueError("HREC and CTSF require four DMS-TPE stages")

        # HREC calibration on Stage 3 and Stage 4.
        self.use_hrec = bool(use_hrec)
        self.hrec_stage3  = (
            HRECBlock(
                channels=int(stage_channels[2]),
                groups=int(hrec_groups),
                gamma_init=float(hrec_gamma_init),
                gamma_max=float(hrec_stage3_gamma_max),
            )
            if self.use_hrec else nn.Identity()
        )
        self.hrec_stage4 = (
            HRECBlock(
                channels=self.backbone.out_channels,
                groups=int(hrec_groups),
                gamma_init=float(hrec_gamma_init),
                gamma_max=float(hrec_stage4_gamma_max),
            )
            if self.use_hrec else nn.Identity()
        )

        # Global average pooling used in the paper.
        self.gap     = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        # CTSF: concatenate calibrated Stage-3 and Stage-4 features.
        self.use_ctsf = bool(use_ctsf)
        ms_fc_in = self.backbone.out_channels + (int(stage_channels[2]) if self.use_ctsf else 0)
        self.fc1 = nn.Linear(ms_fc_in, fc_dim)
        self.act = nn.ReLU(inplace=True)

        # QGBN Neck (or ablation alternatives).
        self.neck_type = str(neck_type).lower()
        if self.neck_type == "none":
            self.neck = nn.Identity()
        elif self.neck_type == "bn":
            self.neck = nn.BatchNorm1d(fc_dim)
        elif self.neck_type == "qgbn":
            self.neck = QGBNNeck(
                dim=fc_dim,
                hidden_dim=qgbn_hidden,
                qgbn_init_bias=qgbn_init_bias,
                qgbn_gate_max=qgbn_gate_max,
            )
        else:
            raise ValueError(f"Unsupported neck_type={neck_type}; choose none, bn, or qgbn")

        self.drop = nn.Dropout(dropout)

        # QACM Head: one center per class and a quality-adaptive cosine margin.
        self.head = QACMHead(
            in_features=fc_dim,
            num_classes=num_classes,
            scale=qacm_scale,
            base_margin=qacm_base_margin,
            qacm_strength=qacm_strength,
            q_ema=qacm_quality_ema,
            q_clip=qacm_quality_clip,
            min_margin=qacm_min_margin,
            max_margin=qacm_max_margin,
        )

        self._init_weights()
        # Restore the identity-preserving initializations after generic weight initialization.
        if isinstance(self.neck, QGBNNeck):
            self.neck.reset_gate(qgbn_init_bias)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Zero-initialize the LKA output projection so each LKA branch starts as identity.
        for m in self.modules():
            if isinstance(m, LargeKernelAttention):
                m.reset_to_near_identity()

    def forward_raw_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.front_stem(x)
        x = self.backbone.stem(x)

        stage3_feat: torch.Tensor | None = None
        for stage_idx, stage in enumerate(self.backbone.stages):
            x = stage(x)
            if stage_idx == 2:  # Stage 3: [B, 192, 16, 16] for 128x128 input
                x = self.hrec_stage3(x)
                if self.use_ctsf:
                    stage3_feat = x
            elif stage_idx == 3:  # Stage 4: [B, 256, 16, 16] for 128x128 input
                x = self.hrec_stage4(x)

        # CTSF concatenates Stage-3 and Stage-4 features at the same spatial resolution.
        if self.use_ctsf and stage3_feat is not None:
            x = torch.cat([x, stage3_feat], dim=1)  # [B, 448, 16, 16]

        x        = self.gap(x)                       # [B, 448/256, 1, 1]
        x        = self.flatten(x)                   # [B, 448/256]
        x        = self.fc1(x)                       # [B, 256]
        raw_feat = self.act(x)
        return raw_feat


    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.neck(self.forward_raw_features(x))

    def logits_from_features(
        self,
        feat: torch.Tensor,
        targets: torch.Tensor | None = None,
        apply_dropout: bool = True,
        margin_feat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        feat_head = self.drop(feat) if apply_dropout else feat
        return self.head(feat_head, targets=targets, margin_feat=margin_feat)

    def forward(self, x: torch.Tensor, targets: torch.Tensor | None = None) -> torch.Tensor:
        raw_feat  = self.forward_raw_features(x)
        feat_neck = self.neck(raw_feat)
        return self.logits_from_features(
            feat_neck,
            targets=targets,
            apply_dropout=self.training,
            margin_feat=raw_feat,
        )



# =====================================================================================
# Train / eval
# =====================================================================================


@dataclass
class EpochResult:
    loss: float
    cls_loss: float
    acc: float
    bAcc: float
    macro_precision: float
    macro_f1: float
    kappa: float


def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler,
    use_amp: bool,
    max_grad_norm: float,
    label_smoothing: float,
) -> EpochResult:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_cls_loss = 0.0
    total_num = 0
    all_logits: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []

    autocast_enabled = bool(use_amp and device.type == "cuda")

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with amp_autocast_cuda(autocast_enabled):
            raw_feat = model.forward_raw_features(xb)
            feat_neck = model.neck(raw_feat)
            logits = model.logits_from_features(
                feat_neck,
                targets=yb if is_train else None,
                apply_dropout=is_train,
                margin_feat=raw_feat,
            )
            cls_loss = F.cross_entropy(logits, yb, label_smoothing=float(label_smoothing))
            loss = cls_loss

        if is_train:
            if scaler is not None and autocast_enabled:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

        bs = xb.size(0)
        total_loss += float(loss.detach().item()) * bs
        total_cls_loss += float(cls_loss.detach().item()) * bs
        total_num += bs
        all_logits.append(logits.detach().cpu())
        all_targets.append(yb.detach().cpu())

    logits_cat = torch.cat(all_logits, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)
    metrics = compute_metrics_from_logits(logits_cat, targets_cat)

    return EpochResult(
        loss=total_loss / max(total_num, 1),
        cls_loss=total_cls_loss / max(total_num, 1),
        acc=metrics["acc"],
        bAcc=metrics["bAcc"],
        macro_precision=metrics["macro_precision"],
        macro_f1=metrics["macro_f1"],
        kappa=metrics["kappa"],
    )


# =====================================================================================
# argparse
# =====================================================================================

def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Train TPQ-Net: DCT-3T + DMS-TPE/LKA + HREC + CTSF + "
            "QGBN Neck + QACM Head."
        )
    )

    parser.add_argument("--data_npz", type=str, default=DEFAULT_DATA_NPZ)
    parser.add_argument("--split_npz", type=str, default=DEFAULT_SPLIT_NPZ)
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--use_amp", action="store_true", default=False)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--save_every", type=int, default=0)
    parser.add_argument("--label_smoothing", type=float, default=0.0)

    parser.add_argument("--frontend_stem_channels", type=int, default=16)
    parser.add_argument("--stage_channels", type=int, nargs="+", default=[64, 128, 192, 256])
    parser.add_argument("--stage_depths", type=int, nargs="+", default=[1, 2, 2, 2])
    parser.add_argument("--branch_dilations", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--fc_dim", type=int, default=256)
    parser.add_argument("--normalization", choices=["none", "std", "zscore"], default="std")

    # QGBN Neck.
    parser.add_argument("--neck_type", choices=["none", "bn", "qgbn"], default="qgbn")
    parser.add_argument("--qgbn_hidden", type=int, default=64)
    parser.add_argument("--qgbn_init_bias", type=float, default=-2.0)
    parser.add_argument("--qgbn_gate_max", type=float, default=0.75)

    # HREC.
    parser.add_argument("--use_hrec", action="store_true", default=True)
    parser.add_argument("--disable_hrec", action="store_false", dest="use_hrec")
    parser.add_argument("--hrec_groups", type=int, default=8)
    parser.add_argument("--hrec_gamma_init", type=float, default=0.0)
    parser.add_argument("--hrec_stage3_gamma_max", type=float, default=0.25)
    parser.add_argument("--hrec_stage4_gamma_max", type=float, default=0.50)

    # QACM Head.
    parser.add_argument("--qacm_scale", type=float, default=24.0)
    parser.add_argument("--qacm_base_margin", type=float, default=0.12)
    parser.add_argument("--qacm_strength", type=float, default=0.15)
    parser.add_argument("--qacm_quality_ema", type=float, default=0.99)
    parser.add_argument("--qacm_quality_clip", type=float, default=1.0)
    parser.add_argument("--qacm_min_margin", type=float, default=0.04)
    parser.add_argument("--qacm_max_margin", type=float, default=0.20)

    # DMS-TPE/LKA and CTSF ablation switches.
    parser.add_argument("--use_lka", action="store_true", default=True)
    parser.add_argument("--disable_lka", action="store_false", dest="use_lka")
    parser.add_argument("--use_ctsf", action="store_true", default=True)
    parser.add_argument("--disable_ctsf", action="store_false", dest="use_ctsf")

    args = parser.parse_args()
    return RunConfig(
        data_npz=args.data_npz,
        split_npz=args.split_npz,
        out_dir=args.out_dir,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        use_amp=args.use_amp,
        max_grad_norm=args.max_grad_norm,
        save_every=args.save_every,
        label_smoothing=args.label_smoothing,
        frontend_stem_channels=args.frontend_stem_channels,
        stage_channels=tuple(args.stage_channels),
        stage_depths=tuple(args.stage_depths),
        branch_dilations=tuple(args.branch_dilations),
        dropout=args.dropout,
        fc_dim=args.fc_dim,
        neck_type=args.neck_type,
        qgbn_hidden=args.qgbn_hidden,
        qgbn_init_bias=args.qgbn_init_bias,
        qgbn_gate_max=args.qgbn_gate_max,
        normalization=args.normalization,
        use_hrec=args.use_hrec,
        hrec_groups=args.hrec_groups,
        hrec_gamma_init=args.hrec_gamma_init,
        hrec_stage3_gamma_max=args.hrec_stage3_gamma_max,
        hrec_stage4_gamma_max=args.hrec_stage4_gamma_max,
        qacm_scale=args.qacm_scale,
        qacm_base_margin=args.qacm_base_margin,
        qacm_strength=args.qacm_strength,
        qacm_quality_ema=args.qacm_quality_ema,
        qacm_quality_clip=args.qacm_quality_clip,
        qacm_min_margin=args.qacm_min_margin,
        qacm_max_margin=args.qacm_max_margin,
        use_lka=args.use_lka,
        use_ctsf=args.use_ctsf,
    )


# =====================================================================================
# main
# =====================================================================================

def main() -> None:
    cfg     = parse_args()
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 100)
    print("[MODEL] TPQ-Net | DCT-3T + DMS-TPE/LKA + HREC + CTSF + QGBN Neck + QACM Head")
    print(f"[PATH ] data_npz  = {cfg.data_npz}")
    print(f"[PATH ] split_npz = {cfg.split_npz}")
    print(f"[PATH ] out_dir   = {out_dir}")
    print(f"[ENV  ] device    = {device}")
    print(f"[DET  ] seed={cfg.seed} | cudnn.benchmark=False | deterministic=True | TF32=False")
    print("=" * 100)

    loaded = load_data_npz(cfg.data_npz)
    x, y   = loaded.x, loaded.y
    idx_train, idx_val, idx_test = load_locked_splits(cfg.split_npz)
    validate_split_indices(idx_train, idx_val, idx_test, n_total=int(x.shape[0]))

    num_classes = int(np.unique(y).size)
    in_channels = int(x.shape[1])
    H, W        = int(x.shape[2]), int(x.shape[3])

    print(f"[DATA ] x={x.shape}, y={y.shape}")
    print("[DATA ] channel_names=['density','segment_density','ordered_segment_density']")
    print(f"[DATA ] num_classes={num_classes}, in_channels={in_channels}, H={H}, W={W}")
    print(f"[SPLIT] train={len(idx_train)}  val={len(idx_val)}  test={len(idx_test)}")
    print(f"[CONF ] stage_channels={cfg.stage_channels} | stage_depths={cfg.stage_depths} | "
          f"branch_dilations={cfg.branch_dilations}")
    print(f"[CONF ] use_lka={cfg.use_lka} | use_hrec={cfg.use_hrec} | "
          f"use_ctsf={cfg.use_ctsf} | pooling=GAP")
    print(f"[CONF ] neck={cfg.neck_type} | hrec={cfg.use_hrec} | "
          f"QACM scale={cfg.qacm_scale} | base_margin={cfg.qacm_base_margin} | "
          f"range=[{cfg.qacm_min_margin},{cfg.qacm_max_margin}]")

    dl_train, dl_val, dl_test, stats = build_dataloaders(
        x=x, y=y,
        idx_train=idx_train, idx_val=idx_val, idx_test=idx_test,
        batch_size=cfg.batch_size, num_workers=cfg.num_workers,
        normalization=cfg.normalization, seed=cfg.seed,
    )

    save_json(
        {
            "config": asdict(cfg),
            "data_info": {
                "data_key": loaded.data_key, "label_key": loaded.label_key,
                "shape_x": list(x.shape), "num_classes": num_classes,
                "channel_names": ["density", "segment_density", "ordered_segment_density"],
            },
            "split_info": {
                "n_train": int(len(idx_train)),
                "n_val":   int(len(idx_val)),
                "n_test":  int(len(idx_test)),
            },
        },
        out_dir / "run_info.json",
    )
    save_json(
        {"normalization": cfg.normalization,
         "train_channel_mean": stats["mean"],
         "train_channel_std":  stats["std"]},
        out_dir / "normalization_stats.json",
    )

    model = TPQNet(
        in_channels=in_channels,
        num_classes=num_classes,
        frontend_stem_channels=cfg.frontend_stem_channels,
        stage_channels=cfg.stage_channels,
        stage_depths=cfg.stage_depths,
        branch_dilations=cfg.branch_dilations,
        dropout=cfg.dropout,
        fc_dim=cfg.fc_dim,
        neck_type=cfg.neck_type,
        qgbn_hidden=cfg.qgbn_hidden,
        qgbn_init_bias=cfg.qgbn_init_bias,
        qgbn_gate_max=cfg.qgbn_gate_max,
        qacm_scale=cfg.qacm_scale,
        qacm_base_margin=cfg.qacm_base_margin,
        qacm_strength=cfg.qacm_strength,
        qacm_quality_ema=cfg.qacm_quality_ema,
        qacm_quality_clip=cfg.qacm_quality_clip,
        qacm_min_margin=cfg.qacm_min_margin,
        qacm_max_margin=cfg.qacm_max_margin,
        use_hrec=cfg.use_hrec,
        hrec_groups=cfg.hrec_groups,
        hrec_gamma_init=cfg.hrec_gamma_init,
        hrec_stage3_gamma_max=cfg.hrec_stage3_gamma_max,
        hrec_stage4_gamma_max=cfg.hrec_stage4_gamma_max,
        use_lka=cfg.use_lka,
        use_ctsf=cfg.use_ctsf,
    ).to(device)

    n_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    print(f"[MODEL] trainable params = {n_params:,}")
    print("[CONF ] classifier = QACM Head")


    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    scaler = make_grad_scaler(cfg.use_amp, device)

    best_val_bacc  = -1.0
    best_epoch     = -1
    best_ckpt_path = out_dir / "best_model.pt"
    history: List[Dict[str, object]] = []

    t0 = time.time()

    for epoch in range(1, cfg.epochs + 1):
        train_res = run_one_epoch(
            model=model, loader=dl_train, device=device,
            optimizer=optimizer, scaler=scaler,
            use_amp=cfg.use_amp, max_grad_norm=cfg.max_grad_norm,
            label_smoothing=cfg.label_smoothing,
        )
        val_res = run_one_epoch(
            model=model, loader=dl_val, device=device,
            optimizer=None, scaler=None,
            use_amp=cfg.use_amp, max_grad_norm=0.0,
            label_smoothing=cfg.label_smoothing,
        )
        scheduler.step()
        lr_cur = float(optimizer.param_groups[0]["lr"])

        history.append({
            "epoch": epoch, "lr": lr_cur,
            "train": asdict(train_res), "val": asdict(val_res),
        })
        print(
            f"[EPOCH {epoch:03d}] lr={lr_cur:.6g} | "
            f"train loss={train_res.loss:.4f} acc={train_res.acc:.4f} bAcc={train_res.bAcc:.4f} | "
            f"val loss={val_res.loss:.4f} acc={val_res.acc:.4f} "
            f"Macro-F1={val_res.macro_f1:.4f} Kappa={val_res.kappa:.4f}"
        )

        if val_res.bAcc > best_val_bacc:
            best_val_bacc = float(val_res.bAcc)
            best_epoch    = int(epoch)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_val_bacc":    best_val_bacc,
                    "config":           asdict(cfg),
                    "normalization_stats": {"mean": stats["mean"], "std": stats["std"]},
                },
                best_ckpt_path,
            )
            print(f"[SAVE ] best ckpt → {best_ckpt_path}  (val bAcc={best_val_bacc:.4f})")

        if cfg.save_every > 0 and epoch % cfg.save_every == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": asdict(cfg),
                    "normalization_stats": {"mean": stats["mean"], "std": stats["std"]},
                },
                out_dir / f"epoch_{epoch:03d}.pt",
            )

    total_train_time = time.time() - t0
    save_json({"history": history}, out_dir / "history.json")

    # ── 用最优 checkpoint 在测试集评估 ──────────────────────────────────────────────
    print("=" * 100)
    print(f"[BEST ] epoch={best_epoch}  val bAcc={best_val_bacc:.4f}")

    ckpt = torch_load_compat(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    hrec_s3_gamma = (
        model.hrec_stage3.effective_alpha_value()
        if hasattr(model.hrec_stage3, "effective_alpha_value") else None
    )
    hrec_s4_gamma = (
        model.hrec_stage4.effective_alpha_value()
        if hasattr(model.hrec_stage4, "effective_alpha_value") else None
    )
    if hrec_s3_gamma is not None:
        print(f"[HREC ] stage3 effective_alpha={hrec_s3_gamma:.4f}")
    if hrec_s4_gamma is not None:
        print(f"[HREC ] stage4 effective_alpha={hrec_s4_gamma:.4f}")


    test_res = run_one_epoch(
        model=model, loader=dl_test, device=device,
        optimizer=None, scaler=None,
        use_amp=cfg.use_amp, max_grad_norm=0.0,
        label_smoothing=cfg.label_smoothing,
    )
    print(f"[TEST ] loss={test_res.loss:.4f} acc={test_res.acc:.4f} "
          f"Macro-P={test_res.macro_precision:.4f} Macro-F1={test_res.macro_f1:.4f} "
          f"Kappa={test_res.kappa:.4f}")
    print(f"[TIME ] total_train_time={total_train_time:.1f}s")
    print("=" * 100)

    save_json(
        {
            "best_epoch":         best_epoch,
            "best_val_bAcc":      best_val_bacc,
            "test":               asdict(test_res),
            "total_train_time_sec": round(total_train_time, 2),
            "n_params":           n_params,
            "use_lka":            cfg.use_lka,
            "use_hrec":           cfg.use_hrec,
            "use_ctsf":           cfg.use_ctsf,
            "method_name":        "TPQ-Net",
            "final_head":         "QACM Head",
            "stage_depths":       list(cfg.stage_depths),
            "branch_dilations":   list(cfg.branch_dilations),
            "hrec_stage3_effective_alpha": hrec_s3_gamma,
            "hrec_stage4_effective_alpha": hrec_s4_gamma,
        },
        out_dir / "metrics_summary.json",
    )


if __name__ == "__main__":
    main()
