# -*- coding: utf-8 -*-
"""
Build the Delay-Conjugate Tri-density Trajectory Tensor (DCT-3T) used by TPQ-Net.

Input
-----
The input is one of the released clean or multi-SNR effective-window I/Q files
under ``./data``. Each file contains a padded complex array ``X`` and
``X_valid_len`` so zero-padding samples are excluded from trajectory construction.

DCT-3T construction
-------------------
For each valid I/Q sequence:
    1. compute the delay-conjugate product with delay tau;
    2. estimate the complex-plane coordinate range from the training split only;
    3. rasterize three 128 x 128 channels:
       - density: global point density;
       - segment_density: local consecutive-segment density;
       - ordered_segment_density: time-weighted segment density.

Using training-only coordinate statistics avoids validation/test information leakage.
Run this script from the repository root.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np


DEFAULT_IN_NPZ = "./data/LFM_SEI_clean_iq.npz"
DEFAULT_SPLIT_NPZ = "./data/split_622.npz"
DEFAULT_OUT_DIR = "./generated/dct3"

DEFAULT_TAU = 2
DEFAULT_GRID_SIZE = 128
DEFAULT_GRID_PERCENTILE = 99.9
DEFAULT_GRID_PAD_RATIO = 0.15
DEFAULT_DISP_STRIDE = 1

DEFAULT_SEGMENT_SAMPLES = 5
DEFAULT_PROGRESS_EVERY = 100
DEFAULT_ORDER_WEIGHT_MIN = 0.2
DEFAULT_ORDER_WEIGHT_MAX = 1.0

DEFAULT_SAVE_COMPLEX = False


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def resolve_existing_path(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    if os.path.exists(path):
        return path
    raise FileNotFoundError(f"Path not found: {path}")


def detect_xy_keys(npz_obj: np.lib.npyio.NpzFile) -> Tuple[str, str]:
    keys = set(npz_obj.files)

    x_key = None
    for cand in ["X", "x", "signals", "data"]:
        if cand in keys:
            x_key = cand
            break
    if x_key is None:
        raise KeyError(f"No signal key found. Available keys: {sorted(keys)}")

    y_key = None
    for cand in ["y", "cls", "labels", "label"]:
        if cand in keys:
            y_key = cand
            break
    if y_key is None:
        raise KeyError(f"No label key found. Available keys: {sorted(keys)}")

    return x_key, y_key


def load_input_npz(npz_path: str, tau: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object], np.lib.npyio.NpzFile]:
    data = np.load(npz_path, allow_pickle=True)
    x_key, y_key = detect_xy_keys(data)

    X = data[x_key]
    y = data[y_key].astype(np.int64, copy=False)

    if not np.iscomplexobj(X):
        raise ValueError(
            f"This script requires padded complex effective-window I/Q data [N,L]. "
            f"Got dtype={X.dtype}, shape={X.shape}."
        )
    if X.ndim != 2:
        raise ValueError(f"X must have shape [N,L], got {X.shape}")

    if "X_valid_len" in data.files:
        valid_len = data["X_valid_len"].astype(np.int64, copy=False)
    elif "win_len" in data.files:
        valid_len = data["win_len"].astype(np.int64, copy=False)
    else:
        raise KeyError(
            f"{npz_path} does not contain X_valid_len or win_len, so padding "
            f"samples cannot be excluded. Available keys: {sorted(data.files)}"
        )

    if valid_len.ndim != 1 or valid_len.shape[0] != X.shape[0]:
        raise ValueError(f"X_valid_len must have shape [N]; got {valid_len.shape}, N={X.shape[0]}")

    if np.any(valid_len <= int(tau)):
        bad = int(np.sum(valid_len <= int(tau)))
        raise ValueError(f"{bad} samples have valid_len <= tau and cannot be delay-conjugated")

    meta = {
        "npz_path": npz_path,
        "x_key": x_key,
        "y_key": y_key,
        "x_dtype": str(X.dtype),
        "x_shape": list(X.shape),
        "y_shape": list(y.shape),
        "valid_len_mean": float(np.mean(valid_len)),
        "valid_len_min": int(np.min(valid_len)),
        "valid_len_max": int(np.max(valid_len)),
        "keys": sorted(data.files),
    }
    return X.astype(np.complex64, copy=False), y, valid_len, meta, data


def load_split_indices(split_npz: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(split_npz, allow_pickle=True)

    train_key = next((k for k in ["idx_train", "train_idx"] if k in data.files), None)
    val_key = next((k for k in ["idx_val", "val_idx"] if k in data.files), None)
    test_key = next((k for k in ["idx_test", "test_idx"] if k in data.files), None)

    if train_key is None or val_key is None or test_key is None:
        raise KeyError(
            f"{split_npz} 中未找到 idx_train/idx_val/idx_test 或兼容键名，"
            f"keys={sorted(data.files)}"
        )

    return (
        data[train_key].astype(np.int64, copy=False),
        data[val_key].astype(np.int64, copy=False),
        data[test_key].astype(np.int64, copy=False),
    )


def validate_split_indices(
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    idx_test: np.ndarray,
    n_total: int,
) -> None:
    for name, idx in [("train", idx_train), ("val", idx_val), ("test", idx_test)]:
        if idx.ndim != 1:
            raise ValueError(f"{name} split 必须是一维")
        if idx.size == 0:
            raise ValueError(f"{name} split 为空")
        if np.any(idx < 0) or np.any(idx >= n_total):
            raise IndexError(f"{name} split 存在越界索引，允许范围 [0,{n_total - 1}]")

    s_train = set(idx_train.tolist())
    s_val = set(idx_val.tolist())
    s_test = set(idx_test.tolist())
    if (s_train & s_val) or (s_train & s_test) or (s_val & s_test):
        raise RuntimeError("split 中 train/val/test 存在重叠索引")


def delay_conjugate_one(z: np.ndarray, tau: int) -> np.ndarray:
    if z.ndim != 1 or not np.iscomplexobj(z):
        raise ValueError("z 必须是一维复数数组")
    if tau < 1:
        raise ValueError(f"tau 必须 >=1，当前 {tau}")
    if z.shape[0] <= tau:
        raise ValueError(f"z 长度必须 > tau，len={z.shape[0]}, tau={tau}")
    return (z[tau:] * np.conjugate(z[:-tau])).astype(np.complex64, copy=False)


def build_delay_conj_from_padded_iq(
    X: np.ndarray,
    valid_len: np.ndarray,
    tau: int,
) -> Tuple[np.ndarray, np.ndarray]:
    d_list: List[np.ndarray] = []
    d_len = np.zeros((X.shape[0],), dtype=np.int64)

    for i in range(X.shape[0]):
        l = int(valid_len[i])
        z = X[i, :l]
        d = delay_conjugate_one(z, tau=tau)
        d_list.append(d)
        d_len[i] = int(d.shape[0])

    max_len = int(d_len.max())
    D = np.full((X.shape[0], max_len), np.complex64(np.nan + 1j * np.nan), dtype=np.complex64)

    for i, d in enumerate(d_list):
        D[i, :d.shape[0]] = d

    return D, d_len


def estimate_complex_coord_range(
    D: np.ndarray,
    percentile: float,
    pad_ratio: float,
) -> Dict[str, float]:
    if D.ndim != 2 or not np.iscomplexobj(D):
        raise ValueError("D 必须是 [N,L] complex")

    valid = np.isfinite(D.real) & np.isfinite(D.imag)
    if not np.any(valid):
        raise ValueError("No finite delay-conjugate points are available for coordinate-range estimation")

    re_abs = np.abs(D.real[valid].astype(np.float64, copy=False))
    im_abs = np.abs(D.imag[valid].astype(np.float64, copy=False))

    re_lim = float(np.percentile(re_abs, float(percentile)))
    im_lim = float(np.percentile(im_abs, float(percentile)))

    re_lim = max(re_lim * (1.0 + float(pad_ratio)), 1e-6)
    im_lim = max(im_lim * (1.0 + float(pad_ratio)), 1e-6)

    return {
        "re_min": -re_lim,
        "re_max": re_lim,
        "im_min": -im_lim,
        "im_max": im_lim,
        "percentile": float(percentile),
        "pad_ratio": float(pad_ratio),
    }


def save_coord_range_json(coord_range: Dict[str, float], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(coord_range, f, ensure_ascii=False, indent=2)


def _map_complex_to_pixel_coords(
    z: np.ndarray,
    grid_size: int,
    coord_range: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    re = z.real.astype(np.float64, copy=False)
    im = z.imag.astype(np.float64, copy=False)

    re_min = float(coord_range["re_min"])
    re_max = float(coord_range["re_max"])
    im_min = float(coord_range["im_min"])
    im_max = float(coord_range["im_max"])

    re_span = max(re_max - re_min, 1e-12)
    im_span = max(im_max - im_min, 1e-12)

    px = (re - re_min) / re_span * (grid_size - 1)
    py = (im - im_min) / im_span * (grid_size - 1)

    valid = (
        np.isfinite(px) & np.isfinite(py) &
        (px >= 0.0) & (px <= grid_size - 1) &
        (py >= 0.0) & (py <= grid_size - 1)
    )
    return px, py, valid


def _bilinear_splat(img: np.ndarray, px: np.ndarray, py: np.ndarray, weight: np.ndarray) -> None:
    if px.size == 0:
        return

    x0 = np.floor(px).astype(np.int64)
    y0 = np.floor(py).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, img.shape[1] - 1)
    y1 = np.clip(y0 + 1, 0, img.shape[0] - 1)

    dx = px - x0
    dy = py - y0

    w00 = (1.0 - dx) * (1.0 - dy) * weight
    w01 = dx * (1.0 - dy) * weight
    w10 = (1.0 - dx) * dy * weight
    w11 = dx * dy * weight

    np.add.at(img, (y0, x0), w00)
    np.add.at(img, (y0, x1), w01)
    np.add.at(img, (y1, x0), w10)
    np.add.at(img, (y1, x1), w11)


def _rasterize_segment_density_fixed(
    px0: np.ndarray,
    py0: np.ndarray,
    px1: np.ndarray,
    py1: np.ndarray,
    seg_weight: np.ndarray,
    grid_size: int,
    n_sample: int,
) -> np.ndarray:
    img = np.zeros((grid_size, grid_size), dtype=np.float32)
    if px0.size == 0:
        return img
    if n_sample < 2:
        raise ValueError(f"segment_samples 必须 >=2，当前={n_sample}")

    ts = np.linspace(0.0, 1.0, int(n_sample), dtype=np.float64)[None, :]
    dx = (px1 - px0)[:, None]
    dy = (py1 - py0)[:, None]

    px = px0[:, None] + dx * ts
    py = py0[:, None] + dy * ts
    weight = np.broadcast_to(seg_weight[:, None] / float(n_sample), px.shape)

    _bilinear_splat(img, px.reshape(-1), py.reshape(-1), weight.reshape(-1))
    return img


def build_dct3_from_complex(
    D: np.ndarray,
    grid_size: int,
    coord_range: Dict[str, float],
    disp_stride: int,
    segment_samples: int,
    progress_every: int,
    order_weight_min: float,
    order_weight_max: float,
) -> np.ndarray:
    if disp_stride < 1:
        raise ValueError(f"disp_stride 必须 >=1，当前={disp_stride}")
    if segment_samples < 2:
        raise ValueError(f"segment_samples 必须 >=2，当前={segment_samples}")

    n, _ = D.shape
    X = np.zeros((n, 3, grid_size, grid_size), dtype=np.float32)

    for i in range(n):
        px_all, py_all, valid_all = _map_complex_to_pixel_coords(D[i], grid_size, coord_range)

        density = np.zeros((grid_size, grid_size), dtype=np.float32)
        if np.any(valid_all):
            _bilinear_splat(
                density,
                px_all[valid_all],
                py_all[valid_all],
                np.ones(int(valid_all.sum()), dtype=np.float64),
            )
            density /= max(float(valid_all.sum()), 1.0)

        src = D[i, :-disp_stride]
        dst = D[i, disp_stride:]

        px0, py0, valid0 = _map_complex_to_pixel_coords(src, grid_size, coord_range)
        px1, py1, valid1 = _map_complex_to_pixel_coords(dst, grid_size, coord_range)

        valid_seg = valid0 & valid1

        segment_density = np.zeros((grid_size, grid_size), dtype=np.float32)
        ordered_segment_density = np.zeros((grid_size, grid_size), dtype=np.float32)

        if np.any(valid_seg):
            px0_v = px0[valid_seg]
            py0_v = py0[valid_seg]
            px1_v = px1[valid_seg]
            py1_v = py1[valid_seg]

            n_seg = int(valid_seg.sum())
            eq_weight = np.ones((n_seg,), dtype=np.float64)
            ordered_weight = np.linspace(
                float(order_weight_min),
                float(order_weight_max),
                n_seg,
                dtype=np.float64,
            )

            segment_density = _rasterize_segment_density_fixed(
                px0_v, py0_v, px1_v, py1_v, eq_weight, grid_size, n_sample=segment_samples
            )
            ordered_segment_density = _rasterize_segment_density_fixed(
                px0_v, py0_v, px1_v, py1_v, ordered_weight, grid_size, n_sample=segment_samples
            )

            segment_density /= max(float(eq_weight.sum()), 1.0)
            ordered_segment_density /= max(float(ordered_weight.sum()), 1.0)

        X[i, 0] = density
        X[i, 1] = segment_density
        X[i, 2] = ordered_segment_density

        if progress_every > 0 and (((i + 1) % progress_every == 0) or (i + 1 == n)):
            print(f"[DCT3 ] processed {i + 1}/{n}")

    return X.astype(np.float32, copy=False)


def infer_output_name(in_npz: str, tau: int, grid_size: int, disp_stride: int) -> str:
    stem = os.path.splitext(os.path.basename(in_npz))[0]
    return f"{stem}__dct3_tau{tau}_grid{grid_size}_disp{disp_stride}.npz"


def copy_optional_fields(data: np.lib.npyio.NpzFile, keys: List[str]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for k in keys:
        if k in data.files:
            out[k] = data[k]
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the TPQ-Net DCT-3T tensor from clean or noisy effective-window I/Q data."
    )
    parser.add_argument("--in_npz", type=str, default=DEFAULT_IN_NPZ)
    parser.add_argument("--split_npz", type=str, default=DEFAULT_SPLIT_NPZ)
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)

    parser.add_argument("--tau", type=int, default=DEFAULT_TAU)
    parser.add_argument("--grid_size", type=int, default=DEFAULT_GRID_SIZE)
    parser.add_argument("--grid_percentile", type=float, default=DEFAULT_GRID_PERCENTILE)
    parser.add_argument("--grid_pad_ratio", type=float, default=DEFAULT_GRID_PAD_RATIO)
    parser.add_argument("--disp_stride", type=int, default=DEFAULT_DISP_STRIDE)

    parser.add_argument("--segment_samples", type=int, default=DEFAULT_SEGMENT_SAMPLES)
    parser.add_argument("--progress_every", type=int, default=DEFAULT_PROGRESS_EVERY)
    parser.add_argument("--order_weight_min", type=float, default=DEFAULT_ORDER_WEIGHT_MIN)
    parser.add_argument("--order_weight_max", type=float, default=DEFAULT_ORDER_WEIGHT_MAX)

    parser.add_argument("--save_complex", action="store_true", default=DEFAULT_SAVE_COMPLEX)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    args.in_npz = resolve_existing_path(args.in_npz)
    args.split_npz = resolve_existing_path(args.split_npz)
    ensure_dir(args.out_dir)

    X_iq, y, valid_len, input_meta, data = load_input_npz(args.in_npz, tau=args.tau)
    n_total = int(X_iq.shape[0])

    idx_train, idx_val, idx_test = load_split_indices(args.split_npz)
    validate_split_indices(idx_train, idx_val, idx_test, n_total)

    print("=" * 100)
    print("[TASK ] Build TPQ-Net DCT-3T representation")
    print(f"[PATH ] in_npz     = {args.in_npz}")
    print(f"[PATH ] split_npz  = {args.split_npz}")
    print(f"[PATH ] out_dir    = {args.out_dir}")
    print(f"[DATA ] X_iq       = {X_iq.shape}, dtype={X_iq.dtype}")
    print(f"[DATA ] y          = {y.shape}, classes={np.unique(y).tolist()}")
    print(
        f"[DATA ] valid_len  mean/min/max = "
        f"{float(np.mean(valid_len)):.2f}/{int(np.min(valid_len))}/{int(np.max(valid_len))}"
    )
    print(f"[CONF ] tau={args.tau} | grid={args.grid_size} | disp_stride={args.disp_stride}")
    print(
        f"[CONF ] segment_samples={args.segment_samples} | "
        f"time_weight={args.order_weight_min}->{args.order_weight_max}"
    )
    print("=" * 100)

    D, D_valid_len = build_delay_conj_from_padded_iq(X_iq, valid_len=valid_len, tau=args.tau)

    coord_range = estimate_complex_coord_range(
        D[idx_train],
        percentile=args.grid_percentile,
        pad_ratio=args.grid_pad_ratio,
    )

    coord_json = os.path.join(
        args.out_dir,
        f"coord_range_dct3_tau{args.tau}_grid{args.grid_size}_disp{args.disp_stride}.json",
    )
    save_coord_range_json(coord_range, coord_json)

    print(f"[COORD] train-only coord_range = {coord_range}")
    print(f"[SAVE ] coord_range -> {coord_json}")

    X_out = np.zeros((n_total, 3, args.grid_size, args.grid_size), dtype=np.float32)

    for split_name, idx in (("train", idx_train), ("val", idx_val), ("test", idx_test)):
        X_out[idx] = build_dct3_from_complex(
            D[idx],
            grid_size=args.grid_size,
            coord_range=coord_range,
            disp_stride=args.disp_stride,
            segment_samples=args.segment_samples,
            progress_every=args.progress_every,
            order_weight_min=args.order_weight_min,
            order_weight_max=args.order_weight_max,
        )
        print(f"[DCT3 ] filled split={split_name} | n={int(idx.size)}")

    out_npz = os.path.join(
        args.out_dir,
        infer_output_name(args.in_npz, tau=args.tau, grid_size=args.grid_size, disp_stride=args.disp_stride),
    )

    save_dict: Dict[str, np.ndarray] = {
        "X": X_out.astype(np.float32, copy=False),
        "y": y.astype(np.int64, copy=False),
        "coord_range": np.array(
            [coord_range["re_min"], coord_range["re_max"], coord_range["im_min"], coord_range["im_max"]],
            dtype=np.float32,
        ),
        "D_valid_len": D_valid_len.astype(np.int64, copy=False),
        "X_valid_len": valid_len.astype(np.int64, copy=False),
        "source_iq_npz": np.array(args.in_npz),
        "source_split_npz": np.array(args.split_npz),
        "build_protocol": np.array("effective_window_iq_to_dct3_training_coordinate_range"),
        "tau": np.array(int(args.tau), dtype=np.int64),
        "grid_size": np.array(int(args.grid_size), dtype=np.int64),
        "disp_stride": np.array(int(args.disp_stride), dtype=np.int64),
        "segment_samples": np.array(int(args.segment_samples), dtype=np.int64),
        "order_weight_min": np.array(float(args.order_weight_min), dtype=np.float32),
        "order_weight_max": np.array(float(args.order_weight_max), dtype=np.float32),
    }

    optional_keys = [
        "cls", "raw_idx", "raw_len", "selected_rank", "class_rank",
        "start", "end", "span", "win_start", "win_end", "win_len",
        "target_snr_db", "noise_seed", "noise_protocol", "noise_applied_on",
        "signal_power_full_pulse", "noise_power_full_pulse", "measured_snr_db_full_pulse",
        "signal_power_effective_window", "noise_power_effective_window", "measured_snr_db_effective_window",
        "signal_power_rawfull", "noise_power_rawfull", "measured_snr_db_rawfull",
        "signal_power_window", "noise_power_window", "measured_snr_db_window",
    ]
    save_dict.update(copy_optional_fields(data, optional_keys))

    if args.save_complex:
        save_dict["D_complex"] = D.astype(np.complex64, copy=False)

    np.savez(out_npz, **save_dict)

    summary = {
        "input_npz": args.in_npz,
        "split_npz": args.split_npz,
        "output_npz": out_npz,
        "coord_json": coord_json,
        "input_meta": input_meta,
        "tau": int(args.tau),
        "grid_size": int(args.grid_size),
        "grid_percentile": float(args.grid_percentile),
        "grid_pad_ratio": float(args.grid_pad_ratio),
        "disp_stride": int(args.disp_stride),
        "segment_samples": int(args.segment_samples),
        "order_weight_min": float(args.order_weight_min),
        "order_weight_max": float(args.order_weight_max),
        "num_samples": int(n_total),
        "num_classes": int(len(np.unique(y))),
        "labels": np.unique(y).tolist(),
        "split": {
            "n_train": int(len(idx_train)),
            "n_val": int(len(idx_val)),
            "n_test": int(len(idx_test)),
        },
        "coord_range": coord_range,
        "D_shape": list(D.shape),
        "D_valid_len_mean": float(np.mean(D_valid_len)),
        "D_valid_len_min": int(np.min(D_valid_len)),
        "D_valid_len_max": int(np.max(D_valid_len)),
        "output_shape": list(X_out.shape),
        "x_mean": float(X_out.mean()),
        "x_std": float(X_out.std()),
        "x_min": float(X_out.min()),
        "x_max": float(X_out.max()),
    }

    full_snr_key = next(
        (key for key in ("measured_snr_db_full_pulse", "measured_snr_db_rawfull") if key in data.files),
        None,
    )
    window_snr_key = next(
        (key for key in ("measured_snr_db_effective_window", "measured_snr_db_window") if key in data.files),
        None,
    )
    if full_snr_key is not None:
        summary["measured_snr_db_full_pulse_mean"] = float(np.mean(data[full_snr_key]))
    if window_snr_key is not None:
        summary["measured_snr_db_effective_window_mean"] = float(np.mean(data[window_snr_key]))

    summary_path = os.path.splitext(out_npz)[0] + "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("-" * 100)
    print(f"[SAVE ] DCT-3T -> {out_npz}")
    print(f"[SAVE ] summary -> {summary_path}")
    print(f"[SHAPE] X={X_out.shape}")
    print("=" * 100)


if __name__ == "__main__":
    main()
