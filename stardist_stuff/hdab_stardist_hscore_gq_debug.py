#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HDAB StarDist H-score (Global Quantiles) + Optional Debug Visuals

- Core pipeline: segments nuclei (StarDist), measures DAB OD per nucleus,
  sets Low/Med/High via dataset-global quantiles, computes H-score, writes CSV & overlays.
- Visual debug (opt-in via flags): save StarDist input (hematoxylin/gray), probability heatmap
  + overlay, and thin instance boundaries. Visuals are side effects only and do not alter flow.

Dependencies: numpy, pandas, matplotlib, scikit-image, csbdeep, stardist

Ex. 
python hdab_stardist_hscore_gq_debug.py \
  --input /Users/jamieannemortel/Documents/IHC_dataset --output /Users/jamieannemortel/Documents/IHC_out \
  --prob-thresh 0.6 --nms-thresh 0.25 --rescale 1.5 \
  --vis --vis-prob --vis-bounds --debug-hist

"""

from __future__ import annotations
from time import time

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from skimage import io as skio, measure, exposure, filters, morphology, transform
from skimage.color import rgb2hed
from skimage.segmentation import find_boundaries
from csbdeep.utils import normalize
from stardist.models import StarDist2D

# Config dataclasses
@dataclass(frozen=True)
class Paths:
    input_dir: Path
    output_dir: Path
    overlay_dir: Optional[Path] = None
    side_by_side_dir: Optional[Path] = None

@dataclass(frozen=True)
class SegmentationConfig:
    model_name: str = "2D_versatile_fluo"
    prob_thresh: float = 0.5
    nms_thresh: float = 0.3
    tile: Optional[Tuple[int, int]] = None
    sd_norm_percentiles: Optional[Tuple[float, float]] = None
    sd_input: str = "hematoxylin"        # "gray" or "hematoxylin"
    rescale: Optional[float] = None       # e.g., 0.5 or 2.0
    use_clahe: bool = False
    median_radius: Optional[int] = None
    min_area: Optional[int] = None        # at ORIGINAL scale
    max_area: Optional[int] = None

@dataclass(frozen=True)
class GlobalQuantileConfig:
    q1: float = 33.0
    q2: float = 66.0
    min_od: float = 0.0
    weights: Tuple[int, int, int] = (1, 2, 3)

@dataclass(frozen=True)
class OverlayConfig:
    mode: str = "qupath"
    alpha: float = 0.35
    draw_boundaries_on_fill: bool = True
    boundary_thickness: int = 1

# NEW: lightweight, optional visualization config
@dataclass(frozen=True)
class DebugVisConfig:
    enable: bool = False               # master switch
    out_dir: Optional[Path] = None     # default: <output>/debug
    save_input: bool = False           # StarDist input (hematoxylin/gray)
    save_prob: bool = False            # probability heatmap + overlay
    save_bounds: bool = True           # RGB with thin instance boundaries
    alpha: float = 0.45                # prob overlay alpha
    max_images: Optional[int] = None   # limit images visualized (None = all)


# I/O + normalization
def ensure_dirs(paths: Paths) -> Paths:
    out = paths.output_dir
    out.mkdir(parents=True, exist_ok=True)
    overlay = paths.overlay_dir or (out / "overlays")
    overlay.mkdir(parents=True, exist_ok=True)
    sbs = paths.side_by_side_dir or (out / "side_by_side")
    sbs.mkdir(parents=True, exist_ok=True)
    return Paths(input_dir=paths.input_dir, output_dir=out, overlay_dir=overlay, side_by_side_dir=sbs)

def discover_images(input_dir: Path) -> List[Path]:
    exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
    return [p for p in sorted(input_dir.rglob("*")) if p.suffix.lower() in exts]

def read_image_rgb01(path: Path) -> np.ndarray:
    arr = skio.imread(str(path))
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.float32) / np.iinfo(arr.dtype).max
    else:
        arr = arr.astype(np.float32)
        vmax = float(np.nanmax(arr)) if arr.size else 1.0
        if vmax > 1.5:
            arr = arr / 255.0
    return np.clip(arr, 0, 1)

# Stains & measures
def rgb_to_dab_od(rgb01: np.ndarray) -> np.ndarray:
    hed = rgb2hed(np.clip(rgb01, 0, 1))
    return hed[..., 2]

def hematoxylin_channel(rgb01: np.ndarray) -> np.ndarray:
    hed = rgb2hed(np.clip(rgb01, 0, 1))
    h = hed[..., 0]
    h = (h - np.min(h)) / (np.max(h) - np.min(h) + 1e-8)
    return h.astype(np.float32)

def per_label_mean_with_ids(values: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    props = measure.regionprops_table(labels, intensity_image=values, properties=("label", "mean_intensity", "area"))
    order = np.argsort(props["label"])
    means = np.asarray(props["mean_intensity"])[order]
    label_ids = np.asarray(props["label"])[order]
    areas = np.asarray(props["area"])[order]
    return means, label_ids, areas


# StarDist + preprocessing
def load_stardist_model(name: str) -> StarDist2D:
    return StarDist2D.from_pretrained(name) if "/" not in name else StarDist2D(None, name=name)

def preprocess_for_stardist(rgb01: np.ndarray, seg_cfg: SegmentationConfig) -> np.ndarray:
    if seg_cfg.sd_input == "hematoxylin":
        im = hematoxylin_channel(rgb01)
    else:
        im = rgb01.mean(axis=-1).astype(np.float32)

    if seg_cfg.use_clahe:
        im = exposure.equalize_adapthist(im, clip_limit=0.01)
    if seg_cfg.median_radius and seg_cfg.median_radius > 0:
        im = filters.median(im, morphology.disk(seg_cfg.median_radius))

    if seg_cfg.sd_norm_percentiles is not None:
        lo, hi = seg_cfg.sd_norm_percentiles
        im = normalize(im, lo, hi, axis=None)

    return np.clip(im, 0, 1).astype(np.float32)

def stardist_segment(
    rgb01: np.ndarray,
    model: StarDist2D,
    seg_cfg: SegmentationConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      labels (H,W,int32), prob (H,W,float32 0..1), sd_input (H,W,float32 0..1)
    """
    sd_input = preprocess_for_stardist(rgb01, seg_cfg)
    scale = seg_cfg.rescale if seg_cfg.rescale and seg_cfg.rescale != 1.0 else None

    if scale is not None:
        im_small = transform.rescale(sd_input, scale, order=1, anti_aliasing=True,
                                     channel_axis=None, preserve_range=True)
        labels_small, _ = model.predict_instances(
            im_small, prob_thresh=seg_cfg.prob_thresh, nms_thresh=seg_cfg.nms_thresh, n_tiles=seg_cfg.tile
        )
        labels = transform.resize(labels_small.astype(np.int32), sd_input.shape,
                                  order=0, anti_aliasing=False, preserve_range=True).astype(np.int32)

        prob_small, _ = model.predict(im_small)
        prob = transform.resize(prob_small.astype(np.float32), sd_input.shape,
                                order=1, anti_aliasing=True, preserve_range=True).astype(np.float32)
    else:
        labels, _ = model.predict_instances(
            sd_input, prob_thresh=seg_cfg.prob_thresh, nms_thresh=seg_cfg.nms_thresh, n_tiles=seg_cfg.tile
        )
        prob, _ = model.predict(sd_input)
        prob = prob.astype(np.float32)

    return labels.astype(np.int32), np.clip(prob, 0, 1), sd_input.astype(np.float32)

def area_filter_labels(labels: np.ndarray, min_area: Optional[int], max_area: Optional[int]) -> np.ndarray:
    if min_area is None and max_area is None:
        return labels
    keep = np.ones(labels.max() + 1, dtype=bool)
    keep[0] = False
    for r in measure.regionprops(labels):
        a = r.area
        if (min_area is not None and a < min_area) or (max_area is not None and a > max_area):
            keep[r.label] = False
    mask = keep[labels]
    labels_filtered = labels.copy()
    labels_filtered[~mask] = 0
    labels_filtered = measure.label(labels_filtered > 0, connectivity=1)
    return labels_filtered.astype(np.int32)


# Global thresholds, binning, H-score
def compute_global_thresholds(all_nuc_od: np.ndarray, q1: float, q2: float, min_od: float) -> Tuple[float, float]:
    mask = np.isfinite(all_nuc_od) & (all_nuc_od > min_od)
    vals = all_nuc_od[mask]
    if vals.size == 0:
        raise ValueError("No nuclei passed the positive DAB filter; try lowering --min-od.")
    t1, t2 = np.percentile(vals, [q1, q2])
    return float(t1), float(t2)

def bin_od(od_values: np.ndarray, t1: float, t2: float, min_od: float) -> np.ndarray:
    bins = np.full_like(od_values, fill_value=-1, dtype=np.int32)
    valid = np.isfinite(od_values)
    neg_mask = valid & (od_values <= min_od)
    bins[neg_mask] = 0
    pos_mask = valid & (od_values > min_od)
    sub = od_values[pos_mask]
    b_sub = np.digitize(sub, [t1, t2], right=False) + 1
    bins[pos_mask] = b_sub
    return bins

def hscore_from_bins(bins: np.ndarray, weights: Tuple[int,int,int]) -> Tuple[float, Dict[str, float]]:
    pos_mask = bins > 0
    n_all = int(np.sum(bins >= 0))
    n_pos = int(np.sum(pos_mask))
    if n_pos == 0:
        return 0.0, {"n_nuclei": n_all, "p_neg": 1.0, "p_low": 0.0, "p_med": 0.0, "p_high": 0.0}
    counts = np.bincount(bins[pos_mask], minlength=4).astype(float)
    p_low, p_med, p_high = counts[1]/n_pos, counts[2]/n_pos, counts[3]/n_pos
    w1, w2, w3 = weights
    h = 100.0 * (p_low*w1 + p_med*w2 + p_high*w3)
    p_neg = (n_all - n_pos) / n_all if n_all > 0 else 0.0
    return float(h), {"n_nuclei": n_all, "p_neg": float(p_neg), "p_low": float(p_low),
                      "p_med": float(p_med), "p_high": float(p_high)}


# Overlays & composites

def qupath_palette() -> Dict[int, Tuple[float, float, float]]:
    return {0: (0.85, 0.85, 0.9), 1: (0.2, 0.6, 1.0), 2: (1.0, 0.8, 0.2), 3: (1.0, 0.2, 0.2)}

def labels_to_binmap(labels: np.ndarray, label_ids: np.ndarray, bins: np.ndarray) -> np.ndarray:
    binmap = np.full(labels.shape, fill_value=-1, dtype=np.int16)
    for lab, b in zip(label_ids, bins):
        if b >= 0:
            binmap[labels == lab] = int(b)
    return binmap

def compose_side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.shape != right.shape:
        right = transform.resize(right, left.shape, order=1, anti_aliasing=True, preserve_range=True).astype(left.dtype)
    return np.concatenate([left, right], axis=1)

def render_qupath_overlay(rgb01: np.ndarray, binmap: np.ndarray, alpha: float, draw_boundaries: bool) -> np.ndarray:
    colors = qupath_palette()
    overlay = rgb01.copy()
    for cls, color in colors.items():
        m = (binmap == cls)
        if not np.any(m): continue
        color_arr = np.array(color, dtype=np.float32).reshape(1, 1, 3)
        overlay[m] = (1 - alpha) * overlay[m] + alpha * color_arr
    if draw_boundaries:
        bounds = find_boundaries(binmap >= 0, mode="outer")
        overlay[bounds] = np.clip(overlay[bounds] * 0.25 + 0.75, 0, 1)
    return np.clip(overlay, 0, 1)


# Debug helpers (pure side effects; gated by DebugVisConfig)

def _rescale01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - np.nanmin(x)
    denom = np.nanmax(x) + 1e-8
    return x / denom


def _u8(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)

def colorize_probability(prob01: np.ndarray) -> np.ndarray:
    p = prob01.astype(np.float32)
    p -= p.min()
    p /= (p.max() - p.min() + 1e-8)
    cmap = plt.get_cmap("viridis")
    return cmap(p)[..., :3].astype(np.float32)

def overlay_on_rgb(rgb01: np.ndarray, heat_rgb01: np.ndarray, alpha: float) -> np.ndarray:
    if heat_rgb01.shape[:2] != rgb01.shape[:2]:
        heat_rgb01 = transform.resize(heat_rgb01, rgb01.shape, order=1, anti_aliasing=True, preserve_range=True).astype(np.float32)
    return np.clip((1 - alpha) * rgb01 + alpha * heat_rgb01, 0, 1)

def render_boundaries_on_rgb(rgb01: np.ndarray, labels: np.ndarray) -> np.ndarray:
    out = rgb01.copy()
    bounds = find_boundaries(labels > 0, mode="outer")
    out[bounds] = np.clip(out[bounds] * 0.2 + 0.8, 0, 1)
    return out

def ensure_debug_dir(base_out: Path, vis: DebugVisConfig) -> Path:
    return (vis.out_dir or (base_out / "debug"))

def maybe_save_debug(
    rgb: np.ndarray,
    labels: np.ndarray,
    prob01: np.ndarray,
    sd_input01: np.ndarray,
    out_base: Path,
    stem: str,
    vis: DebugVisConfig
) -> None:
    if not vis.enable:
        return
    out_dir = ensure_debug_dir(out_base, vis)
    out_dir.mkdir(parents=True, exist_ok=True)

    if vis.save_input:
        # already saving the StarDist input as RGB
        skio.imsave(str(out_dir / f"{stem}_sdinput.png"), _u8(np.repeat(sd_input01[..., None], 3, axis=-1)))

        # NEW: save Hematoxylin (H) and DAB (D) components
        h = hematoxylin_channel(rgb)       # H channel (nuclei)
        d = rgb_to_dab_od(rgb)             # D channel (DAB optical density)

        # Rescale to 0..1 for visualization; flip D for nicer viewing (bright=more DAB)
        h_viz = _rescale01(h)
        d_viz = _rescale01(-d)

        # Save as 3-channel grayscale PNGs for consistency
        skio.imsave(str(out_dir / f"{stem}_H.png"), _u8(np.repeat(h_viz[..., None], 3, axis=-1)))
        skio.imsave(str(out_dir / f"{stem}_D.png"), _u8(np.repeat(d_viz[..., None], 3, axis=-1)))

    if vis.save_prob:
        heat = colorize_probability(prob01)
        skio.imsave(str(out_dir / f"{stem}_prob.png"), _u8(heat))
        ov = overlay_on_rgb(rgb, heat, vis.alpha)
        skio.imsave(str(out_dir / f"{stem}_prob_overlay.png"), _u8(ov))

    if vis.save_bounds:
        b = render_boundaries_on_rgb(rgb, labels)
        skio.imsave(str(out_dir / f"{stem}_bounds.png"), _u8(b))


# Diagnostics (optional)

def plot_dab_histogram(all_nuc_od: np.ndarray, min_od: float, t1: float, t2: float, out_dir: Path) -> Path:
    vals = all_nuc_od[np.isfinite(all_nuc_od)]
    if vals.size == 0:
        raise ValueError("No valid DAB OD values to plot.")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(vals, bins=100, edgecolor="none", alpha=0.85)
    ax.axvline(min_od, color="C0", linestyle="--", label=f"min-od={min_od:.3f}")
    ax.axvline(t1,     color="C1", linestyle="--", label=f"t1={t1:.3f}")
    ax.axvline(t2,     color="C3", linestyle="--", label=f"t2={t2:.3f}")
    ax.set_xlabel("Per-nucleus mean DAB OD"); ax.set_ylabel("Count")
    ax.set_title("DAB OD distribution (all nuclei)"); ax.legend()
    out_path = out_dir / "dab_histogram.png"
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    return out_path

def plot_dab_cdf(all_nuc_od: np.ndarray, min_od: float, t1: float, t2: float, out_dir: Path) -> Path:
    vals = all_nuc_od[np.isfinite(all_nuc_od)]
    if vals.size == 0:
        raise ValueError("No valid DAB OD values to plot.")
    vals_sorted = np.sort(vals)
    cdf = np.arange(1, len(vals_sorted) + 1) / len(vals_sorted)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(vals_sorted, cdf, color="k", lw=1.5)
    ax.axvline(min_od, color="C0", linestyle="--", label=f"min-od={min_od:.3f}")
    ax.axvline(t1, color="C1", linestyle="--", label=f"t1={t1:.3f}")
    ax.axvline(t2, color="C3", linestyle="--", label=f"t2={t2:.3f}")
    ax.set_xlabel("Per-nucleus mean DAB OD"); ax.set_ylabel("Cumulative fraction")
    ax.set_ylim(0, 1.05); ax.set_title("Cumulative DAB OD distribution (all nuclei)"); ax.legend()
    out_path = out_dir / "dab_cdf.png"
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    return out_path


def compute_segmentation_metrics(labels: np.ndarray, prob_map: np.ndarray) -> Dict[str, float]:
    """Compute quick quality metrics for each image."""
    regions = measure.regionprops(labels)
    if not regions:
        return {
            "mean_prob": 0.0, "std_prob": np.nan,
            "n_nuclei": 0, "mean_area": np.nan, "median_circularity": np.nan
        }

    # Compute per-nucleus mean prob directly
    props = measure.regionprops_table(labels, intensity_image=prob_map,
                                      properties=('label', 'mean_intensity', 'area', 'perimeter'))
    probs = np.asarray(props["mean_intensity"])
    areas = np.asarray(props["area"])
    perim = np.asarray(props["perimeter"])
    circ = 4 * np.pi * areas / (perim ** 2 + 1e-8)

    return {
        "mean_prob": float(np.mean(probs)),
        "std_prob": float(np.std(probs)),
        "n_nuclei": len(probs),
        "mean_area": float(np.mean(areas)),
        "median_circularity": float(np.median(circ))
    }





# Main pipeline
def run_global_quantile_pipeline(
    paths: Paths,
    seg_cfg: SegmentationConfig,
    gq_cfg: GlobalQuantileConfig,
    ov_cfg: OverlayConfig,
    save_side_by_side: bool = True,
    debug_hist: bool = False,
    vis: Optional[DebugVisConfig] = None,
) -> Path:
    paths = ensure_dirs(paths)
    imgs = discover_images(paths.input_dir)
    if not imgs:
        raise FileNotFoundError(f"No images found under {paths.input_dir}")

    model = load_stardist_model(seg_cfg.model_name)
    vis = vis or DebugVisConfig(enable=False)

    nuc_ods: List[np.ndarray] = []
    nuc_label_ids: List[np.ndarray] = []
    labels_per_img: List[np.ndarray] = []
    rgb_paths: List[Path] = []
    prob_maps: List[np.ndarray] = []
    runtimes: List[float] = []
    image_mpx: List[float] = []   # megapixels for runtime normalization

    n_vis = 0
    for p in imgs:
        rgb = read_image_rgb01(p)

        # segmentation + prob + sd_input (DRY, single call)
        t0 = time()
        labels, prob01, sd_input01 = stardist_segment(rgb, model, seg_cfg)
        runtime = time() - t0
        prob_maps.append(prob01)
        runtimes.append(runtime)
        image_mpx.append((rgb.shape[0] * rgb.shape[1]) / 1e6)

        labels = area_filter_labels(labels, seg_cfg.min_area, seg_cfg.max_area)

        # per-nucleus DAB OD
        dab = rgb_to_dab_od(rgb)
        means, ids, areas = per_label_mean_with_ids(dab, labels)

        nuc_ods.append(means); nuc_label_ids.append(ids)
        labels_per_img.append(labels); rgb_paths.append(p)

        # Optional debug saves (first N images if vis.max_images is set)
        if vis.enable and (vis.max_images is None or n_vis < vis.max_images):
            maybe_save_debug(
                rgb=rgb, labels=labels, prob01=prob01, sd_input01=sd_input01,
                out_base=paths.output_dir, stem=p.stem, vis=vis
            )
            n_vis += 1

    all_od = np.concatenate(nuc_ods) if nuc_ods else np.array([])
    t1, t2 = compute_global_thresholds(all_od, gq_cfg.q1, gq_cfg.q2, gq_cfg.min_od)

    if debug_hist:
        hist_path = plot_dab_histogram(all_od, gq_cfg.min_od, t1, t2, paths.output_dir)
        cdf_path = plot_dab_cdf(all_od, gq_cfg.min_od, t1, t2, paths.output_dir)
        print(f"[debug] DAB histogram: {hist_path}")
        print(f"[debug] DAB CDF:       {cdf_path}")

    # Per-image results + overlays
    rows: List[Dict[str, object]] = []
    #for p, labels, means, ids in zip(rgb_paths, labels_per_img, nuc_ods, nuc_label_ids):
    for p, labels, means, ids, prob01, runtime, mpx in zip(
        rgb_paths, labels_per_img, nuc_ods, nuc_label_ids, prob_maps, runtimes, image_mpx
    ):
        bins = bin_od(means, t1, t2, gq_cfg.min_od)
        h, details = hscore_from_bins(bins, gq_cfg.weights)
    
        prob_stats = compute_segmentation_metrics(labels, prob_map=prob01)


        # row = {
        #     "file": p.name,
        #     "n_nuclei": details["n_nuclei"],
        #     "p_low": details["p_low"], "p_med": details["p_med"], "p_high": details["p_high"],
        #     "hscore_0_300": h,
        #     "global_t1": t1, "global_t2": t2,
        #     "q1_percent": gq_cfg.q1, "q2_percent": gq_cfg.q2,
        #     "min_od": gq_cfg.min_od,
        #     "sd_input": seg_cfg.sd_input,
        #     "rescale": seg_cfg.rescale if seg_cfg.rescale is not None else 1.0,
        #     "min_area": seg_cfg.min_area, "max_area": seg_cfg.max_area,
        # }
        row = {
            "file": p.name,
            "hscore_0_300": h,
            "p_low": details["p_low"], "p_med": details["p_med"], "p_high": details["p_high"],
            "n_nuclei": prob_stats["n_nuclei"],
            "mean_prob": prob_stats["mean_prob"],
            "std_prob": prob_stats["std_prob"],
            "mean_area": prob_stats["mean_area"],
            "median_circularity": prob_stats["median_circularity"],
            "runtime_sec": runtime,
            "runtime_s_per_mp": runtime / max(mpx, 1e-9),  # guard divide-by-zero just in case
            "global_t1": t1, "global_t2": t2,
            "q1_percent": gq_cfg.q1, "q2_percent": gq_cfg.q2,
            "min_od": gq_cfg.min_od,
            "sd_input": seg_cfg.sd_input,
            "rescale": seg_cfg.rescale if seg_cfg.rescale is not None else 1.0,
            "min_area": seg_cfg.min_area, "max_area": seg_cfg.max_area,
        }

        rows.append(row)

        if ov_cfg.mode != "none":
            rgb = read_image_rgb01(p)
            binmap = labels_to_binmap(labels, ids, bins)
            ov = render_qupath_overlay(rgb, binmap, alpha=ov_cfg.alpha, draw_boundaries=ov_cfg.draw_boundaries_on_fill)
            ov_path = paths.overlay_dir / f"{p.stem}_overlay.png"
            skio.imsave(str(ov_path), _u8(ov))
            if save_side_by_side:
                sbs = compose_side_by_side(rgb, ov)
                sbs_path = paths.side_by_side_dir / f"{p.stem}_sbs.png"
                skio.imsave(str(sbs_path), _u8(sbs))

    df = pd.DataFrame(rows).sort_values("file")
    csv_path = paths.output_dir / "hscore_global_quantile_results.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


# CLI
def parse_tile(s: Optional[str]) -> Optional[Tuple[int, int]]:
    if not s:
        return None
    try:
        a, b = s.split(",")
        return (int(a), int(b))
    except Exception:
        raise argparse.ArgumentTypeError("tile must be like '2,2'")

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Global-quantile HDAB H-score via StarDist2D (with optional debug visuals).")
    ap.add_argument("--input", type=Path, required=True, help="Folder with input images")
    ap.add_argument("--output", type=Path, required=True, help="Folder to write CSV, overlays/, side_by_side/")
    ap.add_argument("--model", type=str, default="2D_versatile_fluo", help="StarDist2D model alias or path")
    ap.add_argument("--prob-thresh", type=float, default=0.5, help="StarDist probability threshold")
    ap.add_argument("--nms-thresh", type=float, default=0.3, help="StarDist NMS threshold")
    ap.add_argument("--tile", type=str, default=None, help="Tiling as 'ny,nx' (e.g., '2,2')")
    ap.add_argument("--sd-norm-percentiles", type=float, nargs=2, default=None, metavar=("PLOW","PHIGH"),
                    help="Percentile-normalize SD input grayscale (e.g., 1 99.8); leave unset for minimal normalization.")
    ap.add_argument("--sd-input", type=str, choices=["gray","hematoxylin"], default="hematoxylin",
                    help="StarDist input channel (hematoxylin recommended for HDAB).")
    ap.add_argument("--rescale", type=float, default=None, help="Rescale factor for segmentation (e.g., 0.5 or 2.0).")
    ap.add_argument("--clahe", action="store_true", help="Apply CLAHE to StarDist input.")
    ap.add_argument("--median-radius", type=int, default=None, help="Median denoise radius (pixels) before SD.")
    ap.add_argument("--min-area", type=int, default=None, help="Minimum nucleus area (pixels) to keep.")
    ap.add_argument("--max-area", type=int, default=None, help="Maximum nucleus area (pixels) to keep.")
    ap.add_argument("--q1", type=float, default=33.0, help="Quantile 1 percent (0-100), low/med split")
    ap.add_argument("--q2", type=float, default=66.0, help="Quantile 2 percent (0-100), med/high split")
    ap.add_argument("--min-od", type=float, default=0.0, help="OD floor; nuclei <= min-od excluded from quantile fit and binning")
    ap.add_argument("--overlay", type=str, default="qupath", choices=["qupath","boundaries","none"], help="Overlay mode")
    ap.add_argument("--overlay-alpha", type=float, default=0.35, help="Alpha for filled class masks")
    ap.add_argument("--overlay-boundaries", action="store_true", help="Draw thin boundaries over filled masks")
    ap.add_argument("--side-by-side", dest="side_by_side", action="store_true", help="Save [original|overlay] composite")
    ap.add_argument("--no-side-by-side", dest="side_by_side", action="store_false", help="Do not save composite")
    ap.add_argument("--debug-hist", action="store_true", help="Save a DAB OD histogram and CDF with min-od/t1/t2 lines.")
    ap.set_defaults(side_by_side=True)

    # Visualization (off by default)
    ap.add_argument("--vis", action="store_true", help="Enable saving debug visuals (prob/bounds/input).")
    ap.add_argument("--vis-input", action="store_true", help="Save the StarDist input image (hematoxylin/gray).")
    ap.add_argument("--vis-prob", action="store_true", help="Save probability heatmap and overlay.")
    ap.add_argument("--vis-bounds", action="store_true", help="Save RGB with thin instance boundaries.")
    ap.add_argument("--vis-alpha", type=float, default=0.45, help="Alpha for probability overlay.")
    ap.add_argument("--vis-max", type=int, default=None, help="Max #images to visualize (default: all).")
    ap.add_argument("--vis-dir", type=Path, default=None, help="Custom dir for debug outputs (default: <output>/debug).")
    return ap

def main(argv: Optional[Iterable[str]] = None) -> None:
    ap = build_argparser()
    args = ap.parse_args(argv)

    paths = Paths(input_dir=args.input, output_dir=args.output)
    seg_cfg = SegmentationConfig(
        model_name=args.model,
        prob_thresh=args.prob_thresh,
        nms_thresh=args.nms_thresh,
        tile=parse_tile(args.tile),
        sd_norm_percentiles=tuple(args.sd_norm_percentiles) if args.sd_norm_percentiles is not None else None,
        sd_input=args.sd_input,
        rescale=args.rescale,
        use_clahe=args.clahe,
        median_radius=args.median_radius,
        min_area=args.min_area,
        max_area=args.max_area,
    )
    gq_cfg = GlobalQuantileConfig(q1=args.q1, q2=args.q2, min_od=args.min_od)
    ov_cfg = OverlayConfig(mode=args.overlay, alpha=args.overlay_alpha, draw_boundaries_on_fill=args.overlay_boundaries)

    # If user just enabled --vis with no sub-flags, default to bounds
    default_bounds = (not args.vis_input and not args.vis_prob)
    vis_cfg = DebugVisConfig(
        enable=args.vis,
        out_dir=args.vis_dir,
        save_input=args.vis_input,
        save_prob=args.vis_prob,
        save_bounds=args.vis_bounds or default_bounds,
        alpha=args.vis_alpha,
        max_images=args.vis_max,
    )

    csv_path = run_global_quantile_pipeline(
        paths, seg_cfg, gq_cfg, ov_cfg,
        save_side_by_side=args.side_by_side,
        debug_hist=args.debug_hist,
        vis=vis_cfg,
    )
    print(f"Wrote: {csv_path}")

if __name__ == "__main__":
    main()
