
"""
REFACTORED.

New options for better segmentation quality on IHC:
  --sd-input {gray,hematoxylin}   : use hematoxylin channel (recommended for HDAB) for StarDist input
  --rescale FLOAT                 : rescale image before segmentation (e.g., 0.5 or 2.0); masks scaled back
  --clahe                         : enable local contrast (CLAHE) on StarDist input
  --median-radius INT             : median denoise radius (pixels) on StarDist input
  --min-area / --max-area         : filter out tiny or huge labels post-segmentation (in pixels at original scale)
"""

from __future__ import annotations

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



# Config
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
    sd_input: str = "hematoxylin"  # "gray" or "hematoxylin"
    rescale: Optional[float] = None  # e.g., 0.5 or 2.0
    use_clahe: bool = False
    median_radius: Optional[int] = None
    min_area: Optional[int] = None  # area filter at ORIGINAL scale
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
        arr = np.stack([arr]*3, axis=-1)
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
    # normalize to 0..1 for network
    h = (h - np.min(h)) / (np.max(h) - np.min(h) + 1e-8)
    return h.astype(np.float32)


def per_label_mean_with_ids(values: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    props = measure.regionprops_table(labels, intensity_image=values, properties=("label", "mean_intensity", "area"))
    order = np.argsort(props["label"])
    means = np.asarray(props["mean_intensity"])[order]
    label_ids = np.asarray(props["label"])[order]
    areas = np.asarray(props["area"])[order]
    return means, label_ids, areas



# StarDist
def load_stardist_model(name: str) -> StarDist2D:
    return StarDist2D.from_pretrained(name) if "/" not in name else StarDist2D(None, name=name)


def preprocess_for_stardist(rgb01: np.ndarray, seg_cfg: SegmentationConfig) -> np.ndarray:
    # Choose input
    if seg_cfg.sd_input == "hematoxylin":
        im = hematoxylin_channel(rgb01)
    else:
        im = rgb01.mean(axis=-1).astype(np.float32)

    # Optional local contrast or denoise
    if seg_cfg.use_clahe:
        im = exposure.equalize_adapthist(im, clip_limit=0.01)
    if seg_cfg.median_radius and seg_cfg.median_radius > 0:
        im = filters.median(im, morphology.disk(seg_cfg.median_radius))

    # Optional percentile normalization for SD
    if seg_cfg.sd_norm_percentiles is not None:
        lo, hi = seg_cfg.sd_norm_percentiles
        im = normalize(im, lo, hi, axis=None)

    return np.clip(im, 0, 1).astype(np.float32)


def stardist_segment(rgb01: np.ndarray, model: StarDist2D, seg_cfg: SegmentationConfig) -> np.ndarray:
    im = preprocess_for_stardist(rgb01, seg_cfg)

    # Optional rescaling to match training object scale
    scale = seg_cfg.rescale if seg_cfg.rescale and seg_cfg.rescale != 1.0 else None
    if scale is not None:
        im_small = transform.rescale(im, scale, order=1, anti_aliasing=True, channel_axis=None, preserve_range=True)
        labels_small, _ = model.predict_instances(
            im_small,
            prob_thresh=seg_cfg.prob_thresh,
            nms_thresh=seg_cfg.nms_thresh,
            n_tiles=seg_cfg.tile
        )
        # scale labels back to original size (nearest to keep labels integer)
        labels = transform.resize(labels_small.astype(np.int32), im.shape, order=0, anti_aliasing=False, preserve_range=True).astype(np.int32)
    else:
        labels, _ = model.predict_instances(
            im,
            prob_thresh=seg_cfg.prob_thresh,
            nms_thresh=seg_cfg.nms_thresh,
            n_tiles=seg_cfg.tile
        )

    return labels.astype(np.int32)


def area_filter_labels(labels: np.ndarray, min_area: Optional[int], max_area: Optional[int]) -> np.ndarray:
    if min_area is None and max_area is None:
        return labels
    keep = np.ones(labels.max()+1, dtype=bool)
    keep[0] = False
    props = measure.regionprops(labels)
    for r in props:
        a = r.area
        if (min_area is not None and a < min_area) or (max_area is not None and a > max_area):
            keep[r.label] = False
    # relabel
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
    """
    Returns 4 bins:
      -1 = excluded (NaN, invalid)
       0 = negative (<= min_od)
       1 = low / weak (min_od < od < t1)
       2 = medium (t1 <= od < t2)
       3 = high (>= t2)
    """
    bins = np.full_like(od_values, fill_value=-1, dtype=np.int32)
    valid = np.isfinite(od_values)

    # negatives
    neg_mask = valid & (od_values <= min_od)
    bins[neg_mask] = 0

    # positive nuclei
    pos_mask = valid & (od_values > min_od)
    sub = od_values[pos_mask]
    b_sub = np.digitize(sub, [t1, t2], right=False) + 1  # +1 so bins are 1..3
    bins[pos_mask] = b_sub

    return bins



def hscore_from_bins(bins: np.ndarray, weights: Tuple[int,int,int]) -> Tuple[float, Dict[str, float]]:
    # Ignore negatives (bin 0) for scoring
    pos_mask = bins > 0
    n_all = np.sum(bins >= 0)
    n_pos = np.sum(pos_mask)

    if n_pos == 0:
        return 0.0, {"n_nuclei": int(n_all), "p_neg": 1.0, "p_low": 0.0, "p_med": 0.0, "p_high": 0.0}

    counts = np.bincount(bins[pos_mask], minlength=4).astype(float)
    p_low, p_med, p_high = counts[1]/n_pos, counts[2]/n_pos, counts[3]/n_pos
    w1, w2, w3 = weights
    h = 100.0 * (p_low*w1 + p_med*w2 + p_high*w3)
    p_neg = (n_all - n_pos) / n_all if n_all > 0 else 0.0

    return float(h), {
        "n_nuclei": int(n_all),
        "p_neg": float(p_neg),
        "p_low": float(p_low),
        "p_med": float(p_med),
        "p_high": float(p_high),
    }


# Overlays & composites
def qupath_palette() -> Dict[int, Tuple[float, float, float]]:
    return {
        0: (0.85, 0.85, 0.9),  # negative
        1: (0.2, 0.6, 1.0),    # low
        2: (1.0, 0.8, 0.2),    # med
        3: (1.0, 0.2, 0.2),    # high
    }


def labels_to_binmap(labels: np.ndarray, label_ids: np.ndarray, bins: np.ndarray) -> np.ndarray:
    binmap = np.full(labels.shape, fill_value=-1, dtype=np.int16)
    for lab, b in zip(label_ids, bins):
        if b >= 0:
            binmap[labels == lab] = int(b)
    return binmap

def compose_side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """
    Create a side-by-side composite [left | right].
    If sizes differ slightly, resize right to match left.
    """
    if left.shape != right.shape:
        right = transform.resize(
            right,
            left.shape,
            order=1,
            anti_aliasing=True,
            preserve_range=True
        ).astype(left.dtype)
    return np.concatenate([left, right], axis=1)


def render_qupath_overlay(rgb01: np.ndarray, binmap: np.ndarray, alpha: float, draw_boundaries: bool) -> np.ndarray:
    colors = qupath_palette()
    overlay = rgb01.copy()
    for cls, color in colors.items():
        m = (binmap == cls)
        if not np.any(m):
            continue
        color_arr = np.array(color, dtype=np.float32).reshape(1,1,3)
        overlay[m] = (1 - alpha) * overlay[m] + alpha * color_arr
    if draw_boundaries:
        bounds = find_boundaries(binmap >= 0, mode="outer")
        overlay[bounds] = np.clip(overlay[bounds] * 0.25 + 0.75, 0, 1)
    return np.clip(overlay, 0, 1)\
    
def plot_dab_histogram(all_nuc_od: np.ndarray, min_od: float, t1: float, t2: float, out_dir: Path) -> Path:
    """
    Plot histogram of per-nucleus mean DAB OD across the dataset.
    Shows min-od (negative/positive gate) and global t1/t2 (weak/med/strong splits).
    """
    vals = all_nuc_od[np.isfinite(all_nuc_od)]
    if vals.size == 0:
        raise ValueError("No valid DAB OD values to plot.")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(vals, bins=100, edgecolor="none", alpha=0.85)
    ax.axvline(min_od, color="C0", linestyle="--", label=f"min-od={min_od:.3f}")
    ax.axvline(t1,     color="C1", linestyle="--", label=f"t1={t1:.3f}")
    ax.axvline(t2,     color="C3", linestyle="--", label=f"t2={t2:.3f}")
    ax.set_xlabel("Per-nucleus mean DAB OD")
    ax.set_ylabel("Count")
    ax.set_title("DAB OD distribution (all nuclei)")
    ax.legend()
    out_path = out_dir / "dab_histogram.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path

def plot_dab_cdf(all_nuc_od: np.ndarray, min_od: float, t1: float, t2: float, out_dir: Path) -> Path:
    """
    Plot cumulative distribution (CDF) of per-nucleus mean DAB OD.
    Visualizes what fraction of nuclei lie below each OD level.
    """
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
    ax.set_xlabel("Per-nucleus mean DAB OD")
    ax.set_ylabel("Cumulative fraction")
    ax.set_ylim(0, 1.05)
    ax.set_title("Cumulative DAB OD distribution (all nuclei)")
    ax.legend()

    out_path = out_dir / "dab_cdf.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# Main two-phase pipeline
def run_global_quantile_pipeline(
    paths: Paths,
    seg_cfg: SegmentationConfig,
    gq_cfg: GlobalQuantileConfig,
    ov_cfg: OverlayConfig,
    save_side_by_side: bool = True,
    debug_hist: bool = False,
) -> Path:
    paths = ensure_dirs(paths)
    imgs = discover_images(paths.input_dir)
    if not imgs:
        raise FileNotFoundError(f"No images found under {paths.input_dir}")

    model = load_stardist_model(seg_cfg.model_name)

    nuc_ods: List[np.ndarray] = []
    nuc_label_ids: List[np.ndarray] = []
    labels_per_img: List[np.ndarray] = []
    rgb_paths: List[Path] = []

    for p in imgs:
        rgb = read_image_rgb01(p)
        labels = stardist_segment(rgb, model, seg_cfg)
        labels = area_filter_labels(labels, seg_cfg.min_area, seg_cfg.max_area)
        dab = rgb_to_dab_od(rgb)
        means, ids, areas = per_label_mean_with_ids(dab, labels)
        nuc_ods.append(means)
        nuc_label_ids.append(ids)
        labels_per_img.append(labels)
        rgb_paths.append(p)

    all_od = np.concatenate(nuc_ods) if nuc_ods else np.array([])
    t1, t2 = compute_global_thresholds(all_od, gq_cfg.q1, gq_cfg.q2, gq_cfg.min_od)

    if debug_hist:
        hist_path = plot_dab_histogram(all_od, gq_cfg.min_od, t1, t2, paths.output_dir)
        cdf_path = plot_dab_cdf(all_od, gq_cfg.min_od, t1, t2, paths.output_dir)
        print(f"[debug] DAB histogram saved to: {hist_path}")
        print(f"[debug] DAB CDF saved to: {cdf_path}")


    rows: List[Dict[str, object]] = []
    for p, labels, means, ids in zip(rgb_paths, labels_per_img, nuc_ods, nuc_label_ids):
        bins = bin_od(means, t1, t2, gq_cfg.min_od)
        h, details = hscore_from_bins(bins, gq_cfg.weights)

        row = {
            "file": p.name,
            "n_nuclei": details["n_nuclei"],
            "p_low": details["p_low"],
            "p_med": details["p_med"],
            "p_high": details["p_high"],
            "hscore_0_300": h,
            "global_t1": t1,
            "global_t2": t2,
            "q1_percent": gq_cfg.q1,
            "q2_percent": gq_cfg.q2,
            "min_od": gq_cfg.min_od,
            "sd_input": seg_cfg.sd_input,
            "rescale": seg_cfg.rescale if seg_cfg.rescale is not None else 1.0,
            "min_area": seg_cfg.min_area,
            "max_area": seg_cfg.max_area,
        }
        rows.append(row)

        if ov_cfg.mode != "none":
            rgb = read_image_rgb01(p)
            binmap = labels_to_binmap(labels, ids, bins)
            ov = render_qupath_overlay(rgb, binmap, alpha=ov_cfg.alpha, draw_boundaries=ov_cfg.draw_boundaries_on_fill)
            ov_path = paths.overlay_dir / f"{p.stem}_overlay.png"
            skio.imsave(str(ov_path), (np.clip(ov, 0, 1) * 255).astype(np.uint8))
            if save_side_by_side:
                sbs = compose_side_by_side(rgb, ov)
                sbs_path = paths.side_by_side_dir / f"{p.stem}_sbs.png"
                skio.imsave(str(sbs_path), (np.clip(sbs, 0, 1) * 255).astype(np.uint8))

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
    ap = argparse.ArgumentParser(description="Global-quantile HDAB H-score via StarDist2D (improved segmentation).")
    ap.add_argument("--input", type=Path, required=True, help="Folder with input images")
    ap.add_argument("--output", type=Path, required=True, help="Folder to write CSV, overlays/, side_by_side/")
    ap.add_argument("--model", type=str, default="2D_versatile_fluo", help="StarDist2D model alias or path")
    ap.add_argument("--prob-thresh", type=float, default=0.5, help="StarDist probability threshold")
    ap.add_argument("--nms-thresh", type=float, default=0.3, help="StarDist NMS threshold")
    ap.add_argument("--tile", type=str, default=None, help="Tiling as 'ny,nx' (e.g., '2,2')")
    ap.add_argument("--sd-norm-percentiles", type=float, nargs=2, default=None, metavar=("PLOW","PHIGH"),
                    help="If set, percentile-normalize StarDist input grayscale (e.g., 1 99.8). Leave unset for minimal normalization.")
    ap.add_argument("--sd-input", type=str, choices=["gray","hematoxylin"], default="hematoxylin",
                    help="Choose StarDist input channel (hematoxylin recommended for HDAB).")
    ap.add_argument("--rescale", type=float, default=None, help="Rescale factor for segmentation (e.g., 0.5 or 2.0).")
    ap.add_argument("--clahe", action="store_true", help="Apply CLAHE (local contrast) to StarDist input.")
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
    ap.add_argument("--debug-hist", action="store_true",
                help="If set, save a DAB OD histogram (per-nucleus means) with min-od/t1/t2 lines.")
    ap.set_defaults(side_by_side=True)
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

    csv_path = run_global_quantile_pipeline(
        paths, seg_cfg, gq_cfg, ov_cfg, save_side_by_side=args.side_by_side, debug_hist=args.debug_hist,    
    )
    print(f"Wrote: {csv_path}")


# if __name__ == "__main__":
#     import sys
#     import argparse

#     # Parse debug-only flags without disturbing your normal CLI
#     pv = argparse.ArgumentParser(add_help=False)
#     pv.add_argument("--probvis", action="store_true",
#                     help="Run a one-off StarDist probability-map visualization and exit.")
#     pv.add_argument("--probvis-image", type=Path, default=None,
#                     help="Path to one image to visualize.")
#     pv.add_argument("--probvis-out", type=Path, default=None,
#                     help="Output folder (default: alongside the image).")
#     pv.add_argument("--probvis-alpha", type=float, default=0.45,
#                     help="Alpha for probability heatmap overlay.")
#     # (Optional) a few SD knobs just for the debug call
#     pv.add_argument("--probvis-model", type=str, default="2D_versatile_fluo")
#     pv.add_argument("--probvis-sd-input", choices=["gray","hematoxylin"], default="hematoxylin")
#     pv.add_argument("--probvis-rescale", type=float, default=None)
#     pv.add_argument("--probvis-clahe", action="store_true")
#     pv.add_argument("--probvis-median-radius", type=int, default=None)
#     pv.add_argument("--probvis-prob-thresh", type=float, default=0.5)
#     pv.add_argument("--probvis-nms-thresh", type=float, default=0.3)
#     dbg, remaining = pv.parse_known_args(sys.argv[1:])

#     if not dbg.probvis:
#         # run your normal pipeline
#         main(remaining)
#         sys.exit(0)

#     # prob map debug
#     if dbg.probvis_image is None:
#         pv.error("--probvis-image is required when using --probvis")

#     def _u8(x: np.ndarray) -> np.ndarray:
#         return (np.clip(x, 0, 1) * 255).astype(np.uint8)

#     def _overlay(rgb01: np.ndarray, heat01_rgb: np.ndarray, alpha: float) -> np.ndarray:
#         return np.clip((1.0 - alpha) * rgb01 + alpha * heat01_rgb, 0, 1)

#     # Build minimal segmentation config ad-hoc run
#     seg_cfg = SegmentationConfig(
#         model_name=dbg.probvis_model,
#         prob_thresh=dbg.probvis_prob_thresh,
#         nms_thresh=dbg.probvis_nms_thresh,
#         tile=None,
#         sd_norm_percentiles=None,   # minimal normalization
#         sd_input=dbg.probvis_sd_input,
#         rescale=dbg.probvis_rescale,
#         use_clahe=dbg.probvis_clahe,
#         median_radius=dbg.probvis_median_radius,
#         min_area=None,
#         max_area=None,
#     )

#     # I/O
#     img_path: Path = dbg.probvis_image
#     out_dir: Path = dbg.probvis_out or (img_path.parent / f"{img_path.stem}_PROBVIS")
#     out_dir.mkdir(parents=True, exist_ok=True)

#     # Load data + model
#     rgb = read_image_rgb01(img_path)
#     model = load_stardist_model(seg_cfg.model_name)

#     # Get the exact SD input and probability map
#     im = preprocess_for_stardist(rgb, seg_cfg)
#     # prob/dist via StarDist; index [0] = prob
#     if seg_cfg.rescale and seg_cfg.rescale != 1.0:
#         im_small = transform.rescale(im, seg_cfg.rescale, order=1, anti_aliasing=True, channel_axis=None, preserve_range=True)
#         prob_small, _ = model.predict(im_small)
#         prob = transform.resize(prob_small, im.shape, order=1, anti_aliasing=True, preserve_range=True).astype(np.float32)
#     else:
#         prob, _ = model.predict(im)
#         prob = prob.astype(np.float32)

#     # Colorize probability map (viridis), normalize 0..1
#     p = prob.copy()
#     p -= p.min()
#     p /= (p.max() - p.min() + 1e-8)
#     cmap = plt.get_cmap("viridis")
#     p_rgb = cmap(p)[..., :3]  # RGBA->RGB, still 0..1

#     # Labels for boundaries (use your existing instance predictor)
#     labels = stardist_segment(rgb, model, seg_cfg)
#     if isinstance(labels, tuple):  # in case stardist_segment returns (labels, prob)
#         labels = labels[0]
#     labels = labels.astype(np.int32)

#     # Save outputs
#     prob_path = out_dir / f"{img_path.stem}_prob.png"
#     skio.imsave(str(prob_path), _u8(p_rgb))

#     overlay = _overlay(rgb, p_rgb, dbg.probvis_alpha)
#     ov_path = out_dir / f"{img_path.stem}_prob_overlay.png"
#     skio.imsave(str(ov_path), _u8(overlay))

#     bounds = find_boundaries(labels > 0, mode="outer")
#     overlay_b = rgb.copy()
#     overlay_b[bounds] = np.clip(overlay_b[bounds] * 0.2 + 0.8, 0, 1)
#     b_path = out_dir / f"{img_path.stem}_bounds.png"
#     skio.imsave(str(b_path), _u8(overlay_b))

#     print(f"[probvis] Saved:\n  {prob_path}\n  {ov_path}\n  {b_path}")
#     sys.exit(0)

# -------------------------------------------------------------------
# Batch probability-map visualizer for multiple random images
# Usage:
#   python hdab_stardist_hscore_global_quant.py \
#       --probvis-batch --probvis-dir /path/to/images --probvis-out /tmp/debug --n 3

if __name__ == "__main__":
    import sys, random, argparse

    pv = argparse.ArgumentParser(add_help=False)
    pv.add_argument("--probvis-batch", action="store_true",
                    help="Run probability map visualization on N random images from a directory.")
    pv.add_argument("--probvis-dir", type=Path, default=None,
                    help="Input directory containing images.")
    pv.add_argument("--probvis-out", type=Path, default=None,
                    help="Output directory (default: 'PROBVIS' inside input dir).")
    pv.add_argument("-n", type=int, default=3,
                    help="Number of random images to visualize (default=3).")
    pv.add_argument("--probvis-alpha", type=float, default=0.45,
                    help="Alpha for probability overlay.")
    pv.add_argument("--probvis-model", type=str, default="2D_versatile_fluo")
    pv.add_argument("--probvis-sd-input", choices=["gray","hematoxylin"], default="hematoxylin")
    pv.add_argument("--probvis-rescale", type=float, default=None)
    pv.add_argument("--probvis-clahe", action="store_true")
    pv.add_argument("--probvis-median-radius", type=int, default=None)
    pv.add_argument("--probvis-prob-thresh", type=float, default=0.5)
    pv.add_argument("--probvis-nms-thresh", type=float, default=0.3)
    dbg, remaining = pv.parse_known_args(sys.argv[1:])

    if not dbg.probvis_batch:
        # fall back to normal execution
        main(remaining)
        sys.exit(0)

    if dbg.probvis_dir is None:
        pv.error("--probvis-dir is required when using --probvis-batch")

    # ---------- helpers ----------
    def _u8(x: np.ndarray) -> np.ndarray:
        return (np.clip(x, 0, 1) * 255).astype(np.uint8)

    def _overlay(rgb01: np.ndarray, heat01_rgb: np.ndarray, alpha: float) -> np.ndarray:
        return np.clip((1.0 - alpha) * rgb01 + alpha * heat01_rgb, 0, 1)

    def _match_shape(img: np.ndarray, ref: np.ndarray) -> np.ndarray:
        """Resize img (H,W[,C]) to match ref.shape using bilinear resampling, keep 0..1 range."""
        if img.shape == ref.shape:
            return img.astype(np.float32, copy=False)
        out = transform.resize(
            img, ref.shape, order=1, anti_aliasing=True, preserve_range=True
        ).astype(np.float32)
        return np.clip(out, 0, 1)

    # ---------- config & model ----------
    seg_cfg = SegmentationConfig(
        model_name=dbg.probvis_model,
        prob_thresh=dbg.probvis_prob_thresh,
        nms_thresh=dbg.probvis_nms_thresh,
        tile=None,
        sd_norm_percentiles=None,
        sd_input=dbg.probvis_sd_input,
        rescale=dbg.probvis_rescale,
        use_clahe=dbg.probvis_clahe,
        median_radius=dbg.probvis_median_radius,
    )
    model = load_stardist_model(seg_cfg.model_name)

    # ---------- select images ----------
    imgs = discover_images(dbg.probvis_dir)
    if len(imgs) == 0:
        print("No images found.")
        sys.exit(1)

    chosen = random.sample(imgs, min(dbg.n, len(imgs)))
    out_dir = dbg.probvis_out or (dbg.probvis_dir / "PROBVIS")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[probvis-batch] Selected {len(chosen)} images.")
    for p in chosen:
        print(f"[probvis] processing {p.name}")
        rgb = read_image_rgb01(p).astype(np.float32)

        # preprocess & probability map
        im = preprocess_for_stardist(rgb, seg_cfg)
        if seg_cfg.rescale and seg_cfg.rescale != 1.0:
            im_small = transform.rescale(im, seg_cfg.rescale, order=1, anti_aliasing=True,
                                         channel_axis=None, preserve_range=True)
            prob_small, _ = model.predict(im_small)
            prob = transform.resize(prob_small, im.shape, order=1, anti_aliasing=True,
                                    preserve_range=True).astype(np.float32)
        else:
            prob, _ = model.predict(im)
            prob = prob.astype(np.float32)

        # normalize + colorize prob
        pnorm = prob - prob.min()
        pnorm /= (pnorm.max() - pnorm.min() + 1e-8)
        cmap = plt.get_cmap("viridis")
        p_rgb = cmap(pnorm)[..., :3].astype(np.float32)  # 0..1

        # match sizes to RGB for blending
        p_rgb = _match_shape(p_rgb, rgb)

        # labels & boundaries (for thin edges)
        labels = stardist_segment(rgb, model, seg_cfg)
        if isinstance(labels, tuple):  # backward compatible if function returns (labels, prob)
            labels = labels[0]
        labels = labels.astype(np.int32)
        bounds = find_boundaries(labels > 0, mode="outer")
        if bounds.shape != rgb.shape[:2]:
            bounds = transform.resize(bounds.astype(float), rgb.shape[:2],
                                      order=0, anti_aliasing=False, preserve_range=True) > 0.5

        # overlay
        overlay = _overlay(rgb, p_rgb, dbg.probvis_alpha)
        overlay[bounds] = np.clip(overlay[bounds] * 0.2 + 0.8, 0, 1)

        # save outputs
        base = p.stem
        skio.imsave(str(out_dir / f"{base}_prob.png"), _u8(p_rgb))
        skio.imsave(str(out_dir / f"{base}_prob_overlay.png"), _u8(overlay))
        print(f"  saved {base}_prob.png and {base}_prob_overlay.png")

    print(f"[probvis-batch] done. results in {out_dir}")
    sys.exit(0)
