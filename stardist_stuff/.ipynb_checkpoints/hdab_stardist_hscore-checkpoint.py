#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HDAB → StarDist → Dataset-wide Low/Medium/High binning → H-score

Beginner-friendly, readable script.

Pipeline (per image):
1) Read RGB HDAB image
2) Color deconvolution (rgb2hed) → Hematoxylin (H), DAB (D)
3) StarDist nuclei segmentation on H
4) Per-nucleus DAB OD metrics: mean, median, p99
5) (Later) Classify each nucleus as Low / Medium / High using dataset-wide thresholds
6) Compute image-level H-score and save outputs

Dataset steps:
A) First pass: collect chosen metric (e.g., p99) for all nuclei across the dataset
B) Clip extremes via percentiles (background/outlier removal) + save histogram/CDF plots
C) Learn dataset-wide thresholds:
   - "multiotsu": two cut points from the global distribution
   - "quantile":   two cut points from quantiles (e.g., 33% and 66%)
D) Second pass: classify each nucleus with learned thresholds, compute H-score, save results

Outputs:
- per-image:
    <out_dir>/<image_stem>/<image_stem>_per_nucleus_metrics.csv
    <out_dir>/<image_stem>/<image_stem>_labels.tif
    <out_dir>/<image_stem>/<image_stem>_overlay.png
- dataset:
    <out_dir>/dataset_summary.csv
    <out_dir>/nuclei_all.csv
    <out_dir>/global_metric_hist_cdf_before.png
    <out_dir>/global_metric_hist_cdf_after.png
"""

import argparse
import traceback
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tifffile import imread, imsave
from skimage import io as skio, exposure, util, measure
from skimage.color import rgb2hed
from skimage.filters import threshold_multiotsu

from csbdeep.utils import normalize
from stardist.models import StarDist2D

# ------------------------- Configuration -------------------------

IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}

# Colors for Low / Medium / High overlay (BGR-ish but used as RGB here)
COLOR_LOW    = (180, 130, 255)   # light purple
COLOR_MEDIUM = (120, 220, 120)   # green
COLOR_HIGH   = (255, 160,  70)   # orange


# ------------------------- Utility functions -------------------------

def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def list_images(data_dir: Path, recursive: bool) -> List[Path]:
    if recursive:
        return sorted([p for p in data_dir.rglob("*") if is_image_file(p)])
    else:
        return sorted([p for p in data_dir.iterdir() if is_image_file(p)])


def read_image_rgb(path: Path) -> np.ndarray:
    """Load image as RGB. If grayscale, replicate to 3 channels."""
    if path.suffix.lower() in (".tif", ".tiff"):
        img = imread(str(path))
    else:
        img = skio.imread(str(path))
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    return img


def normalize_0_1(arr: np.ndarray) -> np.ndarray:
    """Robust 1–99 percentile normalization to [0, 1] for visualization/metrics."""
    p1, p99 = np.percentile(arr, 1), np.percentile(arr, 99)
    return np.clip((arr - p1) / (p99 - p1 + 1e-8), 0.0, 1.0)


def save_histogram_and_cdf(values: np.ndarray, out_png: Path, title: str) -> None:
    """Save histogram and CDF plot for QC."""
    values = np.asarray(values)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    axes[0].hist(values, bins=60)
    axes[0].set_title(f"Histogram: {title}")
    axes[0].set_xlabel("Intensity (chosen metric)")
    axes[0].set_ylabel("Count")
    sorted_vals = np.sort(values)
    cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    axes[1].plot(sorted_vals, cdf)
    axes[1].set_title(f"CDF: {title}")
    axes[1].set_xlabel("Intensity (chosen metric)")
    axes[1].set_ylabel("Cumulative fraction")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ------------------------- Core image processing -------------------------

def segment_and_measure_single_image(
    image_path: Path,
    stardist_model: StarDist2D,
    probability_threshold: float,
    nms_threshold: float,
    minimum_area: int,
    minimum_mean_od: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Returns:
      labels (H×W int): instance labels
      display_rgb (H×W×3 uint8): contrast-stretched RGB for overlays
      dab_normalized (H×W float [0..1]): normalized DAB OD-like map
      nuclei_table (DataFrame): one row per kept nucleus (no class yet)
    """
    # Read and prepare image
    rgb = read_image_rgb(image_path)
    display_rgb = exposure.rescale_intensity(rgb, out_range=np.uint8).astype(np.uint8)

    # Deconvolution: HED → Hematoxylin (H), DAB (D)
    hed = rgb2hed(util.img_as_float(np.clip(rgb, 0, 1)))
    hematoxylin = hed[..., 0]
    dab = hed[..., 2]
    dab_normalized = normalize_0_1(dab)

    # StarDist segmentation on Hematoxylin channel
    hematoxylin_for_net = normalize(hematoxylin, 1, 99.8)
    labels, _ = stardist_model.predict_instances(
        hematoxylin_for_net, prob_thresh=probability_threshold, nms_thresh=nms_threshold
    )

    # Per-nucleus DAB stats
    props = measure.regionprops(labels, intensity_image=dab_normalized)
    rows = []
    for prop in props:
        if prop.area < minimum_area:
            continue
        vals = dab_normalized[labels == prop.label]
        mean_value = float(vals.mean())
        median_value = float(np.median(vals))
        p99_value = float(np.percentile(vals, 99))
        # Remove tiny or near-background nuclei (helps reduce background bias)
        if (mean_value < minimum_mean_od) and (p99_value < minimum_mean_od):
            continue
        centroid_y, centroid_x = prop.centroid
        rows.append(dict(
            label=int(prop.label),
            center_x=float(centroid_x),
            center_y=float(centroid_y),
            area_pixels=int(prop.area),
            mean_dab_od=mean_value,
            median_dab_od=median_value,
            p99_dab_od=p99_value
        ))

    nuclei_table = pd.DataFrame(rows)
    return labels, display_rgb, dab_normalized, nuclei_table


def make_overlay(
    display_rgb: np.ndarray,
    labels: np.ndarray,
    nuclei_with_classes: Optional[pd.DataFrame],
    title: str,
    out_png: Path
) -> None:
    """Save a color overlay showing Low/Medium/High classes."""
    overlay = np.zeros((*labels.shape, 3), np.uint8)
    if nuclei_with_classes is not None and len(nuclei_with_classes) > 0:
        color_map = {1: COLOR_LOW, 2: COLOR_MEDIUM, 3: COLOR_HIGH}
        for _, row in nuclei_with_classes.iterrows():
            overlay[labels == int(row["label"])] = color_map[int(row["class_lmh"])]
    else:
        overlay[labels > 0] = COLOR_MEDIUM

    plt.figure(figsize=(6, 6))
    plt.imshow(display_rgb)
    plt.imshow(overlay, alpha=0.45)
    plt.axis("off")
    plt.title(title)
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


# ------------------------- Dataset runner -------------------------

def run_dataset(args) -> None:
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list_images(data_dir, args.recursive)
    if not image_paths:
        print(f"No images found in: {data_dir}")
        return

    print(f"Found {len(image_paths)} images. Loading StarDist model…")
    model = StarDist2D.from_pretrained("2D_versatile_he")

    # ---------- Pass 1: collect metric from all nuclei ----------
    metric_column_map = {"p99": "p99_dab_od", "mean": "mean_dab_od", "median": "median_dab_od"}
    chosen_metric_column = metric_column_map[args.bin_metric]

    print("Pass 1/2 — collecting per-nucleus metrics across dataset…")
    all_nuclei_tables: List[pd.DataFrame] = []
    for image_path in image_paths:
        try:
            labels, display_rgb, dab_normalized, nuclei_table = segment_and_measure_single_image(
                image_path=image_path,
                stardist_model=model,
                probability_threshold=args.probability_threshold,
                nms_threshold=args.nms_threshold,
                minimum_area=args.minimum_area,
                minimum_mean_od=args.minimum_mean_od
            )
            if len(nuclei_table) > 0:
                nuclei_table.insert(0, "image_name", image_path.name)
                all_nuclei_tables.append(nuclei_table)
                print(f"[OK] {image_path.name}: kept nuclei = {len(nuclei_table)}")
            else:
                print(f"[OK] {image_path.name}: no nuclei kept after filtering")
        except Exception as e:
            print(f"[WARN] {image_path.name}: {e}")
            if args.trace:
                traceback.print_exc()

    if len(all_nuclei_tables) == 0:
        print("No nuclei found in pass 1 — nothing to bin. Exiting.")
        return

    nuclei_all = pd.concat(all_nuclei_tables, ignore_index=True)

    # ---------- Background/outlier clipping using histogram + CDF ----------
    global_values_before = nuclei_all[chosen_metric_column].values.copy()
    save_histogram_and_cdf(global_values_before, out_dir / "global_metric_hist_cdf_before.png", "BEFORE clipping")

    lower_clip_value = np.percentile(global_values_before, args.clip_lower_percent)
    upper_clip_value = np.percentile(global_values_before, args.clip_upper_percent)

    if args.drop_out_of_clip:
        nuclei_all = nuclei_all[
            (nuclei_all[chosen_metric_column] >= lower_clip_value) &
            (nuclei_all[chosen_metric_column] <= upper_clip_value)
        ].reset_index(drop=True)
        print(f"Clipped nuclei outside [{args.clip_lower_percent}%, {args.clip_upper_percent}%] "
              f"→ kept {len(nuclei_all)} nuclei.")
    else:
        # If not dropping, we still show the AFTER plot on the clipped range for reference
        pass

    global_values_after = nuclei_all[chosen_metric_column].values
    save_histogram_and_cdf(global_values_after, out_dir / "global_metric_hist_cdf_after.png", "AFTER clipping")

    # ---------- Learn dataset-wide thresholds ----------
    if args.bin_strategy == "multiotsu":
        thresholds = threshold_multiotsu(global_values_after, classes=3)
        threshold_low, threshold_high = float(thresholds[0]), float(thresholds[1])
    else:
        # quantile strategy
        threshold_low  = float(np.quantile(global_values_after, args.quantile_1))
        threshold_high = float(np.quantile(global_values_after, args.quantile_2))

    print(f"Dataset thresholds on '{args.bin_metric}' "
          f"(after clipping): low<{threshold_low:.4f}, medium<{threshold_high:.4f}, high≥{threshold_high:.4f}")

    # ---------- Pass 2: classify nuclei, compute H-score, save outputs ----------
    print("Pass 2/2 — classifying nuclei, computing H-scores, and saving results…")
    per_image_summary_rows = []

    # Pre-load again for each image, classify with final thresholds, save outputs
    for image_path in image_paths:
        try:
            labels, display_rgb, dab_normalized, nuclei_table = segment_and_measure_single_image(
                image_path=image_path,
                stardist_model=model,
                probability_threshold=args.probability_threshold,
                nms_threshold=args.nms_threshold,
                minimum_area=args.minimum_area,
                minimum_mean_od=args.minimum_mean_od
            )

            image_stem = image_path.stem
            image_out_dir = out_dir / image_stem
            image_out_dir.mkdir(parents=True, exist_ok=True)

            number_detected = int(labels.max())
            number_kept = int(len(nuclei_table))

            if number_kept > 0:
                # Optionally drop out-of-clip nuclei here too (to mirror dataset filtering)
                if args.drop_out_of_clip:
                    mask = (
                        (nuclei_table[chosen_metric_column] >= lower_clip_value) &
                        (nuclei_table[chosen_metric_column] <= upper_clip_value)
                    )
                    nuclei_table = nuclei_table[mask].reset_index(drop=True)
                    number_kept = int(len(nuclei_table))

                # Classify Low/Medium/High on the chosen metric using dataset thresholds
                classes_lmh = np.zeros(number_kept, dtype=np.int32)
                values = nuclei_table[chosen_metric_column].values
                classes_lmh[values < threshold_low] = 1
                classes_lmh[(values >= threshold_low) & (values < threshold_high)] = 2
                classes_lmh[values >= threshold_high] = 3
                nuclei_table["class_lmh"] = classes_lmh

                # Compute H-score
                percent_low  = 100.0 * np.mean(classes_lmh == 1)
                percent_med  = 100.0 * np.mean(classes_lmh == 2)
                percent_high = 100.0 * np.mean(classes_lmh == 3)
                h_score = 1 * percent_low + 2 * percent_med + 3 * percent_high

                # Save per-image CSV and overlay
                nuclei_table_out = nuclei_table.copy()
                nuclei_table_out.insert(0, "image_name", image_path.name)
                nuclei_table_out.to_csv(image_out_dir / f"{image_stem}_per_nucleus_metrics.csv", index=False)

                imsave(str(image_out_dir / f"{image_stem}_labels.tif"), labels.astype(np.uint16))
                make_overlay(
                    display_rgb,
                    labels,
                    nuclei_table,
                    title=f"H-score {h_score:.1f}",
                    out_png=image_out_dir / f"{image_stem}_overlay.png"
                )
            else:
                # No nuclei kept → save labels and a neutral overlay
                nuclei_table_out = nuclei_table  # empty
                imsave(str(image_out_dir / f"{image_stem}_labels.tif"), labels.astype(np.uint16))
                make_overlay(
                    display_rgb,
                    labels,
                    None,
                    title="No kept nuclei",
                    out_png=image_out_dir / f"{image_stem}_overlay.png"
                )
                percent_low = percent_med = percent_high = 0.0
                h_score = 0.0

            per_image_summary_rows.append(dict(
                image_name=image_path.name,
                number_detected=number_detected,
                number_kept=number_kept,
                threshold_low=threshold_low,
                threshold_high=threshold_high,
                percent_low=percent_low,
                percent_medium=percent_med,
                percent_high=percent_high,
                h_score=h_score
            ))

            print(f"[OK] {image_path.name}: detected={number_detected} kept={number_kept} H-score={h_score:.1f}")

        except Exception as e:
            print(f"[ERROR] {image_path.name}: {e}")
            if args.trace:
                traceback.print_exc()

    # Save dataset-level tables
    pd.DataFrame(per_image_summary_rows).to_csv(out_dir / "dataset_summary.csv", index=False)
    nuclei_all.to_csv(out_dir / "nuclei_all.csv", index=False)
    print("Saved dataset_summary.csv and nuclei_all.csv")
    print("Done.")


# ------------------------- CLI -------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HDAB → StarDist → Dataset-wide binning (Low/Med/High) → H-score",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Folder with images (PNG/JPG/TIF). Example: /Users/jamieannemortel/Documents/IHC_dataset"
    )
    parser.add_argument(
        "--out-dir",
        default="outputs_dataset",
        help="Where to write results"
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search subfolders for images"
    )

    # StarDist settings
    parser.add_argument("--probability-threshold", type=float, default=0.5, help="StarDist prob_thresh")
    parser.add_argument("--nms-threshold", type=float, default=0.4, help="StarDist nms_thresh")

    # Quality filters
    parser.add_argument("--minimum-area", type=int, default=20, help="Min pixels to keep a nucleus")
    parser.add_argument("--minimum-mean-od", type=float, default=0.02, help="Drop nuclei with both mean and p99 below this")

    # Dataset-wide binning choices
    parser.add_argument(
        "--bin-metric",
        choices=["p99", "mean", "median"],
        default="p99",
        help="Per-nucleus metric used for class binning"
    )
    parser.add_argument(
        "--bin-strategy",
        choices=["multiotsu", "quantile"],
        default="multiotsu",
        help="How to derive dataset cut points"
    )
    parser.add_argument("--quantile-1", type=float, default=0.33, help="Lower cut for quantile strategy")
    parser.add_argument("--quantile-2", type=float, default=0.66, help="Upper cut for quantile strategy")

    # Background / outlier clipping (on dataset distribution before thresholding)
    parser.add_argument("--clip-lower-percent", type=float, default=1.0, help="Lower percentile to keep (e.g., 1.0)")
    parser.add_argument("--clip-upper-percent", type=float, default=99.0, help="Upper percentile to keep (e.g., 99.0)")
    parser.add_argument("--drop-out-of-clip", action="store_true",
                        help="If set, drop nuclei outside the clipped range")

    # Debug
    parser.add_argument("--trace", action="store_true", help="Print tracebacks on errors")

    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    run_dataset(args)


"""
python hdab_stardist_hscore.py \
  --data-dir "/Users/jamieannemortel/Documents/IHC_dataset" \
  --out-dir "results_dataset" \
  --bin-metric p99 \
  --bin-strategy multiotsu \
  --clip-lower-percent 1.0 \
  --clip-upper-percent 99.0 \
  --drop-out-of-clip
"""