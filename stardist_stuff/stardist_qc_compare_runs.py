#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare multiple StarDist runs by plotting:
1) Confidence metrics per run (mean_prob, std_prob; bg_highprob_frac if present) — histogram + run chart + cross-run comparison
2) n_nuclei histograms per run + run charts
3) Cross-run comparison plots (distributions and scatter comparisons)
Also writes summary CSVs with Coefficient of Variation, skew, outlier counts, and confidence aggregates.

Usage:
  python stardist_qc_compare_runs.py --out /path/to/out runA.csv runB.csv [runC.csv ...]


 Example: 
python stardist_qc_compare_runs.py \
  --out /Users/jamieannemortel/Documents/QC_out_2 \
  /Users/jamieannemortel/Documents/IHC_out_v2/hscore_global_quantile_results_runa.csv \
  /Users/jamieannemortel/Documents/IHC_out_v7/hscore_global_quantile_results_runb.csv \
  /Users/jamieannemortel/Documents/IHC_out_vAB/hscore_global_quantile_results_runab.csv

 
"""
import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew

#  helpers -

def safe_cv(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0: return np.nan
    m = np.mean(x); s = np.std(x, ddof=0)
    return (s/m) if m > 0 else np.nan

def iqr_fences(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0: return (np.nan, np.nan)
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    return (q1 - 1.5 * iqr, q3 + 1.5 * iqr)

def count_runchart_violations(y):
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if y.size == 0: return np.nan
    m = np.mean(y); s = np.std(y, ddof=0)
    upper = m + 2*s; lower = m - 2*s
    return int(np.sum((y > upper) | (y < lower)))

# def pretty_run_label(raw: str) -> str:
#     s = raw.strip().lower()
#     # Common cases: "runa", "runb", "runc", etc.
#     m = re.fullmatch(r"run([a-z])", s)
#     if m:
#         return f"Run {m.group(1).upper()}"
#     # "v3", "v4", "v7" -> "Run V3" etc.
#     m = re.fullmatch(r"v(\d+)", s)
#     if m:
#         return f"Run V{m.group(1)}"
#     # "run_a", "run-a" -> "Run A"
#     m = re.fullmatch(r"run[_-]?([a-z])", s)
#     if m:
#         return f"Run {m.group(1).upper()}"
#     # Fall back to title-cased
#     return raw

def pretty_run_label(raw: str) -> str:
    s = raw.strip().lower()

    # Matches "runa", "runab", "run_a", "run-ab"
    m = re.fullmatch(r"run[_-]?([a-z]+)", s)
    if m:
        return f"Run {m.group(1).upper()}"

    # Matches "v3", "v10", etc.
    m = re.fullmatch(r"v(\d+)", s)
    if m:
        return f"Run V{m.group(1)}"

    # Default fallback
    return raw

def load_runs(files):
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        stem = Path(f).stem
        run_raw = stem.replace("hscore_global_quantile_results_", "")
        df["run_raw"] = run_raw
        df["run"] = pretty_run_label(run_raw)
        dfs.append(df)
    df_all = pd.concat(dfs, ignore_index=True)
    df_all["image_idx"] = df_all.groupby("run").cumcount() + 1
    return df_all

def hist_with_stats(vals, title, xlabel, out_path):
    v = np.asarray(vals, dtype=float); v = v[np.isfinite(v)]
    if v.size == 0: return
    fig = plt.figure(figsize=(7,5))
    plt.hist(v, bins=30)
    m = np.mean(v); s = np.std(v, ddof=0)
    for x in [m, m-s, m+s, m-2*s, m+2*s]:
        plt.axvline(x, linestyle="--")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Number of images")
    fig.tight_layout(); plt.savefig(out_path, dpi=150); plt.close(fig)

def run_chart(x, y, title, ylabel, out_path):
    y = np.asarray(y, dtype=float)
    if not np.isfinite(y).any(): return
    fig = plt.figure(figsize=(8,4))
    plt.plot(x, y, marker="o")
    m = np.nanmean(y); s = np.nanstd(y, ddof=0)
    if np.isfinite(m):
        plt.axhline(m, linestyle="--")
        if np.isfinite(s):
            plt.axhline(m+2*s, linestyle=":")
            plt.axhline(m-2*s, linestyle=":")
    plt.title(title)
    plt.xlabel("Image index")
    plt.ylabel(ylabel)
    fig.tight_layout(); plt.savefig(out_path, dpi=150); plt.close(fig)

def boxplot_by_run(df_all, metric, title, fname, out_dir):
    data, labels = [], []
    for run, g in df_all.groupby("run"):
        vals = g[metric].dropna().values if metric in g.columns else np.array([])
        if vals.size:
            data.append(vals); labels.append(run)
    if not data: return None
    fig = plt.figure(figsize=(8,5))
    plt.boxplot(data, labels=labels, showfliers=True)
    plt.xlabel("Run")
    plt.ylabel(metric)
    plt.title(title)
    fig.tight_layout(); path = out_dir / fname; plt.savefig(path, dpi=150); plt.close(fig); return path

def scatter_two(df_all, x_metric, y_metric, title, fname, out_dir):
    if (x_metric not in df_all.columns) or (y_metric not in df_all.columns): return None
    d = df_all.dropna(subset=[x_metric, y_metric])
    if d.empty: return None
    fig = plt.figure(figsize=(7,6))
    for run, g in d.groupby("run"):
        plt.scatter(g[x_metric], g[y_metric], label=run, s=12)
    plt.xlabel(x_metric)
    plt.ylabel(y_metric)
    plt.title(title)
    plt.legend(markerscale=2, title="Run")
    fig.tight_layout(); path = out_dir / fname; plt.savefig(path, dpi=150); plt.close(fig); return path

#  main 

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True, help="Output directory")
    ap.add_argument("csvs", nargs="+", help="One or more result CSVs")
    args = ap.parse_args()

    out_dir = args.out; out_dir.mkdir(parents=True, exist_ok=True)
    df_all = load_runs(args.csvs)

    # Ensure optional columns exist
    for col in ["n_nuclei","mean_area","median_circularity","mean_prob","std_prob","bg_highprob_frac"]:
        if col not in df_all.columns: df_all[col] = np.nan

    #  Summaries per run 
    rows = []
    for run, g in df_all.groupby("run"):
        counts = g["n_nuclei"].values
        areas = g["mean_area"].values
        circs = g["median_circularity"].values
        mp = g["mean_prob"].values
        sp = g["std_prob"].values
        bg = g["bg_highprob_frac"].values

        low_f, high_f = iqr_fences(areas)
        n_small = int(np.sum(areas < low_f)) if np.isfinite(low_f) else np.nan
        n_large = int(np.sum(areas > high_f)) if np.isfinite(high_f) else np.nan

        rows.append({
            "run": run,
            "images": len(g),
            "n_nuclei_mean": float(np.nanmean(counts)),
            "n_nuclei_sd": float(np.nanstd(counts, ddof=0)),
            "n_nuclei_cv": float(safe_cv(counts)),
            "n_nuclei_runchart_viol_±2SD": float(count_runchart_violations(counts)),
            "mean_area_mean": float(np.nanmean(areas)),
            "mean_area_skew": float(skew(areas[np.isfinite(areas)])) if np.isfinite(areas).any() else np.nan,
            "mean_area_small_outliers_IQR": float(n_small) if isinstance(n_small,(int,float)) else np.nan,
            "mean_area_large_outliers_IQR": float(n_large) if isinstance(n_large,(int,float)) else np.nan,
            "median_circularity_mean": float(np.nanmean(circs)),
            "mean_prob_mean": float(np.nanmean(mp)),
            "mean_prob_sd": float(np.nanstd(mp, ddof=0)),
            "std_prob_mean": float(np.nanmean(sp)),
            "bg_highprob_frac_mean": float(np.nanmean(bg)),
        })
    pd.DataFrame(rows).sort_values("run").to_csv(out_dir / "compare_summary_per_run.csv", index=False)

    #  1) Confidence metrics per run 
    for run, g in df_all.groupby("run"):
        # mean_prob
        hist_with_stats(g["mean_prob"], f"Distribution of mean probability — {run}", "Mean probability (per image)", out_dir / f"conf_meanprob_hist_{run}.png")
        run_chart(g["image_idx"].values, g["mean_prob"].values, f"Mean probability over images — {run}", "Mean probability (per image)", out_dir / f"conf_meanprob_runchart_{run}.png")
        # std_prob
        hist_with_stats(g["std_prob"], f"Distribution of probability spread — {run}", "Std. of probability (per image)", out_dir / f"conf_stdprob_hist_{run}.png")
        run_chart(g["image_idx"].values, g["std_prob"].values, f"Probability spread over images — {run}", "Std. of probability (per image)", out_dir / f"conf_stdprob_runchart_{run}.png")
        # bg_highprob_frac (optional)
        if g["bg_highprob_frac"].notna().any():
            hist_with_stats(g["bg_highprob_frac"], f"Distribution of background high-probability fraction — {run}", "Background high-probability fraction", out_dir / f"conf_bg_highprob_hist_{run}.png")
            run_chart(g["image_idx"].values, g["bg_highprob_frac"].values, f"Background high-probability over images — {run}", "Background high-probability fraction", out_dir / f"conf_bg_highprob_runchart_{run}.png")

    # Cross-run comparisons (clean titles)
    boxplot_by_run(df_all, "mean_prob", "Mean probability across runs", "conf_box_mean_prob.png", out_dir)
    boxplot_by_run(df_all, "std_prob", "Probability spread across runs", "conf_box_std_prob.png", out_dir)
    if df_all["bg_highprob_frac"].notna().any():
        boxplot_by_run(df_all, "bg_highprob_frac", "Background high-probability across runs", "conf_box_bg_highprob_frac.png", out_dir)

    #  2) Object counts per run 
    for run, g in df_all.groupby("run"):
        hist_with_stats(g["n_nuclei"], f"Distribution of object counts — {run}", "Objects per image (n_nuclei)", out_dir / f"counts_hist_{run}.png")
        run_chart(g["image_idx"].values, g["n_nuclei"].values, f"Object counts over images — {run}", "Objects per image (n_nuclei)", out_dir / f"counts_runchart_{run}.png")
    boxplot_by_run(df_all, "n_nuclei", "Object counts across runs", "counts_box_n_nuclei.png", out_dir)

    #  3) Shape/size & relationships 
    boxplot_by_run(df_all, "mean_area", "Mean area across runs", "shape_box_mean_area.png", out_dir)
    boxplot_by_run(df_all, "median_circularity", "Median circularity across runs", "shape_box_median_circularity.png", out_dir)
    scatter_two(df_all, "mean_area", "median_circularity", "Mean area vs. median circularity (by run)", "shape_scatter_area_vs_circ.png", out_dir)
    scatter_two(df_all, "mean_prob", "n_nuclei", "Mean probability vs. object counts (by run)", "conf_scatter_meanprob_vs_counts.png", out_dir)
    scatter_two(df_all, "mean_area", "n_nuclei", "Mean area vs. object counts (by run)", "shape_scatter_area_vs_counts.png", out_dir)

if __name__ == "__main__":
    main()
