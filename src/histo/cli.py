from __future__ import annotations
import argparse
import os
from typing import Optional

from .config import PreprocessConfig, PipelineConfig, CellMaskParams
from .helpers import ensure_dir, load_image_bgr
from .preprocess import ImagePreprocessor
from .tissue import TissueSegmenter
from .cells import CellMaskBuilder
from .augment import StainAugConfig, StainAugmenter
from .viz import Visualizer
import cv2
import numpy as np

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Histopathology image tools")
    g_io = p.add_argument_group("I/O")
    g_io.add_argument("--image", type=str, help="Path to a single image")
    g_io.add_argument("--dir", type=str, help="Path to a directory of images")
    g_io.add_argument("--save_dir", type=str, default=None, help="Output folder")
    g_io.add_argument("--show", action="store_true", help="Display figures")

    g_pre = p.add_argument_group("Preprocess")
    g_pre.add_argument("--blur_ksize", type=int, default=5)
    g_pre.add_argument("--blur_sigma", type=float, default=0.0)
    g_pre.add_argument("--no_equalize", action="store_true")

    g_aug = p.add_argument_group("Stain Augmentation")
    g_aug.add_argument("--n_stain_aug", type=int, default=0)
    g_aug.add_argument("--no_aug_mask_stats", action="store_true")

    g_cell = p.add_argument_group("Cell Mask (HED+SLIC)")
    g_cell.add_argument("--cell_mask", action="store_true", help="Produce cells-vs-background mask")
    g_cell.add_argument("--cell_segments", type=int, default=1000)
    g_cell.add_argument("--cell_compactness", type=float, default=10.0)
    g_cell.add_argument("--cell_sigma", type=float, default=1.0)
    g_cell.add_argument("--cell_invert", action="store_true")
    g_cell.add_argument("--cell_sauvola", action="store_true")

    return p

def _process_one(path: str, pipe_cfg: PipelineConfig, pre: ImagePreprocessor,
                 tissue: TissueSegmenter, augmenter: StainAugmenter,
                 cell_builder: Optional[CellMaskBuilder], viz: Visualizer):
    # load & convert to RGB
    img_bgr = load_image_bgr(path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # augmentation preview stats (exclude background with quick Otsu on gray)
    rough_gray = pre.to_gray(img_bgr)
    rough_mask = tissue.threshold_otsu(rough_gray)  # 0/255
    mask_out = (rough_mask == 0)

    # augment if requested
    augmented_rgbs = augmenter.run(img_rgb, mask_out=mask_out)
    variants = [("orig", img_rgb)] + [(f"aug{i+1}", a) for i, a in enumerate(augmented_rgbs)]

    base = os.path.splitext(os.path.basename(path))[0]
    for tag, rgb in variants:
        gray = pre.run(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))  # preprocess pipeline
        # tissue mask (kept simple)
        seg = tissue.segment(gray)
        overlay = tissue.overlay(rgb, seg.mask)

        # optional: cell mask (our HED+SLIC)
        cell_mask_u8 = None
        if cell_builder is not None:
            mask_bool, dbg = cell_builder.run(rgb)
            cell_mask_u8 = (mask_bool.astype(np.uint8) * 255)

        # show
        if pipe_cfg.show and tag == "orig":
            panels = [rgb, gray, seg.mask, overlay]
            titles = ["RGB", "Preprocessed Gray", "Tissue Mask", "Overlay"]
            cmaps = [None, "gray", "gray", None]
            if cell_mask_u8 is not None:
                panels.append(cell_mask_u8)
                titles.append("Cell Mask (HED+SLIC)")
                cmaps.append("gray")
            viz.show_side_by_side(panels, titles, cmaps, figsize=(18, 6))

        # save
        if pipe_cfg.save_dir:
            ensure_dir(pipe_cfg.save_dir)
            suffix = "" if tag == "orig" else f"_{tag}"
            cv2.imwrite(os.path.join(pipe_cfg.save_dir, f"{base}_rgb{suffix}.png"),
                        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(pipe_cfg.save_dir, f"{base}_gray{suffix}.png"), gray)
            cv2.imwrite(os.path.join(pipe_cfg.save_dir, f"{base}_tissue_mask{suffix}.png"), seg.mask)
            cv2.imwrite(os.path.join(pipe_cfg.save_dir, f"{base}_overlay{suffix}.png"),
                        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            if cell_mask_u8 is not None:
                cv2.imwrite(os.path.join(pipe_cfg.save_dir, f"{base}_cellmask{suffix}.png"), cell_mask_u8)

def main() -> None:
    args = build_argparser().parse_args()

    pre_cfg = PreprocessConfig(
        blur_ksize=args.blur_ksize,
        blur_sigma=args.blur_sigma,
        equalize_hist=not args.no_equalize,
    )
    pipe_cfg = PipelineConfig(preprocess=pre_cfg, save_dir=args.save_dir, show=args.show)

    augmenter = StainAugmenter(StainAugConfig(
        n_aug=max(0, args.n_stain_aug),
        use_mask_for_stats=not args.no_aug_mask_stats,
    ))

    pre = ImagePreprocessor(pre_cfg)
    tissue = TissueSegmenter()                  # tissue/background only
    viz = Visualizer()

    cell_builder = None
    if args.cell_mask:
        params = CellMaskParams(
            n_segments=args.cell_segments,
            compactness=args.cell_compactness,
            sigma=args.cell_sigma,
            invert_result=args.cell_invert,
            use_sauvola_if_uneven=args.cell_sauvola,
            return_debug=pipe_cfg.show,
        )
        cell_builder = CellMaskBuilder(params)

    if args.image:
        _process_one(args.image, pipe_cfg, pre, tissue, augmenter, cell_builder, viz)
    elif args.dir:
        for fname in sorted(os.listdir(args.dir)):
            if os.path.splitext(fname)[1].lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
                _process_one(os.path.join(args.dir, fname), pipe_cfg, pre, tissue, augmenter, cell_builder, viz)
    else:
        raise SystemExit("Provide either --image or --dir")
