from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .helpers import load_image_bgr, ensure_dir, PreprocessConfig, PipelineConfig

#  HistomicsTK: stain augmentation 
from histomicstk.preprocessing.augmentation.color_augmentation import (
    rgb_perturb_stain_concentration,
)


# Stain augmentation module

@dataclass
class StainAugConfig:
    n_aug: int = 0                       # how many augmented variants to create per image (0 = off)
    use_mask_for_stats: bool = True      # exclude background from stain stats if True
    extra_kwargs: dict | None = None     # pass-through knobs to HTK if desired


class StainAugmenter:
    def __init__(self, cfg: Optional[StainAugConfig] = None) -> None:
        self.cfg = cfg or StainAugConfig()

    def run(self, img_rgb: np.ndarray, mask_out: Optional[np.ndarray]) -> list[np.ndarray]:
        """Return a list of augmented RGBs (length = cfg.n_aug)."""
        if self.cfg.n_aug <= 0:
            return []
        kwargs = self.cfg.extra_kwargs or {}
        # mask_out=True means "ignore these pixels when estimating stats"
        m = mask_out if self.cfg.use_mask_for_stats else None
        out: list[np.ndarray] = []
        for _ in range(self.cfg.n_aug):
            aug = rgb_perturb_stain_concentration(img_rgb, mask_out=m, **kwargs)
            out.append(aug.astype(np.uint8))
        return out


# Preprocessing


class ImagePreprocessor:
    """Preprocess images (BGR in, grayscale/normalized out)."""

    def __init__(self, config: Optional[PreprocessConfig] = None) -> None:
        self.config = config or PreprocessConfig()

    def to_gray(self, img_bgr: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    def denoise(self, gray: np.ndarray) -> np.ndarray:
        k = self.config.blur_ksize
        return cv2.GaussianBlur(gray, (k, k), self.config.blur_sigma)

    def enhance_contrast(self, gray_or_blur: np.ndarray) -> np.ndarray:
        if self.config.equalize_hist:
            return cv2.equalizeHist(gray_or_blur)
        return gray_or_blur

    def run(self, img_bgr: np.ndarray) -> np.ndarray:
        gray = self.to_gray(img_bgr)
        blur = self.denoise(gray)
        enhanced = self.enhance_contrast(blur)
        return enhanced


# Segmentation


@dataclass
class SegmentationResult:
    mask: np.ndarray  # uint8 0/255
    contours: List[np.ndarray]


class TissueSegmenter:
    """Local/Adaptive or Otsu threshold-based tissue/background segmentation."""

    def __init__(
        self,
        method: str = "adaptive_gaussian",  # 'adaptive_gaussian' | 'adaptive_mean' | 'otsu'
        block_size: int = 51,               # odd, >= 3
        C: int = 5,                         # subtractor; higher -> fewer pixels pass
        invert: bool = True,                # white mask
    ) -> None:
        if block_size % 2 == 0 or block_size < 3:
            raise ValueError("block_size must be odd and >= 3")
        self.method = method
        self.block_size = block_size
        self.C = C
        self.invert = invert

    def threshold_adaptive(self, img_gray: np.ndarray) -> np.ndarray:
        maxval = 255
        adaptive = (
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C
            if self.method == "adaptive_gaussian"
            else cv2.ADAPTIVE_THRESH_MEAN_C
        )
        thresh_type = cv2.THRESH_BINARY_INV if self.invert else cv2.THRESH_BINARY
        mask = cv2.adaptiveThreshold(
            img_gray, maxval, adaptive, thresh_type, self.block_size, self.C
        )
        return mask

    def threshold_otsu(self, img_gray: np.ndarray) -> np.ndarray:
        ttype = cv2.THRESH_BINARY_INV if self.invert else cv2.THRESH_BINARY
        _t, mask = cv2.threshold(img_gray, 0, 255, ttype + cv2.THRESH_OTSU)
        return mask

    def find_regions(self, mask: np.ndarray) -> List[np.ndarray]:
        contours, _hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    # def segment(self, img_gray: np.ndarray) -> SegmentationResult:
    #     if self.method in ("adaptive_gaussian", "adaptive_mean"):
    #         mask = self.threshold_adaptive(img_gray)
    #     elif self.method == "otsu":
    #         mask = self.threshold_otsu(img_gray)
    #     else:
    #         raise ValueError(f"Unknown method: {self.method}")
    #     contours = self.find_regions(mask)
    #     return SegmentationResult(mask=mask, contours=contours)
    
    # def segment(self, img_gray: np.ndarray) -> SegmentationResult:
    #     if self.method in ("adaptive_gaussian", "adaptive_mean"):
    #         mask = self.threshold_adaptive(img_gray)
    #     elif self.method == "otsu":
    #         mask = self.threshold_otsu(img_gray)
    #     else:
    #         raise ValueError(f"Unknown method: {self.method}")

    def segment(self, img_gray: np.ndarray) -> SegmentationResult:
        if self.method in ("adaptive_gaussian", "adaptive_mean"):
            mask = self.threshold_adaptive(img_gray)
        elif self.method == "otsu":
            mask = self.threshold_otsu(img_gray)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Cleanup mask

        # Convert to binary 0/1
        mask_bin = (mask > 0).astype(np.uint8)

        # closing; fill small holes inside tissue
        kernel_close = np.ones((7, 7), np.uint8)
        mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kernel_close)

        # opening; remove small bright noise specks
        kernel_open = np.ones((5, 5), np.uint8)
        mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel_open)

        # connected-component filtering; remove very small blobs
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
        min_area = 500  # adjust depending on resolution
        cleaned = np.zeros_like(mask_bin)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                cleaned[labels == i] = 1

        mask_clean = (cleaned * 255).astype(np.uint8)

        contours = self.find_regions(mask_clean)
        return SegmentationResult(mask=mask_clean, contours=contours)

    def overlay(self, img_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return cv2.bitwise_and(img_rgb, img_rgb, mask=mask)


# Visualization

class Visualizer:
    @staticmethod
    def show_image(img_rgb: np.ndarray, title: str = "Image") -> None:
        plt.figure()
        plt.imshow(img_rgb)
        plt.title(title)
        plt.axis("off")
        plt.show()

    @staticmethod
    def show_side_by_side(images, titles=None, cmap_list=None, figsize=(18, 6)) -> None:
        n = len(images)
        titles = titles or [f"Image {i+1}" for i in range(n)]
        cmap_list = cmap_list or [None] * n

        fig, axes = plt.subplots(1, n, figsize=figsize)
        if n == 1:
            axes = [axes]

        for ax, img, title, cmap in zip(axes, images, titles, cmap_list):
            if getattr(img, "ndim", 2) == 2 and cmap is None:
                cmap = "gray"
            ax.imshow(img, cmap=cmap)
            ax.set_title(title)
            ax.axis("off")

        plt.tight_layout()
        plt.show()


# Orchestrator


class TissuePipeline:
    """End-to-end orchestration for single images or folders."""

    def __init__(self, config: Optional[PipelineConfig] = None, stain_aug_cfg: Optional[StainAugConfig] = None) -> None:
        self.config = config or PipelineConfig()
        self.pre = ImagePreprocessor(self.config.preprocess)
        self.segmenter = TissueSegmenter()
        self.viz = Visualizer()
        self.augmenter = StainAugmenter(stain_aug_cfg or StainAugConfig())

    def process_image(self, path: str) -> Tuple[np.ndarray, np.ndarray, SegmentationResult, np.ndarray]:
        # Load & RGB
        img_bgr = load_image_bgr(path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Rough mask for augmentation stats (exclude background)
        rough_gray = self.pre.to_gray(img_bgr)
        rough_mask = self.segmenter.threshold_otsu(rough_gray)  # 0/255
        mask_out = (rough_mask == 0)  # boolean: True where background

        # Generate stain-perturbed variants (RGB)
        augmented_rgbs = self.augmenter.run(img_rgb, mask_out=mask_out)

        # Optional quick preview when --show and we have at least 1 augmentation
        if self.config.show and len(augmented_rgbs) > 0:
            self.viz.show_side_by_side(
                [img_rgb, augmented_rgbs[0]],
                titles=["Original RGB", "Stain-Augmented RGB (preview)"],
                cmap_list=[None, None],
                figsize=(10, 5),
            )

        # Variant 0 = original; others = stain-augmented
        variants = [("orig", img_rgb)] + [(f"aug{i+1}", a) for i, a in enumerate(augmented_rgbs)]

        # Process each variant end-to-end
        last = None
        for tag, rgb in variants:
            gray = self.pre.run(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            seg = self.segmenter.segment(gray)
            overlay = self.segmenter.overlay(rgb, seg.mask)


            # Visualize the first (original) variant if requested
            if self.config.show and tag == "orig":
                self.viz.show_side_by_side(
                    [rgb, gray, seg.mask, overlay],
                    titles=[f"{tag.upper()} RGB", "Preprocessed (Gray)", "Tissue Mask", "Overlay"],
                    cmap_list=[None, "gray", "gray", None],
                    figsize=(18, 6),
                )

            # Save artifacts if requested
            if self.config.save_dir:
                ensure_dir(self.config.save_dir)
                base = os.path.splitext(os.path.basename(path))[0]
                suffix = "" if tag == "orig" else f"_{tag}"
                cv2.imwrite(os.path.join(self.config.save_dir, f"{base}_rgb{suffix}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(self.config.save_dir, f"{base}_gray{suffix}.png"), gray)
                cv2.imwrite(os.path.join(self.config.save_dir, f"{base}_mask{suffix}.png"), seg.mask)
                cv2.imwrite(os.path.join(self.config.save_dir, f"{base}_overlay{suffix}.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

            last = (rgb, gray, seg, overlay)

        # Return the last processed variant (keeps API compatible)
        return last

    def process_dir(self, dir_path: str, extensions=(".png", ".jpg", ".jpeg", ".tif", ".tiff")) -> list[str]:
        paths = [
            os.path.join(dir_path, f)
            for f in sorted(os.listdir(dir_path))
            if os.path.splitext(f)[1].lower() in extensions
        ]
        for p in paths:
            self.process_image(p)
        return paths


# CLI


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tissue preprocessing, stain augmentation & segmentation (OOP)")
    parser.add_argument("--image", type=str, help="Path to a single image")
    parser.add_argument("--dir", type=str, help="Path to a directory of images")
    parser.add_argument("--save_dir", type=str, default=None, help="Where to save artifacts")
    parser.add_argument("--show", action="store_true", help="Display figures")
    parser.add_argument("--blur_ksize", type=int, default=5, help="Gaussian blur kernel size (odd)")
    parser.add_argument("--blur_sigma", type=float, default=0.0, help="Gaussian sigma (0 -> auto)")
    parser.add_argument("--no_equalize", action="store_true", help="Disable histogram equalization")

    # stain augmentation controls
    parser.add_argument("--n_stain_aug", type=int, default=0,
                        help="Number of stain-perturbed variants to generate per image (0=off)")
    parser.add_argument("--no_aug_mask_stats", action="store_true",
                        help="Do NOT exclude background when estimating stain stats for augmentation")
    # if you want to expose HTK-specific magnitude knobs later, add flags and pass via StainAugConfig.extra_kwargs
    return parser


def main() -> None:
    parser = _build_argparser()
    args = parser.parse_args()

    pre_cfg = PreprocessConfig(
        blur_ksize=args.blur_ksize,
        blur_sigma=args.blur_sigma,
        equalize_hist=not args.no_equalize,
    )
    pipe_cfg = PipelineConfig(preprocess=pre_cfg, save_dir=args.save_dir, show=args.show)

    # wire stain augmentation from CLI
    aug_cfg = StainAugConfig(
        n_aug=max(0, args.n_stain_aug),
        use_mask_for_stats=not args.no_aug_mask_stats,
    )

    pipeline = TissuePipeline(pipe_cfg, stain_aug_cfg=aug_cfg)

    if args.image:
        pipeline.process_image(args.image)
    elif args.dir:
        pipeline.process_dir(args.dir)
    else:
        parser.error("Provide either --image or --dir")


if __name__ == "__main__":
    main()
