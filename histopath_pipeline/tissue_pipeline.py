from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .helpers import load_image_bgr, ensure_dir, PreprocessConfig, PipelineConfig

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
    """Otsu threshold-based tissue/background segmentation."""

    def threshold_otsu(self, img_gray: np.ndarray) -> np.ndarray:
        _t, mask = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return mask

    def find_regions(self, mask: np.ndarray) -> List[np.ndarray]:
        contours, _hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def segment(self, img_gray: np.ndarray) -> SegmentationResult:
        mask = self.threshold_otsu(img_gray)
        contours = self.find_regions(mask)
        return SegmentationResult(mask=mask, contours=contours)

    def overlay(self, img_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        # Bitwise mask; ensure mask is single channel uint8
        return cv2.bitwise_and(img_rgb, img_rgb, mask=mask)

#Visualization

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
        """
        Display multiple images side by side.

        Args:
            images: list of np.ndarray images (RGB or grayscale)
            titles: list of titles for each subplot
            cmap_list: list of colormaps for each image (e.g., 'gray' for grayscale)
            figsize: tuple defining the figure size
        """
        n = len(images)
        titles = titles or [f"Image {i+1}" for i in range(n)]
        cmap_list = cmap_list or [None] * n

        fig, axes = plt.subplots(1, n, figsize=figsize)
        if n == 1:
            axes = [axes]

        for ax, img, title, cmap in zip(axes, images, titles, cmap_list):
            # if a grayscale image comes in without a cmap, default to gray
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

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()
        self.pre = ImagePreprocessor(self.config.preprocess)
        self.segmenter = TissueSegmenter()
        self.viz = Visualizer()

    def process_image(self, path: str) -> Tuple[np.ndarray, np.ndarray, SegmentationResult, np.ndarray]:
        # Load & RGB
        img_bgr = load_image_bgr(path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Preprocess (gray)
        gray = self.pre.run(img_bgr)

        # Segment
        seg = self.segmenter.segment(gray)

        # Overlay
        overlay = self.segmenter.overlay(img_rgb, seg.mask)

        # Visualize if requested
        # if self.config.show:
        #     self.viz.show_image(img_rgb, title="Original (RGB)")
        #     self.viz.show_gray(gray, title="Preprocessed (Gray)")
        #     self.viz.show_mask(seg.mask, title="Tissue Mask (Otsu)")
        #     self.viz.show_image(overlay, title="Overlay (Masked)")
        if self.config.show:
            self.viz.show_side_by_side(
            [img_rgb, gray, seg.mask, overlay],
            titles=["Original (RGB)", "Preprocessed (Gray)", "Tissue Mask (Otsu)", "Overlay (Masked)"],
            cmap_list=[None, "gray", "gray", None],
            figsize=(18, 6),
    )
    
        

        # Save artifacts if requested
        if self.config.save_dir:
            ensure_dir(self.config.save_dir)
            base = os.path.splitext(os.path.basename(path))[0]
            rgb_path = os.path.join(self.config.save_dir, f"{base}_rgb.png")
            gray_path = os.path.join(self.config.save_dir, f"{base}_gray.png")
            mask_path = os.path.join(self.config.save_dir, f"{base}_mask.png")
            over_path = os.path.join(self.config.save_dir, f"{base}_overlay.png")

            cv2.imwrite(rgb_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(gray_path, gray)
            cv2.imwrite(mask_path, seg.mask)
            cv2.imwrite(over_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        return img_rgb, gray, seg, overlay

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
    parser = argparse.ArgumentParser(description="Tissue preprocessing & segmentation (OOP)")
    parser.add_argument("--image", type=str, help="Path to a single image")
    parser.add_argument("--dir", type=str, help="Path to a directory of images")
    parser.add_argument("--save_dir", type=str, default=None, help="Where to save artifacts")
    parser.add_argument("--show", action="store_true", help="Display figures")
    parser.add_argument("--blur_ksize", type=int, default=5, help="Gaussian blur kernel size (odd)")
    parser.add_argument("--blur_sigma", type=float, default=0.0, help="Gaussian sigma (0 -> auto)")
    parser.add_argument("--no_equalize", action="store_true", help="Disable histogram equalization")
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
    pipeline = TissuePipeline(pipe_cfg)

    if args.image:
        pipeline.process_image(args.image)
    elif args.dir:
        pipeline.process_dir(args.dir)
    else:
        parser.error("Provide either --image or --dir")


if __name__ == "__main__":
    main()
