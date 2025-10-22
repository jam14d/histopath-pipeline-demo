from __future__ import annotations
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np


@dataclass
class SegmentationResult:
    mask: np.ndarray            # uint8, 0/255
    contours: List[np.ndarray]


class TissueSegmenter:
    """Tissue vs background using adaptive or Otsu thresholding + light cleanup."""

    def __init__(
        self,
        method: str = "adaptive_gaussian",  # 'adaptive_gaussian' | 'adaptive_mean' | 'otsu'
        block_size: int = 51,               # odd, >= 3
        C: int = 5,                         # subtractor; higher -> fewer pixels pass
        invert: bool = True,                # True -> white tissue on black bg
        min_area: int = 500,                # remove tiny specks
    ) -> None:
        if method in ("adaptive_gaussian", "adaptive_mean"):
            if block_size % 2 == 0 or block_size < 3:
                raise ValueError("block_size must be odd and >= 3")
        self.method = method
        self.block_size = block_size
        self.C = C
        self.invert = invert
        self.min_area = int(min_area)

    def threshold_adaptive(self, img_gray: np.ndarray) -> np.ndarray:
        maxval = 255
        adaptive = (
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C
            if self.method == "adaptive_gaussian"
            else cv2.ADAPTIVE_THRESH_MEAN_C
        )
        ttype = cv2.THRESH_BINARY_INV if self.invert else cv2.THRESH_BINARY
        mask = cv2.adaptiveThreshold(
            img_gray, maxval, adaptive, ttype, self.block_size, self.C
        )
        return mask

    def threshold_otsu(self, img_gray: np.ndarray) -> np.ndarray:
        ttype = cv2.THRESH_BINARY_INV if self.invert else cv2.THRESH_BINARY
        _t, mask = cv2.threshold(img_gray, 0, 255, ttype + cv2.THRESH_OTSU)
        return mask

    def _cleanup(self, mask: np.ndarray) -> np.ndarray:
        # work in 0/1
        m = (mask > 0).astype(np.uint8)

        # closing then opening
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  np.ones((5, 5), np.uint8))

        # cc filter by area
        num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        keep = np.zeros_like(m)
        for i in range(1, num):
            if stats[i, cv2.CC_STAT_AREA] >= self.min_area:
                keep[labels == i] = 1
        return (keep * 255).astype(np.uint8)

    @staticmethod
    def _find_regions(mask_u8: np.ndarray) -> List[np.ndarray]:
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def segment(self, img_gray: np.ndarray) -> SegmentationResult:
        if self.method in ("adaptive_gaussian", "adaptive_mean"):
            raw = self.threshold_adaptive(img_gray)
        elif self.method == "otsu":
            raw = self.threshold_otsu(img_gray)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        clean = self._cleanup(raw)
        contours = self._find_regions(clean)
        return SegmentationResult(mask=clean, contours=contours)

    @staticmethod
    def overlay(img_rgb: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
        return cv2.bitwise_and(img_rgb, img_rgb, mask=mask_u8)
