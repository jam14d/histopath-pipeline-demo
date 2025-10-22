from __future__ import annotations
from typing import Optional

import cv2
import numpy as np

from .helpers import PreprocessConfig


class ImagePreprocessor:
    """Preprocess images (BGR in, grayscale/normalized out)."""

    def __init__(self, config: Optional[PreprocessConfig] = None) -> None:
        self.config = config or PreprocessConfig()
        if self.config.blur_ksize % 2 == 0 or self.config.blur_ksize < 3:
            raise ValueError("blur_ksize must be odd and >= 3")

    @staticmethod
    def to_gray(img_bgr: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    def denoise(self, gray: np.ndarray) -> np.ndarray:
        k = self.config.blur_ksize
        return cv2.GaussianBlur(gray, (k, k), self.config.blur_sigma)

    def enhance_contrast(self, gray_or_blur: np.ndarray) -> np.ndarray:
        if self.config.equalize_hist:
            return cv2.equalizeHist(gray_or_blur)
        return gray_or_blur

    def run(self, img_bgr: np.ndarray) -> np.ndarray:
        """BGR -> gray -> blur -> (optional) equalize -> uint8 gray."""
        gray = self.to_gray(img_bgr)
        blur = self.denoise(gray)
        enhanced = self.enhance_contrast(blur)
        return enhanced
