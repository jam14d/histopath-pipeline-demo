import os
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def load_image_bgr(path: str) -> np.ndarray:
    """Load an image from disk (BGR). Raises if path is invalid."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at: {path}")
    return img


@dataclass
class PreprocessConfig:
    blur_ksize: int = 5
    blur_sigma: float = 0.0
    equalize_hist: bool = True


@dataclass
class PipelineConfig:
    preprocess: PreprocessConfig = PreprocessConfig()
    save_dir: Optional[str] = None
    show: bool = False
