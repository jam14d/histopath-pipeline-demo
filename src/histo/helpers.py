from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple, Optional

import cv2
import numpy as np
import os


# Config dataclasses (lightweight & reusable) 

@dataclass
class PreprocessConfig:
    blur_ksize: int = 5         # odd, >= 3
    blur_sigma: float = 0.0     # 0 => auto in OpenCV
    equalize_hist: bool = True  # CLAHE could be added later


@dataclass
class PipelineConfig:
    preprocess: PreprocessConfig = PreprocessConfig()
    save_dir: Optional[str] = None
    show: bool = False


# I/O & filesystem helpers 

def ensure_dir(path: str | os.PathLike) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def load_image_bgr(path: str | os.PathLike) -> np.ndarray:
    """Load an image as BGR uint8 (OpenCV default). Raises on failure."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def list_images(
    dir_path: str | os.PathLike,
    extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff"),
) -> List[str]:
    p = Path(dir_path)
    return [
        str(fp) for fp in sorted(p.iterdir())
        if fp.is_file() and fp.suffix.lower() in extensions
    ]
