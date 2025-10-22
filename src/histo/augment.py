from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import numpy as np


@dataclass
class StainAugConfig:
    n_aug: int = 0                       # augmented variants per image (0 = off)
    use_mask_for_stats: bool = True      # ignore background when estimating stats
    extra_kwargs: Optional[Dict[str, Any]] = None  # passthrough to HTK


class StainAugmenter:
    """
    Thin wrapper around HistomicsTK stain augmentation:
    https://digitalslidearchive.github.io/HistomicsTK/
    """

    def __init__(self, cfg: Optional[StainAugConfig] = None) -> None:
        self.cfg = cfg or StainAugConfig()

        # Defer import to give a nicer error if package is missing
        try:
            from histomicstk.preprocessing.augmentation.color_augmentation import (
                rgb_perturb_stain_concentration,
            )
        except Exception as e:
            raise ImportError(
                "HistomicsTK is required for StainAugmenter. "
                "Install with: pip install histomicstk"
            ) from e
        self._augment_fn = rgb_perturb_stain_concentration  # type: ignore

    def run(self, img_rgb: np.ndarray, mask_out: Optional[np.ndarray]) -> List[np.ndarray]:
        """
        Return a list of augmented RGBs (length = cfg.n_aug).
        mask_out: boolean mask where True means 'ignore for stain stats'.
        """
        if self.cfg.n_aug <= 0:
            return []
        kwargs = self.cfg.extra_kwargs or {}
        m = mask_out if self.cfg.use_mask_for_stats else None
        out: List[np.ndarray] = []
        for _ in range(self.cfg.n_aug):
            aug = self._augment_fn(img_rgb, mask_out=m, **kwargs)
            out.append(aug.astype(np.uint8))
        return out
