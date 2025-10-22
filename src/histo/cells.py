from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
from skimage.color import rgb2hed
from skimage.segmentation import slic, mark_boundaries
from skimage.util import img_as_float
from skimage.filters import threshold_otsu, threshold_sauvola
from skimage.morphology import binary_opening, remove_small_holes, remove_small_objects, disk


@dataclass
class CellMaskParams:
    # SLIC (tuned for BreakHis 400×; tweak per dataset)
    n_segments: int = 1000
    compactness: float = 10.0
    sigma: float = 1.0
    convert2lab: bool = True
    slic_zero: bool = True
    start_label: int = 1

    # Thresholding over per-superpixel Hematoxylin means
    use_sauvola_if_uneven: bool = False
    sauvola_window: int = 41

    # Flip result if your dataset trends opposite
    invert_result: bool = False

    # Morphology cleanup (pixels)
    open_radius: int = 2
    min_hole_area: int = 64
    min_object_area: int = 200

    # Debug artifacts
    return_debug: bool = False


class CellMaskBuilder:
    """
    H&E-optimized binary cell mask:
      1) H = rgb2hed(...)[...,0]
      2) SLIC superpixels on RGB
      3) Otsu on per-superpixel H means (Sauvola optional rescue)
      4) Light morphology cleanup
    Returns: (mask_bool, debug_dict or None)
    """

    def __init__(self, params: Optional[CellMaskParams] = None) -> None:
        self.p = params or CellMaskParams()

    @staticmethod
    def _hematoxylin(img_rgb: np.ndarray) -> np.ndarray:
        H = rgb2hed(img_rgb)[..., 0].astype(np.float32)
        return (H - H.min()) / (H.max() - H.min() + 1e-8)

    def _slic(self, img_rgb: np.ndarray) -> np.ndarray:
        imf = img_as_float(img_rgb)
        labels = slic(
            imf,
            n_segments=self.p.n_segments,
            compactness=self.p.compactness,
            sigma=self.p.sigma,
            convert2lab=self.p.convert2lab,
            slic_zero=self.p.slic_zero,
            start_label=self.p.start_label,
            channel_axis=-1,
        )
        return labels.astype(np.int32)

    @staticmethod
    def _per_label_mean(values: np.ndarray, labels: np.ndarray) -> np.ndarray:
        L = labels.ravel()
        V = values.ravel()
        mx = int(labels.max())
        sums = np.bincount(L, weights=V, minlength=mx + 1)
        cnts = np.bincount(L, minlength=mx + 1)
        means = sums / np.maximum(cnts, 1)
        return means

    def run(self, img_rgb: np.ndarray) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        # 1) Hematoxylin channel feature
        H = self._hematoxylin(img_rgb)

        # 2) Superpixels
        labels = self._slic(img_rgb)
        means = self._per_label_mean(H, labels)

        # 3) Threshold over SP means
        start = max(0, self.p.start_label)
        valid = means[start:] if start <= labels.max() else means
        thr = float(threshold_otsu(valid)) if valid.size > 0 else float(H.mean())

        fg_ids = np.where(means >= thr)[0]  # flip meaning with invert_result later
        mask = np.isin(labels, fg_ids)

        if self.p.use_sauvola_if_uneven:
            Tloc = threshold_sauvola(H, window_size=self.p.sauvola_window)
            mask = np.logical_or(mask, H >= Tloc)

        if self.p.invert_result:
            mask = ~mask

        # 4) Cleanup
        if self.p.open_radius > 0:
            mask = binary_opening(mask, disk(self.p.open_radius))
        if self.p.min_hole_area > 0:
            mask = remove_small_holes(mask, area_threshold=self.p.min_hole_area)
        if self.p.min_object_area > 0:
            mask = remove_small_objects(mask, min_size=self.p.min_object_area)

        debug = None
        if self.p.return_debug:
            bnd = (mark_boundaries(img_rgb, labels, mode="thick") * 255).astype(np.uint8)
            debug = dict(H=H, labels=labels, boundaries=bnd, thr=thr, means=means)

        return mask.astype(bool), debug
