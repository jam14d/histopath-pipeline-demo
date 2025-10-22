from __future__ import annotations
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


class Visualizer:
    """Plot helpers (only used when --show). No implicit showing in library paths."""

    @staticmethod
    def show_image(img_rgb: np.ndarray, title: str = "Image", show: bool = True) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img_rgb)
        ax.set_title(title)
        ax.axis("off")
        if show:
            plt.show()
        return fig

    @staticmethod
    def show_side_by_side(
        images: Sequence[np.ndarray],
        titles: Optional[Sequence[str]] = None,
        cmap_list: Optional[Sequence[Optional[str]]] = None,
        figsize: Tuple[int, int] = (18, 6),
        show: bool = True,
    ) -> plt.Figure:
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

        fig.tight_layout()
        if show:
            plt.show()
        return fig
