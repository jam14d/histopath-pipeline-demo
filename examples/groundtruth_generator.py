from pathlib import Path
import cv2
import numpy as np
from histo.helpers import load_image_bgr, ensure_dir
from histo.preprocess import ImagePreprocessor
from histo.tissue import TissueSegmenter
from histo.cells import CellMaskBuilder, CellMaskParams

def main(image_path: str, out_dir: str = "out"):
    ensure_dir(out_dir)
    bgr = load_image_bgr(image_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    pre = ImagePreprocessor()
    gray = pre.run(bgr)

    tissue = TissueSegmenter()
    seg = tissue.segment(gray)

    cell_params = CellMaskParams(n_segments=1000, compactness=10.0, sigma=1.0)
    cell_mask, dbg = CellMaskBuilder(cell_params).run(rgb)

    base = Path(image_path).stem
    cv2.imwrite(f"{out_dir}/{base}_tissue_mask.png", seg.mask)
    cv2.imwrite(f"{out_dir}/{base}_cellmask.png", (cell_mask.astype(np.uint8) * 255))

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="out")
    args = ap.parse_args()
    main(args.image, args.out)
