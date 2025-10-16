from histomicstk.preprocessing.color_normalization import reinhard_stats, reinhard
from skimage import io, img_as_ubyte
from pathlib import Path
import numpy as np

in_dir  = Path("/Users/jamieannemortel/Documents/Projects/histopath-pipeline-demo/synth_data/synthetic_histology")
out_dir = in_dir.parent / "normalized_reinhard"
out_dir.mkdir(parents=True, exist_ok=True)

# pick a reference image
ref_path = in_dir / "fake_slide_1.png"
ref_img  = io.imread(ref_path)

# compute target LAB stats — sample 10% of pixels
target_mu, target_sigma = reinhard_stats(ref_img, sample_fraction=0.1)

for p in sorted(in_dir.glob("*.png")):
    src = io.imread(p)
    src_norm = reinhard(src, target_mu, target_sigma)
    io.imsave(out_dir / p.name, img_as_ubyte(np.clip(src_norm, 0, 1)))

print("Saved normalized images to:", out_dir)
