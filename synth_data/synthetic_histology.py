import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import disk
from skimage.filters import gaussian
from skimage.util import random_noise
from skimage import io
import os

# Output directory
os.makedirs("synthetic_histology", exist_ok=True)

def make_fake_histology(seed, color_shift=(1.0, 1.0, 1.0)):
    """Generate a synthetic RGB 'histology-like' patch with variable colors."""
    rng = np.random.default_rng(seed)
    H, W = 256, 256
    img = np.ones((H, W, 3), np.float32) * rng.uniform(0.9, 1.0)
    
    # Add random nuclei
    for _ in range(40):
        y, x = rng.integers(0, H), rng.integers(0, W)
        r = rng.integers(5, 15)
        rr, cc = disk((y, x), r, shape=(H, W))
        color = np.array([0.5*rng.uniform(0.8,1.2), 0.3*rng.uniform(0.8,1.2), 0.8*rng.uniform(0.8,1.2)]) * color_shift
        img[rr, cc] = color
    
    # Blur, add noise, clip
    img = gaussian(img, sigma=rng.uniform(0.5, 1.5), channel_axis=-1)
    img = random_noise(img, mode="gaussian", var=0.002)
    img = np.clip(img, 0, 1)
    return (img*255).astype(np.uint8)

# Generate 3 versions with different stain tints
imgs = [
    make_fake_histology(1, (1.2, 0.9, 1.0)),   # pinkish
    make_fake_histology(2, (0.9, 1.0, 1.2)),   # bluish
    make_fake_histology(3, (1.1, 1.1, 0.9)),   # yellowish
]

for i, img in enumerate(imgs):
    io.imsave(f"synthetic_histology/fake_slide_{i+1}.png", img)

# Show them side by side
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for i, ax in enumerate(axes):
    ax.imshow(imgs[i])
    ax.set_title(f"Fake Slide {i+1}")
    ax.axis("off")
plt.tight_layout()
plt.show()
