"""
Synthetic DAB-like IHC image generator (spread-out with Poisson-disk sampling)
-----------------------------------------------------------------------------

Creates PNGs with cell-like blobs: brown (DAB+) and blue (negative).
Set PLACEMENT_MODE = "poisson" for evenly spread nuclei (little clumping).

Requires: Pillow, numpy
pip install pillow numpy
"""

import os, random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

# ---------------------------
# CONFIG
# ---------------------------
W, H = 1024, 1024          # image size
N_IMAGES = 5               # how many images to generate
SEED = 9                   # set to None for full randomness
SAVE_DIR = "dab_synthetic"

PLACEMENT_MODE = "poisson"  # "poisson" (spread out) or "clustered"

CFG = {
    "pos_fraction_range": (0.25, 0.55),    # fraction of DAB-positive (brown)
    "nuclei_count_range": (900, 1500),     # *target* count; Poisson may yield slightly less
    "radius_px_range": (6, 14),            # nucleus radius range
    "ellipticity_range": (0.85, 1.25),     # 1.0 = circle; >1 elongated
    "blur_sigma_range": (0.6, 1.8),        # per-nucleus blur
    "ring_darkness": (0.06, 0.16),         # rim strength
    "intensity_jitter": (0.75, 1.15),      # stain variability

    # CLUSTERED placement only (kept for completeness)
    "cluster_strength": (0.15, 0.45),      # ignored in "poisson" mode

    # POISSON placement controls
    # Minimum spacing between nuclei centers; you can tie this to size for realism.
    # A good rule of thumb is ~ 1.8× median radius for mild touching, 2.2× for less overlap.
    "min_spacing_px": 24,                  # try between 18–30 depending on size/density
    "poisson_attempts": 30,                # Bridson k parameter (more = denser but slower)
    "background_vignette": True,
}

# Brown palette (DAB) and blue palette (hematoxylin-ish)
BROWN_BASES = [(120,84,45), (140,98,50), (160,115,60), (110,78,40), (132,92,52)]
BLUE_BASES  = [(70,95,170), (85,110,185), (65,90,160), (95,120,190), (60,85,150)]

if SEED is not None:
    random.seed(SEED)
    np.random.seed(SEED)

os.makedirs(SAVE_DIR, exist_ok=True)

# ---------------------------
# Utility
# ---------------------------
def jitter_color(rgb, s=0.08):
    r = np.clip(np.array(rgb) * (1 + np.random.uniform(-s, s, 3)), 0, 255)
    return tuple(r.astype(np.uint8))

def make_background(w, h):
    base = np.ones((h, w, 3), dtype=np.float32) * np.array([247,246,241], dtype=np.float32)
    noise = np.random.normal(0, 1, (h, w)).astype(np.float32)
    for _ in range(2):
        noise = (np.pad(noise, ((2,2),(2,2)), mode='reflect')[2:-2,2:-2] +
                 np.pad(noise, ((4,4),(4,4)), mode='reflect')[4:-4,4:-4] + noise)/3.0
    noise = (noise - noise.min())/(noise.max()-noise.min()+1e-6)
    noise = (noise*8 - 4)
    base += noise[..., None]
    base = np.clip(base, 0, 255).astype(np.uint8)
    img = Image.fromarray(base, mode="RGB")
    if CFG["background_vignette"]:
        vign = Image.new("L", (w, h), 255)
        d = ImageDraw.Draw(vign)
        r_max = int(1.1 * max(w, h))
        for r in range(r_max, 0, -20):
            val = int(255 * (r / r_max))
            d.ellipse((w//2 - r, h//2 - r, w//2 + r, h//2 + r), fill=val)
        vign = vign.filter(ImageFilter.GaussianBlur(radius=80))
        img = Image.composite(img, ImageOps.colorize(vign, (245,245,240), (250,250,246)), vign)
    return img

def draw_soft_ellipse(canvas, center, rx, ry, color, intensity_scale=1.0, blur_sigma=1.0, ring_darkness=0.1):
    cx, cy = center
    rpad = int(max(rx, ry) * 3)
    x0, y0 = max(0, cx - rpad), max(0, cy - rpad)
    x1, y1 = min(canvas.width, cx + rpad), min(canvas.height, cy + rpad)
    w, h = x1 - x0, y1 - y0
    if w <= 2 or h <= 2:
        return
    patch = Image.new("RGBA", (w, h), (0,0,0,0))
    pdraw = ImageDraw.Draw(patch)
    fill = tuple(int(np.clip(c * intensity_scale, 0, 255)) for c in color) + (255,)
    rim_color = tuple(int(v * (1 - ring_darkness)) for v in color) + (255,)
    pdraw.ellipse((rpad-rx, rpad-ry, rpad+rx, rpad+ry), fill=fill)
    width = max(1, int(min(rx, ry) * 0.15))
    pdraw.ellipse((rpad-rx, rpad-ry, rpad+rx, rpad+ry), outline=rim_color, width=width)
    for _ in range(np.random.randint(2, 6)):
        tx = np.random.randint(rpad-rx+2, rpad+rx-2)
        ty = np.random.randint(rpad-ry+2, rpad+ry-2)
        tr = np.random.randint(1, max(2, int(min(rx, ry) * 0.25)))
        alpha = np.random.randint(120, 200)
        tcol = tuple(int(np.clip(c * (0.8 + 0.4 * np.random.rand()), 0, 255)) for c in color) + (alpha,)
        pdraw.ellipse((tx-tr, ty-tr, tx+tr, ty+tr), fill=tcol)
    patch = patch.filter(ImageFilter.GaussianBlur(radius=blur_sigma))
    canvas.alpha_composite(patch, dest=(x0, y0))

# ---------------------------
# Poisson-disk sampling (Bridson)
# ---------------------------
def poisson_disk(width, height, r, k=30, target=None, rng=np.random):
    """
    width, height: domain size
    r: minimum distance between points
    k: attempts per active sample
    target: stop once we reach this many points (optional)
    Returns: list[(x,y)]
    """
    cell = r / np.sqrt(2)
    grid_w = int(np.ceil(width / cell))
    grid_h = int(np.ceil(height / cell))
    grid = -np.ones((grid_h, grid_w), dtype=int)

    def grid_coords(p):
        return int(p[1] // cell), int(p[0] // cell)

    def fits(p):
        gy, gx = grid_coords(p)
        y0, y1 = max(gy-2, 0), min(gy+3, grid_h)
        x0, x1 = max(gx-2, 0), min(gx+3, grid_w)
        for yy in range(y0, y1):
            for xx in range(x0, x1):
                gi = grid[yy, xx]
                if gi != -1:
                    q = pts[gi]
                    if (p[0]-q[0])**2 + (p[1]-q[1])**2 < r*r:
                        return False
        return (0 <= p[0] < width) and (0 <= p[1] < height)

    pts = []
    active = []

    # seed
    p0 = (rng.uniform(0, width), rng.uniform(0, height))
    pts.append(p0)
    active.append(0)
    gy, gx = grid_coords(p0)
    grid[gy, gx] = 0

    while active and (target is None or len(pts) < target):
        idx = rng.choice(active)
        base = pts[idx]
        found = False
        for _ in range(k):
            rad = rng.uniform(r, 2*r)
            ang = rng.uniform(0, 2*np.pi)
            p = (base[0] + rad * np.cos(ang), base[1] + rad * np.sin(ang))
            if fits(p):
                pts.append(p)
                active.append(len(pts)-1)
                gyy, gxx = grid_coords(p)
                grid[gyy, gxx] = len(pts)-1
                found = True
                if target is not None and len(pts) >= target:
                    break
        if not found:
            active.remove(idx)
    return [(int(x), int(y)) for x, y in pts]

# ---------------------------
# Clustered positions (fallback/option)
# ---------------------------
def clustered_positions(n, w, h, cluster_strength):
    k = max(2, int(8 * cluster_strength) + 1)
    centers = [(random.randint(0,w-1), random.randint(0,h-1)) for _ in range(k)]
    weights = np.random.dirichlet([1]*k)
    pts = []
    for _ in range(n):
        ci = np.random.choice(k, p=weights)
        cx, cy = centers[ci]
        s = int((min(w,h)/10) * (0.5 + 0.8*(1-cluster_strength)))
        x = int(np.clip(np.random.normal(cx, s), 0, w-1))
        y = int(np.clip(np.random.normal(cy, s), 0, h-1))
        pts.append((x,y))
    return pts

# ---------------------------
# Main generator per image
# ---------------------------
def generate_image():
    img = make_background(W, H).convert("RGBA")
    total_target = int(np.random.randint(*CFG["nuclei_count_range"]))
    pos_frac = np.random.uniform(*CFG["pos_fraction_range"])
    n_pos = int(total_target * pos_frac)

    # Choose positions
    if PLACEMENT_MODE == "poisson":
        # You can auto-tie spacing to radius range if you prefer:
        # CFG["min_spacing_px"] = int(2.0 * np.median(CFG["radius_px_range"]))
        positions = poisson_disk(
            W, H,
            r=CFG["min_spacing_px"],
            k=CFG["poisson_attempts"],
            target=total_target
        )
        # If we got fewer than target (packing limit), that's okay.
    else:
        cluster_strength = np.random.uniform(*CFG["cluster_strength"])
        positions = clustered_positions(total_target, W, H, cluster_strength)

    random.shuffle(positions)

    for i, (x, y) in enumerate(positions):
        rx = np.random.randint(*CFG["radius_px_range"])
        ry = int(rx * np.random.uniform(*CFG["ellipticity_range"]))
        blur_sigma = np.random.uniform(*CFG["blur_sigma_range"])
        ring_dark = np.random.uniform(*CFG["ring_darkness"])
        inten = np.random.uniform(*CFG["intensity_jitter"])

        base = random.choice(BROWN_BASES if i < n_pos else BLUE_BASES)
        col = jitter_color(base, s=0.12)

        draw_soft_ellipse(
            img, (x, y), rx, ry, col,
            intensity_scale=inten,
            blur_sigma=blur_sigma,
            ring_darkness=ring_dark
        )

    out = img.convert("RGB").filter(ImageFilter.GaussianBlur(radius=0.5))
    cast = Image.new("RGB", (W, H), (248, 247, 245))
    out = Image.blend(out, cast, alpha=0.08)
    return out

# ---------------------------
# Run
# ---------------------------
def main():
    if SEED is not None:
        random.seed(SEED)
        np.random.seed(SEED)
    for i in range(N_IMAGES):
        im = generate_image()
        path = os.path.join(SAVE_DIR, f"synthetic_dab_{i+1:02d}.png")
        im.save(path, format="PNG", optimize=True)
        print("Saved:", path)

if __name__ == "__main__":
    main()
