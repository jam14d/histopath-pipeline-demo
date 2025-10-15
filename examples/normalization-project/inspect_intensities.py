from skimage import io, img_as_float
from pathlib import Path
import matplotlib.pyplot as plt

input_dir = Path("/Users/jamieannemortel/Documents/Projects/histopath-pipeline-demo/synth_data/synthetic_histology")

# Define RGB colors for plotting
colors = ("r", "g", "b")

for path in sorted(input_dir.glob("*.png")):
    img = img_as_float(io.imread(path))

    #  Create a figure with 2 panels: image + histogram
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: show image
    axes[0].imshow(img)
    axes[0].set_title(path.name)
    axes[0].axis("off")

    # Right: RGB histograms
    for i, color in enumerate(colors):
        axes[1].hist(img[..., i].ravel(), bins=50, color=color, alpha=0.6, label=f"{color.upper()} channel")

    axes[1].set_title("RGB Intensity Histogram")
    axes[1].set_xlabel("Pixel Intensity (0–1)")
    axes[1].set_ylabel("Frequency")
    axes[1].legend()

    plt.tight_layout()
    plt.show()
