from stardist.models import StarDist2D
from stardist.data import test_image_nuclei_2d
from stardist.plot import render_label
from csbdeep.utils import normalize
import matplotlib.pyplot as plt
import numpy as np

# list available models 
#StarDist2D.from_pretrained()

# load pretrained 1-channel model and test image
model = StarDist2D.from_pretrained('2D_versatile_fluo')
img = test_image_nuclei_2d()

labels, _ = model.predict_instances(normalize(img), axes="YX")  # 2D input

plt.figure(figsize=(10,4))
plt.subplot(1,2,1); plt.imshow(img, cmap="gray"); plt.axis("off"); plt.title("input image")
plt.subplot(1,2,2); plt.imshow(render_label(labels, img=img)); plt.axis("off"); plt.title("prediction + input")

plt.tight_layout()
plt.show(block=True)   # <-- keep window open
# plt.savefig("stardist_demo.png", dpi=200, bbox_inches="tight")  # <-- or save instead
