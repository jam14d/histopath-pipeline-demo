# early sketch stage 

def load_image(path):
    # return ndarray
    ...

def resize(img, target):
    ...
    return img_resized

def denoise(img, sigma):
    ...
    return img_denoised

def enhance_contrast(img, clip):
    ...
    return img_contrast

# Simple config dict to hold parameters
CONFIG = {
    "resize": {"target": (512, 512)},
    "denoise": {"sigma": 10},
    "enhance_contrast": {"clip": 2.0}
}

# List of pipeline steps (order matters)
STEPS = [resize, denoise, enhance_contrast]

def run_prototype_pipeline(path):
    img = load_image(path)
    trace = []

    for fn in STEPS:
        params = CONFIG[fn.__name__]
        img = fn(img, **params)
        trace.append((fn.__name__, params))

    save_image(img, "output.jpg")
    return trace


---
OOP:
# build same steps but as objects
pipeline = [
    Transform("resize", resize, target=(512,512)),
    Transform("denoise", denoise, sigma=10),
    Transform("enhance_contrast", enhance_contrast, clip=2.0)
]

def run_pipeline(path):
    img = load_image(path)
    for t in pipeline:
        img = t(img)
    return img