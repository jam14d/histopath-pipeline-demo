
"""
TensorFlow training demo (Malignant vs Benign) reusing pipeline's stain augmentation.
Directory layout expected:
  DATA_ROOT/
    train/{malignant,benign}/*.png
    test/{malignant,benign}/*.png

Validation = the test split (deterministic; no augmentation).
"""

from __future__ import annotations
import os, glob
from typing import List, Tuple

import numpy as np
import tensorflow as tf
import cv2

# 0) optional import pipeline stain augmentation

TRY_HTK = True
try:
    from histopath_pipeline.tissue_pipeline import StainAugmenter, StainAugConfig
    AUG = StainAugmenter(StainAugConfig(n_aug=1, use_mask_for_stats=False))
except Exception:
    TRY_HTK = False
    AUG = None

# 1) Data config
#DATA_ROOT  = '/Users/jamieannemortel/Downloads/BreaKHis 400X'   
DATA_ROOT = "/content/drive/MyDrive/BreaKHis 400X"

TRAIN_DIR  = os.path.join(DATA_ROOT, "train")
TEST_DIR   = os.path.join(DATA_ROOT, "test")

CLASSES = ["malignant", "benign"]
CLASS_TO_LABEL = {"malignant": 1, "benign": 0}

IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
AUTOTUNE   = tf.data.AUTOTUNE

# Number of stain-aug copies per training image (in addition to the original)
N_AUG_TRAIN = 2
N_AUG_TEST  = 0   # keep test deterministic


# 2) Utilities
def collect_pairs(root: str) -> List[Tuple[str, int]]:
    pairs: List[Tuple[str, int]] = []
    for cls in CLASSES:
        paths = sorted(glob.glob(os.path.join(root, cls, "*.png")))
        pairs.extend((p, CLASS_TO_LABEL[cls]) for p in paths)
    rng = np.random.default_rng(42)
    rng.shuffle(pairs)
    return pairs

def _load_rgb_numpy(path_tensor, aug_index_tensor, training: bool):
    """Robust TF→NumPy bridge + optional pipeline stain augmentation."""
    # decode path
    if isinstance(path_tensor, (bytes, bytearray)):
        path = path_tensor.decode("utf-8")
    elif hasattr(path_tensor, "numpy"):
        path = path_tensor.numpy().decode("utf-8")
    else:
        path = str(path_tensor)

    # decode idx
    if hasattr(aug_index_tensor, "numpy"):
        idx = int(aug_index_tensor.numpy())
    else:
        idx = int(aug_index_tensor)

    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Pipeline stain augmentation only for training copies (idx>0)
    if training and TRY_HTK and AUG is not None and idx > 0:
        aug_list = AUG.run(rgb, mask_out=None)  # no tissue mask
        if len(aug_list) > 0:
            rgb = aug_list[0]

    # resize + normalize [0,1]
    rgb = cv2.resize(rgb, IMG_SIZE, interpolation=cv2.INTER_AREA)
    rgb = (rgb.astype(np.float32) / 255.0)
    return rgb

def _map_with_bridge(path: tf.Tensor, label: tf.Tensor, idx: tf.Tensor, training: bool):
    rgb = tf.numpy_function(
        func=_load_rgb_numpy,
        inp=[path, idx, training],
        Tout=tf.float32,
    )
    rgb.set_shape(IMG_SIZE + (3,))

    if training:
        # lightweight TF jitter (deterministic test has none)
        rgb = tf.image.random_flip_left_right(rgb)
        rgb = tf.image.random_brightness(rgb, max_delta=0.05)

    return rgb, tf.cast(label, tf.int32)


def make_dataset(root: str, training: bool) -> tf.data.Dataset:
    pairs = collect_pairs(root)
    if not pairs:
        raise RuntimeError(f"No images found under {root} for classes {CLASSES}")

    paths  = tf.constant([p for p, _ in pairs], dtype=tf.string)
    labels = tf.constant([l for _, l in pairs], dtype=tf.int32)
    base   = tf.data.Dataset.from_tensor_slices((paths, labels))

    n_aug = N_AUG_TRAIN if training else N_AUG_TEST

    # Expand each path to (1 + n_aug) samples via an index
    def expand(p, y):
        idxs = tf.range(n_aug + 1, dtype=tf.int32)
        return tf.data.Dataset.from_tensor_slices(idxs).map(lambda i: (p, y, i))

    ds = base.flat_map(expand)
    ds = ds.map(lambda p, y, i: _map_with_bridge(p, y, i, training),
                num_parallel_calls=AUTOTUNE)

    if training:
        ds = ds.shuffle(buffer_size=max(2048, len(pairs) * (n_aug + 1)), reshuffle_each_iteration=True)
        ds = ds.repeat()  # optional if you want steps_per_epoch-driven training

    # cache helps a lot when data fits RAM; remove if memory is tight
    ds = ds.cache()
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds, len(pairs) * (n_aug + 1)

# 3) Build datasets

train_ds, train_len = make_dataset(TRAIN_DIR, training=True)
test_ds,  test_len  = make_dataset(TEST_DIR,  training=False)


# 4) Model

base = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet",
)
base.trainable = False

inputs = tf.keras.Input(shape=IMG_SIZE + (3,))  # feeding tensors directly (not a dict)
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
x = base(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

model = tf.keras.Model(inputs, outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="binary_crossentropy",
              metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])

model.summary()


# 5) Train + evaluate

# If you used ds.repeat() for train, set steps_per_epoch
steps_per_epoch = max(1, train_len // BATCH_SIZE)
validation_steps = max(1, test_len // BATCH_SIZE)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath="./checkpoints/mnv2_{epoch:02d}_{val_accuracy:.3f}.keras",
        monitor="val_accuracy", save_best_only=True),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=5, restore_best_weights=True),
]

history = model.fit(
    train_ds,
    epochs=20,
    steps_per_epoch=steps_per_epoch,     # because train_ds is repeated
    validation_data=test_ds,
    validation_steps=validation_steps,
    callbacks=callbacks,
    verbose=2,
)

print("\nFinal test evaluation:")
model.evaluate(test_ds, verbose=2)