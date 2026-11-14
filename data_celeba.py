"""Load CelebA and attach the Blond_Hair attribute to each filename.

The course notebook reads list_attr_celeba.csv and passes that label list
to image_dataset_from_directory. Those two sequences match only when the
directory walk is in the same order as the attribute file. This loader
joins on the image filename, so the blond label stays on the right face.

Images are resized to 64x64 with bilinear interpolation, which is what
image_dataset_from_directory does with interpolation='bilinear'. Pixels
are stored as uint8. Training rescales them to [-1, 1].
"""

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from cgan import BATCH_SIZE, CLASSES, IMAGE_SIZE, LABEL

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / "cache"
ATTR_FILE = DATA_DIR / "list_attr_celeba.txt"
IMAGE_DIR = DATA_DIR / "img_align_celeba"
IMAGE_MEMMAP = CACHE_DIR / "images.uint8"
LABEL_FILE = CACHE_DIR / "labels.npy"
FILENAME_FILE = CACHE_DIR / "filenames.npy"


def read_attributes(path=ATTR_FILE):
    """Return filenames and 0/1 blond labels aligned to those filenames.

    CelebA stores attributes as -1 or +1. Blond is +1. Everything else,
    including the file's -1, is the negative class.
    """
    frame = pd.read_csv(path, sep=r"\s+", skiprows=1)
    filenames = frame.index.astype(str).to_numpy()
    blond = (frame[LABEL].to_numpy() == 1).astype(np.int32)
    return filenames, blond, frame


def prepare_cache(attr_file=ATTR_FILE, image_dir=IMAGE_DIR):
    """Resize every aligned face once and write a uint8 memmap plus labels."""
    filenames, labels, _ = read_attributes(attr_file)
    paths = [str(image_dir / name) for name in filenames]
    missing = [p for p in paths[:5] if not Path(p).exists()]
    if missing or not Path(paths[-1]).exists():
        raise FileNotFoundError(
            f"CelebA images are not in {image_dir}. "
            "See README.md for the download."
        )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    count = len(paths)
    images = np.memmap(
        IMAGE_MEMMAP, dtype=np.uint8, mode="w+", shape=(count, IMAGE_SIZE, IMAGE_SIZE, 3)
    )

    def load(path):
        raw = tf.io.read_file(path)
        image = tf.io.decode_jpeg(raw, channels=3)
        image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE], method="bilinear")
        image = tf.clip_by_value(tf.round(image), 0, 255)
        return tf.cast(image, tf.uint8)

    dataset = tf.data.Dataset.from_tensor_slices(paths)
    dataset = dataset.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(512).prefetch(tf.data.AUTOTUNE)

    cursor = 0
    for batch in dataset:
        batch_np = batch.numpy()
        images[cursor : cursor + len(batch_np)] = batch_np
        cursor += len(batch_np)
        if cursor % 20480 == 0 or cursor == count:
            print(f"cached {cursor}/{count} faces", flush=True)
    if cursor != count:
        raise RuntimeError(f"cache wrote {cursor} images, expected {count}")

    images.flush()
    np.save(LABEL_FILE, labels)
    np.save(FILENAME_FILE, np.asarray(filenames, dtype="U16"))
    print(
        f"blond {int(labels.sum())} / {count} "
        f"({labels.mean():.2%}) written to {CACHE_DIR}",
        flush=True,
    )
    return images, labels


def open_cache():
    if not IMAGE_MEMMAP.exists() or not LABEL_FILE.exists():
        prepare_cache()
    labels = np.load(LABEL_FILE)
    count = len(labels)
    images = np.memmap(
        IMAGE_MEMMAP, dtype=np.uint8, mode="r", shape=(count, IMAGE_SIZE, IMAGE_SIZE, 3)
    )
    return images, labels


def make_training_dataset(images, labels, batch_size=BATCH_SIZE, seed=42):
    """Infinite shuffled stream of (image in [-1, 1], one-hot blond label)."""
    count = len(labels)
    rng = np.random.default_rng(seed)

    def batches():
        while True:
            index = rng.integers(0, count, size=batch_size)
            pixels = images[index].astype(np.float32)
            pixels = (pixels - 127.5) / 127.5
            one_hot = np.zeros((batch_size, CLASSES), dtype=np.float32)
            one_hot[np.arange(batch_size), labels[index]] = 1.0
            yield pixels, one_hot

    signature = (
        tf.TensorSpec((batch_size, IMAGE_SIZE, IMAGE_SIZE, 3), tf.float32),
        tf.TensorSpec((batch_size, CLASSES), tf.float32),
    )
    return tf.data.Dataset.from_generator(batches, output_signature=signature).prefetch(
        tf.data.AUTOTUNE
    )
