"""Measurements behind the written reading of the samples.

Fidelity is judged by looking at the pictures next to real faces.
Diversity is the spread across noise vectors of one class, compared
with the spread of real faces of that class.
Conditioning is judged on paired noise: same z, label flipped.
A hair-band luminance gap is only a proxy. The same measurement on the
center of the face shows whether the label only turned a brightness knob.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from cgan import IMAGE_SIZE, Z_DIM, build_generator, one_hot
from data_celeba import open_cache

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"
MODEL_DIR = ROOT / "models"

# Boxes on the 64x64 aligned face, checked against the mean difference
# image. The hair band is the arch above the eyes. The face box is the
# control: nose, mouth, and cheeks, not the hair.
HAIR_BOX = (slice(10, 26), slice(16, 48))
FACE_BOX = (slice(30, 46), slice(22, 42))


def to_float(images):
    images = np.asarray(images, dtype=np.float32)
    if images.min() >= 0 and images.max() > 1.5:
        images = images / 255.0
    elif images.min() < 0:
        images = (images + 1.0) / 2.0
    return images


def region_mean(images, box):
    rows, cols = box
    patch = to_float(images)[:, rows, cols, :]
    return patch.mean(axis=(1, 2, 3))


def pairwise_rmse(images, cap=64):
    flat = to_float(images[:cap]).reshape(len(images[:cap]), -1)
    # Mean RMSE between distinct pairs.
    diff = flat[:, None, :] - flat[None, :, :]
    rmse = np.sqrt((diff ** 2).mean(axis=-1))
    mask = ~np.eye(len(flat), dtype=bool)
    return float(rmse[mask].mean())


def pixel_std(images):
    return float(to_float(images).std(axis=0).mean())


def load_generator():
    generator = build_generator()
    dummy_z = tf.zeros((1, Z_DIM))
    dummy_y = one_hot(0, 1)
    generator([dummy_z, dummy_y], training=False)
    generator.load_weights(MODEL_DIR / "generator.weights.h5")
    return generator


def sample(generator, class_index, count, seed):
    noise = tf.random.normal((count, Z_DIM), seed=seed)
    images = generator([noise, one_hot(class_index, count)], training=False).numpy()
    return noise, images


def _outline(canvas, box, color):
    rows, cols = box
    canvas[rows, cols.start : cols.start + 2] = color
    canvas[rows, cols.stop - 2 : cols.stop] = color
    canvas[rows.start : rows.start + 2, cols] = color
    canvas[rows.stop - 2 : rows.stop, cols] = color


def draw_box_guide(image, path):
    canvas = (to_float(image) * 255).astype(np.uint8).copy()
    _outline(canvas, HAIR_BOX, (255, 210, 60))
    _outline(canvas, FACE_BOX, (80, 170, 255))
    fig, axis = plt.subplots(figsize=(2.6, 3.0))
    axis.imshow(canvas)
    axis.axis("off")
    axis.set_title("Yellow: hair band\nBlue: face center", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def save_difference_map(path, not_blond, blond):
    """Mean paired difference. Red means the blond label made that pixel brighter."""
    difference = to_float(blond) - to_float(not_blond)
    luminance = difference.mean(axis=(0, -1))
    color = np.clip(difference.mean(axis=0) / 0.25 + 0.5, 0, 1)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
    heat = axes[0].imshow(luminance, cmap="coolwarm", vmin=-0.05, vmax=0.20)
    axes[0].set_title("Blond minus not blond, luminance")
    fig.colorbar(heat, ax=axes[0], fraction=0.046)
    axes[1].imshow(color)
    axes[1].set_title("Same difference, drawn as color")
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle("Where the label moves pixels, averaged over 64 shared noise vectors", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    hair = difference[:, HAIR_BOX[0], HAIR_BOX[1], :].mean(axis=(0, 1, 2))
    face = difference[:, FACE_BOX[0], FACE_BOX[1], :].mean(axis=(0, 1, 2))
    return {
        "hair_rgb_blond_minus_not": [round(float(v), 4) for v in hair],
        "face_rgb_blond_minus_not": [round(float(v), 4) for v in face],
    }


def save_labeled_pair(path, not_blond, blond):
    """Two rows, same noise down each column, with the row label outside the photos."""
    not_blond = (np.clip(not_blond, -1, 1) + 1) / 2
    blond = (np.clip(blond, -1, 1) + 1) / 2
    count = not_blond.shape[0]
    fig, axes = plt.subplots(
        2, count, figsize=(1.45 * count, 3.5), gridspec_kw={"wspace": 0.05, "hspace": 0.08}
    )
    for column in range(count):
        axes[0, column].imshow(not_blond[column])
        axes[1, column].imshow(blond[column])
        axes[0, column].axis("off")
        axes[1, column].axis("off")
    fig.suptitle(
        "Same noise in each column. Only the Blond_Hair label changes.",
        fontsize=12,
    )
    fig.text(0.005, 0.70, "Not blond\n[1, 0]", va="center", ha="left", fontsize=9)
    fig.text(0.005, 0.28, "Blond\n[0, 1]", va="center", ha="left", fontsize=9)
    fig.subplots_adjust(left=0.08, right=0.995, top=0.86, bottom=0.02, wspace=0.05, hspace=0.12)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_class_grid(path, images, title, columns=8):
    images = (np.clip(images, -1, 1) + 1) / 2
    count = images.shape[0]
    rows = int(np.ceil(count / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(1.4 * columns, 1.5 * rows))
    axes = np.atleast_2d(axes)
    for index, axis in enumerate(axes.ravel()):
        axis.axis("off")
        if index < count:
            axis.imshow(images[index])
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_real_vs_fake(path, real_not, fake_not, real_blond, fake_blond):
    rows = [
        ("Real, not blond", real_not),
        ("Generated, not blond", fake_not),
        ("Real, blond", real_blond),
        ("Generated, blond", fake_blond),
    ]
    columns = 8
    fig, axes = plt.subplots(4, columns, figsize=(1.4 * columns, 6.6))
    for row, (title, batch) in enumerate(rows):
        batch = to_float(batch)
        for column in range(columns):
            axes[row, column].imshow(batch[column])
            axes[row, column].axis("off")
        axes[row, 0].set_title(title, loc="left", fontsize=9)
    fig.suptitle("Fidelity: real CelebA faces against samples from the same label", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_gap_histogram(path, hair_gap, cheek_gap):
    fig, axis = plt.subplots(figsize=(6.2, 3.4))
    bins = np.linspace(-0.35, 0.35, 29)
    axis.hist(hair_gap, bins=bins, alpha=0.85, label="Hair band", color="#ca8a04")
    axis.hist(cheek_gap, bins=bins, alpha=0.55, label="Face center", color="#0284c7")
    axis.axvline(0, color="black", lw=0.8)
    axis.set_xlabel("Blond minus not-blond, mean luminance, same noise")
    axis.set_ylabel("Latent vectors")
    axis.set_title("Hair band against the center of the face")
    axis.legend(frameon=False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    images, labels = open_cache()
    generator = load_generator()

    real_index = {}
    rng = np.random.default_rng(1)
    for class_index in (0, 1):
        pool = np.flatnonzero(labels == class_index)
        chosen = rng.choice(pool, size=64, replace=False)
        real_index[class_index] = images[chosen]

    draw_box_guide(real_index[1][0], OUTPUT_DIR / "measurement_boxes.png")

    report = {"classes": {}}
    generated = {}
    for class_index, name in ((0, "not_blond"), (1, "blond")):
        _, fake = sample(generator, class_index, 64, seed=1000 + class_index)
        generated[class_index] = fake
        real = real_index[class_index]
        report["classes"][name] = {
            "fake_pairwise_rmse": round(pairwise_rmse(fake), 4),
            "real_pairwise_rmse": round(pairwise_rmse(real), 4),
            "fake_pixel_std": round(pixel_std(fake), 4),
            "real_pixel_std": round(pixel_std(real), 4),
            "fake_hair_mean": round(float(region_mean(fake, HAIR_BOX).mean()), 4),
            "real_hair_mean": round(float(region_mean(real, HAIR_BOX).mean()), 4),
            "fake_face_mean": round(float(region_mean(fake, FACE_BOX).mean()), 4),
            "real_face_mean": round(float(region_mean(real, FACE_BOX).mean()), 4),
        }

    # Paired noise: the label is the only thing that changes.
    noise = tf.random.normal((64, Z_DIM), seed=123)
    paired_not = generator([noise, one_hot(0, 64)], training=False).numpy()
    paired_blond = generator([noise, one_hot(1, 64)], training=False).numpy()
    hair_gap = region_mean(paired_blond, HAIR_BOX) - region_mean(paired_not, HAIR_BOX)
    face_gap = region_mean(paired_blond, FACE_BOX) - region_mean(paired_not, FACE_BOX)
    report["difference_rgb"] = save_difference_map(
        OUTPUT_DIR / "label_difference_map.png", paired_not, paired_blond
    )

    save_labeled_pair(OUTPUT_DIR / "final_paired.png", paired_not[:8], paired_blond[:8])
    save_class_grid(
        OUTPUT_DIR / "final_not_blond.png",
        generated[0][:16],
        "Generated class 0 — not blond, one-hot [1, 0]",
    )
    save_class_grid(
        OUTPUT_DIR / "final_blond.png",
        generated[1][:16],
        "Generated class 1 — blond, one-hot [0, 1]",
    )
    save_real_vs_fake(
        OUTPUT_DIR / "real_vs_generated.png",
        real_index[0],
        generated[0],
        real_index[1],
        generated[1],
    )
    save_gap_histogram(OUTPUT_DIR / "label_luminance_gap.png", hair_gap, face_gap)

    report["paired_label_flip"] = {
        "n": 64,
        "hair_mean_blond_minus_not": round(float(hair_gap.mean()), 4),
        "hair_gap_positive_fraction": round(float((hair_gap > 0).mean()), 4),
        "face_mean_blond_minus_not": round(float(face_gap.mean()), 4),
        "face_gap_positive_fraction": round(float((face_gap > 0).mean()), 4),
    }

    # Real-data gap, same boxes, unpaired. This is what "blond" looks like
    # in CelebA, and the ceiling for the proxy.
    real_hair_gap = region_mean(real_index[1], HAIR_BOX).mean() - region_mean(
        real_index[0], HAIR_BOX
    ).mean()
    real_face_gap = region_mean(real_index[1], FACE_BOX).mean() - region_mean(
        real_index[0], FACE_BOX
    ).mean()
    report["real_unpaired_gap"] = {
        "hair_mean_blond_minus_not": round(float(real_hair_gap), 4),
        "face_mean_blond_minus_not": round(float(real_face_gap), 4),
    }
    (OUTPUT_DIR / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
