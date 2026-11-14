"""Train the conditional WGAN-GP and write the samples the report reads.

Run from the repository root:

    python train.py
    python train.py --steps 2000

The default step count is the course notebook's: 20 * 100 updates,
one batch each. Weights, the loss history, and sample grids land in
models/ and output/.
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from cgan import (
    ADAM_BETA_1,
    ADAM_BETA_2,
    BATCH_SIZE,
    CRITIC_STEPS,
    GP_WEIGHT,
    IMAGE_SIZE,
    LABEL,
    LEARNING_RATE,
    TRAIN_STEPS,
    Z_DIM,
    build_cgan,
    expand_label_to_map,
    one_hot,
)
from data_celeba import make_training_dataset, open_cache, read_attributes

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"
MODEL_DIR = ROOT / "models"


def to_uint8(images):
    images = np.clip(np.asarray(images), -1.0, 1.0)
    return ((images + 1.0) * 127.5).astype(np.uint8)


def save_strip(path, images, title):
    images = to_uint8(images)
    count = images.shape[0]
    fig, axes = plt.subplots(1, count, figsize=(1.35 * count, 1.7))
    for axis, image in zip(axes, images):
        axis.imshow(image)
        axis.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def save_paired(path, not_blond, blond, title):
    """Same noise in each column. Row 0 is class 0, row 1 is class 1."""
    not_blond = to_uint8(not_blond)
    blond = to_uint8(blond)
    count = not_blond.shape[0]
    fig, axes = plt.subplots(2, count, figsize=(1.35 * count, 3.3))
    for column in range(count):
        axes[0, column].imshow(not_blond[column])
        axes[1, column].imshow(blond[column])
        axes[0, column].axis("off")
        axes[1, column].axis("off")
    axes[0, 0].set_title("not blond  [1, 0]", loc="left", fontsize=10)
    axes[1, 0].set_title("blond      [0, 1]", loc="left", fontsize=10)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def save_real_references(images, labels):
    rng = np.random.default_rng(0)
    for class_index, name in ((0, "real_not_blond"), (1, "real_blond")):
        pool = np.flatnonzero(labels == class_index)
        chosen = rng.choice(pool, size=10, replace=False)
        pixels = images[chosen].astype(np.float32)
        pixels = (pixels - 127.5) / 127.5
        save_strip(
            OUTPUT_DIR / f"{name}.png",
            pixels,
            f"Real CelebA faces, {LABEL} class {class_index} ({name.replace('real_', '').replace('_', ' ')})",
        )


def generate_pair(generator, noise):
    count = noise.shape[0]
    not_blond = generator([noise, one_hot(0, count)], training=False).numpy()
    blond = generator([noise, one_hot(1, count)], training=False).numpy()
    return not_blond, blond


def save_checkpoint_samples(generator, noise, step):
    not_blond, blond = generate_pair(generator, noise)
    tag = f"{step:04d}"
    save_paired(
        OUTPUT_DIR / f"paired_step_{tag}.png",
        not_blond,
        blond,
        f"Same noise, flipped Blond_Hair label, after {step} updates",
    )
    save_strip(
        OUTPUT_DIR / f"generated_not_blond_step_{tag}.png",
        not_blond,
        f"Generated class 0, not blond, step {step}",
    )
    save_strip(
        OUTPUT_DIR / f"generated_blond_step_{tag}.png",
        blond,
        f"Generated class 1, blond, step {step}",
    )


def write_summaries(cgan):
    with open(OUTPUT_DIR / "critic_summary.txt", "w") as handle:
        cgan.critic.summary(print_fn=lambda line: handle.write(line + "\n"))
    with open(OUTPUT_DIR / "generator_summary.txt", "w") as handle:
        cgan.generator.summary(print_fn=lambda line: handle.write(line + "\n"))


def shape_audit(cgan, dataset):
    """Show the format mismatch, then the expansion that removes it."""
    real_images, one_hot_labels = next(iter(dataset.take(1)))
    label_map = expand_label_to_map(one_hot_labels)
    noise = tf.random.normal((BATCH_SIZE, Z_DIM))
    fake_images = cgan.generator([noise, one_hot_labels], training=False)
    scores = cgan.critic([fake_images, label_map], training=False)

    rejected = "the call was accepted"
    try:
        cgan.critic([fake_images, one_hot_labels], training=False)
    except Exception as error:
        rejected = f"{type(error).__name__}: {error}"

    lines = [
        "Generator noise shape:        " + str(tuple(noise.shape)),
        "Generator one-hot shape:      " + str(tuple(one_hot_labels.shape)),
        "Generator image shape:        " + str(tuple(fake_images.shape)),
        "Critic label-map shape:       " + str(tuple(label_map.shape)),
        "Critic score shape:           " + str(tuple(scores.shape)),
        "Map pixel (0, 0) equals vector: "
        + str(bool(tf.reduce_all(label_map[:, 0, 0, :] == one_hot_labels))),
        "Map pixel (63, 63) equals vector: "
        + str(bool(tf.reduce_all(label_map[:, -1, -1, :] == one_hot_labels))),
        "Critic called with the unexpanded vector: " + rejected,
    ]
    report = "\n".join(lines)
    (OUTPUT_DIR / "shape_audit.txt").write_text(report + "\n")
    print(report, flush=True)
    return report


def plot_history(history, path):
    steps = np.arange(1, len(history["c_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    axes[0].plot(steps, history["c_wass_loss"], color="#1f4e79", lw=1.0)
    axes[0].set_title("Critic Wasserstein estimate")
    axes[0].set_xlabel("Update")
    axes[1].plot(steps, history["c_gp"], color="#9a3412", lw=1.0)
    axes[1].set_title("Gradient penalty")
    axes[1].set_xlabel("Update")
    axes[2].plot(steps, history["g_loss"], color="#166534", lw=1.0)
    axes[2].set_title("Generator loss")
    axes[2].set_xlabel("Update")
    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def train(steps, seed):
    tf.keras.utils.set_random_seed(seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    filenames, labels, _ = read_attributes()
    print(
        f"{LABEL}: {int(labels.sum())} blond / {len(labels)} faces "
        f"({labels.mean():.2%})",
        flush=True,
    )
    images, labels = open_cache()
    save_real_references(images, labels)
    dataset = make_training_dataset(images, labels, seed=seed)

    cgan = build_cgan()
    write_summaries(cgan)
    shape_audit(cgan, dataset)

    noise = tf.random.normal((10, Z_DIM), seed=7)
    save_checkpoint_samples(cgan.generator, noise, step=0)

    history = {key: [] for key in ("c_loss", "c_wass_loss", "c_gp", "g_loss")}
    started = time.time()
    print(f"training {steps} updates, batch {BATCH_SIZE}", flush=True)

    for step, batch in enumerate(dataset, start=1):
        metrics = cgan.train_step(batch)
        for key in history:
            history[key].append(float(metrics[key]))
        if step % 25 == 0 or step == 1:
            elapsed = time.time() - started
            print(
                f"step {step:4d}/{steps}  "
                f"c_wass {history['c_wass_loss'][-1]:+.3f}  "
                f"c_gp {history['c_gp'][-1]:.3f}  "
                f"g_loss {history['g_loss'][-1]:+.3f}  "
                f"elapsed {elapsed/60:.1f} min",
                flush=True,
            )
        if step % 100 == 0 or step == steps:
            save_checkpoint_samples(cgan.generator, noise, step=step)
            plot_history(history, OUTPUT_DIR / "loss_curves.png")
            cgan.generator.save_weights(MODEL_DIR / "generator.weights.h5")
            cgan.critic.save_weights(MODEL_DIR / "critic.weights.h5")
        if step >= steps:
            break

    elapsed = time.time() - started
    table = np.column_stack([np.arange(1, steps + 1)] + [history[k] for k in history])
    header = "step," + ",".join(history)
    np.savetxt(
        OUTPUT_DIR / "history.csv",
        table,
        delimiter=",",
        header=header,
        comments="",
    )
    metadata = {
        "label": LABEL,
        "class_0": "not blond, one-hot [1, 0], CelebA attribute != 1",
        "class_1": "blond, one-hot [0, 1], CelebA attribute == 1",
        "images": int(len(labels)),
        "blond_images": int(labels.sum()),
        "blond_rate": float(labels.mean()),
        "image_size": IMAGE_SIZE,
        "z_dim": Z_DIM,
        "batch_size": BATCH_SIZE,
        "steps": steps,
        "critic_steps": CRITIC_STEPS,
        "gp_weight": GP_WEIGHT,
        "learning_rate": LEARNING_RATE,
        "adam_beta_1": ADAM_BETA_1,
        "adam_beta_2": ADAM_BETA_2,
        "seed": seed,
        "minutes": round(elapsed / 60, 2),
        "filenames_in_attribute_file": int(len(filenames)),
    }
    (OUTPUT_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"finished in {elapsed/60:.1f} min", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=TRAIN_STEPS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(steps=args.steps, seed=args.seed)


if __name__ == "__main__":
    main()
