"""Reproduce every dataset number quoted in docs/dataset_analysis.md.

    python scripts/analyze_dataset.py

Writes results/dataset_stats.json and charts in results/figures/. A grid of sample images
goes to results/local/ (git-ignored: the competition rules do not allow redistributing the data).
Needs numpy, pandas, pillow, scipy, matplotlib. Takes about 2 minutes on a laptop.
"""

import hashlib
import json
import os
import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import find_objects, label as connected
from scipy.signal import convolve2d

ROOT = os.environ.get("SG10_DATA_ROOT",
                      os.path.join("data", "shift-guard-10-robust-image-classification-challenge"))
OUT = "results"
FIG = os.path.join(OUT, "figures")
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]
OCCLUDER = np.array([125, 123, 114])          # CIFAR mean colour in 0-255
NOISE_KERNEL = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]])


def load_images(folder, ids):
    return np.stack([np.asarray(Image.open(os.path.join(ROOT, folder, f"{i}.png")).convert("RGB"))
                     for i in ids])


def noise_sigma(channel):
    """Immerkaer (1996) fast noise estimate on one 32x32 channel."""
    r = convolve2d(channel, NOISE_KERNEL, mode="valid")
    return np.sqrt(np.pi / 2) * np.abs(r).sum() / (6 * r.size)


def chroma_noise(X):
    """Noise in colour-difference channels. Natural texture is mostly the same
    in R, G and B, so it cancels in R-G and B-G. Added per-pixel colour noise
    does not cancel, so this separates added noise from busy textures."""
    X = X.astype(np.float64)
    return np.array([0.5 * (noise_sigma(x[..., 0] - x[..., 1]) + noise_sigma(x[..., 2] - x[..., 1]))
                     for x in X])


def luma_noise(X):
    return np.array([noise_sigma(x.astype(np.float64).mean(-1)) for x in X])


def occlusion_patches(X):
    """Largest solid block of the occluder colour in each image."""
    rows = []
    for n, img in enumerate(X):
        mask = (img == OCCLUDER).all(-1)
        if mask.sum() < 16:
            continue
        lab, _ = connected(mask)
        best = None
        for sl in find_objects(lab):
            h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
            if h >= 4 and w >= 4 and (best is None or h * w > best[1] * best[2]):
                best = (n, h, w, sl[0].start, sl[1].start)
        if best:
            rows.append(best)
    return pd.DataFrame(rows, columns=["idx", "h", "w", "y", "x"])


def q(v):
    return {p: round(float(np.percentile(v, p)), 2) for p in (10, 25, 50, 75, 90)}


def grid(rows, path, titles):
    fig, axes = plt.subplots(len(rows), len(rows[0]), figsize=(len(rows[0]) * 1.1, len(rows) * 1.25))
    for r, row in enumerate(rows):
        for c, img in enumerate(row):
            ax = axes[r, c]
            ax.imshow(img, interpolation="nearest")
            ax.axis("off")
        axes[r, 0].set_title(titles[r], fontsize=8, loc="left")
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()


def main():
    if not os.path.isfile(os.path.join(ROOT, "train_labels.csv")):
        sys.exit(f"Dataset not found at {ROOT}. See data/README.md.")
    plt.switch_backend("Agg")
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(os.path.join(OUT, "local"), exist_ok=True)
    S = {}

    tr = pd.read_csv(os.path.join(ROOT, "train_labels.csv"), dtype=str)
    ss = pd.read_csv(os.path.join(ROOT, "sample_submission.csv"), dtype=str)
    S["files"] = {
        "train_rows": len(tr), "test_rows": len(ss),
        "train_png": len(os.listdir(os.path.join(ROOT, "train_images"))),
        "test_png": len(os.listdir(os.path.join(ROOT, "test_images"))),
        "classes_txt": open(os.path.join(ROOT, "classes.txt")).read().split(),
        "sample_submission_labels": ss.label.value_counts().to_dict(),
        "id_format": "6-digit zero-padded string; train and test ids both start at 000001",
    }
    counts = tr.label.value_counts().reindex(CLASSES)
    S["class_counts"] = counts.to_dict()
    S["imbalance_ratio_max_over_min"] = float(counts.max() / counts.min())

    X_tr = load_images("train_images", tr.id)
    X_te = load_images("test_images", ss.id)
    S["image_shape"] = list(X_tr.shape[1:])
    S["pixel_mean_0_1"] = {"train": (X_tr / 255).mean((0, 1, 2)).round(4).tolist(),
                           "test": (X_te / 255).mean((0, 1, 2)).round(4).tolist(),
                           "cifar_constants": [0.4914, 0.4822, 0.4465]}
    S["pixel_std_0_1"] = {"train": (X_tr / 255).std((0, 1, 2)).round(4).tolist(),
                          "test": (X_te / 255).std((0, 1, 2)).round(4).tolist(),
                          "cifar_constants": [0.2470, 0.2435, 0.2616]}

    h_tr = [hashlib.md5(a.tobytes()).hexdigest() for a in X_tr]
    h_te = [hashlib.md5(a.tobytes()).hexdigest() for a in X_te]
    S["exact_duplicates"] = {
        "within_train": len(h_tr) - len(set(h_tr)),
        "within_test": len(h_te) - len(set(h_te)),
        "test_images_also_in_train": len(set(h_te) & set(h_tr)),
    }

    p_tr, p_te = occlusion_patches(X_tr), occlusion_patches(X_te)
    S["occlusion"] = {
        "colour_rgb": OCCLUDER.tolist(),
        "train_images_with_patch": len(p_tr), "train_pct": round(100 * len(p_tr) / len(X_tr), 2),
        "train_patch_sizes": {f"{h}x{w}": n for (h, w), n in Counter(zip(p_tr.h, p_tr.w)).items()},
        "test_images_with_patch": len(p_te), "test_pct": round(100 * len(p_te) / len(X_te), 2),
        "test_patch_sizes": {f"{h}x{w}": n for (h, w), n in Counter(zip(p_te.h, p_te.w)).items()},
        "train_pct_by_class": (tr.iloc[p_tr.idx].label.value_counts() / tr.label.value_counts() * 100)
        .reindex(CLASSES).round(2).to_dict(),
    }

    c_tr, c_te = chroma_noise(X_tr), chroma_noise(X_te)
    l_tr, l_te = luma_noise(X_tr), luma_noise(X_te)
    S["noise"] = {
        "method": "Immerkaer noise estimate on R-G and B-G (chroma) and on gray (luma)",
        "chroma_quantiles": {"train": q(c_tr), "test": q(c_te)},
        "luma_quantiles": {"train": q(l_tr), "test": q(l_te)},
        "pct_images_chroma_above_3.5": {"train": round(100 * float((c_tr > 3.5).mean()), 1),
                                        "test": round(100 * float((c_te > 3.5).mean()), 1)},
        "pct_images_chroma_above_6": {"train": round(100 * float((c_tr > 6).mean()), 1),
                                      "test": round(100 * float((c_te > 6).mean()), 1)},
        "luma_median_by_train_class": pd.Series(l_tr).groupby(tr.label.values).median()
        .reindex(CLASSES).round(2).to_dict(),
    }

    with open(os.path.join(OUT, "dataset_stats.json"), "w") as f:
        json.dump(S, f, indent=1)

    # Figures
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(CLASSES, counts.values, color="#4C72B0")
    ax.set_yscale("log")
    ax.set_ylabel("training images (log scale)")
    for i, v in enumerate(counts.values):
        ax.text(i, v * 1.1, str(v), ha="center", fontsize=8)
    ax.set_title("Training class counts: 5,000 cats vs 100 trucks (50:1)")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "class_counts.png"), dpi=110)
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 3))
    bins = np.linspace(0, 12, 49)
    ax.hist(c_tr, bins, density=True, alpha=0.6, label="train")
    ax.hist(c_te, bins, density=True, alpha=0.6, label="test")
    ax.set_xlabel("estimated colour-noise level (higher = noisier)")
    ax.set_ylabel("density")
    ax.set_title("Colour noise: about 21% of train vs about 75% of test images are noisy")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "noise_train_vs_test.png"), dpi=110)
    plt.close()

    rng = np.random.RandomState(0)
    grid([X_tr[rng.choice(len(X_tr), 10, replace=False)],
          X_te[rng.choice(len(X_te), 10, replace=False)],
          X_tr[p_tr.idx.values[:10]],
          X_te[p_te.idx.values[:10]],
          X_te[np.argsort(-c_te)[:10]]],
         os.path.join(OUT, "local", "train_vs_test_samples.png"),
         ["random train", "random test", "train with 6x6 patch", "test with 10x10 patch",
          "noisiest test"])
    print(json.dumps(S, indent=1))


if __name__ == "__main__":
    main()
