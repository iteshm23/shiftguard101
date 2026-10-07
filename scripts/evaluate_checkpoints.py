"""Measure the parts of the pipeline that were never measured, using saved checkpoints.

Post-competition analysis, not part of the submitted solution. It imports the model,
dataset and TTA code from notebook.py unchanged, and evaluates on the 5% validation
split (macro F1), optionally with test-like perturbations added.

Examples (checkpoints are the wrn_seed{seed}.pth files from the Kaggle Model):
    python scripts/evaluate_checkpoints.py --ckpt-dir ckpts --seeds 42 --tta 0
    python scripts/evaluate_checkpoints.py --ckpt-dir ckpts --seeds 42 137 7 --tta 30
    python scripts/evaluate_checkpoints.py --ckpt-dir ckpts --seeds 42 --tta 30 --shifted

Comparing --tta 0 vs 30, one seed vs three, and clean vs --shifted gives real numbers for
"how much did TTA / the ensemble add" and "how robust is it to the test-like shift".
"""

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import notebook as nb  # noqa: E402

OCCLUDER = (125, 123, 114)


class TestLikeShift(Dataset):
    """Validation images with fixed per-image test-like changes: a 10x10 grey patch on 5% of
    images and Gaussian colour noise (std 3.5 to 6 on 0-255) on 75%, as measured on the
    test set (docs/dataset_analysis.md)."""

    def __init__(self, base, seed=0):
        self.base, self.seed = base, seed
        self.labels, self.transform = base.labels, None

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        tf, self.base.transform = self.transform, None
        img, y = self.base[i]
        self.base.transform = tf
        rng = np.random.default_rng(self.seed * 100003 + i)
        a = np.asarray(img).astype(np.float32)
        if rng.random() < 0.75:
            a = a + rng.normal(0, rng.uniform(3.5, 6.0), a.shape)
        if rng.random() < 0.05:
            yy, xx = rng.integers(0, 23, 2)
            a[yy:yy + 10, xx:xx + 10] = OCCLUDER
        img = Image.fromarray(np.clip(a, 0, 255).round().astype(np.uint8))
        return (self.transform(img) if self.transform else img), y


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=os.path.join(
        "data", "shift-guard-10-robust-image-classification-challenge"))
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 137, 7])
    p.add_argument("--tta", type=int, default=30, help="random views on top of the clean view; 0 = clean only")
    p.add_argument("--shifted", action="store_true")
    args = p.parse_args()

    nb.seed_everything(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val = nb.ShiftGuard10Dataset(args.data_root, "val", nb.get_val_transforms(), val_ratio=0.05, seed=42)
    data = TestLikeShift(val) if args.shifted else val
    targets = val.labels

    total = None
    for seed in args.seeds:
        ckpt = torch.load(os.path.join(args.ckpt_dir, f"wrn_seed{seed}.pth"),
                          map_location=device, weights_only=False)
        state = ckpt.get("best_state", ckpt.get("model_state", ckpt))
        model = nb.WideResNet(depth=28, widen_factor=10, num_classes=10, dropout=0.3).to(device)
        model.load_state_dict(state)
        probs, _ = nb.predict_with_tta(model, data, device, n_views=args.tta, batch_size=512)
        f1 = nb.compute_macro_f1(probs.argmax(1).numpy(), targets)
        print(f"seed {seed}: val macro F1 {f1:.4f} (tta={args.tta}, shifted={args.shifted})")
        total = probs if total is None else total + probs
    pred = (total / len(args.seeds)).argmax(1).numpy()
    print(f"ensemble of {len(args.seeds)}: val macro F1 {nb.compute_macro_f1(pred, targets):.4f}")
    from sklearn.metrics import classification_report
    print(classification_report(targets, pred, labels=list(range(10)),
                                target_names=nb.CLASS_NAMES, zero_division=0))


if __name__ == "__main__":
    main()
