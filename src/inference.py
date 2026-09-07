"""
ShiftGuard10 Inference — single model or multi-checkpoint ensemble with TTA.

Usage:
  # Single model, no TTA
  python src/inference.py --checkpoint checkpoints/best_wrn_s42.pth

  # 31-view TTA (default)
  python src/inference.py --checkpoint checkpoints/best_wrn_s42.pth --tta 31

  # 3-seed ensemble + 31-view TTA
  python src/inference.py \\
      --checkpoint checkpoints/best_wrn_s42.pth \\
                   checkpoints/best_wrn_s7.pth \\
                   checkpoints/best_wrn_s13.pth \\
      --tta 31
"""

import os
import sys
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import (
    ShiftGuard10Dataset, get_val_transforms, get_tta_transforms,
    IDX_TO_CLASS, CLASS_NAMES,
)
from src.models.wideresnet import wrn_28_10


def load_model(checkpoint_path, device):
    ckpt    = torch.load(checkpoint_path, map_location=device)
    cfg     = ckpt["config"]
    wrn_cfg = cfg["model"]["wrn"]
    model   = wrn_28_10(num_classes=cfg["model"]["num_classes"],
                        dropout=wrn_cfg["dropout"])
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    seed  = ckpt.get("seed", "?")
    f1    = ckpt.get("best_f1", 0.0)
    print(f"  Loaded WRN-28-10 | seed={seed} | best_f1={f1:.4f} | {checkpoint_path}")
    return model


@torch.no_grad()
def predict_single(model, loader, device):
    probs, ids = [], []
    for images, img_ids in tqdm(loader, desc="Predicting", leave=False):
        logits = model(images.to(device))
        probs.append(F.softmax(logits, dim=1).cpu())
        ids.extend(img_ids)
    return torch.cat(probs, dim=0), ids


@torch.no_grad()
def predict_tta(model, dataset, device, n_views=31, batch_size=256):
    """
    31-view TTA: 1 clean pass + 30 stochastic augmented passes.
    Final probability is the arithmetic mean across all views.
    """
    clean_tf, aug_tf, _ = get_tta_transforms(n_views)

    # View 1: clean (no augmentation)
    dataset.transform = clean_tf
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    accumulated, all_ids = predict_single(model, loader, device)

    # Views 2..n_views: stochastic augmentation
    dataset.transform = aug_tf
    for v in range(1, n_views):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        aug_probs, _ = predict_single(model, loader, device)
        accumulated = accumulated + aug_probs
        if (v + 1) % 5 == 0 or v == n_views - 1:
            print(f"    TTA view {v + 1}/{n_views}")

    return accumulated / n_views, all_ids


def generate_submission(probs, ids, output_path):
    preds  = probs.argmax(dim=1).numpy()
    labels = [IDX_TO_CLASS[p] for p in preds]
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write("id,label\n")
        for img_id, label in zip(ids, labels):
            f.write(f"{img_id},{label}\n")
    print(f"\n  Submission saved → {output_path}  ({len(ids)} rows)")
    from collections import Counter
    dist = Counter(labels)
    for cls in CLASS_NAMES:
        print(f"    {cls:12s}: {dist.get(cls, 0):5d}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", nargs="+", required=True,
                        help="Checkpoint path(s); multiple = ensemble")
    parser.add_argument("--data-root",  type=str,
                        default="shift-guard-10-robust-image-classification-challenge")
    parser.add_argument("--output",     type=str, default="submission.csv")
    parser.add_argument("--tta",        type=int, default=31,
                        help="TTA views (default 31; 0 = disabled)")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--gpu",        type=int, default=0)
    args = parser.parse_args()

    device = (torch.device(f"cuda:{args.gpu}")
              if torch.cuda.is_available() else torch.device("cpu"))
    print(f"\n{'='*60}")
    print(f"  ShiftGuard10 Inference")
    print(f"  Device: {device} | TTA views: {args.tta or 'disabled'}")
    print(f"  Checkpoints ({len(args.checkpoint)}):")
    for c in args.checkpoint:
        print(f"    {c}")
    print(f"{'='*60}\n")

    test_ds = ShiftGuard10Dataset(args.data_root, "test",
                                  transform=get_val_transforms())
    print(f"  Test samples: {len(test_ds)}")

    ensemble_probs = None
    for ckpt_path in args.checkpoint:
        model = load_model(ckpt_path, device)
        if args.tta > 0:
            probs, ids = predict_tta(
                model, test_ds, device,
                n_views=args.tta, batch_size=args.batch_size,
            )
        else:
            loader = DataLoader(test_ds, batch_size=args.batch_size,
                                shuffle=False, num_workers=4)
            probs, ids = predict_single(model, loader, device)

        ensemble_probs = probs if ensemble_probs is None else ensemble_probs + probs

    ensemble_probs /= len(args.checkpoint)
    generate_submission(ensemble_probs, ids, args.output)


if __name__ == "__main__":
    main()
