"""Utility functions: metrics, MixUp/CutMix, seeding, checkpointing."""

import os
import random
import numpy as np
import torch
from sklearn.metrics import f1_score, classification_report


def seed_everything(seed=42):
    """Deterministic seeding for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def compute_macro_f1(preds, targets, num_classes=10):
    """Macro F1 — the official competition metric."""
    return f1_score(targets, preds, average="macro", zero_division=0)


def get_classification_report(preds, targets, class_names):
    """Full per-class precision / recall / F1."""
    return classification_report(
        targets, preds,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
    )


# ─── MixUp & CutMix ──────────────────────────────────────────────────────────

def mixup_data(x, y, alpha=1.0):
    """MixUp: convex combination of random image pairs."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def cutmix_data(x, y, alpha=1.0):
    """CutMix: paste a rectangular crop from one image into another."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    W, H = x.size(3), x.size(2)
    rw = int(W * np.sqrt(1.0 - lam))
    rh = int(H * np.sqrt(1.0 - lam))
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1, x2 = max(0, cx - rw // 2), min(W, cx + rw // 2)
    y1, y2 = max(0, cy - rh // 2), min(H, cy + rh // 2)
    x = x.clone()
    x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam = 1 - (y2 - y1) * (x2 - x1) / (W * H)
    return x, y, y[idx], lam


def mixup_cutmix_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ─── Checkpoint I/O ──────────────────────────────────────────────────────────

def save_checkpoint(state, filepath):
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    torch.save(state, filepath)
    print(f"  [SAVED] {filepath}")


def load_checkpoint(filepath, model, optimizer=None):
    ckpt = torch.load(filepath, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt.get("epoch", 0), ckpt.get("best_f1", 0.0)


class AverageMeter:
    """Running mean tracker."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count
