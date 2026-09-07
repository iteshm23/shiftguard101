"""
ShiftGuard10 Dataset and Augmentation Pipeline.

Augmentation strategy (training):
  AutoAugment (CIFAR10 policy) → Random Crop+Flip → Cutout → MixUp/CutMix

Test-Time Augmentation (TTA):
  31 stochastic views (flip + crop + rotation + colour jitter) averaged at inference.
"""

import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import AutoAugment, AutoAugmentPolicy


# CIFAR-10 normalisation statistics
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for name, idx in CLASS_TO_IDX.items()}


class Cutout:
    """Zero-mask a random square patch of size (length × length)."""
    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length  = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = torch.ones(h, w, dtype=img.dtype, device=img.device)
        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = max(0, y - self.length // 2)
            y2 = min(h, y + self.length // 2)
            x1 = max(0, x - self.length // 2)
            x2 = min(w, x + self.length // 2)
            mask[y1:y2, x1:x2] = 0.0
        return img * mask.unsqueeze(0)


def get_train_transforms():
    """
    Full training augmentation pipeline.

    1. AutoAugment (CIFAR10 learned policy — 25 sub-policies from NAS search)
    2. Random crop with 4-pixel padding
    3. Random horizontal flip
    4. Tensor conversion and CIFAR-10 normalisation
    5. Cutout (16×16 random patch zeroed)
    """
    return transforms.Compose([
        transforms.AutoAugment(AutoAugmentPolicy.CIFAR10),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        Cutout(n_holes=1, length=16),
    ])


def get_val_transforms():
    """Clean validation/test transforms — no augmentation."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])


def get_tta_transforms(n_views=31):
    """
    Test-Time Augmentation: n_views stochastic views + 1 clean view.

    Augmented views apply random crop, flip, rotation, and colour jitter to
    simulate distribution shift at test time. Probabilities are averaged over
    all views before taking argmax.
    """
    aug = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2,
                               saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    clean = get_val_transforms()
    return clean, aug, n_views


class ShiftGuard10Dataset(Dataset):
    """
    Dataset loader for ShiftGuard10.

    Args:
        root      : path to 'shift-guard-10-robust-image-classification-challenge/'
        split     : 'train', 'val', or 'test'
        transform : torchvision transform pipeline
        val_ratio : fraction held out for validation (stratified by class)
        seed      : random seed for reproducible splits
    """
    def __init__(self, root, split="train", transform=None,
                 val_ratio=0.1, seed=42):
        self.root      = root
        self.split     = split
        self.transform = transform

        if split in ("train", "val"):
            labels_df = pd.read_csv(os.path.join(root, "train_labels.csv"))
            labels_df["label"] = labels_df["label"].str.strip()
            labels_df["id"]    = labels_df["id"].astype(str).str.zfill(6)

            # Stratified split — equal class representation in both folds
            rng = np.random.RandomState(seed)
            train_idx, val_idx = [], []
            for cls in CLASS_NAMES:
                idx = labels_df[labels_df["label"] == cls].index.tolist()
                rng.shuffle(idx)
                n_val = max(1, int(len(idx) * val_ratio))
                val_idx.extend(idx[:n_val])
                train_idx.extend(idx[n_val:])

            labels_df = labels_df.iloc[train_idx if split == "train" else val_idx]
            labels_df = labels_df.reset_index(drop=True)

            self.image_ids = labels_df["id"].tolist()
            self.labels    = [CLASS_TO_IDX[l] for l in labels_df["label"]]
            self.image_dir = os.path.join(root, "train_images")

        elif split == "test":
            sub_df = pd.read_csv(os.path.join(root, "sample_submission.csv"))
            sub_df["id"] = sub_df["id"].astype(str).str.zfill(6)
            self.image_ids = sub_df["id"].tolist()
            self.labels    = None
            self.image_dir = os.path.join(root, "test_images")
        else:
            raise ValueError(f"Unknown split: {split!r}")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        path  = os.path.join(self.image_dir, f"{self.image_ids[idx]}.png")
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        if self.labels is not None:
            return image, self.labels[idx]
        return image, self.image_ids[idx]

    def get_class_counts(self):
        """Number of samples per class (length = num_classes)."""
        return np.bincount(self.labels, minlength=len(CLASS_NAMES))

    def get_sampler(self):
        """Inverse-frequency WeightedRandomSampler for balanced batches."""
        counts = self.get_class_counts()
        class_w = 1.0 / (counts + 1e-6)
        sample_w = [class_w[l] for l in self.labels]
        return WeightedRandomSampler(
            weights=sample_w,
            num_samples=len(self.labels),
            replacement=True,
        )
