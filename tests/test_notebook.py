"""CPU checks of the numbers quoted in the docs, run against notebook.py itself.

    python -m pytest -q

Tests that need the dataset are skipped if it is not in data/.
"""

import math
import os
import random
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import notebook as nb  # noqa: E402

DATA = os.path.join("data", "shift-guard-10-robust-image-classification-challenge")
HAS_DATA = os.path.isfile(os.path.join(DATA, "train_labels.csv"))
TRAIN_COUNTS = [4750, 4750, 4750, 4750, 3800, 3800, 475, 475, 285, 95]


def test_wrn_28_10_parameters_and_shapes():
    model = nb.WideResNet().eval()
    assert sum(p.numel() for p in model.parameters()) == 36_479_194
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        a = model.conv1(x)
        b = model.group1(a)
        c = model.group2(b)
        d = model.group3(c)
        assert model(x).shape == (2, 10)
    assert [a.shape[1:], b.shape[1:], c.shape[1:], d.shape[1:]] == [
        (16, 32, 32), (160, 32, 32), (320, 16, 16), (640, 8, 8)]


def test_kaiming_fan_out():
    torch.manual_seed(0)
    w = nb.WideResNet().group2[1].conv1.weight
    assert abs(w.std().item() - math.sqrt(2 / (320 * 9))) < 0.001


def test_balanced_softmax_equal_logits_give_training_prior():
    crit = nb.BalancedSoftmaxLoss(TRAIN_COUNTS)
    probs = F.softmax(crit.log_freq[None], dim=1)[0]
    prior = torch.tensor(TRAIN_COUNTS, dtype=torch.float32) / sum(TRAIN_COUNTS)
    assert torch.allclose(probs, prior, atol=1e-6)


def test_balanced_softmax_is_plain_ce_on_balanced_counts():
    crit = nb.BalancedSoftmaxLoss([100] * 10, label_smoothing=0.0)
    z, y = torch.randn(8, 10), torch.randint(0, 10, (8,))
    assert torch.allclose(crit(z, y), F.cross_entropy(z, y), atol=1e-6)


def test_cutmix_lambda_equals_kept_area():
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)
    for _ in range(200):
        x = torch.zeros(4, 1, 32, 32)
        x[1:] = 1.0
        out, ya, yb, lam = nb.cutmix_data(x, torch.arange(4), 1.0)
        if yb[0] != 0:
            assert abs((1 - lam) - (out[0] != 0).float().mean().item()) < 1e-6


def test_cutout_is_mean_grey():
    random.seed(1)
    out = nb.Cutout(16)(torch.randn(3, 32, 32))
    assert (out == 0).all(0).any()
    assert [round(m * 255) for m in nb.CIFAR_MEAN] == [125, 123, 114]


def test_tta_average_uses_clean_view_plus_n_views():
    # predict_with_tta divides by n_views + 1: one clean pass plus n random passes
    import inspect
    src = inspect.getsource(nb.predict_with_tta)
    assert "get_val_transforms()" in src and "accumulated / (n_views + 1)" in src


def test_final_defaults():
    import inspect
    src = inspect.getsource(nb.main)
    for s in ['"--epochs", type=int, default=450', '"--batch-size", type=int, default=512',
              '"--lr", type=float, default=0.1', '"--wd", type=float, default=5e-4',
              '"--swa-start", type=int, default=360', '"--swa-lr", type=float, default=0.005',
              '"--tta", type=int, default=30', 'default=[42, 137, 7]',
              '"--mix-prob", type=float, default=0.5', '"--label-smoothing", type=float, default=0.1']:
        assert s in src, s


@pytest.mark.skipif(not HAS_DATA, reason="dataset not in data/")
def test_split_95_5_class_wise():
    tr = nb.ShiftGuard10Dataset(DATA, "train", None, val_ratio=0.05, seed=42)
    va = nb.ShiftGuard10Dataset(DATA, "val", None, val_ratio=0.05, seed=42)
    assert len(tr) == 27_930 and len(va) == 1_470
    assert tr.get_class_counts().tolist() == TRAIN_COUNTS
    assert va.get_class_counts().tolist() == [250, 250, 250, 250, 200, 200, 25, 25, 15, 5]
    assert not set(tr.image_ids) & set(va.image_ids)


@pytest.mark.skipif(not HAS_DATA, reason="dataset not in data/")
def test_known_issue_classification_report_needs_labels():
    # Documented bug: crashes when fewer than 10 classes are present (what --debug hits).
    with pytest.raises(ValueError):
        nb.get_classification_report([0] * 5, [0] * 5)
