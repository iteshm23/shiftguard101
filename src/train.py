"""
ShiftGuard10 Training Script — WideResNet-28-10 with Balanced Softmax.

Pipeline:
  AutoAugment → MixUp/CutMix → Balanced Softmax Loss → Cosine LR Warmup
  → Nesterov SGD → SWA (final 25% of training)

Usage:
  python src/train.py                         # default: WRN-28-10, 300 epochs
  python src/train.py --seed 7                # different seed for ensemble member
  python src/train.py --debug                 # CPU sanity check (2 epochs)
  python src/train.py --resume checkpoints/best_wrn_s42.pth
"""

import os
import sys
import time
import argparse
import yaml
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import (
    ShiftGuard10Dataset, get_train_transforms, get_val_transforms, CLASS_NAMES
)
from src.models.wideresnet import wrn_28_10
from src.loss import BalancedSoftmaxLoss
from src.utils import (
    seed_everything, compute_macro_f1, get_classification_report,
    mixup_data, cutmix_data, mixup_cutmix_criterion,
    save_checkpoint, load_checkpoint, AverageMeter,
)


def build_model(num_classes, cfg, device):
    wrn_cfg = cfg["model"]["wrn"]
    model = wrn_28_10(num_classes=num_classes, dropout=wrn_cfg["dropout"])
    model = model.to(device)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: WideResNet-28-10 | Params: {n:,}")
    return model


def train_one_epoch(model, loader, criterion, optimizer, device, cfg, epoch):
    model.train()
    loss_meter = AverageMeter()
    correct = total = 0

    mix_prob     = cfg["training"]["mix_prob"]
    mixup_alpha  = cfg["training"]["mixup_alpha"]
    cutmix_alpha = cfg["training"]["cutmix_alpha"]
    warmup_ep    = cfg["training"]["warmup_epochs"]

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)

        use_mix = (np.random.rand() < mix_prob) and (epoch >= warmup_ep)
        if use_mix:
            if np.random.rand() < 0.5:
                images, ya, yb, lam = mixup_data(images, targets, mixup_alpha)
            else:
                images, ya, yb, lam = cutmix_data(images, targets, cutmix_alpha)

        outputs = model(images)
        loss    = (mixup_cutmix_criterion(criterion, outputs, ya, yb, lam)
                   if use_mix else criterion(outputs, targets))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        loss_meter.update(loss.item(), images.size(0))
        if not use_mix:
            _, pred = outputs.max(1)
            correct += pred.eq(targets).sum().item()
            total   += targets.size(0)

    acc = 100.0 * correct / total if total > 0 else 0.0
    return loss_meter.avg, acc


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    loss_meter = AverageMeter()
    all_preds, all_targets = [], []

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        loss    = criterion(outputs, targets)
        loss_meter.update(loss.item(), images.size(0))
        _, pred = outputs.max(1)
        all_preds.extend(pred.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    macro_f1 = compute_macro_f1(all_preds, all_targets)
    acc = 100.0 * np.mean(np.array(all_preds) == np.array(all_targets))
    return loss_meter.avg, acc, macro_f1, all_preds, all_targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     type=str,  default="configs/default.yaml")
    parser.add_argument("--seed",       type=int,  default=None,
                        help="Override seed (used for 3-seed ensemble)")
    parser.add_argument("--epochs",     type=int,  default=None)
    parser.add_argument("--batch-size", type=int,  default=None)
    parser.add_argument("--lr",         type=float, default=None)
    parser.add_argument("--debug",      action="store_true")
    parser.add_argument("--resume",     type=str,  default=None)
    parser.add_argument("--gpu",        type=int,  default=0)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.seed       is not None: cfg["seed"]                    = args.seed
    if args.epochs     is not None: cfg["training"]["epochs"]      = args.epochs
    if args.batch_size is not None: cfg["training"]["batch_size"]  = args.batch_size
    if args.lr         is not None: cfg["training"]["lr"]          = args.lr

    if args.debug:
        cfg["training"]["epochs"]    = 2
        cfg["training"]["batch_size"] = 32
        cfg["training"]["swa_start"] = 9999
        cfg["data"]["num_workers"]   = 0
        print("=" * 60)
        print("  DEBUG MODE — 2 epochs, no SWA")
        print("=" * 60)

    seed = cfg["seed"]
    seed_everything(seed)
    device = (torch.device(f"cuda:{args.gpu}")
              if torch.cuda.is_available() else torch.device("cpu"))

    total_epochs = cfg["training"]["epochs"]
    warmup_ep    = cfg["training"]["warmup_epochs"]
    swa_start    = cfg["training"]["swa_start"]

    print(f"\n{'='*60}")
    print(f"  ShiftGuard10 — WideResNet-28-10 Training")
    print(f"  Seed: {seed} | Device: {device}")
    print(f"  Epochs: {total_epochs} | Batch: {cfg['training']['batch_size']}")
    print(f"  SWA from epoch {swa_start}")
    print(f"{'='*60}\n")

    # ── Data ─────────────────────────────────────────────────────────────────
    root = cfg["data"]["root"]
    train_ds = ShiftGuard10Dataset(root, "train", get_train_transforms(),
                                   val_ratio=cfg["data"]["val_ratio"], seed=seed)
    val_ds   = ShiftGuard10Dataset(root, "val",   get_val_transforms(),
                                   val_ratio=cfg["data"]["val_ratio"], seed=seed)

    sampler = train_ds.get_sampler() if (cfg["training"]["use_balanced_sampler"]
                                         and not args.debug) else None

    if args.debug:
        from torch.utils.data import Subset
        train_ds = Subset(train_ds, range(min(200, len(train_ds))))
        val_ds   = Subset(val_ds,   range(min(50,  len(val_ds))))

    nw = cfg["data"]["num_workers"]
    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"],
                              shuffle=(sampler is None), sampler=sampler,
                              num_workers=nw, pin_memory=cfg["data"]["pin_memory"],
                              drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"] * 2,
                              shuffle=False, num_workers=nw,
                              pin_memory=cfg["data"]["pin_memory"])

    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ── Model / Loss / Optimiser ──────────────────────────────────────────────
    model = build_model(cfg["model"]["num_classes"], cfg, device)

    full_train = ShiftGuard10Dataset(root, "train",
                                     val_ratio=cfg["data"]["val_ratio"], seed=seed)
    cls_counts = full_train.get_class_counts()
    criterion  = BalancedSoftmaxLoss(cls_counts).to(device)
    print(f"  Loss: Balanced Softmax (class counts: min={cls_counts.min()}, max={cls_counts.max()})")

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg["training"]["lr"],
        momentum=cfg["training"]["momentum"],
        weight_decay=cfg["training"]["weight_decay"],
        nesterov=True,
    )

    def lr_lambda(ep):
        if ep < warmup_ep:
            return (ep + 1) / warmup_ep
        prog = (ep - warmup_ep) / (total_epochs - warmup_ep)
        return 0.5 * (1 + np.cos(np.pi * prog))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    use_swa = swa_start < total_epochs
    if use_swa:
        swa_model     = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=cfg["training"]["swa_lr"])
        print(f"  SWA enabled from epoch {swa_start}")

    start_epoch = 0
    best_f1     = 0.0
    if args.resume:
        start_epoch, best_f1 = load_checkpoint(args.resume, model, optimizer)
        print(f"  Resumed from epoch {start_epoch}, best F1: {best_f1:.4f}")

    # ── Training Loop ─────────────────────────────────────────────────────────
    ckpt_dir   = cfg["output"]["checkpoint_dir"]
    model_tag  = f"wrn_s{seed}"

    for epoch in range(start_epoch, total_epochs):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, cfg, epoch)

        if use_swa and epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

        v_loss, v_acc, v_f1, preds, tgts = validate(
            model, val_loader, criterion, device)

        lr  = optimizer.param_groups[0]["lr"]
        tag = " [SWA]" if (use_swa and epoch >= swa_start) else ""
        print(f"  Ep {epoch+1:3d}/{total_epochs} | "
              f"Tr {tr_loss:.4f}/{tr_acc:.1f}% | "
              f"Val {v_loss:.4f}/{v_acc:.1f}% F1:{v_f1:.4f} | "
              f"LR:{lr:.5f} | {time.time()-t0:.1f}s{tag}")

        if (epoch + 1) % 25 == 0 or (epoch + 1) == total_epochs:
            print(get_classification_report(preds, tgts, CLASS_NAMES))

        if v_f1 > best_f1:
            best_f1 = v_f1
            save_checkpoint({
                "epoch": epoch + 1, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_f1": best_f1, "config": cfg, "seed": seed,
            }, os.path.join(ckpt_dir, f"best_{model_tag}.pth"))

        if (epoch + 1) % 50 == 0:
            save_checkpoint({
                "epoch": epoch + 1, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_f1": best_f1, "config": cfg, "seed": seed,
            }, os.path.join(ckpt_dir, f"{model_tag}_ep{epoch+1}.pth"))

    # ── SWA BN Update ─────────────────────────────────────────────────────────
    if use_swa:
        print("\n  Updating SWA batch normalization statistics...")
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
        _, _, swa_f1, _, _ = validate(swa_model, val_loader, criterion, device)
        print(f"  SWA Val F1: {swa_f1:.4f}")
        save_checkpoint({
            "epoch": total_epochs,
            "model_state_dict": swa_model.module.state_dict(),
            "best_f1": swa_f1, "config": cfg, "seed": seed,
        }, os.path.join(ckpt_dir, f"swa_{model_tag}.pth"))
        if swa_f1 > best_f1:
            best_f1 = swa_f1
            save_checkpoint({
                "epoch": total_epochs,
                "model_state_dict": swa_model.module.state_dict(),
                "best_f1": best_f1, "config": cfg, "seed": seed,
            }, os.path.join(ckpt_dir, f"best_{model_tag}.pth"))

    print(f"\n{'='*60}")
    print(f"  Done. Best Macro F1: {best_f1:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
