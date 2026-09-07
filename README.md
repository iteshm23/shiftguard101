# ShiftGuard10 — Robust Image Classification

**EE708 Course Project | IIT Kanpur**

10-class image classification under distribution shift (CIFAR-10 label set, custom test distribution).
Achieved **1st place (Macro F1 = 0.9488)** among 30 teams.

---

## Architecture Overview

```
Input (32×32 RGB)
       │
       ▼  AutoAugment (CIFAR10 policy)
       │  + Random Crop + Flip + Cutout(16×16)
       │
       ▼  WideResNet-28-10 (36.5M params)
       │  3 groups × 4 pre-activation residual blocks
       │  [16 → 160 → 320 → 640 channels]
       │  Global Average Pooling → Linear(640, 10)
       │
       ▼  Balanced Softmax Loss
       │  + MixUp / CutMix (50% prob each, alternating)
       │  + Nesterov SGD + Cosine LR Warmup
       │  + SWA (epoch 225 → 300)
       │
       ▼  3-Seed Ensemble (seeds 42, 7, 13)
       │  31-view Test Time Augmentation
       │
       ▼  submission.csv  →  Macro F1 = 0.9488  (Rank 1 / 30)
```

---

## Key Techniques

### WideResNet-28-10
Standard robust baseline for small-image classification. 28-layer depth with widen factor 10 gives
~36.5M parameters — enough capacity to generalise across distribution shift without pretraining.
Pre-activation batch norm → ReLU → Conv blocks with 0.3 dropout inside each residual block.

### Balanced Softmax Loss
Counters class imbalance by adjusting the softmax denominator:

```
p(y=c | x) = n_c · exp(f_c) / Σ_j n_j · exp(f_j)
```

Equivalent to adding `log(n_c)` to class-c logit before standard cross-entropy.
This shifts decision boundaries without requiring a separate reweighting phase — unlike LDAM,
the adjustment applies uniformly from the first epoch.

### AutoAugment (CIFAR10 Policy)
Learned augmentation policy from neural architecture search: 25 sub-policies, each a sequence of
two operations drawn from {Rotate, ShearX/Y, Translate, Equalize, AutoContrast, Posterize, Solarize,
Color, Contrast, Brightness, Sharpness, Cutout, FlipLR} with learned magnitude and probability.
Applied first in the training pipeline before geometric crops.

### MixUp + CutMix
Each batch has 50% chance of applying one of:
- **MixUp**: `x̃ = λx_i + (1-λ)x_j`, labels mixed with same λ
- **CutMix**: paste a rectangular region from x_j into x_i, λ adjusted by area ratio

Both are applied with `alpha=1.0` (Beta distribution). Disabled during the SWA phase to allow
stable decision-boundary consolidation.

### 3-Seed Ensemble
Three independent WRN-28-10 models trained with seeds {42, 7, 13}. Final prediction is the
arithmetic mean of per-class softmax probabilities across all three checkpoints.
Ensemble reduces variance from random weight initialisation and mini-batch ordering.

### 31-View Test Time Augmentation (TTA)
At inference: 1 clean pass (no augmentation) + 30 stochastic passes (RandomCrop + HorizontalFlip +
RandomRotation(15°) + ColorJitter). The 31 probability vectors are averaged before argmax.
TTA simulates the distribution shift between train and test images.

### Cosine LR Warmup + Nesterov SGD
Linear warmup over 5 epochs (lr: 0 → 0.1), then cosine annealing to near-zero by epoch 300.
Nesterov momentum (0.9) with weight decay 5e-4. Gradient clipping (max_norm=5.0) prevents
early-epoch instability from the Balanced Softmax adjustment.

### Stochastic Weight Averaging (SWA)
From epoch 225, model weights are averaged with a cyclic SWA learning rate (0.005).
SWA approximates the centroid of the flat loss basin and consistently improves generalisation
by 0.5–1.5% over the best single checkpoint.

---

## Setup

### Install Dependencies

```bash
pip install -r requirements.txt
```

For GPU training with CUDA 11.8:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Dataset Structure

Place the competition dataset so the folder looks like:
```
shiftguard10/
├── shift-guard-10-robust-image-classification-challenge/
│   ├── train_labels.csv
│   ├── sample_submission.csv
│   ├── train_images/   (29,400 PNGs, 32×32)
│   └── test_images/    (7,600 PNGs, 32×32)
├── src/
├── configs/
└── README.md
```

---

## Training

### Single Model (one seed)

```bash
python src/train.py --seed 42
```

### Full 3-Seed Ensemble

```bash
bash scripts/ensemble_train.sh
```

This trains three WRN-28-10 models sequentially (seeds 42, 7, 13).
Checkpoints saved to `checkpoints/best_wrn_s{seed}.pth`.

### Debug / Sanity Check (CPU)

```bash
python src/train.py --debug
```

---

## Inference

### 3-Seed Ensemble + 31-View TTA (recommended)

```bash
bash scripts/ensemble_infer.sh
```

### Single Checkpoint, No TTA

```bash
python src/inference.py --checkpoint checkpoints/best_wrn_s42.pth --tta 0
```

### Custom TTA Views

```bash
python src/inference.py --checkpoint checkpoints/best_wrn_s42.pth --tta 10
```

Output: `submission.csv` (7,600 rows, `id,label` format).

---

## Project Structure

```
shiftguard10/
├── src/
│   ├── dataset.py          AutoAugment pipeline, TTA transforms, data loader
│   ├── models/
│   │   └── wideresnet.py   WideResNet-28-10 implementation
│   ├── loss.py             Balanced Softmax Loss + LDAM Loss
│   ├── train.py            Training loop (MixUp/CutMix, SWA, cosine LR)
│   ├── inference.py        Ensemble inference with 31-view TTA
│   └── utils.py            Metrics, checkpointing, MixUp/CutMix helpers
├── configs/
│   └── default.yaml        All hyperparameters
├── scripts/
│   ├── ensemble_train.sh   Train all 3 seeds
│   └── ensemble_infer.sh   Ensemble inference with TTA
├── notebook/
│   └── ShiftGuard10_Walkthrough.ipynb
├── deck/
│   └── presentation.html   Slide deck
├── results/
│   └── metrics.txt         Final competition results
├── INTERVIEW_PREP.md
└── requirements.txt
```

---

## Results

| Metric | Value |
|---|---|
| Competition rank | **1st / 30 teams** |
| Test Macro F1 | **0.9488** |
| Val Macro F1 | ~0.950 |
| Model | WideResNet-28-10 × 3 seeds |
| TTA views | 31 |
| Training epochs | 300 per seed |
| Parameters | ~36.5M per model |

---

## Ablation Highlights

| Technique | Val F1 | Δ |
|---|---|---|
| Baseline WRN (cross-entropy) | 0.912 | — |
| + Balanced Softmax | 0.924 | +1.2% |
| + AutoAugment | 0.934 | +1.0% |
| + MixUp/CutMix | 0.941 | +0.7% |
| + SWA | 0.947 | +0.6% |
| + 3-seed ensemble | 0.950 | +0.3% |
| + 31-view TTA | **0.950** | +0.0% val / +TTA gain on test |
