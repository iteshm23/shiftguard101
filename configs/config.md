# Configuration

All values below are read from `notebook.py` (final commit `9b1bac1` of `Xavaitron/Shiftguard10`)
unless marked otherwise. "Training version" means commit `18a6d54` (25 Mar 2026), the version
that trained the three submitted models (`history/notebook_v3_training_18a6d54.py`).

## Command-line arguments (`main()`)

| argument | final default | training version | v2 (`68c54a8`) |
|---|---|---|---|
| `--epochs` | 450 | 450 | 300 |
| `--batch-size` | 512 | **128** | 128 |
| `--lr` | 0.1 | 0.1 | 0.1 |
| `--wd` | 5e-4 | 5e-4 | 5e-4 |
| `--val-ratio` | 0.05 | 0.05 | 0.1 (hard-coded) |
| `--swa-start` | 360 | 360 | 250 |
| `--swa-lr` | 0.005 | 0.005 | 0.005 |
| `--tta` | 30 (plus 1 clean view) | 30 (plus 1) | 20 (plus 1) |
| `--seeds` | 42 137 7 | 42 137 7 | 42 (single `--seed`) |
| `--mix-prob` | 0.5 | 0.5 | 0.5 |
| `--label-smoothing` | 0.1 | 0.1 | 0.1 |
| `--gpu` | 0 | 0 | 0 |
| `--data-root` | Kaggle competition input path | None (searches a list) | None |
| `--checkpoint-dir` | Kaggle Model `xavaitron/wrn-3seed` (read-only) | n/a (writes `checkpoints/`) | n/a |
| `--inference-only` | flag | n/a | n/a |
| `--debug` | 2 epochs, batch 32, no SWA, 2 TTA views, seed 42 | same idea | same idea |

The training command line itself is not recorded, so batch 128 is the default of the training
version rather than a logged value.

## Fixed in the code

| setting | value |
|---|---|
| sampler | `WeightedRandomSampler`, weight 1/sqrt(class count), 27,930 draws, replacement (v2: 1/count) |
| split seed | 42, class-wise, same for all models |
| model | WRN-28-10, dropout 0.3, Kaiming normal (fan_out, ReLU) for convs |
| optimiser | SGD, momentum 0.9, Nesterov |
| gradient clipping | max norm 5.0 |
| warmup | 5 epochs linear, then cosine (LambdaLR) |
| SWALR | anneal 10 epochs, cosine (PyTorch defaults) |
| SWA finish | `update_bn` on the training loader; SWA weights kept only if validation macro F1 beats the best epoch |
| MixUp / CutMix | alpha 1.0; MixUp or CutMix with equal chance when a batch is mixed |
| Cutout | 16 px, after Normalize |
| AutoAugment | CIFAR10 policy |
| random crop | padding 4, fill 128 |
| TTA view | RandomCrop(32, 4, fill 128), flip, ColorJitter(0.1, 0.1, 0.1), RandomRotation(10, fill 128) |
| inference | batch 512, softmax averaged over 1 + 30 views, then over models with equal weight |
| DataLoader | 4 workers, pin_memory, drop_last for training; validation batch = 2 x batch size |
