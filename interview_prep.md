# ShiftGuard10 Interview Preparation

One file to revise the whole project. The source of truth for the implementation is the
complete repository `Xavaitron/Shiftguard10` (final commit `9b1bac1`, 29 Mar 2026), which is
`notebook.py` in this repo, plus its git history. Dataset numbers come from the competition
files. Rules come from the competition page. Public claims come from the resume.

Labels used throughout:

- **Verified**: read in `notebook.py`, its git history, the dataset or the competition page, or computed by running the code.
- **Inference**: reasoned from evidence, not proven. Say "I believe" or "the evidence suggests".
- **Unknown**: not in any file. Do not state these as facts.

What changed after reading the complete repository is listed in Appendix A. The short
version: the TTA uses 1 clean view plus 30 random views, `update_bn` is called, each seed's
final weights are whichever of "best epoch" and "SWA average" scored higher on validation,
the seeds are 42, 137 and 7, the models were trained with the 25 March version of the script
(batch size 128), and the test set is very probably not balanced but tilted towards the
classes that are rare in training.

## How to use this file

| time you have | read |
|---|---|
| 10 minutes before the interview | 30 (cheat sheet), 29 (what to say), 22 (resume defense) |
| 1 hour | 1, 3, 5, 6, then 9, 10, 13, 15, then 23 |
| a full revision | everything in order, then answer 23 to 26 out loud without looking |

Sections 7 to 18 explain each technique. Sections 23 to 26 are the questions, each with a full answer.

## Contents

1. Project Overview
2. What the Competition Asked
3. Dataset
4. Evaluation Metric
5. Exact Approach
6. Complete Pipeline
7. Model Architecture
8. Data Processing
9. Class Imbalance
10. Loss Function
11. Augmentation
12. Optimization
13. SWA
14. Ensemble
15. TTA
16. Validation
17. Distribution Shift
18. Complete Code Explanation
19. Why Each Decision Was Made
20. Why Not Alternative Approaches
21. Limitations
22. Resume Bullet Defense
23. Likely Interview Questions
24. Hard Technical Questions
25. Cross Questions
26. Debugging Questions
27. ML Concepts Connected to the Project
28. AmEx Transfer Questions
29. What I Should Say
30. Final Cheat Sheet

Appendix A: What the complete repository changed.
Appendix B: Project timeline from the git history.
Appendix C: Source-of-truth audit.

How each technical topic is written (sections 4 and 7 to 18): **WHAT** it is, **WHY** it
is here (the problem it solves), **HOW** it works (including the exact code), **WHY THIS**
instead of the alternatives, **TRADEOFF**, **LIMITATION** (when it fails), **INTERVIEW**
(the question and a good answer) and **CROSS QUESTION** (the next "why" and the answer).

---

## 1. Project Overview

**The competition.** A Kaggle competition run for the EE708 course at IIT Kanpur (22 Jan to
11 Apr 2026). Classify 32x32 colour images into 10 classes. Scored by macro F1 on a hidden
test set that was deliberately shifted away from the training data. No pretrained weights, no
outside data, notebook-based submission.

**The team and the code.** The competition was ranked by team. The final code is in the
GitHub repo `Xavaitron/Shiftguard10`; every commit there is by the account Xavaitron (commit
author name Pratyush Singh). Be ready to say exactly which parts you did yourself. The
`ee708-img` Kaggle notebooks in your Downloads (a small CNN and a WideResNet run) look like
your own experiments (these notes assume so; correct it if not); the final pipeline's
commits are not under your name.

**The data (verified).** 29,400 training images, 7,600 test images, all 32x32 RGB. Training
classes: 5,000 each of airplane, automobile, bird, cat; 4,000 each of deer, dog; 500 frog, 500
horse, 300 ship, **100 truck** (50 to 1).

**The shift (measured on the data).**

- 5% of images in both sets have a grey 6x6 (train) or 10x10 (test) square of RGB (125, 123, 114).
- About 21% of training images and about 75% of test images have colour noise; 38% of test
  images carry a strong level that almost never appears in training.
- The test class mix is very probably reversed: every strong submission in the repo history
  predicts about 510 to 550 images for each of the four biggest training classes and about 910
  to 980 for each of the four rarest. A balanced test set would cap those submissions below
  their own scores (section 3.4).

**The final method (verified from `notebook.py`).**

| part | exact choice |
|---|---|
| split | class-wise 95/5, split seed 42, same for every model: 27,930 train / 1,470 validation |
| model | WideResNet-28-10 from scratch, 36,479,194 parameters, dropout 0.3, Kaiming normal (fan_out) |
| imbalance | `WeightedRandomSampler` with weight 1/sqrt(class count), plus Balanced Softmax loss |
| loss | Balanced Softmax with label smoothing 0.1 |
| augmentation | RandomCrop(32, padding 4, fill 128), horizontal flip, AutoAugment (CIFAR10 policy), Cutout 16 |
| batch mixing | with probability 0.5 per batch; then MixUp or CutMix with equal chance; alpha 1.0 |
| optimiser | SGD, lr 0.1, momentum 0.9, Nesterov, weight decay 5e-4, gradient norm clipped at 5.0 |
| schedule | 5 warmup epochs, cosine decay, 450 epochs; SWA from epoch 360 with SWALR to 0.005 |
| batch size | 128 in the version that trained the models (the final file's default is 512, set later) |
| per-seed weights | best validation epoch or the SWA average (after `update_bn`), whichever has the higher validation macro F1 |
| ensemble | seeds 42, 137 and 7; softmax probabilities averaged with equal weight |
| TTA | per model: 1 clean pass + 30 random passes (crop, flip, colour jitter, rotation up to 10 degrees), averaged |
| inference cost | 3 models x 31 passes = 93 forward passes per test image |

**Results.**

| | value | source |
|---|---|---|
| v2 (single WRN, full inverse-frequency sampler, 300 epochs, 20+1 TTA) | 0.9348 | written in the v3 code header ("v2 (which scored 0.9348)") |
| v3 final (this method) | 0.9488, 1st of 27 teams | resume; not in any file |
| your first small CNN | 0.4812 validation macro F1 | your own notebook output |

**Five things to understand cold.**

1. Why macro F1 with a 50:1 imbalance, and a test set tilted the other way, makes the rare classes decide the score.
2. Balanced Softmax: train on `logits + log(prior)`, predict on raw logits.
3. How the sampler and Balanced Softmax overlap, and why that overlap probably helped here.
4. What SWA, the ensemble and TTA each average (weights over time, models over seeds, predictions over views), and how the code chooses between the SWA and best-epoch weights.
5. What the shift actually is (label mix, occlusion size, noise) and which part of the pipeline targets each.

**Biggest honest weaknesses.**

- No controlled ablations. The only before/after number is v2 0.9348 to v3, with six things changed at once.
- Validation has 5 trucks and 15 ships, and it is not shifted like the test set.
- No augmentation adds pixel noise, the most common test perturbation.
- The models were trained off Kaggle ("server train + Kaggle inference workflow" in the commit
  message), and the final Kaggle notebook only ran inference from those self-trained checkpoints.
- Several small code issues, including a crash in the documented `--debug` run (verified).

---

## 2. What the Competition Asked

**Task (verified, competition page).** Predict one of airplane, automobile, bird, cat, deer,
dog, frog, horse, ship, truck for each of 7,600 test images. Submit a CSV with columns `id`
and `label`.

**Metric.** Macro F1 over the 10 classes (section 4).

**What makes it non-trivial (page wording).** The test set "contains challenging samples and
distribution shift relative to training"; test images "may include subtle perturbations and
partial occlusions". The page does not say what the perturbations are or how the test classes
are distributed.

**Rules (page wording, paraphrased).**

1. Competition data only, no redistribution.
2. No pretrained models: no ImageNet weights, no self-supervised checkpoints, no external checkpoints.
3. No external datasets or data of any kind.
4. Notebook submission: the Kaggle notebook must train and run inference, reproducible from
   scratch using only competition data and standard packages.

**Leaderboards.** Public leaderboard on part of the test set during the competition; private
leaderboard on the rest for the final ranking.

**How the final submission was produced (verified from the git history).** The three models
were trained with the 25 March version of the script. A commit on 28 March is titled "Add
--inference-only and --checkpoint-dir for server train + Kaggle inference workflow", and the
final script's default checkpoint folder is a Kaggle Model input
(`/kaggle/input/models/xavaitron/wrn-3seed/pytorch/default/1`). When the checkpoint files are
present, `main()` skips training and only runs inference. So the final Kaggle notebook loaded
self-trained checkpoints (trained with this same script, no pretrained weights) and generated
the predictions. The notebook header documents this and says the script trains from scratch
if the checkpoints are removed.

How to say it in an interview: "The models were trained with our own script from random
initialisation, outside the Kaggle session limit, and the Kaggle notebook loaded those
checkpoints and ran inference. No pretrained weights or outside data were used anywhere."
Do not say training ran inside the Kaggle notebook.

**Why clean accuracy is not enough here.**

1. The metric gives the four rare training classes 40% of the score.
2. The test class mix is very probably tilted towards those same rare classes (section 3.4).
3. Test images carry bigger occlusions and more and stronger noise than training images.

**Why it is hard even with 10 classes.** Look-alike pairs at 32x32 (cat/dog,
automobile/truck, deer/horse, airplane/bird, airplane/ship), rare classes sitting next to big
look-alikes (100 trucks vs 5,000 automobiles), almost nothing to learn "truck" from, and no
pretrained features.

---

## 3. Dataset

### 3.1 Files (verified)

| file | content |
|---|---|
| `train_images/` | 29,400 PNG, 32x32 RGB |
| `train_labels.csv` | `id,label` |
| `test_images/` | 7,600 PNG, 32x32 RGB |
| `sample_submission.csv` | 7,600 ids, every label `airplane` (template only) |
| `classes.txt` | the 10 names |

Ids are 6-digit strings. Train and test ids both start at `000001`; the code pads with
`zfill(6)` and reads from the right folder. No exact duplicate images inside or across the sets.

### 3.2 Training classes (verified)

| class | images | share |
|---|---|---|
| airplane, automobile, bird, cat | 5,000 each | 17.0% each |
| deer, dog | 4,000 each | 13.6% each |
| frog, horse | 500 each | 1.7% each |
| ship | 300 | 1.0% |
| truck | 100 | 0.34% |

Head classes (the six big ones) hold 95.2% of the data; tail classes (frog, horse, ship,
truck) hold 4.8%.

### 3.3 Perturbations (measured with `scripts/analyze_dataset.py`)

| | train | test |
|---|---|---|
| grey patch RGB (125, 123, 114) | 6x6 on 1,470 images (5.00%) | 10x10 on 380 images (5.00%) |
| clean (no added colour noise) | 79.1% | 24.8% |
| mild colour noise | 20.0% | 36.8% |
| strong colour noise | 0.9% | 38.4% |
| mean and std of R, G, B | 0.4925, 0.4832, 0.4474 / 0.2467, 0.2428, 0.2614 | 0.4940, 0.4841, 0.4476 / 0.2454, 0.2413, 0.2596 |

The grey is the CIFAR-10 mean colour. Noise was measured with a high-pass noise estimate on
the colour-difference channels (R-G, B-G), where natural texture mostly cancels and added
per-pixel noise does not. Colour statistics do not shift, and they match the CIFAR-10
constants the code uses (within about 0.003). Blur is not claimed: some images look soft in
both sets but a deliberate blur subset could not be separated.

### 3.4 The test class mix (inference, with strong evidence)

Test labels are hidden. But the repo history has three submissions from strong models:

| class | v2 (24 Mar) | v3 2-seed (25 Mar) | v3 3-seed (27 Mar, the repo's `submission.csv`) |
|---|---|---|---|
| airplane | 534 | 524 | 520 |
| automobile | 553 | 553 | 537 |
| bird | 525 | 519 | 510 |
| cat | 549 | 570 | 554 |
| deer | 814 | 801 | 805 |
| dog | 876 | 812 | 818 |
| frog | 946 | 962 | 972 |
| horse | 913 | 937 | 932 |
| ship | 982 | 983 | 979 |
| truck | 908 | 939 | 973 |

The 2-seed and 3-seed files agree on 98.0% of test images; v2 and the 3-seed file agree on 96.2%.

The argument:

- A model's predicted count for a class can't be far from the true count if its macro F1 is
  high. For each class, F1 is at most 2 x min(predicted, true) / (predicted + true).
- If the test set were balanced (760 per class), the 3-seed file could score at most 0.875
  and the v2 file at most 0.880. v2 scored 0.9348 (stated in the code), and the final scored
  0.9488 (resume). So the test set cannot be balanced, if these files are the scored ones
  (the v2 file was committed right after the v2 code; the 3-seed file is labelled "3 ensemble
  results").
- If the test set mirrored training (50 to 1), the cap would be about 0.48. Impossible.
- All three strong models agree on the shape: about 500 for each of the four biggest training
  classes, about 800 for deer and dog, about 950 for the four rarest. One design that fits and
  sums to exactly 7,600 is 500 x 4 + 800 x 2 + 1,000 x 4. That exact design is a guess.

So the shift includes a **reversed label shift**: truck is 0.34% of training but probably
around 13% of test. The four rarest training classes may make up about half the test set.

Caveat to say out loud: these counts come from our own predictions, so they can't also be used
to test whether the model over-predicts the tail. They do rule out a balanced test set.

Your earlier notebooks fit this picture. A WRN trained with no rebalancing predicted only 447
trucks; the March CCT/LDAM-era submission predicted 224 trucks and 1,220 deer.

---

## 4. Evaluation Metric

**WHAT.** Macro F1: for each class, F1 = 2PR / (P + R) with precision P = TP / (TP + FP) and
recall R = TP / (TP + FN); then the plain average over the 10 classes. Equivalent form:
F1 = 2TP / (2TP + FP + FN).

**WHY.** It is the leaderboard metric, and it decides what "good" means: each class is 10% of
the score no matter how many images it has.

**HOW (in the code).** `f1_score(targets, preds, average="macro", zero_division=0)` on the
validation set every epoch; used to keep the best epoch and to choose between the best epoch and
the SWA weights. `zero_division=0` means a class that is never predicted gets precision 0 and
F1 0 (no warning), so a dead class costs a full 0.1.

**WHY THIS (why not accuracy).** On a training-like mix, a model that is perfect except it calls
every truck an automobile scores 99.66% accuracy but 0.899 macro F1. On a set with 1,000 images
per class and 100 trucks, a model that never predicts truck gets 98.9% accuracy and 0.895 macro
F1, while a model with 3% errors everywhere gets 97.0% accuracy and 0.960 macro F1.

Micro F1 equals accuracy in single-label multiclass problems. Weighted F1 weights classes by
size, so big classes dominate again.

**TRADEOFF.** Fair to every class, but very noisy for small classes and not differentiable.

**LIMITATION.** With 5 trucks in validation, one truck mistake moves validation macro F1 by about
0.011 (accuracy moves by 0.0007). In a simple simulation with 5% random errors, validation macro
F1 had a standard deviation of about 0.015 from sampling alone.

**INTERVIEW.** "Why wasn't accuracy enough?" "Because four classes are under 2% of training data
each, and macro F1 gives each of them 10% of the score. You can ignore them and still have high
accuracy. Here it mattered even more, because the test set appears to have many more of those
classes than training did."

**CROSS QUESTION.** "Then why train with cross-entropy, not F1?" "F1 is built from argmax counts
over a whole dataset, so it has no useful gradient and can't be computed well on a batch of 128
with two or three trucks in it. Cross-entropy is smooth and per-example, and when minimised it
gives good probability estimates. Balanced Softmax then makes the argmax act as if the classes
were balanced. We didn't tune per-class thresholds; with 5 validation trucks they would overfit."

---

## 5. Exact Approach

Line numbers refer to `notebook.py`.

| component | exact setting | where |
|---|---|---|
| data root | `/kaggle/input/competitions/shift-guard-10-robust-image-classification-challenge` (default) | `--data-root`, L605 |
| split | per class `max(1, int(n * 0.05))` to validation, `RandomState(42)`, same for all seeds | L147-177, L395-398 |
| sizes | 27,930 train / 1,470 val (val: 250 x 4, 200 x 2, 25, 25, 15, 5) | computed |
| train transform | RandomCrop(32, padding=4, fill=128), RandomHorizontalFlip, AutoAugment(CIFAR10), ToTensor, Normalize(CIFAR mean/std), Cutout(16) | L111-120 |
| val/test transform | ToTensor, Normalize | L123-127 |
| TTA transform | RandomCrop(32, 4, fill 128), RandomHorizontalFlip, ColorJitter(0.1, 0.1, 0.1), RandomRotation(10, fill 128), ToTensor, Normalize | L130-139 |
| sampler | `WeightedRandomSampler(1/sqrt(n_c) per image, num_samples=27,930, replacement=True)` | L205-212 |
| loaders | train: batch `args.batch_size`, 4 workers, pin_memory, drop_last; val: batch x2 | L405-414 |
| model | WRN-28-10, dropout 0.3, 36,479,194 params | L219-273, L416 |
| init | Kaiming normal, fan_out, ReLU for convs; BN weight 1, bias 0; Linear default | L259-264 |
| loss | Balanced Softmax, class counts of the 27,930-image train split, label smoothing 0.1 | L280-291, L677-678 |
| mixing | `mix_prob` 0.5 per batch, then MixUp or CutMix 50/50, alpha 1.0 | L325-344 |
| optimiser | SGD lr 0.1, momentum 0.9, Nesterov, weight decay 5e-4 | L421-424, L608-609 |
| clipping | `clip_grad_norm_(max_norm=5.0)` every step | L348 |
| LR schedule | LambdaLR: (e+1)/5 for e < 5, then 0.5(1 + cos(pi (e-5)/445)) | L426-432 |
| epochs | 450 | L606 |
| SWA | `AveragedModel`, `SWALR(swa_lr=0.005)` (anneal 10 epochs, cosine; defaults), update every epoch from 360: 90 snapshots | L434-438, L472-474 |
| SWA finish | `update_bn(train_loader, swa_model)`, validate, use SWA weights only if val F1 beats the best epoch | L509-518 |
| per-epoch checkpoint | full state, `swa_state` only once SWA has started, `completed: False` | L495-507 |
| final checkpoint | same keys, `completed: True` | L522-535 |
| seeds | 42, 137, 7 | L616 |
| batch size | 128 when the models were trained (commit `18a6d54`); 512 default in the final file | history, L607 |
| inference | per model: clean pass + 30 TTA passes, softmax averaged over 31; then average over 3 models | L544-576, L707-724 |
| inference batch | 512 (hard-coded) | L715 |
| output | `submission.csv`, `id,label`, ids in `sample_submission.csv` order | L579-594 |

---

## 6. Complete Pipeline

### 6.1 Training one seed (`train_single_seed`)

```
seed_everything(seed)                                   # seeds 42, 137, 7
train_ds = 27,930 images, val_ds = 1,470 images         # split seed fixed at 42
sampler: weight 1/sqrt(n_c), 27,930 draws per epoch with replacement
model = WRN-28-10 (Kaiming init), criterion = BalancedSoftmax(train counts, smoothing 0.1)
optimizer = SGD(lr 0.1, momentum 0.9, Nesterov, wd 5e-4)
scheduler = warmup 5 + cosine;  swa_model = AveragedModel(model);  swa_scheduler = SWALR(0.005)

for epoch in 0..449:
    for each batch (augmented: crop, flip, AutoAugment, normalise, Cutout):
        with prob 0.5: MixUp or CutMix (50/50) the batch, mixed loss
        else: plain loss
        backward, clip grad norm to 5.0, SGD step
    if epoch >= 360: swa_model.update_parameters(model); swa_scheduler.step()
    else:            scheduler.step()
    validate live model (raw logits, macro F1); keep best_state if improved
    save checkpoint (final code only)

update_bn(train_loader, swa_model)                      # recompute BN statistics for the average
validate swa_model; if its F1 > best F1: best_state = SWA weights
return best_state
```

### 6.2 Inference (`main`, final file)

```
for seed in 42, 137, 7:
    if wrn_seed{seed}.pth exists in checkpoint dir: load its best_state (skip training)
    else: train_single_seed(...)
for each model:
    probs = softmax(model(clean test images))                         # 1 pass
    for v in 1..30: probs += softmax(model(random TTA view of test))  # 30 passes
    model_probs = probs / 31
ensemble = mean of the 3 model_probs
label = argmax(ensemble) -> class name -> submission.csv
```

The final Kaggle run took the first branch for every seed, because the checkpoints were
attached as a Kaggle Model.

---

## 7. Model Architecture

### 7.1 Shapes and sizes (verified by building the model)

| stage | what happens | output for one image | parameters |
|---|---|---|---|
| input | normalised image | 3 x 32 x 32 | |
| conv1 | 3x3 conv, 3 -> 16, stride 1 | 16 x 32 x 32 | 432 |
| group1 | 4 blocks, 16 -> 160, stride 1 (first block has a 1x1 shortcut) | 160 x 32 x 32 | 1,640,672 |
| group2 | 4 blocks, 160 -> 320, first block stride 2 | 320 x 16 x 16 | 6,968,000 |
| group3 | 4 blocks, 320 -> 640, first block stride 2 | 640 x 8 x 8 | 27,862,400 |
| bn + ReLU | BatchNorm(640) | 640 x 8 x 8 | 1,280 |
| pool | global average over 8 x 8 | 640 | 0 |
| fc | Linear 640 -> 10 | 10 logits | 6,410 |
| total | | | **36,479,194** |

28 conv layers (25 of size 3x3, three 1x1 shortcuts), 25 BatchNorm layers, 12 dropout layers,
one linear layer. About 5.24 billion multiply-adds per image. 139 MB of float32 weights. The
theoretical receptive field of the last stage is about 109 pixels, more than the whole image.

### 7.2 WideResNet-28-10

**WHAT.** A residual CNN for 32x32 images (Zagoruyko and Komodakis, 2016). Depth 28 gives
n = (28 - 4) / 6 = 4 blocks per group; widen factor 10 makes the group widths 160, 320, 640
instead of 16, 32, 64.

**WHY.** The model has to learn everything from 29,400 small images with no pretrained weights.
WRN-28-10 is one of the most reliable from-scratch architectures for exactly this input size.

**HOW.** `WideResNet.__init__` builds a 3x3 stem conv, three groups with `_make_group`
(the first block of each group changes channels and, for groups 2 and 3, halves the size with
stride 2), a final BatchNorm, ReLU, global average pooling and a linear layer.

**WHY THIS.** The team tried other things first (git history): a Compact Convolutional
Transformer with LDAM loss in mid-March, then PyramidNet-272 with ShakeDrop plus WRN in the
first all-in-one script on 23 March, and the next day simplified to a single WRN-28-10 ("simplify:
single WRN-28-10 training run"). ResNet-50-style ImageNet models downsample a 32x32 image far too
early; ViTs need more data or pretraining; ResNet-18 is cheaper but usually a bit weaker on
CIFAR-style data when trained long.

**TRADEOFF.** Strong and stable, but heavy: 36.5M parameters, 5.24 GMACs per image, about 60
hours to train three models (header), and 93 forward passes per test image at inference.

**LIMITATION.** With 95 training trucks seen about 6 times per epoch for 450 epochs (around 2,750
times each), a model this size can memorise them. Augmentation and regularisation are what keep
those views different.

**INTERVIEW.** "Why WRN-28-10?" "It's built for 32x32 images: the first stage keeps the full
32x32 resolution, and it's one of the strongest architectures trained from scratch on
CIFAR-sized data. We did try a transformer-style model and PyramidNet before settling on it,
and the single WRN was simpler and scored 0.9348 on its own as v2."

**CROSS QUESTION.** "Is 36M parameters too many for 28K images?" "Parameter count alone isn't
a good measure of overfitting in CNNs. The model is heavily regularised (dropout, weight decay,
AutoAugment, Cutout, MixUp, CutMix, label smoothing), and validation was checked every epoch.
The real risk is the four rare classes, which is why the augmentation matters most there."

### 7.3 Residual blocks (pre-activation)

**WHAT.** Each block computes `out = conv2(relu(bn2(dropout(conv1(relu(bn1(x))))))) + shortcut(x)`.

**WHY.** Without shortcuts, deep stacks of convs are hard to train (gradients shrink, even
training error gets worse). The shortcut lets a block learn only a change on top of its input,
and gives gradients a direct path back.

**HOW.** `WRNBlock.forward`. The shortcut is the identity (`nn.Sequential()`), except in the
first block of each group, where channels change (and in groups 2 and 3 the size halves), so it
is a 1x1 conv with the same stride to make shapes match. "Pre-activation" means BN and ReLU come
before each conv, so the shortcut path carries x completely untouched.

**WHY THIS.** Pre-activation residual blocks are the WRN design and train most easily. A 1x1
projection only where needed keeps parameters low (only three 1x1 convs in the whole network).

**TRADEOFF.** Very little; the 1x1 projections add 258,560 parameters in total.

**LIMITATION.** Residual connections make training easy; they don't by themselves make the
model robust to shift.

**INTERVIEW.** "Why does the shortcut become a 1x1 convolution?" "Because x has 16 channels at
32x32 going into group 1, or 160 at 32x32 going into group 2 whose output is 320 at 16x16. You
can't add tensors of different shapes, so a 1x1 conv with stride 2 re-maps the channels and
downsamples."

**CROSS QUESTION.** "Why does adding x help gradients?" "The derivative of F(x) + x with respect
to x is dF/dx plus the identity, so even if dF/dx is small the gradient still flows back
unchanged through the identity term."

### 7.4 BatchNorm

**WHAT.** Per channel: subtract the batch mean, divide by the batch standard deviation, then
apply a learned scale and shift. In evaluation mode it uses running averages collected during
training.

**WHY.** Keeps activations in a stable range in a 28-layer network trained with a high learning
rate (0.1) and heavily augmented, mixed batches.

**HOW.** `bn1` and `bn2` in every block, plus a final `bn` before pooling; initialised to scale
1, shift 0 in `_init_weights`. `validate` calls `model.eval()` so BN uses running statistics.
For the SWA model, `update_bn(train_loader, swa_model)` recomputes the running statistics for
the averaged weights.

**WHY THIS.** Standard in WRN; makes high learning rates and deep stacks trainable.

**TRADEOFF.** Depends on batch statistics: small batches give noisy statistics (one reason for
`drop_last=True`), and running statistics come from augmented training images, not clean test
images.

**LIMITATION.** If test images are distributed differently (noise), the running statistics no
longer match exactly. Test-time BN adaptation exists for this but was not used.

**INTERVIEW.** "What happens if you forget `model.eval()`?" "BatchNorm would normalise each test
batch with its own statistics and dropout would stay on, so predictions would depend on what else
is in the batch and be noisier. The code calls `model.eval()` in both validation and inference."

**CROSS QUESTION.** "Why was `update_bn` needed for SWA?" "The averaged weights are a new network
that never ran forward during training. Its BN running statistics would otherwise be copied from
the last training snapshot, which doesn't match the averaged weights. `update_bn` does one pass
over the training loader to recompute them."

### 7.5 ReLU

**WHAT.** max(0, x).

**WHY.** The non-linearity; without it stacked convs collapse into one linear map.

**HOW.** `F.relu(..., inplace=True)` after each BN (in-place to save memory).

**WHY THIS.** Standard for ResNets, cheap, no saturation for positive inputs, and Kaiming init is
derived for it.

**TRADEOFF.** Zero gradient for negative inputs ("dead" units are possible but BN makes this rare).

**LIMITATION.** None that matters here.

**INTERVIEW.** "Why not GELU or SiLU?" "Those are common in transformers and EfficientNets; for a
WRN trained from scratch ReLU is the standard and the init is designed for it. There was no reason to change it."

**CROSS QUESTION.** "Why in-place?" "It overwrites the BN output instead of allocating a new tensor,
saving activation memory, which matters at batch 128 with 160-channel 32x32 maps."

### 7.6 Kaiming initialisation

**WHAT.** Conv weights drawn from a normal distribution with std sqrt(2 / fan), here
fan = fan_out = C_out x 3 x 3.

**WHY.** Training from scratch, 28 layers deep: if weights start too small or too large, the
signal shrinks or explodes layer by layer before training even starts.

**HOW.** `_init_weights`: `kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")` for every
conv; BN weight 1, bias 0. The Linear layer keeps PyTorch's default init. Measured std for a
320-channel conv: 0.0263 (expected 0.0264).

**WHY THIS.** The factor 2 compensates for ReLU zeroing half its inputs. Xavier init uses about
1/fan and is designed for tanh-like activations, so with ReLU the signal shrinks every layer.
fan_out keeps the backward (gradient) variance stable; it is what torchvision's ResNets use. In
most blocks fan_in equals fan_out anyway.

**TRADEOFF.** None really; with BN everywhere, init matters less than in a plain network.

**LIMITATION.** Only the start of training; it doesn't affect the final solution much.

**INTERVIEW.** "Why does init matter if BN normalises anyway?" "BN rescales activations, but the
first steps, the shortcut paths and the gradient sizes still depend on the weight scale. Kaiming
gives sensible scales from step one, which matters with lr 0.1."

**CROSS QUESTION.** "Derive the 2." "For y = Wx with x the output of a ReLU, Var(y) = fan x Var(w) x
E[x squared], and E[x squared] is half the variance of the pre-activation. To keep Var(y) equal to
that variance you need Var(w) = 2 / fan."

### 7.7 Dropout

**WHAT.** During training, zero 30% of the activations between the two convs of each block and
scale the rest by 1/0.7. Off in `model.eval()`.

**WHY.** 36M parameters and very few tail images: the model must not depend on any single
channel.

**HOW.** `nn.Dropout(p=0.3)` after `conv1` in every `WRNBlock` (12 layers).

**WHY THIS.** That is where the WRN paper put it, and 0.3 is the paper's CIFAR value. No sweep
in the files; it is a standard default, not a tuned number.

**TRADEOFF.** Extra regularisation slows fitting; interacts a little with BN statistics.

**LIMITATION.** Combined with everything else here (AutoAugment, Cutout, mixing, smoothing,
weight decay), too much regularisation could underfit; the long 450-epoch schedule compensates.

**INTERVIEW.** "Wouldn't BatchNorm already regularise?" "Only a little, through batch noise. Dropout
is an explicit regulariser. With 95 training trucks, we wanted both."

**CROSS QUESTION.** "Why inside the block and not before the classifier?" "The WRN paper found
dropout between the convs of a residual block works well; before the classifier it would act only
on the 640 pooled features."

---

## 8. Data Processing

**WHAT.** Read each PNG with PIL, `.convert("RGB")`, turn it into a float tensor in [0, 1]
(3 x 32 x 32), and normalise each channel with fixed CIFAR-10 constants: mean (0.4914, 0.4822,
0.4465), std (0.2470, 0.2435, 0.2616).

**WHY.** The network needs a fixed 3-channel float input with roughly zero mean and unit scale,
the same way every time.

**HOW.** `ShiftGuard10Dataset.__getitem__` opens `{id}.png` from `train_images/` or
`test_images/`. Ids are padded with `zfill(6)`. Test ids are read from `sample_submission.csv`,
so predictions come out in the template's order. Validation and the clean TTA pass use only
`ToTensor` + `Normalize`; training and the 30 TTA views add random transforms before them. Cutout
runs after `Normalize` (so its square becomes 0, the mean colour).

**WHY THIS.** CIFAR constants are standard and match this data within about 0.003 (section 3.3).
`.convert("RGB")` guards against grayscale or RGBA files (all files here are already RGB).

**TRADEOFF.** Fixed constants are simple and identical for train and test. Per-image
standardisation would remove brightness information and amplify noise in flat images.

**LIMITATION.** None material here.

**INTERVIEW.** "Why is validation preprocessing different from training?" "Training adds random
changes so the model learns invariance. Validation must measure the model on the real images,
identically every epoch, so it only gets tensor conversion and the same normalisation."

**CROSS QUESTION.** "Isn't using CIFAR-10 statistics external data?" "They're six published
constants, not images or weights. This dataset's own means and standard deviations agree with
them to within about 0.003, and the first conv is followed by BatchNorm, which re-centres
features anyway."

---

## 9. Class Imbalance

### 9.1 The problem

**WHAT.** 50:1 imbalance: six head classes hold 95.2% of training images, four tail classes 4.8%.

**WHY it matters here.** Gradient updates come mostly from head classes, so decision boundaries
drift into tail territory: borderline trucks become automobiles, horses become deer. Macro F1
gives each tail class 10% of the score, and the test set appears to have many more tail images
than training (section 3.4). Your own earlier WRN with no rebalancing predicted 447 trucks; the
team's March CCT/LDAM-era model predicted 224.

**HOW it was handled.** Two mechanisms: a sampler (9.2, 9.3) that changes what the model sees,
and Balanced Softmax (10.1) that changes the decision rule.

### 9.2 WeightedRandomSampler

**WHAT.** A PyTorch sampler that draws dataset indices with probability proportional to given
per-sample weights.

**WHY.** To put tail classes into almost every batch and give them more gradient updates.

**HOW.** `WeightedRandomSampler(sample_weights, len(self.labels), replacement=True)`.
`num_samples` is 27,930, so an epoch is still 27,930 draws: the epoch length doesn't change, only
the class mix inside it. `replacement=True` is required to draw a class more often than it has
images. The DataLoader gets `sampler=` and therefore `shuffle=False`.

**WHY THIS.** Resampling keeps the loss and gradient sizes normal, unlike large class weights in
the loss.

**TRADEOFF.** Head images are not all seen each epoch: each is drawn 0.86 times on average, so
only about 58% of head images appear in a given epoch (verified by simulating the real sampler).

**LIMITATION.** It repeats the same tail images; it creates no new information.

**INTERVIEW.** "Are you changing the dataset size?" "No. Each epoch is still 27,930 draws. What
changes is the class mix in those draws."

**CROSS QUESTION.** "Why replacement=True?" "Without replacement each image can appear at most
once per epoch, so trucks could never be drawn more than 95 times. With replacement a truck can be
drawn several times per epoch."

### 9.3 Square-root inverse weighting

**WHAT.** Each image gets weight 1/sqrt(n_class), so a class is drawn with probability
proportional to sqrt(n_class).

**WHY.** A middle ground between ignoring the tail (natural sampling) and drowning the model in
repeats of 95 trucks (full inverse frequency).

**HOW.** `class_weights = 1.0 / (np.sqrt(counts) + 1e-6)` (the `1e-6` avoids division by zero for
an empty class). With the training counts:

| class | natural share | sampled share | draws per image per epoch | full inverse frequency (v2) |
|---|---|---|---|---|
| airplane, automobile, bird, cat | 17.0% | 14.7% | 0.86 | 10% share, 0.59 per image |
| deer, dog | 13.6% | 13.1% | 0.97 | 10%, 0.74 |
| frog, horse | 1.7% | 4.65% | 2.73 | 10%, 5.9 |
| ship | 1.0% | 3.6% | 3.53 | 10%, 9.8 |
| truck | 0.34% | 2.08% | 6.11 | 10%, 29.4 |

With batch size 128 (the training batch size), about 6.8% of batches have no truck, against
64.7% with natural sampling. Expected trucks per batch: 2.7 instead of 0.4.

**WHY THIS.** The git history shows the progression. v2 (scored 0.9348) used full inverse
frequency, `1.0 / (counts + 1e-6)`, where every class is 10% of draws and each truck image is
shown about 29 times per epoch. v3 switched to sqrt-inverse. Important correction: the v3
header says "More aggressive oversampling of tail classes (sqrt-inverse weighting)" and the
docstring says "more aggressive than inverse frequency". Both are backwards. Square root is
milder than inverse frequency; v3 actually reduced the oversampling, while Balanced Softmax
continued to handle the prior.

**TRADEOFF.** Less repetition of tail images and more use of head data than full inverse
frequency, but the tail is still repeated about 6 times per epoch, around 2,750 times over 450
epochs.

**LIMITATION.** Combined with Balanced Softmax, the two corrections overlap (section 10.3).
v3's change to sqrt came together with five other changes, so its separate effect is unknown.

**INTERVIEW.** "Why sqrt-inverse instead of inverse frequency?" "Full inverse frequency, which
our v2 used, shows each of the 95 trucks about 29 times per epoch and each head image only 0.6
times. That's a lot of repetition of very few images. Square root is in between: trucks go
from 0.34% to about 2% of samples, about 6 times per epoch. Balanced Softmax was already
handling the class prior, so the sampler didn't need to do all the work."

**CROSS QUESTION.** "Why not class weights, focal loss, undersampling or SMOTE?" "Class weights of
up to 50x make one truck dominate a batch's gradient and do nothing when there's no truck in the
batch; with deep nets that fit training data, loss reweighting also loses much of its effect.
Focal loss targets easy versus hard examples, not class priors. Undersampling would throw away
4,900 of 5,000 cats, which kills feature learning from scratch. SMOTE on raw pixels just blends
images, which is roughly MixUp without being class-aware."

---

## 10. Loss Function

### 10.1 Balanced Softmax

**WHAT.** Cross-entropy on `logits + log(class prior)` during training; plain argmax of raw
logits at prediction (Ren et al., 2020).

**WHY.** Training has a 50:1 prior; the metric weights classes equally; the test prior is very
different from training. You want predictions that don't carry the training prior.

**HOW.**

```python
freq = class_counts / class_counts.sum()             # counts of the 27,930-image train split
self.register_buffer("log_freq", torch.log(freq + 1e-12))
adjusted = logits + self.log_freq.unsqueeze(0)       # shape 1 x 10, broadcast over the batch
return F.cross_entropy(adjusted, targets, label_smoothing=0.1)
```

`register_buffer` moves `log_freq` with `.to(device)` and saves it, but it is not trained.
`validate` and inference use `outputs.max(1)` / softmax on the raw logits. In `main()`,
`class_counts` comes from `tmp_ds.get_class_counts()`, i.e. the raw training-split counts
(verified).

Values of log prior: head classes -1.772, deer and dog -1.995, frog and horse -4.074, ship -4.585,
truck -5.684. The truck logit starts 3.91 below the cat logit (a factor of 50).

Why it works (Bayes): with the class appearance p(x|y) shared and only the prior changing,
p(y|x) is proportional to p(x|y) p(y). Training `softmax(z + log pi)` to match the training
posterior means exp(z_y) is proportional to p_train(y|x) / pi_y, which is the posterior under a
uniform prior. The fixed log pi term absorbs the training prior; the raw logits keep the evidence.

**WHY THIS.** It fixes the prior in the decision rule, needs no hyperparameter, starts working
from epoch 1, and doesn't change the model. Alternatives:

- Plain cross-entropy learns the training prior; borderline cases go to head classes.
- Weighted cross-entropy changes gradient sizes, not the decision rule, and big weights are noisy.
- Focal loss is about hard examples, not priors.
- Post-hoc logit adjustment (train normally, subtract log prior at test) is the same idea applied after
  training; Balanced Softmax learns the features under the balanced objective.
- LDAM with deferred re-weighting was the team's March approach (more hyperparameters: margins
  and when to switch).

**TRADEOFF.** Principled and free, but it targets a uniform test prior.

**LIMITATION.** Assumes pure label shift (classes look the same in train and test). If the test
prior is known and not uniform, you'd add log(test prior) to the raw logits. Here the test
prior seems tilted even further towards the tail than uniform, which Balanced Softmax alone
doesn't account for (10.3).

**INTERVIEW.** "How can a loss function account for class priors?" "Softmax exponentiates the
logits. Adding log of the class frequency to a logit multiplies that class's score by its
frequency, which is exactly the prior in Bayes' rule. So during training the prior is a fixed
number we add, not something the network has to learn into its weights. At prediction time we
use the raw logits, which behaves like a model trained on balanced classes."

**CROSS QUESTION.** "Why log frequency and not frequency?" "Because softmax exponentiates.
exp(z + log pi) = pi x exp(z), the Bayes prior factor. Adding pi itself, a number below 1, would
be a meaningless small shift. And if all classes had the same count, log pi would be the same
constant for every class and Balanced Softmax would be ordinary cross-entropy."

### 10.2 Label smoothing (0.1, verified)

**WHAT.** Target 0.91 for the true class and 0.01 for each other class, instead of 1 and 0
(PyTorch: (1 - 0.1) x onehot + 0.1/10).

**WHY.** 32x32 images are often ambiguous and perturbed; driving probabilities to exactly 1
needs ever-larger logits and encourages memorising individual images.

**HOW.** `F.cross_entropy(adjusted, targets, label_smoothing=0.1)` inside Balanced Softmax, also
inside both halves of a MixUp or CutMix loss.

**WHY THIS.** Cheap regulariser that usually helps CIFAR-style training; the value 0.1 is the
common default (v2 and v3 both used it).

**TRADEOFF.** Less overconfidence, but probabilities are no longer calibrated in the usual sense.

**LIMITATION (computed exactly).** The smoothing happens in the prior-adjusted space. For a cat
image the network fits perfectly, the adjusted probabilities are 0.91 cat and 0.01 for each other
class; the raw probabilities are proportional to adjusted / prior, so truck gets 0.01/0.0034
against cat's 0.91/0.170. Result: raw softmax 0.497 cat, 0.273 truck, 0.091 ship. At smoothing
above about 0.17 that image would be predicted as truck.

| smoothing | raw softmax of a perfectly fitted cat image |
|---|---|
| 0.0 | cat 1.000 |
| 0.05 | cat 0.675, truck 0.177 |
| 0.1 (used) | cat 0.497, truck 0.273, ship 0.091 |
| 0.2 | truck 0.375, cat 0.308 (wrong) |

So inside Balanced Softmax, label smoothing quietly adds another push towards rare classes.

**INTERVIEW.** "Why does overconfidence matter if you only take the argmax?" "Because of how the
model gets there. Pushing a probability to exactly 1 needs ever-larger logits, which rewards
fitting individual training images, including odd ones. Under distribution shift an overconfident
model is confidently wrong. Smoothing stops the push at 0.91."

**CROSS QUESTION.** "Does it interact with Balanced Softmax?" Give the calculation above, and the
lesson: on a 50:1 imbalance, smoothing has to stay small.

### 10.3 Sampler plus Balanced Softmax: double correction

**WHAT.** Balanced Softmax assumes the labels it sees follow the counts it was given (raw
counts). The sampler changes the labels seen to roughly sqrt(counts).

**HOW (maths).** The raw logits end up proportional to the balanced posterior times
n_y to the power -1/2. Relative to cat, truck gets +1.96 in logit, a factor of sqrt(4,750/95) =
7.07 in odds, before label smoothing adds its own push (10.2). In v2, with full inverse-frequency
sampling, the theoretical tilt was the full factor 50.

**What this means now (correction of the earlier analysis).** The earlier version of these notes
called this "over-correction", assuming a balanced test set. The repo's submissions show the test
set is very probably tilted towards the tail (section 3.4): perhaps 1,000 trucks against 500
cats. Relative to a balanced test, the extra tilt is an over-correction; relative to the likely
real test prior, it pushes in the right direction. Whether it overshoots is unknowable from the
files, because the only evidence about the test prior is these same predictions.

**INTERVIEW.** "Aren't the sampler and Balanced Softmax redundant?" "They overlap. The sampler
changes what the network sees, so rare classes are in almost every batch and get more updates.
Balanced Softmax changes the decision rule. Strictly, Balanced Softmax assumes the batches
follow the real counts, so combining them tilts predictions further towards rare classes than
a balanced test needs, about 7 to 1 for truck versus cat in theory. In this competition the test
set seems to have more rare-class images than common ones, so that extra tilt probably helped
rather than hurt. But I can't prove that without labels."

**CROSS QUESTION.** "How would you have checked it?" "Run validation with the prior correction at
a few strengths, `z + tau x log pi` for tau between 0 and 1, and compare per-class precision and
recall. And estimate the test prior from the model's own test predictions corrected by its
validation confusion matrix (black-box shift estimation), then add log of that prior at
prediction time."

---

## 11. Augmentation

Training images go through, in order: RandomCrop -> RandomHorizontalFlip -> AutoAugment ->
ToTensor -> Normalize -> Cutout. Then, per batch on the GPU, MixUp or CutMix with probability
0.5.

### 11.1 RandomCrop(32, padding=4, fill=128)

**WHAT.** Pad to 40x40 with grey 128, take a random 32x32 window: the object shifts by up to 4
pixels in each direction (81 possible offsets).

**WHY.** Objects aren't always centred; small shifts shouldn't change the label.

**HOW.** First transform in training and in the 30 TTA views.

**WHY THIS.** The most reliable augmentation for 32x32 data. Grey fill instead of black makes the
border look like neutral background, not a strong edge.

**TRADEOFF.** Adds grey borders real photos don't have.

**LIMITATION.** Larger shifts (8 pixels) would push small objects out of frame.

**INTERVIEW.** "Why fill with 128?" "Grey is close to the average colour, so the padding carries no
class signal. Black borders could become a cue."

**CROSS QUESTION.** "Doesn't convolution already give translation invariance?" "Convolution gives
translation equivariance of features, and pooling helps, but networks still learn position
habits. Random crops make the invariance explicit."

### 11.2 RandomHorizontalFlip

**WHAT.** Mirror left-right with probability 0.5.

**WHY.** A car facing left is still a car; doubles pose variety.

**HOW.** Training and TTA views.

**WHY THIS.** Label-safe for all 10 classes. Vertical flips are not: upside-down horses and ships
never appear in the test set, and sky-above/ground-below is a real cue.

**TRADEOFF / LIMITATION.** None for these classes (it would be wrong for digits or text).

**INTERVIEW.** "Why not vertical flip or large rotations?" "They make images that never occur in
the test set and destroy the sky-on-top cue. AutoAugment includes small rotations (its CIFAR policy
uses about 6.7 degrees, sometimes) and TTA uses up to 10 degrees, which is the mild range."

**CROSS QUESTION.** "Can augmentation change the class?" "Yes if it is too strong, for example a
crop that removes the object. Flips and small crops here are safe; Cutout and CutMix can hide a
small object, which is a known, tolerated source of label noise."

### 11.3 AutoAugment (CIFAR10 policy)

**WHAT.** A fixed list of 25 sub-policies found by an automated search on CIFAR-10 (Cubuk et al.,
2019). Each picks two operations with set probabilities and strengths: Invert, Contrast, Rotate,
TranslateX/Y, ShearY, Sharpness, AutoContrast, Equalize, Posterize, Solarize, Color, Brightness.

**WHY.** Broad colour, contrast, sharpness and geometry variation, without hand-tuning, on data
that looks like CIFAR-10.

**HOW.** `transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10)` after crop and flip; no
`fill` given, so geometric ops fill with black.

**WHY THIS.** Proven on CIFAR-10, costs nothing at test time. The team's first all-in-one script
used RandAugment; the March repo used TrivialAugment; v2 and v3 used AutoAugment. There is no
controlled comparison in the files.

**TRADEOFF.** Strong augmentation means slower fitting (one reason for 450 epochs). Some ops
(Invert, Solarize) make unnatural images.

**LIMITATION.** Searched for clean CIFAR-10 accuracy, not robustness; it has no noise operation,
and noise is the most common test perturbation here. AugMix or plain Gaussian noise would have
targeted that.

**INTERVIEW.** "Why AutoAugment?" "It's a well-tested recipe for exactly this kind of 32x32 data.
Since we couldn't train on the test perturbations, broad augmentation was the main tool for
robustness."

**CROSS QUESTION.** "Wasn't the policy learned on CIFAR-10? Is that external data?" "It's a list of
operations shipped in torchvision, no images or weights. The rules ban external data and pretrained
models, not standard transforms. But yes, it was tuned on very similar data, which is part of why
it suits this one."

### 11.4 Cutout (16) and partial occlusion

**WHAT.** Zero out one square of up to 16x16 at a random position on every training image (DeVries
and Taylor, 2017).

**WHY.** The test set has partial occlusions: 5% of test images carry a 10x10 grey square. The model
must not depend on any one region.

**HOW.** `Cutout.__call__`: random centre over the whole image, square from centre - 8 to centre +
8, clipped at the border, so it is a full 16x16 only 28% of the time; expected side 14 pixels,
expected area 196 pixels (19% of the image). It runs after `Normalize`, so the zeroed pixels are the
CIFAR mean colour, about RGB (125, 123, 114), which is exactly the colour of the dataset's own
occlusion squares. Applied with probability 1.

**WHY THIS.** Direct simulation of occlusion; 16 is the size the Cutout paper used for CIFAR-10. An
earlier notebook of yours used RandomErasing instead.

**TRADEOFF.** Every training image is partly hidden, though only 5% of test images are.

**LIMITATION.** On a small object a 16x16 square can hide all of it; the label then has no evidence
for that one view. Tolerated because the same image appears many times with the square elsewhere.

**INTERVIEW.** "How is Cutout related to partial occlusion?" "It's simulated occlusion. Every
training image has a grey square of up to 16x16 somewhere, so the model learns to classify from
what's still visible. The occluded test images have a 10x10 square of the same grey, so the
training squares are bigger on average than the test ones."

**CROSS QUESTION.** "Did you choose the grey colour to match?" Say the truth. The code puts Cutout
after normalisation, which makes it the mean colour; the match with the dataset's patches was found
in the analysis afterwards. Do not claim it was designed that way unless it was.

### 11.5 MixUp

**WHAT.** Blend two images with weight lambda and blend their labels the same way (Zhang et al., 2018).

**WHY.** Regularisation; smoother, less overconfident boundaries between classes; inputs that are
slightly off the training data get sensible outputs.

**HOW.** `mixup_data`: lambda from Beta(1, 1) (uniform on [0, 1]), partner = random permutation of
the same batch, `lam * x + (1 - lam) * x[perm]`. Loss `lam * CE(y) + (1 - lam) * CE(y_perm)`, which
equals cross-entropy with the mixed soft label. Chosen for a batch with probability 0.5 x 0.5 = 25%.

**WHY THIS.** Proven on CIFAR, cheap, works on the GPU batch.

**TRADEOFF.** Mixed images are unrealistic; mixed batches never show a clean single-label image.
Mixing only some batches is the compromise.

**LIMITATION.** Can blur the small details that separate fine-grained classes (cat vs dog at 32x32).

**INTERVIEW.** "Why mix the labels?" "Because the input really contains both images. If you blend a
cat and a truck 70/30 and keep the label 'cat', you teach the model to ignore the truck."

**CROSS QUESTION.** "Why Beta(1, 1)?" "Every mixing ratio is equally likely. Smaller alpha like 0.2
gives mostly near-0 or near-1 lambda, which is milder mixing; your earlier notebook used 0.4."

### 11.6 CutMix

**WHAT.** Paste a box from another image of the batch and mix labels by the box's area (Yun et al., 2019).

**WHY.** Learn to classify from a partial view with foreign content covering part of the object,
close to occlusion; every pixel stays real.

**HOW.** `cutmix_data`: lambda ~ Beta(1, 1); box side `int(32 * sqrt(1 - lambda))`; random centre
`cx, cy`; box clipped with `max(0, ...)` and `min(W, ...)`; pixels replaced from the permuted
batch; then lambda recomputed as `1 - box_area / 1024`. One box for the whole batch. Chosen for
25% of batches.

**WHY THIS.** Strong CIFAR results and a natural fit with occlusion.

**TRADEOFF.** The pasted box may contain only background of the partner image while the label still
gives it class weight.

**LIMITATION.** Recomputing lambda matters a lot: in a simulation of this exact code, the sampled
and recomputed lambda differ by 0.20 on average (0.45 at the 90th percentile), because boxes near
the edge are clipped. Without it, the labels would claim much more of the partner image than is
visible.

**INTERVIEW.** "Why calculate lambda from the actual patch area?" Give the clipping reason and the
0.20 number.

**CROSS QUESTION.** "Why use Cutout, MixUp and CutMix together? Isn't that excessive?" "They do
different things: Cutout hides information with plain grey, CutMix replaces it with distracting
real content, MixUp smooths the space between classes. It can be excessive; signs would be clean
training accuracy staying low. The long schedule gives the model time to fit. We didn't ablate each
one, so I can't claim each was necessary; v2 already had all three and scored 0.9348."

| | what changes | label | pixels | link to occlusion |
|---|---|---|---|---|
| Cutout | square set to mean grey | unchanged | rest real | direct |
| CutMix | box replaced by another image | mixed by area | all real | object partly hidden by other content |
| MixUp | whole image blended | mixed by lambda | blended | indirect |

---

## 12. Optimization

### 12.1 SGD with momentum 0.9 and Nesterov

**WHAT.** Stochastic gradient descent with a momentum buffer (an exponential average of recent
gradients) and the Nesterov look-ahead.

**WHY.** The standard, well-generalising optimiser for training CNNs from scratch on CIFAR-style data.

**HOW.** `torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)`.
In PyTorch: g = grad + wd x theta; v = 0.9 v + g; step = g + 0.9 v; theta -= lr x step. Momentum
averages over roughly 1/(1 - 0.9) = 10 steps.

**WHY THIS.** lr 0.1, momentum 0.9, Nesterov, weight decay 5e-4 with batch 128 and dropout 0.3 are
exactly the WRN paper's CIFAR settings. SGD also pairs naturally with SWA. Adam converges faster but
on CNN image classification often generalises slightly worse unless carefully tuned; AdamW is the
usual choice for transformers.

**TRADEOFF.** Sensitive to the learning rate; needs warmup and a schedule.

**LIMITATION.** No comparison with Adam in the files.

**INTERVIEW.** "What does momentum do mathematically?" "It keeps a running average of gradients.
Directions that agree across steps add up, so it moves faster there; directions that flip sign
cancel, so there is less zig-zag."

**CROSS QUESTION.** "What is Nesterov?" "It uses the gradient plus the momentum step it's about to
take, so in effect it looks at where momentum is carrying the weights and corrects a bit earlier.
Slightly more stable than plain momentum."

### 12.2 Weight decay 5e-4

**WHAT.** Every step adds 5e-4 x weights to the gradient, pulling weights towards zero.

**WHY.** Regularisation for a 36M-parameter model; also keeps BN-scaled weights from growing.

**HOW.** The `weight_decay` argument of SGD; applied to all parameters, including BN and the
linear bias.

**WHY THIS.** The WRN paper value.

**TRADEOFF / LIMITATION.** Too much underfits; decaying BN parameters is a common simplification.

**INTERVIEW.** "Is weight decay the same as L2 regularisation?" "For SGD, yes: adding wd x theta to
the gradient is exactly the gradient of an L2 penalty, and with momentum it goes through the
momentum buffer the same way. For Adam they differ, because Adam rescales gradients per parameter,
which is why AdamW applies decay separately."

**CROSS QUESTION.** "Why decay BN parameters too?" "It's the common simple setup; excluding BN and
biases is a small refinement that wasn't tried here."

### 12.3 Warmup (5 epochs) and cosine decay

**WHAT.** LR = 0.1 x (e + 1)/5 for epochs 0 to 4 (0.02, 0.04, 0.06, 0.08, 0.1), then
0.1 x 0.5 x (1 + cos(pi x (e - 5)/445)). From epoch 360 SWALR takes over.

**WHY.** Fresh networks with heavy augmentation and mixing can take wild steps at full LR; long runs
need the LR to come down so the weights settle.

**HOW.** `LambdaLR(optimizer, lr_lambda)`, stepped once per epoch until epoch 359. At epoch 359 the
cosine factor is 0.0997, so the LR is about 0.00997 when SWA starts; SWALR then moves it to 0.005
over 10 epochs (cosine) and holds it there to epoch 449.

**WHY THIS.** No milestones to choose (step decay needs them), smooth, and long high-LR phase suits
heavy augmentation. ReduceLROnPlateau would react to a validation score that moves 0.011 per truck.

**TRADEOFF.** Total epochs must be fixed up front. The cosine never finishes because SWA cuts in.

**LIMITATION.** 5 warmup epochs is a common default, not tuned.

**INTERVIEW.** "Why cosine?" "It keeps the learning rate high for a long time, which suits heavily
augmented data, then brings it down smoothly. No milestone epochs to tune, which matters when each
run takes about 20 hours."

**CROSS QUESTION.** "What if the LR is too high or too low?" "Too high: the loss oscillates or
diverges and clipping fires constantly. Too low: slow, and it tends to settle in a sharper minimum
that generalises worse."

### 12.4 Gradient clipping at 5.0

**WHAT.** If the global L2 norm of all gradients exceeds 5.0, scale them all down to norm 5.0.

**WHY.** Mixed batches, heavy augmentation and the prior-shifted loss can produce occasional very
large gradients, especially early.

**HOW.** `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)` between `backward()`
and `step()`. One global norm, so the direction is unchanged. Weight decay is added later inside
`step()`, so it isn't clipped.

**WHY THIS.** A loose safety cap. The value isn't documented or tuned (your earlier notebook used
1.0). Don't invent a reason for 5.0.

**TRADEOFF.** Protects against rare spikes; if it fired on most steps it would change the optimiser.

**LIMITATION.** Doesn't fix a bad LR or bad data. Gradient norms weren't logged, so how often it
fired is unknown.

**INTERVIEW.** "What problem does clipping solve?" "Rare huge gradients that would make one huge
update. It caps the step size but keeps the direction."

**CROSS QUESTION.** "Why 5?" "It's a loose cap for outliers, not a tuned number. Properly, you'd log
the gradient norm and set the cap above its normal range."

### 12.5 Batch size and learning rate

**WHAT.** Batch 128 in the code version that trained the three models (commit `18a6d54`, default
`--batch-size 128`); the final file's default is 512 (changed on 28 March, after the models were
trained, together with a DataParallel experiment that was then removed).

**WHY it matters.** 27,930 / 128 = 218 steps per epoch (drop_last drops 26 images), about 98,100
steps per seed over 450 epochs. At 512 it would be 54 steps per epoch.

**HOW to say it.** "We trained with batch 128 and lr 0.1, the standard WRN setting. The final
script's default became 512 when we set it up for inference; if you retrained at 512, you'd
normally scale the LR up too."

**LIMITATION.** The training run's command line isn't in the files, so 128 is the default of that
version, not a logged value. Memory check: WRN-28-10 stores about 22 MB of activations per image for
the backward pass, so 128 needs about 3 GB, and 512 about 11 to 12 GB (tight on a 16 GB T4).

**INTERVIEW.** "Why batch 128?" "It's the WRN paper setting and fits comfortably in memory. Larger
batches need a larger LR and sometimes generalise slightly worse without adjustments."

**CROSS QUESTION.** "Your final script says 512." "The final script's default is 512, set after training
when we reworked it for Kaggle inference. The checkpoints come from the 128 version."

---

## 13. SWA (Stochastic Weight Averaging)

**WHAT.** From epoch 360 to 449, keep a running average of the model's weights, one snapshot per
epoch, 90 snapshots (Izmailov et al., 2018). Weights, not predictions.

**WHY.** At the end of SGD the weights keep bouncing around a low-loss region; the average sits
nearer its centre, which tends to be flatter and to generalise better, including under small input
shifts. It is also not chosen on the noisy validation score.

**HOW (verified).**

```python
swa_model = AveragedModel(model)                    # created at the start
swa_scheduler = SWALR(optimizer, swa_lr=0.005)      # anneal_epochs=10, cosine (defaults)
...
if epoch >= 360:
    swa_model.update_parameters(model)              # theta_avg += (theta - theta_avg) / (n + 1)
    swa_scheduler.step()
else:
    scheduler.step()
...
update_bn(train_loader, swa_model, device=device)   # recompute BN running stats for the average
_, val_acc, val_f1, ... = validate(swa_model, ...)
if val_f1 > best_f1:                                # SWA must beat the best single epoch
    best_state = copy.deepcopy(swa_model.module.state_dict())
```

The LR goes from about 0.00997 at epoch 360 down to 0.005 over 10 epochs, then stays at 0.005.
`update_bn` runs over the training loader, i.e. sqrt-sampled, augmented images (AutoAugment,
Cutout), without MixUp or CutMix.

**Which weights each seed really used.** Whichever had the higher validation macro F1: the SWA
average or the single best epoch out of 450. The training logs aren't in the repo, so for each of
seeds 42, 137 and 7 it is unknown which one won. Note the comparison is tilted against SWA: the best
epoch is the maximum of 450 noisy validation scores (5 trucks), so it is biased upwards.

**WHY THIS.** Almost free (one extra copy of the weights), no extra inference cost. Alternatives:
keep only the best checkpoint (picked on a noisy score), an exponential moving average of weights
(close cousin), or ensembling snapshots (costs inference).

**TRADEOFF.** Needs a sensible SWA LR, a late start, and `update_bn`. The comparison with the best
epoch adds a safety net but also lets validation noise decide.

**LIMITATION.** If the SWA LR is too high, snapshots fall in different basins and the average sits
on a ridge. BN statistics are computed on augmented images, not clean ones.

**INTERVIEW.** "How is SWA different from saving the best checkpoint?" "The best checkpoint is one
snapshot chosen by validation score, and our validation score is noisy because of the tiny rare
classes. SWA averages the weights of the last 90 epochs into one model without looking at
validation. Our code then compared the two on validation and kept the better one per seed."

**CROSS QUESTION.** "So did your final models actually use SWA?" "SWA was computed for all three
seeds, and each seed used the SWA weights only if they beat the best epoch on validation. I don't
have the logs to say which won for each seed." (Do not claim more.)

More: "Why late in training?" Early weights are far from any good region and from each other;
averaging them would average different solutions. "Weights or predictions?" Weights; the ensemble
and TTA average predictions. "Can SWA hurt?" Yes: no `update_bn` (handled here), too-high SWA LR,
starting too early.

---

## 14. Ensemble

**WHAT.** Three WRN-28-10 models trained with seeds 42, 137 and 7; their softmax probabilities are
averaged with equal weight.

**WHY.** Each run lands in a different solution and makes partly different mistakes, especially on
rare classes and shifted images; averaging cancels the mistakes that aren't shared.

**HOW (verified).** `train_single_seed` is called once per seed; `seed_everything(seed)` sets
weight init, sampler order, augmentation, dropout and mixing randomness. The validation split stays
fixed (split seed 42). In `main()`, each model's 31-view probabilities are added to
`ensemble_probs`, which is then divided by 3.

Agreement between submissions from the history: the 2-seed and 3-seed ensembles agree on 98.0% of
test images; v2 (one model) and the 3-seed ensemble agree on 96.2%.

**WHY THIS.** Ensembling is one of the most reliable ways to add accuracy and robustness when
inference time doesn't matter. The history shows the team went 1 model (v2) -> 2 seeds (first v3
draft, 400 epochs) -> 3 seeds (450 epochs). 3 rather than more because each model costs about 20
hours.

**TRADEOFF.** 3x training and 3x inference; members are correlated because they share data,
architecture and recipe, so the gain is limited (variance of an average of M models with
correlation r is s^2 (r + (1 - r)/M)).

**LIMITATION.** If all three make the same error (say a 10x10 patch hides the only truck-specific
part), averaging can't fix it.

**INTERVIEW.** "Probabilities or logits?" "Softmax probabilities, equal weights, after averaging
each model's 31 TTA views." "Why 3?" "Cost: about 20 hours per model. Most of an ensemble's gain
comes from the first few members."

**CROSS QUESTION.** "Is an ensemble the same as SWA?" "No. SWA averages weights inside one run and
gives one model; the ensemble averages predictions of three separate models. Here they stack: each
ensemble member may itself be an SWA model."

More: "What if members are highly correlated?" The gain shrinks; diversity would come from different
architectures (the team's first script planned PyramidNet + WRN), different augmentation, or
different data splits.

---

## 15. TTA (test-time augmentation)

**WHAT.** For each model: one prediction on the clean test image plus 30 predictions on random light
transformations, averaged. 31 predictions per model, 93 per image for the ensemble, 706,800 forward
passes for the 7,600 test images.

**WHY.** One forward pass depends on the exact framing; averaging over small shifts, flips, colour
changes and rotations gives a steadier answer.

**HOW (verified).** `predict_with_tta`: sets `test_dataset.transform = get_val_transforms()` and
runs the clean pass; then 30 times sets `test_dataset.transform = get_tta_transform()` (crop with
4-pixel padding, flip, ColorJitter brightness/contrast/saturation 0.1, rotation up to 10 degrees
with grey fill) and runs a pass; sums softmax probabilities; divides by 31. Batch 512, 4 workers.

**Important correction.** The resume says "30-view TTA". The code does 30 augmented views plus the
clean view, 31 predictions. That is consistent with "30-view" if you mean augmented views, but say
it precisely: "30 augmented views plus the original image".

**WHY THIS.** Random views cover more cases than a few fixed views (your earlier notebook used 4
fixed views averaging logits). 30 is a cost compromise: the random part of the average shrinks like
1/sqrt(views). v2 used 20 + 1.

**TRADEOFF.** 31x inference per model. Gains from TTA are usually small.

**LIMITATION.** The views don't remove noise or occlusion; they mostly average over position and
colour. Rotation up to 10 degrees is a bit beyond most training rotations. Also: when the
checkpoints are loaded (the final Kaggle run), `main()` never calls `seed_everything`, and PyTorch
seeds itself randomly per process (verified), so the 30 views differ between runs. The submission
is not bit-reproducible, though the differences should be tiny.

**INTERVIEW.** "Why would augmentation at inference help?" "The model's errors on individual views
partly cancel when you average, so the prediction depends less on one framing. The clean view keeps
the original image in the average."

**CROSS QUESTION.** "How much did TTA add?" "I don't have a measured number; TTA was in both v2 and v3,
so the v2-to-v3 comparison doesn't isolate it. I'd measure it by running the same checkpoints on
validation with 0, 5, 10 and 30 views." "Can TTA hurt?" "Yes, if views are off-distribution or not
label-preserving; that's why the TTA transforms are milder than training ones." "Does it break the
rules?" "No; it only uses the test image and the trained model."

---

## 16. Validation

### 16.1 The split

**WHAT.** For each class: shuffle with `RandomState(42)`, put `max(1, int(n * 0.05))` in
validation. 27,930 / 1,470; validation per class 250, 250, 250, 250, 200, 200, 25, 25, 15, 5.

**WHY.** Needed to pick checkpoints and catch overfitting, while keeping as many tail images as
possible for training.

**HOW.** Inside `ShiftGuard10Dataset`; train and val built with the same `seed=42`, so they are
exact complements (zero overlap, tested) and identical for all three model seeds.

**WHY THIS.** v2 used 10%; v3 moved to 5% ("more data for tail classes"): truck keeps 95 training
images instead of 90. Stratified so every class appears in the right proportion. k-fold would cost
5x a 20-hour run. Training on 100% would leave nothing to select checkpoints or compare SWA against.

**TRADEOFF.** More training data, less reliable validation.

**LIMITATION.** 5 trucks and 15 ships: one truck mistake moves macro F1 by 0.011; sampling noise
alone is about 0.015 standard deviation. Validation comes from the training distribution (6x6
patches, mostly clean pixels, 50:1 classes), not the test distribution.

**INTERVIEW.** "Why 95/5?" "Because of truck. With 100 images, every one matters for training. 5%
still gives 250 validation images for each big class, but only 5 trucks, so we treated validation as
a rough signal."

**CROSS QUESTION.** "How do you know validation represents the test?" "It doesn't fully. The test
has bigger occlusions, much more noise and a different class mix. Macro F1 on validation handles the
class mix partly because it weights classes equally anyway. For the image shift, the right fix is a
validation copy with test-like noise and 10x10 patches added, which I'd do next time."

### 16.2 Model selection

**WHAT.** Per seed, keep the epoch with the highest validation macro F1; after training, compare
it with the SWA model and keep the better.

**WHY.** Guards against a bad final epoch or a bad SWA average.

**HOW.** `if val_f1 > best_f1: best_state = copy.deepcopy(model.state_dict())`, then the SWA
comparison (section 13). `deepcopy` matters: `state_dict()` returns references to live tensors.

**WHY THIS.** Simple and aligned with the leaderboard metric.

**TRADEOFF.** Picking the maximum of 450 noisy scores overfits the validation set a little (winner's
curse), and that biased maximum is also the bar the SWA model has to beat.

**LIMITATION.** With 5 validation trucks, the choice can be decided by one or two images.

**INTERVIEW.** "Could you have overfit the validation set?" "Somewhat: choosing the best of 450 epochs
on a small, noisy set biases that choice upward. The SWA model isn't chosen per epoch, which is one
reason to prefer it, but our code only used it when it beat that biased maximum."

**CROSS QUESTION.** "What would leakage look like here?" "The same image in train and validation
(checked: none, no duplicates), normalisation statistics computed on test (no, fixed constants),
or many decisions tuned on the public leaderboard (overfits the public part)."

---

## 17. Distribution Shift

### 17.1 Covariate shift: noise and occlusion

**WHAT.** The images change while the meaning of each class stays the same: test images have 10x10
grey patches instead of 6x6 (on 5% of images), and about 75% of test images carry colour noise
(38% strong) against about 21% of training images (0.9% strong).

**WHY it matters.** Models that rely on fine texture or on one discriminative region lose accuracy.

**HOW the pipeline addresses it.** Cutout (squares up to 16x16, same grey) and CutMix for occlusion;
AutoAugment, crops, flips, MixUp, label smoothing, dropout, weight decay for general robustness; SWA,
ensemble and TTA to average out fragile predictions.

**WHY THIS.** You can't train on test perturbations you don't have; you train under many
label-preserving changes.

**TRADEOFF.** Generic robustness is not targeted robustness.

**LIMITATION.** Nothing adds pixel noise in training, although noise is the most common test
change. The measured shift came from post-competition analysis.

**INTERVIEW.** "What distribution shift did this competition have?" "The page only said
perturbations and partial occlusions. Analysing the data, there were three things: a grey square on
5% of images, 6x6 in training and 10x10 in test; colour noise on about a fifth of training images
but three quarters of test images, and stronger; and a very different class mix."

**CROSS QUESTION.** "How did you measure the noise?" "A standard high-pass noise estimate applied to
the R-G and B-G colour differences. Natural texture like grass looks similar in all three channels
and cancels out there; added per-pixel noise doesn't. The histogram shows clearly separate clean,
mild and strong groups."

### 17.2 Label shift: a reversed class mix

**WHAT.** The class proportions change. Training is 50:1 towards airplane/automobile/bird/cat; the
test set appears tilted the other way (about 500 each for those four and about 950 to 1,000 each for
frog, horse, ship, truck; section 3.4).

**WHY it matters.** A model that learned the training prior would massively under-predict exactly
the classes that dominate the test set. This may be the biggest single reason simple solutions
scored low here.

**HOW the pipeline addresses it.** Balanced Softmax removes the training prior from predictions; the
sqrt sampler and label smoothing (through their interaction with Balanced Softmax) tilt predictions
further towards rare classes (10.3).

**WHY THIS.** Without labels you can't know the test prior in advance; correcting to "no prior" is the
principled default, and the extra tilt happened to point the same way as the real shift.

**TRADEOFF / LIMITATION.** If the test prior had matched training, this setup would over-predict the
tail and lose accuracy. Nothing estimates the test prior explicitly.

**INTERVIEW.** "Was the test set balanced?" "Not as far as I can tell. Our strong submissions all
predict around 500 for each of the four biggest training classes and around 950 for each of the four
rarest. With macro F1 near 0.95, the predicted counts can't be far from the true ones, and a
balanced test set would cap those files below 0.88. So the rare training classes seem to be the
common test classes."

**CROSS QUESTION.** "Isn't that circular, using your own predictions?" "For checking whether the model
over-predicts the tail, yes, and I don't use it for that. But it does rule out a balanced test set,
because a balanced test can't produce a 0.9348 score from v2's prediction counts."

### 17.3 Robustness, in one sentence for the interview

"We couldn't see the test perturbations, so we trained under many label-preserving changes, removed
the training class prior from the decision, and averaged over time (SWA), models (3 seeds) and
inputs (31 views) so that no single fragile cue decided the answer."

---

## 18. Complete Code Explanation (`notebook.py`, 738 lines)

### 18.1 Header and configuration (L1-58)

The docstring lists the v3 features (3 seeds, sqrt-inverse sampling, 95/5, 450 epochs with SWA
from 360, "30-view TTA", checkpoints) and the compliance note: no external checkpoint or dataset,
about 60 hours on 2x T4, trained in batches because of Kaggle's 30-hour limit. That note was added
on 29 March, after training. The earlier version of the same header said "v3 improvements over v2
(which scored 0.9348)".

`DATA_ROOT`, `CLASS_NAMES` (fixes label indices: airplane 0 ... truck 9), `CLASS_TO_IDX`,
`IDX_TO_CLASS`, `CIFAR_MEAN`, `CIFAR_STD`.

### 18.2 Utilities (L65-89)

- `seed_everything(seed)`: seeds Python `random` (Cutout position, CutMix box, mix decisions), NumPy
  (MixUp/CutMix lambda), PyTorch CPU and CUDA (init, sampler, `randperm`, dropout, DataLoader worker
  seeds); sets cuDNN to deterministic and turns off benchmark mode.
- `AverageMeter`: sample-weighted running mean of the loss.
- `compute_macro_f1`: `f1_score(targets, preds, average="macro", zero_division=0)`; note it passes
  `targets` first, which is the order scikit-learn expects.
- `get_classification_report`: per-class precision, recall, F1, printed every 50 epochs and at the
  end. **Bug (verified by running):** it passes `target_names` without `labels`, so it raises an
  error whenever fewer than 10 classes appear in predictions plus targets. Never triggers in a full
  run (every class has at least 5 validation images), but `python notebook.py --debug`, the smoke
  test documented in the README, crashes at the end of epoch 2: its validation subset is 50
  airplanes. Fix: `labels=list(range(10))`.

### 18.3 Augmentation (L96-139)

`Cutout`, `get_train_transforms`, `get_val_transforms`, `get_tta_transform`: see section 11.

### 18.4 Dataset (L146-212)

`ShiftGuard10Dataset(root, split, transform, val_ratio=0.05, seed=42)`: the class-wise split
(section 16.1), `__getitem__` returns `(image, label)` for train/val and `(image, id)` for test,
`get_class_counts` (`np.bincount`, 10 bins), `get_sampler` (section 9). Debug-mode caveat: the
split is built class by class, so `Subset(range(200))` in debug mode is 200 airplanes and the
validation subset 50 airplanes.

### 18.5 Model (L219-273)

`WRNBlock`, `WideResNet`, `_make_group`, `_init_weights`: section 7.

### 18.6 Loss and mixing (L280-318)

`BalancedSoftmaxLoss` (section 10), `mixup_data`, `cutmix_data`, `mix_criterion` (section 11).

### 18.7 `train_one_epoch` (L325-358)

Per batch: move to device; with probability `mix_prob` mix the whole batch (MixUp or CutMix 50/50)
and use the mixed loss, else the plain loss; `zero_grad`, `backward`, clip to norm 5.0, `step`.
Training accuracy counts only unmixed batches, on augmented images, so it understates clean
training accuracy.

### 18.8 `validate` (L361-378)

`model.eval()`, no gradients, loss with the same Balanced Softmax criterion (so the validation loss
includes the log-prior shift and isn't plain cross-entropy), prediction by argmax of raw logits,
macro F1 and accuracy.

### 18.9 `train_single_seed` (L381-537)

1. Seed; checkpoint path `wrn_seed{seed}.pth` in `--checkpoint-dir`.
2. Datasets with the fixed split, sampler, loaders (train: 4 workers, `pin_memory`, `drop_last`;
   val: batch x2).
3. Model, Balanced Softmax with `class_counts`, SGD, LambdaLR, `AveragedModel` and `SWALR` if
   `swa_start < epochs`.
4. Resume block: if the checkpoint exists and has `model_state`, restore model, optimiser,
   scheduler, best F1, best weights, start epoch, SWA model and SWA scheduler; if it is a plain
   state_dict (old format), load weights and start from epoch 0.
5. Epoch loop: train, step the right scheduler, validate, log, report every 50 epochs, keep best,
   save a checkpoint with `completed: False`. `swa_state` and `swa_scheduler_state` are saved only
   once the SWA phase has started.
6. SWA finish: `update_bn`, validate the SWA model, replace `best_state` if it is better.
7. Save the final checkpoint with `completed: True`; return `best_state, best_f1`.

### 18.10 `predict_with_tta` and `generate_submission` (L544-594)

`predict_with_tta(model, test_dataset, device, n_views=30, batch_size=512)`: clean pass, then
`n_views` random passes, each building a fresh DataLoader after swapping `test_dataset.transform`;
sums softmax probabilities and divides by `n_views + 1`. Returns probabilities and ids.
`generate_submission`: argmax, map to class names, write `id,label` in test order, print the
predicted class counts.

### 18.11 `main` (L601-734)

1. Arguments with defaults (all verified): `--epochs 450`, `--batch-size 512`, `--lr 0.1`,
   `--wd 5e-4`, `--val-ratio 0.05`, `--swa-start 360`, `--swa-lr 0.005`, `--tta 30`,
   `--seeds 42 137 7`, `--mix-prob 0.5`, `--label-smoothing 0.1`, `--gpu 0`,
   `--data-root` (Kaggle competition path), `--checkpoint-dir` (Kaggle Model
   `xavaitron/wrn-3seed`), `--inference-only`, `--output submission.csv`, `--debug`.
   `parse_known_args` ignores the extra arguments Jupyter/Kaggle passes.
2. Data root: the default is set, so the list of fallback paths below it is never used.
3. Debug overrides: 2 epochs, batch 32, SWA off (start 9999), 2 TTA views, seed 42.
4. Device `cuda:{gpu}`; `n_gpus` is computed but unused (no DataParallel in the final file; it was
   added and removed on 28 March).
5. Class counts from the 27,930-image training split.
6. For each seed: if `wrn_seed{seed}.pth` exists, load `best_state` (or `model_state`, or the whole
   file if it is a plain state_dict) and skip training; else train (or exit in `--inference-only`).
7. Ensemble inference over all models with `predict_with_tta`, average, write the submission.

### 18.12 Checkpointing

**WHAT.** Saving the training state to disk so work isn't lost.

**WHY.** A 450-epoch WRN-28-10 run takes many hours; Kaggle sessions and GPU quotas are limited.

**HOW (final file).** After every epoch: `epoch`, `model_state`, `optimizer_state` (momentum buffers),
`scheduler_state`, `best_f1`, `best_state`, `swa_state` and `swa_scheduler_state` (once SWA has
started), `completed: False`. At the end, the same with `completed: True`. Measured file size:
438 MB before SWA (model + best weights + momentum), about 580 MB once the SWA average is included.
`main()` reads `best_state` from it, or the whole file if it's an old plain state_dict.

**What actually happened (verified from the git history).** The three models were trained with the
25 March version, which had no per-epoch checkpoints: it saved only `best_state` as a plain file
`checkpoints/wrn_seed{seed}.pth` at the end of each seed. Per-epoch saving and resume were added on
28 March (commit "Save checkpoint every epoch, resume mid-training on restart"), along with
"backward-compatible checkpoint loading for old-format checkpoints", which is what lets the final
script load those plain files.

**Two logic issues in the final file (verified by running).**

1. `main()` skips training whenever the checkpoint file exists, without looking at `completed`. So
   the resume block inside `train_single_seed` can never be reached from `main()`: by the time
   `train_single_seed` runs, the file does not exist. Run on CPU: a `--debug` run crashes (bug
   18.2) after saving an epoch-0 checkpoint; running the same command again prints "Checkpoint found
   ... skipping training" and goes straight to inference with the half-trained weights.
2. The default `--checkpoint-dir` is a read-only Kaggle input folder, so retraining on Kaggle with
   default arguments can't save checkpoints there; you have to pass a writable folder.

Fix for 1: check `ckpt.get('completed')` in `main()` and call `train_single_seed` (which resumes)
when it is False.

**WHY THIS design.** Save everything needed to continue: weights alone would restart momentum at zero
and the LR schedule at warmup.

**TRADEOFF.** Writing about 0.4 to 0.6 GB per epoch per seed; overwriting a single file risks
corrupting the only copy on a crash (safer: write to a temporary file, then rename).

**LIMITATION.** Random number generator states are not saved and the run reseeds at the start, so a
resumed run is valid but not bit-identical to an uninterrupted one.

**INTERVIEW.** "Why save the optimizer state?" "SGD momentum is a running average of past gradients.
Reloading only the weights resets it to zero, so the next steps differ from an uninterrupted run. Same
for the scheduler: without it the LR would restart from warmup."

**CROSS QUESTION.** "So did you train across Kaggle sessions with this resume logic?" Answer honestly:
"The models we submitted were trained in one run with the earlier version of the script, outside
Kaggle's session limit, and saved as best-weight files. We added per-epoch checkpointing afterwards
and set up the Kaggle notebook to load the checkpoints and run inference. Looking at the final code,
the resume path also has a bug: main skips training if any checkpoint file exists."

### 18.13 Reproducibility

**WHAT.** Making a run repeatable.

**WHY.** Comparable seeds, the competition's reproducibility rule, debugging.

**HOW.** `seed_everything(seed)` at the start of each seed's training; fixed split seed 42; cuDNN
deterministic, benchmark off.

**WHY THIS.** Each ensemble member is defined by its seed, and validation is identical for all three.

**TRADEOFF.** Deterministic cuDNN and no benchmark mode cost speed.

**LIMITATION.** Seeds don't guarantee identical bits across GPUs, driver or library versions, or worker
counts. In the inference-only path (the final Kaggle run), nothing is seeded, and PyTorch picks a
random seed per process (verified), so the 30 TTA views and therefore tiny details of the submission
differ from run to run. Fix: call `seed_everything` before inference.

**INTERVIEW.** "Is your submission exactly reproducible?" "The training is seeded. The final inference
run wasn't, because it skipped the training function where seeding happens, so the random TTA views
change between runs. The effect should be tiny but it's not bit-identical; one line fixes it."

**CROSS QUESTION.** "Do seeds guarantee identical results?" "No. GPU kernels can add numbers in
different orders, and results change with hardware and library versions. You can promise
statistically equivalent results, not identical bits."

### 18.14 Small issues list (for "find a bug in your code")

| issue | effect | fix |
|---|---|---|
| `classification_report` without `labels` | `--debug` crashes at epoch 2 (verified) | pass `labels=list(range(10))` |
| `main()` ignores `completed` | resume unreachable; half-trained checkpoint used silently (verified) | check the flag, call `train_single_seed` |
| no seeding in inference-only path | TTA views differ per run (verified) | `seed_everything(0)` before inference |
| sampler docstring and header say "more aggressive" | wrong: sqrt is milder than inverse frequency | fix the text |
| SWA-scheduler resume line not guarded by `use_swa` | NameError in an edge case | add `use_swa and` |
| debug subsets are all airplanes | debug run can't show learning | sample across classes |
| default checkpoint dir is read-only on Kaggle | retraining with defaults fails to save | pass a writable dir |
| fallback data-root list unreachable | harmless dead code | remove or make the default `None` |
| `n_gpus` unused, header says 2x T4 | one GPU per process | document; run seeds on separate GPUs with `--gpu` |

---

## 19. Why Each Decision Was Made

| problem in this data | what targets it | evidence it helped | honest status |
|---|---|---|---|
| 50:1 imbalance, macro F1, tail-heavy test | Balanced Softmax + sqrt sampler | v2 and v3 both had it, and both predict roughly 950 per rare class; earlier models without it predicted 224 or 447 trucks | strong reasoning, no ablation |
| repeated tail images | sqrt instead of full inverse sampling (v2 -> v3) | part of the v2 0.9348 -> v3 0.9488 change bundle | effect not isolated |
| overconfidence, ambiguous 32x32 images | label smoothing 0.1 | none | standard default |
| occlusion (10x10 grey in test) | Cutout 16 (same grey), CutMix | none | plausible, untested |
| perturbations in general | AutoAugment, crop, flip, MixUp | none | standard strong recipe |
| memorising 95 trucks with 36M params | dropout, weight decay, augmentation, mixing | none | standard |
| unstable steps | warmup, clipping at 5.0 | none | safety measures |
| noisy final weights | SWA (used per seed only if it beat the best epoch) | none; which won per seed is unknown | partly verified |
| run-to-run variance | 3-seed ensemble | 1 -> 2 -> 3 seeds in the history; agreement 96-98% between versions | no score per step in the files |
| framing variance | 1 clean + 30 TTA views | none (v2 already had 20 + 1) | no measurement |
| more training data for the tail | 95/5 split instead of 90/10 | part of the v2 -> v3 bundle | effect not isolated |
| compute and Kaggle limits | server training, checkpoint files, Kaggle inference-only run | git history | verified |

**The one real before/after number.** v2 scored 0.9348 (stated in the code). v3 (resume: 0.9488)
changed six things at once: 1 -> 3 seeds, full inverse -> sqrt-inverse sampling, 90/10 -> 95/5
split, 300 -> 450 epochs, SWA start 250 -> 360, TTA 20+1 -> 30+1. So "v3 was +0.014 over v2" is
true; "the ensemble added 0.014" is not something you can say.

---

## 20. Why Not Alternative Approaches

Things the team actually tried (from the git history) are marked "tried".

| choice | alternative | why the choice makes sense | when the alternative is better | limitation |
|---|---|---|---|---|
| WRN-28-10 | PyramidNet-272 + ShakeDrop (tried, 23 Mar) | simpler, faster, strong; the single WRN v2 already scored 0.9348 | more compute and time | 36M params |
| WRN-28-10 | Compact Convolutional Transformer (tried, March) | CNN priors matter with 29K small images, no pretraining | large data or pretraining | |
| WRN-28-10 | ResNet-18 | usually stronger on CIFAR-style data when trained long | tight compute | slower |
| from scratch | ImageNet weights | required by the rules | any real project where allowed | everything learned from 29K images |
| Balanced Softmax | LDAM + deferred re-weighting (tried, March) | no extra hyperparameters, works from epoch 1 | when margins are tuned well | targets a uniform prior |
| Balanced Softmax | weighted CE / focal loss | fixes the decision rule, normal gradient sizes | mild imbalance | |
| sqrt-inverse sampling | full inverse frequency (v2, tried) | less repetition of 95 trucks (6 vs 29 per epoch), more head data used | when tail recall is all that matters | still repeats; overlaps with Balanced Softmax |
| sampling | undersampling | keeps head data for feature learning | huge head classes | head images not all seen each epoch |
| AutoAugment | RandAugment (tried, first script), TrivialAugment (tried, March), AugMix | proven CIFAR-10 recipe | AugMix targets corruption robustness, which fits the noise | no noise op |
| Cutout + CutMix + MixUp | one of them | different kinds of regularisation; v2 already had all three | small compute budget | possible over-regularisation |
| SGD + Nesterov | Adam / AdamW | WRN's own settings; good generalisation; fits SWA | transformers, quick prototypes | LR-sensitive |
| warmup + cosine | step decay / plateau | no milestones; plateau would react to noisy validation | known good milestones | epochs fixed up front |
| SWA vs best epoch, pick by val | always SWA / always best | safety net | always SWA avoids winner's-curse selection | validation noise decides |
| 3 seeds | 1 model / more seeds | lower variance; about 20 h per model | latency-bound systems | correlated members |
| 1 + 30 TTA | none / few fixed views | averages framing randomness | latency-bound systems | 31x cost, doesn't remove noise |
| 95/5 split | 90/10 (v2), k-fold | keeps 95 trucks | when reliable validation matters more | 5 validation trucks |
| plain supervised training | supervised contrastive learning (tried, 13 Mar, `src/supcon.py`) | simpler single-stage training won out | when representation quality is the bottleneck | |

Short answers to "why not X":

- **ViT:** too little data and no pretraining allowed; CNNs have the right built-in assumptions.
- **ResNet-50:** built for 224x224; its stem shrinks 32x32 to 8x8 before the first block.
- **Adam:** fine for prototyping; SGD with the WRN settings is the reference and generalises well.
- **Focal loss:** about hard vs easy examples, not class priors.
- **Class weights:** 50x weights make single images dominate a batch and do nothing when the class is absent.
- **Transfer learning:** banned here; in any real project it would be the first thing to try.

---

## 21. Limitations

1. **No controlled ablations.** One before/after number (v2 0.9348 -> v3) with six simultaneous
   changes. Answer: "We changed several things at once because each run took around 20 hours; the
   v3 bundle improved on v2 by 0.014, but I can't attribute that to one change."
2. **Small, unshifted validation.** 5 trucks, 15 ships; one truck moves macro F1 by 0.011; and it
   comes from the training distribution, while the test has bigger occlusions, more noise and a
   different class mix.
3. **Model selection noise.** Best of 450 epochs on that set is biased upwards, and the SWA model
   only replaced it if it beat that biased number.
4. **No noise augmentation.** 75% of test images are noisy; training adds no noise.
5. **Imbalance corrections overlap.** Sampler + Balanced Softmax + label smoothing tilt predictions
   towards rare classes beyond what a balanced test needs. Here the test seems tail-heavy, so this
   probably helped, but on a test set with the training class mix it would cost accuracy.
6. **Test prior never estimated explicitly.** It could have been estimated from predictions (black-box
   shift estimation or EM) and plugged in as log(test prior).
7. **Cost.** About 60 hours of training (header), 93 forward passes per test image.
8. **Workflow vs rule wording.** The rule says the notebook trains and infers; the final Kaggle
   notebook loaded self-trained checkpoints and ran inference. Be factual about it.
9. **Code issues** (section 18.14): `--debug` crash, unreachable resume, unseeded inference.
10. **Reproducibility.** Inference not seeded; resume reseeds; GPU non-determinism.
11. **Calibration not measured.** Label smoothing, MixUp and Balanced Softmax all change confidence.
    Fine for argmax F1, not for any use of the probabilities.
12. **Ownership.** The final code's commits are by Pratyush Singh (Xavaitron). Know your part.

Not real weaknesses (don't over-apologise): WRN-28-10 is right for 32x32 from scratch; CIFAR
normalisation constants match the data; TTA doesn't break rules; self-trained checkpoints are not
pretrained weights.

**Edge cases an interviewer may raise.**

| situation | what happens in this code | what to say |
|---|---|---|
| class with very few examples (95 train, 5 val trucks) | drawn about 6 times per epoch; validation F1 for it takes only a few values | risk of memorising; noisy validation; shifted and repeated validation |
| class missing from a batch | about 7% of batches of 128 have no truck; loss is per example, nothing breaks | fine |
| model predicts only big classes | macro F1 collapses; `zero_division=0` gives 0 for dead classes | that is what Balanced Softmax and the sampler prevent |
| augmentation changes the label | Cutout or CutMix can hide a small object | rare; CutMix mixes labels by area |
| CutMix box covers the key object | label still claims lambda for the base class | tolerated label noise |
| MixUp makes unrealistic images | intended | regularises between classes |
| sampler + Balanced Softmax over-correct | lean towards rare classes (x7 truck vs cat in theory) | right direction for this tail-heavy test; wrong for a training-like test |
| test prior differs from training | Balanced Softmax targets a uniform prior | add log(test prior) if it is known or estimated |
| shift stronger than expected | no noise augmentation | accuracy drops on the noisiest images |
| TTA views not label-preserving | views are mild; clean view included | keep TTA milder than training augmentation |
| ensemble members correlated | same recipe and data; 96-98% agreement between versions | gain is limited; diversify |
| SWA averages incompatible weights | late start (360) and low LR (0.005) make this unlikely | `update_bn` is run |
| scheduler resumes incorrectly | changing `--epochs` changes the cosine curve | keep arguments fixed |
| corrupted or half-trained checkpoint | `main()` uses any existing file without checking `completed` (verified) | check the flag; write to a temp file and rename |
| GPU non-determinism | small differences remain; inference unseeded | seed inference; promise equivalent, not identical |
| high train accuracy, poor validation F1 | overfitting or tail failure | per-class recall, tail first |
| validation up, leaderboard down | validation unshifted; selection noise | shifted validation; fewer leaderboard-driven decisions |
| macro F1 metric, cross-entropy loss | aligned through Balanced Softmax plus argmax | not perfectly; per-class offsets need more validation data |
| poor calibration with high F1 | F1 only uses the argmax | calibrate only if probabilities are used |

**How to design the missing ablation.** Fix the split and seeds; shorten to about 100 epochs; start
from v2; add one change at a time (sqrt sampler, 95/5, more epochs, SWA timing, seeds, TTA views),
3 seeds each, report mean and spread of validation macro F1 on both a clean and a test-like shifted
validation set. Inference-only parts (TTA views, SWA vs best, 1 vs 3 models) need no retraining:
run the existing checkpoints with different settings.

---

## 22. Resume Bullet Defense

### Bullet 1

> Built a WideResNet-28-10 classifier from scratch for 10-class image recognition under distribution
> shift with Kaiming initialization and Dropout regularization

| term | status | detail |
|---|---|---|
| WideResNet-28-10 | verified | 36,479,194 params, `WideResNet(28, 10)` |
| from scratch | verified | random init, no pretrained weights; trained off-Kaggle, inference on Kaggle from those checkpoints |
| 10-class, distribution shift | verified | competition + measured shift |
| Kaiming initialization | verified | `kaiming_normal_`, fan_out, ReLU, convs only |
| Dropout | verified | 0.3 in every block |
| "Built" | check | the repo's commits are by Pratyush Singh (Xavaitron). If this was team work, say "we built" and describe your part |

### Bullet 2

> Trained a 3-seed ensemble using Balanced Softmax Loss with label smoothing, sqrt-inverse class
> resampling, and Stochastic Weight Averaging (SWA) with Cosine LR warmup

| term | status | detail |
|---|---|---|
| 3-seed ensemble | verified | seeds 42, 137, 7; probabilities averaged equally |
| Balanced Softmax | verified | raw training-split counts |
| label smoothing | verified | 0.1 |
| sqrt-inverse resampling | verified | `WeightedRandomSampler`, 1/sqrt(n) |
| SWA | verified with a caveat | computed from epoch 360 with `update_bn`; each seed kept SWA only if it beat the best epoch on validation; which won is unknown |
| Cosine LR warmup | verified | 5 warmup epochs, cosine, SWALR from 360 |

### Bullet 3

> Applied AutoAugment, Mixup, CutMix & Cutout with SGD & gradient clipping to maximize
> generalization under domain shift

All verified: AutoAugment CIFAR10 policy; MixUp and CutMix (alpha 1.0, on 50% of batches, 50/50);
Cutout 16; SGD lr 0.1 Nesterov momentum 0.9 wd 5e-4; clip norm 5.0. "Maximize" describes the goal,
not a measured optimum.

### Bullet 4

> Secured 1st Position among 27 teams with 0.9488 macro-F1 on a held-out perturbed test set via
> 30-view Test Time Augmentation

| term | status | detail |
|---|---|---|
| 1st among 27 teams | not in the files | keep a leaderboard screenshot |
| 0.9488 | not in the files | v2's 0.9348 is in the code; check public vs private for 0.9488 |
| held-out perturbed test set | verified | hidden labels; measured perturbations |
| 30-view TTA | verified with precision | 30 augmented views plus the clean image = 31 predictions per model |
| "via" | wording risk | implies TTA caused the score; it was part of the pipeline and also in v2 |

Safer wording if you edit the resume: "...0.9488 macro-F1 on a held-out perturbed test set, using a
3-model ensemble with 30-view Test Time Augmentation".

### The GitHub link

The resume's GitHub icon currently points to a repo that will be checked. `iteshm23/shiftguard10`
(built earlier in this chat) is wrong in several places: seeds 42/7/13 (real: 42/137/7), 300 epochs
with SWA from 225 (real: 450/360), "MixUp disabled during SWA" (not in the code), an ablation table
and per-class F1 numbers with no source, and a `submission.csv` from the March CCT/LDAM era. Its
"31-view TTA (1 clean + 30)" is actually correct. Replace it with this repo, or link the team repo.

---

## 23. Likely Interview Questions

Every question below has the same layout:

- **Answer**: what to say, in spoken form.
- **Testing**: what the interviewer is checking.
- **Follow-up**: the next question they are likely to ask, with its answer.
- **Avoid**: the weak answer.

"We" is used for team work. Change it to "I" only for parts you did yourself.

### 23.A Project

#### A1. Tell me about the ShiftGuard10 project.

- **Answer:** "It was a Kaggle competition in my EE708 course: classify 32 by 32 images into 10
  classes, scored by macro F1. The training data was 50 to 1 imbalanced, only 100 trucks against
  5,000 cats, and the test set was deliberately shifted: more noise, bigger occlusions and a very
  different class mix. Our team trained a WideResNet-28-10 from scratch, handled the imbalance with
  Balanced Softmax and a square-root sampler, used heavy augmentation, and averaged three models
  over 31 views per test image. Our single-model version scored 0.9348; the final ensemble scored
  0.9488 and placed first of 27 teams."
- **Testing:** can you give problem, approach and result in under a minute.
- **Follow-up:** "What was the hardest part?" -> "The rare classes. Truck had 100 training images
  but counts for 10% of the score, and the test set seems to have far more rare-class images than
  training did."
- **Avoid:** listing techniques without saying which problem each one solves.

#### A2. What exactly was your role?

- **Answer:** "It was a team entry. The final pipeline is in my teammate's repository. My part was
  [fill in exactly what you did]. We discussed every design decision together, which is why I can
  walk you through all of it."
- **Testing:** ownership and honesty. The commit history of the final repo is under the account
  Xavaitron (Pratyush Singh).
- **Follow-up:** "Which part would you say is yours?" -> name concrete pieces (for example the early
  baseline notebooks, specific experiments, Kaggle submissions, analysis, documentation).
- **Avoid:** "I built everything", unless it is true.

#### A3. What would you do differently?

- **Answer:** "Four things. First, a validation set with test-like noise and 10 by 10 occlusions,
  because ours looked like the training data. Second, noise augmentation, since about 75% of test
  images were noisy. Third, estimate the test class mix explicitly and correct for it, instead of only
  removing the training prior. Fourth, run proper ablations on a shorter schedule so each component's
  value is measured."
- **Testing:** reflection; whether you understand the weak points.
- **Follow-up:** "Which one first?" -> "The shifted validation set, because every other decision
  depends on measuring the right thing."
- **Avoid:** "use a bigger model".

#### A4. Why do you think you won?

- **Answer:** "I can't attribute it to one component because we didn't run clean ablations. My best
  explanation is the class mix: the test set appears tilted towards the classes that are rarest in
  training, and our setup removes the training prior and leans towards rare classes. Models trained
  without that would badly under-predict trucks and ships. Augmentation and averaging then add
  robustness to the noise and occlusion."
- **Testing:** humility plus reasoning.
- **Follow-up:** "How do you know the test was tilted?" -> see B3.
- **Avoid:** "because WideResNet is the best model" or "because of TTA".

### 23.B Dataset

#### B1. Describe the dataset.

- **Answer:** "29,400 training and 7,600 test images, all 32 by 32 RGB PNGs. Training classes: 5,000
  each of airplane, automobile, bird and cat; 4,000 each of deer and dog; 500 frog, 500 horse, 300 ship
  and 100 truck. So 50 to 1 between the largest and smallest class. Test labels were hidden."
- **Testing:** whether you know your data, with numbers.
- **Follow-up:** "What does that imbalance do to a normal classifier?" -> "It learns the prior. When
  unsure, it says the big class, so trucks become automobiles and horses become deer."
- **Avoid:** "around 30,000 roughly balanced images".

#### B2. Did you look at the images? What was the shift?

- **Answer:** "Yes. 5% of images in both sets have a solid grey square of exactly RGB 125, 123, 114,
  6 by 6 in training but 10 by 10 in test. About 21% of training images have colour noise, against
  about 75% of test images, and 38% of test images have a strong noise level that almost never appears
  in training. Average colour and contrast didn't change. There were no duplicate images."
- **Testing:** whether you went beyond the competition description.
- **Follow-up:** "Did you know this during the competition?" -> be honest: this detailed measurement
  was done afterwards.
- **Avoid:** "the test images were harder".

#### B3. Was the test set balanced?

- **Answer:** "Very probably not, and it seems to go the other way. Our strong submissions all predict
  about 500 to 550 images for each of the four biggest training classes and about 900 to 980 for each
  of the four rarest. With macro F1 around 0.93 to 0.95, the predicted counts can't be far from the true
  counts: for each class F1 is at most 2 times the smaller count over the sum. If the test were 760 per
  class, those files could score at most 0.88, but v2 scored 0.9348. So the rare training classes are
  probably the common test classes."
- **Testing:** reasoning from evidence without labels.
- **Follow-up:** "Isn't that circular?" -> "For checking whether we over-predict rare classes, yes, so
  I don't use it for that. But it does rule out a balanced test set."
- **Avoid:** stating the exact test counts as fact.

#### B4. How did you measure the noise?

- **Answer:** "A standard high-pass noise estimate, applied to the R minus G and B minus G colour
  differences. Natural texture like grass or fur is similar in all three channels and cancels out there,
  while added per-pixel noise doesn't. The histogram shows separate clean, mild and strong groups."
- **Testing:** whether the claim is grounded.
- **Follow-up:** "How strong is the noise?" -> "Roughly like Gaussian noise with a standard deviation of
  3.5 to 6 on a 0 to 255 scale, but that conversion is approximate."
- **Avoid:** "the test images looked noisy".

#### B5. Is this just CIFAR-10?

- **Answer:** "The class names, image size and colour statistics are consistent with CIFAR-10, but the
  competition never said so, so I don't claim it."
- **Testing:** care with claims.
- **Avoid:** "yes, it was CIFAR-10".

### 23.C Macro F1

#### C1. What is macro F1?

- **Answer:** "For each class, F1 is the harmonic mean of precision and recall. Macro F1 is the plain
  average over the 10 classes, so every class is worth 10% no matter how many images it has."
- **Testing:** definitions.
- **Follow-up:** "Precision and recall?" -> "Precision: of the images I called truck, how many were
  trucks. Recall: of the real trucks, how many I found."
- **Avoid:** confusing macro with weighted F1.

#### C2. Can accuracy be higher while macro F1 is lower?

- **Answer:** "Yes. Take 1,000 images per class and 100 trucks. A model that's perfect except it never
  predicts truck gets 98.9% accuracy but 0.895 macro F1. A model with 3% errors spread across all classes
  gets 97.0% accuracy but 0.960 macro F1."
- **Testing:** whether you understand why the metric was chosen.
- **Follow-up:** "Micro F1?" -> "In single-label multiclass, micro F1 equals accuracy."
- **Avoid:** "they're basically the same".

#### C3. Why a harmonic mean?

- **Answer:** "It's pulled towards the smaller value, so you need both precision and recall. Precision 1.0
  and recall 0.1 give F1 0.18, not the 0.55 an arithmetic mean would give."
- **Follow-up:** "What does zero_division=0 do in your code?" -> "If a class is never predicted,
  precision is 0 over 0; this setting counts it as 0, so a dead class costs a full 0.1."

#### C4. Why train with cross-entropy instead of optimising F1?

- **Answer:** "F1 is built from argmax counts over a whole dataset, so it has no useful gradient, and a
  batch of 128 has only two or three trucks, so batch-level F1 is very noisy. Cross-entropy is smooth and
  per-example, and it gives good probability estimates. Balanced Softmax then makes the argmax act as if
  classes were balanced, which suits a macro-averaged metric."
- **Testing:** loss vs metric.
- **Follow-up:** "Did you tune thresholds?" -> "No, plain argmax. With 5 validation trucks, per-class
  offsets would overfit. A single strength parameter on the prior correction would be the safer version."

### 23.D Class imbalance

#### D1. How did you handle the imbalance?

- **Answer:** "In two places. A sampler that draws each image with weight one over the square root of its
  class size, so trucks go from 0.34% to about 2% of what the model sees. And Balanced Softmax, which
  adds the log of each class's frequency to the logits during training and uses the raw logits for
  prediction, so the training prior is removed from the decision."
- **Testing:** whether you know the mechanism, not just the names.
- **Follow-up:** "Why both?" -> "The sampler changes what the network sees: rare classes appear in almost
  every batch and get more updates. Balanced Softmax changes the decision rule. They overlap, which I can
  explain."
- **Avoid:** "we oversampled the minority classes" and stopping there.

#### D2. Why square root instead of inverse frequency?

- **Answer:** "Inverse frequency, which our v2 used, makes every class 10% of the draws, so each of the 95
  training trucks is shown about 29 times per epoch and each head image only about 0.6 times. Square root
  is in between: about 6 times per truck image. Less repetition of very few images, and Balanced Softmax
  already handles the prior."
- **Follow-up:** "The code's docstring says sqrt is more aggressive." -> "That's wrong; it's milder. v3
  actually reduced the oversampling compared with v2."

#### D3. Why not class weights, focal loss, undersampling or SMOTE?

- **Answer:** "Class weights up to 50 times make one truck dominate a batch and do nothing when no truck
  is in the batch. Focal loss is about hard versus easy examples, not class priors. Undersampling would
  throw away 4,900 of 5,000 cats, which kills feature learning when training from scratch. SMOTE on raw
  pixels just blends images, which is roughly MixUp without being class-aware."

#### D4. Does oversampling add information? Does it change the dataset size?

- **Answer:** "No to both. Each epoch is still 27,930 draws; only the class mix inside them changes. The
  same 95 trucks are repeated; the new variety comes from augmentation, since every repeat gets a different
  crop, flip, AutoAugment policy, Cutout square and maybe MixUp or CutMix."
- **Follow-up:** "Why replacement=True?" -> "Without replacement each image can appear once per epoch,
  so trucks could never be drawn more than 95 times."

### 23.E Augmentation

#### E1. Walk me through your training augmentation.

- **Answer:** "Random crop with 4 pixels of grey padding, horizontal flip, AutoAugment with the CIFAR-10
  policy, then normalisation, then Cutout with a 16-pixel square. On half the batches, the whole batch is
  mixed with MixUp or CutMix, each with equal chance."
- **Follow-up:** "Why Cutout after normalisation?" -> "So the zeroed square equals the mean colour, a
  neutral grey, not black. That grey is exactly the colour of the dataset's own occlusion patches."

#### E2. Why horizontal flip but not vertical flip or large rotations?

- **Answer:** "Mirrored cars and animals still look like real photos. Upside-down horses and ships never
  appear in the test set, and sky-on-top is a useful cue. Rotations are kept small: AutoAugment's are
  about 7 degrees and TTA's at most 10."

#### E3. How does Cutout relate to partial occlusion?

- **Answer:** "It simulates it. Every training image has a grey square of up to 16 by 16 somewhere, so the
  model learns to classify from what's visible. The occluded test images have a 10 by 10 square of the same
  grey, so training occlusions are bigger on average than the test ones."
- **Follow-up:** "What if the square covers the whole object?" -> "Then that one view has a label with no
  evidence. It happens sometimes; the same image appears many times with the square elsewhere."

#### E4. Why use Cutout, MixUp and CutMix together?

- **Answer:** "They regularise in different ways. Cutout hides part of the image with plain grey. CutMix
  replaces part of it with another real image and mixes the labels by area. MixUp blends whole images and
  smooths the space between classes. It can be too much; the sign would be low clean training accuracy.
  We didn't ablate each one, and v2 already had all three."

#### E5. Why does CutMix recompute lambda from the actual box?

- **Answer:** "The box centre is random, so near the edges part of the box falls outside the image and is
  clipped. In a simulation of this code, the real pasted area differs from the sampled lambda by 0.2 on
  average. Recomputing makes the label match what's actually visible."

#### E6. What augmentation was missing?

- **Answer:** "Noise. About 75% of test images are noisy, and nothing in training adds noise. I'd add
  Gaussian noise at the measured strength, or use AugMix."

### 23.F Architecture

#### F1. What do 28 and 10 mean in WRN-28-10?

- **Answer:** "Depth 28 gives (28 minus 4) divided by 6, so 4 residual blocks in each of 3 groups. Widen
  factor 10 multiplies the base widths 16, 32, 64 to 160, 320, 640 channels."

#### F2. Walk me through the tensor shapes.

- **Answer:** "3 by 32 by 32 in. A 3 by 3 conv to 16 by 32 by 32. Group 1 to 160 by 32 by 32. Group 2 to 320
  by 16 by 16. Group 3 to 640 by 8 by 8. BatchNorm and ReLU, global average pooling to 640, then a linear
  layer to 10 logits. 36.48 million parameters, three quarters of them in group 3."

#### F3. Why does the shortcut sometimes become a 1x1 convolution?

- **Answer:** "When a block changes the number of channels or halves the size, the input can't be added to
  the output directly. A 1 by 1 conv with the same stride re-maps the channels and downsamples so the shapes
  match. That happens only in the first block of each group."

#### F4. Why global average pooling instead of flattening?

- **Answer:** "Flattening 640 by 8 by 8 gives 40,960 numbers and a 409,600-weight classifier tied to exact
  positions. Averaging gives 640 numbers saying how much of each feature is present anywhere, which is
  more stable when the object moves or is partly covered."

#### F5. Why WRN and not ResNet-50, a ViT or EfficientNet?

- **Answer:** "ResNet-50's stem is built for 224 by 224 and shrinks a 32 by 32 image to 8 by 8 before the
  first block. ViTs need far more data or pretraining, and pretraining wasn't allowed. EfficientNet is
  mostly used with pretrained weights at larger sizes. WRN-28-10 is built for 32 by 32 inputs and is one of
  the strongest from-scratch models there. We did try PyramidNet and a compact transformer before settling."

#### F6. Isn't 36 million parameters too many for 28,000 images?

- **Answer:** "Parameter count alone isn't a good measure of overfitting for CNNs. The model is heavily
  regularised and we checked validation every epoch. The real risk is the rare classes: each truck image is
  seen about 2,750 times over training, which is why the augmentation matters most there."

### 23.G Loss

#### G1. Explain Balanced Softmax.

- **Answer:** "In training, we add the log of each class's training frequency to its logit before the
  softmax. Softmax exponentiates, so that multiplies each class's score by its frequency, which is the
  prior in Bayes' rule. The network's raw logits then only need to carry the evidence from the image. At
  prediction time we use the raw logits, which behaves like a model trained on balanced classes."
- **Follow-up:** "Why log of the frequency?" -> "Because exp(z + log p) is p times exp(z). Adding p itself
  would mean nothing."
- **Follow-up:** "What if the classes were balanced?" -> "Log prior would be the same for every class, and
  it would reduce to plain cross-entropy."

#### G2. Aren't the sampler and Balanced Softmax double counting?

- **Answer:** "They overlap. Balanced Softmax assumes the batches follow the real class counts, but the
  sampler changes that, so in theory predictions lean further towards rare classes than a balanced test
  needs: about 7 to 1 for truck versus cat. Since our test set appears to have more rare-class images than
  common ones, that lean was probably helpful, but I can't prove the size was right."
- **Follow-up:** "How would you check?" -> "Try the prior correction at a few strengths on validation and
  compare per-class precision and recall; estimate the test prior from predictions and correct for it."

#### G3. What label smoothing did you use and why?

- **Answer:** "0.1, so the target is 0.91 for the true class and 0.01 for each other class. It stops the model
  pushing probabilities to exactly 1, which needs huge logits and encourages memorising odd images."
- **Follow-up:** "Any interaction with Balanced Softmax?" -> "Yes. The smoothing is applied after the prior
  shift, so for a perfectly learned cat image the raw probabilities are about 0.50 cat and 0.27 truck. Above
  about 0.17 smoothing, that image would flip to truck. So with Balanced Softmax on a 50 to 1 imbalance,
  smoothing has to stay small."

### 23.H Optimisation

#### H1. What were your optimiser settings?

- **Answer:** "SGD with learning rate 0.1, Nesterov momentum 0.9 and weight decay 5e-4, batch size 128,
  gradient norm clipped at 5. Five warmup epochs, then cosine decay over 450 epochs, with SWA from epoch
  360. Those base values are the WideResNet paper's CIFAR settings."
- **Follow-up:** "The final script says batch 512." -> "That default was changed after training, when the
  script was reworked for Kaggle inference. The checkpoints came from the 128 version."

#### H2. Why SGD and not Adam?

- **Answer:** "For CNNs trained from scratch on CIFAR-style data, SGD with momentum and a decaying learning
  rate is the reference setup and usually generalises at least as well as Adam. It also pairs naturally
  with SWA. Adam or AdamW would be my choice for transformers or quick prototypes."
- **Follow-up:** "What does momentum do?" -> "It keeps an exponential average of gradients over roughly the
  last 10 steps, so consistent directions speed up and noisy directions cancel."

#### H3. Is weight decay the same as L2 regularisation?

- **Answer:** "For SGD, yes: adding wd times the weights to the gradient is exactly the gradient of an L2
  penalty. For Adam they differ, because Adam rescales gradients per parameter, which is why AdamW exists."

#### H4. Why warmup and cosine?

- **Answer:** "Warmup avoids wild steps at full learning rate while BatchNorm statistics and momentum
  settle. Cosine keeps the rate high for a long time, which suits heavy augmentation, then brings it down
  smoothly, with no milestone epochs to tune."
- **Follow-up:** "What's the learning rate when SWA starts?" -> "About 0.01, then SWALR brings it to 0.005
  over 10 epochs and holds it."

#### H5. Why clip gradients at 5?

- **Answer:** "It's a loose safety cap for rare huge gradients from mixed batches and the prior-shifted
  loss. It isn't a tuned number. Properly, I'd log the gradient norm and set the cap above its normal range."

### 23.I SWA, ensemble and TTA

#### I1. What is SWA and how did you use it?

- **Answer:** "From epoch 360 to 449 we kept a running average of the weights, one snapshot per epoch, 90
  in total, at a small constant learning rate. After training, BatchNorm statistics were recomputed for the
  averaged weights with update_bn. Then each seed kept either the SWA model or its best single epoch,
  whichever scored higher on validation."
- **Follow-up:** "So did your final models use SWA?" -> "SWA was computed for all three seeds, and used
  wherever it beat the best epoch. I don't have the logs to say which won for each seed."

#### I2. How is SWA different from an ensemble?

- **Answer:** "SWA averages weights within one training run and gives one model. The ensemble averages the
  predictions of three separately trained models. They stack: each ensemble member may be an SWA model."

#### I3. How exactly was the ensemble combined?

- **Answer:** "Seeds 42, 137 and 7. Each model's softmax probabilities, after test-time augmentation, were
  averaged with equal weight, and the final label is the argmax."
- **Follow-up:** "Why three?" -> "Cost, about 20 hours per model, and most of an ensemble's gain comes from
  the first few members."

#### I4. How did your TTA work?

- **Answer:** "For each model, one prediction on the clean image plus 30 on random light transformations:
  a 4-pixel shifted crop, a random flip, 10% colour jitter and a rotation up to 10 degrees. The 31 softmax
  outputs are averaged. Over three models that's 93 forward passes per test image."
- **Follow-up:** "How much did TTA add?" -> "I don't have a measured number; v2 also had TTA, so the
  v2-to-v3 comparison doesn't isolate it. I'd measure it by running the checkpoints on validation with
  and without TTA."

### 23.J Validation and leakage

#### J1. How did you validate?

- **Answer:** "For each class we held out 5%, shuffled with a fixed seed, so 27,930 training and 1,470
  validation images, the same split for all three models. Validation macro F1 was checked every epoch."
- **Follow-up:** "Why 95/5 and not 80/20?" -> "Because of truck: with 100 images, keeping 95 for training
  mattered more than a big validation set."

#### J2. How reliable was the validation score?

- **Answer:** "Not very for the rare classes. There were only 5 trucks and 15 ships, so one truck mistake
  moves macro F1 by about 0.011. And validation came from the training distribution, not the shifted test
  distribution."
- **Follow-up:** "Could you have overfit it?" -> "Somewhat. Picking the best of 450 epochs on a noisy score
  biases that choice upwards."

#### J3. Any data leakage?

- **Answer:** "None that I found: no image appears in both training and validation, there are no duplicate
  images, normalisation uses fixed published constants rather than test statistics, and test labels were
  never available. The one mild risk is tuning on the public leaderboard, which overfits the public part."

### 23.K Engineering

#### K1. How did you deal with Kaggle's time limits?

- **Answer:** "The three models were trained with our script on a separate server, from random
  initialisation, and saved as checkpoint files. The Kaggle notebook loaded those self-trained checkpoints
  and ran inference. The script trains everything from scratch if the checkpoints are removed."
- **Follow-up:** "Isn't the rule that the notebook trains?" -> "The checkpoints were trained with the
  submitted code, with no pretrained weights or outside data, and the notebook header documents this. It
  was accepted and ranked."

#### K2. Is your result reproducible?

- **Answer:** "Training is seeded and cuDNN is set to deterministic. The inference-only run isn't seeded,
  so the random TTA views differ slightly between runs. That's a one-line fix. Even with seeds, GPU results
  can differ across hardware and library versions, so I'd promise equivalent, not identical, results."

#### K3. Find a bug in your code.

- **Answer:** "Two real ones. First, the per-class report passes target names without a labels argument,
  so the documented debug run crashes at its last epoch, because its small validation subset contains only
  airplanes. Second, main skips training whenever a checkpoint file exists without checking the completed
  flag, so an interrupted run is never resumed and its half-trained weights get used. I confirmed both by
  running the code."

---

## 24. Hard Technical Questions

#### 24.1 Derive Balanced Softmax from Bayes' rule.

- **Answer:** "Assume each class looks the same in training and test, so only the prior changes. Then
  p(y given x) is proportional to p(x given y) times p(y). Train softmax of z plus log pi to match the
  training posterior. Then exp(z_y) is proportional to p_train(y given x) divided by pi_y, which is the
  posterior under a uniform prior."
- **Follow-up:** "What if the images also change?" -> "Then p(x given y) isn't shared, and the correction
  only handles the prior part of the shift."

#### 24.2 Show the double correction mathematically.

- **Answer:** "With the sampler, labels in the batches follow q proportional to the square root of n, not
  pi. Then exp(z_y) is proportional to the balanced posterior times q_y over pi_y, which is n_y to the minus
  half. Truck versus cat is the square root of 4,750 over 95, about 7. v2, with full inverse sampling, had
  the full factor of 50."

#### 24.3 With label smoothing 0.1, what does a perfectly fitted cat image predict?

- **Answer:** "Adjusted probabilities are 0.91 cat and 0.01 each for the others. Raw ones are proportional to
  adjusted divided by prior: cat 0.91 over 0.17 is 5.35, truck 0.01 over 0.0034 is 2.94. After normalising:
  about 0.50 cat, 0.27 truck, 0.09 ship. Above about 0.17 smoothing, truck wins."

#### 24.4 How can predicted counts bound macro F1?

- **Answer:** "For each class, true positives can't exceed the smaller of predicted and true counts, so F1
  is at most 2 times the minimum over the sum. With our final predicted counts and a balanced test of 760 per
  class, the cap is 0.875, so a 0.9488 score rules out a balanced test."

#### 24.5 Why is cross-entropy a proper scoring rule, and why does it matter?

- **Answer:** "Its expected value is minimised only by the true probabilities, so training with it recovers
  the posterior. Any decision rule, argmax or thresholds, can then be applied on top."

#### 24.6 What is the expected Cutout area?

- **Answer:** "The centre is uniform, so near the edges the square is clipped. The side averages 14 pixels,
  the area about 196 pixels, 19% of the image; a full 16 by 16 only 28% of the time."

#### 24.7 How much of each head class is seen per epoch?

- **Answer:** "Each head image is drawn 0.86 times on average, so about 1 minus e to the minus 0.86, roughly
  58%, of head images appear in a given epoch."

#### 24.8 Averaging probabilities vs averaging logits?

- **Answer:** "Averaging probabilities is an arithmetic mean, a soft vote. Averaging logits then applying
  softmax is a normalised geometric mean, a product of experts, which punishes a class harder if any one
  model rules it out. We averaged probabilities."

#### 24.9 What would the SWA model's BatchNorm contain without update_bn?

- **Answer:** "In current PyTorch, update_parameters copies the live model's buffers, so the statistics
  would belong to the last snapshot, not the averaged weights. update_bn recomputes them with one pass over
  the training loader."

#### 24.10 Why is the best-epoch vs SWA comparison biased?

- **Answer:** "The best epoch is the maximum of 450 noisy validation scores, and the maximum of noisy
  estimates is biased upwards. So the SWA model has to beat an inflated number."

#### 24.11 How would you estimate the test class mix without labels?

- **Answer:** "Black-box shift estimation: take the validation confusion matrix, with entries P(predicted i
  given true j), and the predicted class distribution on test, and solve the linear system for the test
  prior. Or run EM on the predicted probabilities. Then add log of that prior to the raw logits."

#### 24.12 What is the receptive field of the last stage?

- **Answer:** "About 109 pixels: 3 from the stem, plus 16 in group 1, 30 in group 2 and 60 in group 3. So the
  last features can see the whole 32-pixel image."

#### 24.13 Derive the factor 2 in Kaiming initialisation.

- **Answer:** "For y = Wx where x comes from a ReLU, Var(y) is fan times Var(w) times E[x squared], and the
  ReLU halves E[x squared] relative to the pre-activation variance. Keeping variance constant needs Var(w)
  = 2 over fan."

#### 24.14 How many optimisation steps per model?

- **Answer:** "27,930 divided by 128 is 218 steps per epoch, dropping 26 images with drop_last, so about
  98,100 steps over 450 epochs."

#### 24.15 How would you put a confidence interval on validation macro F1?

- **Answer:** "Stratified bootstrap: resample the 1,470 validation images by class with replacement many
  times, recompute macro F1, and take percentiles. With 5 trucks the interval would be wide."

---

## 25. Cross Questions

These are chains: each step is the interviewer's next "why". The pattern for answering: agree with what
is true, give the reasoning and the evidence, and say how you would check.

### 25.1 Imbalance chain

1. **"How did you handle imbalance?"** "A square-root sampler plus Balanced Softmax."
2. **"Why both?"** "The sampler changes what the model sees; Balanced Softmax changes the decision rule."
3. **"Aren't they double counting?"** "In theory, yes: about 7 to 1 extra for truck versus cat."
4. **"Then isn't that wrong?"** "Relative to a balanced test, it over-corrects. Our test set seems to have
   about twice as many images per rare class as per common class, so the extra lean was in the right
   direction. I can't prove it was the right size."
5. **"How would you know?"** "Validate with several correction strengths and compare per-class precision
   and recall, and estimate the test prior explicitly."

### 25.2 SWA chain

1. **"Why SWA?"** "A flatter solution that isn't chosen on the noisy validation score."
2. **"Did your submission use it?"** "Per seed, only if it beat the best epoch on validation. I don't have
   the logs to say which won."
3. **"So you can't claim SWA helped?"** "Correct. It was part of training; I have no measured gain."
4. **"Why keep the comparison at all?"** "As a safety net, though it lets validation noise decide. With a
   bigger, shifted validation set the comparison would mean more."

### 25.3 Validation chain

1. **"How did you validate?"** "A class-wise 5% split."
2. **"Five trucks?"** "Yes; one truck mistake moves macro F1 by 0.011."
3. **"So model selection is noise?"** "Partly, for the rare classes. Big differences are real, small ones
   aren't."
4. **"Validation goes up and the leaderboard goes down: what happened?"** "Validation isn't shifted like
   the test, or we overfit the validation set by picking the best epoch. Fix: a shifted validation set and
   fewer leaderboard-driven decisions."

### 25.4 TTA chain

1. **"How many views?"** "30 augmented plus the clean image."
2. **"Why 30?"** "Diminishing returns: the random part of the average shrinks like 1 over the square root of
   the number of views."
3. **"How much did it help?"** "Not measured; v2 also had TTA."
4. **"What if TTA hurts?"** "Test each transform on validation, drop the 10-degree rotation, keep the clean
   view."

### 25.5 Workflow chain

1. **"Did your Kaggle notebook train the model?"** "It loaded checkpoints we trained with the same script
   from random initialisation, outside the session limit, and ran inference."
2. **"Isn't that against the rule?"** "No pretrained weights or outside data were used; the checkpoints come
   from the submitted code, and the notebook says so. The entry was accepted and ranked."

### 25.6 "What if" chain

- **"The test went back to the training class mix?"** "Our lean towards rare classes would over-predict them
  and accuracy would drop. Estimate the new prior and add log prior at prediction time."
- **"The noise were stronger?"** "Nothing in training adds noise, so it would degrade. Add noise augmentation
  and test on shifted validation."
- **"One class is never predicted?"** "Macro F1 loses 0.1. Check the prior correction, the label mapping and
  that class's logits."
- **"The model is overconfident?"** "Fit temperature scaling on validation and check a reliability diagram."
- **"Ensemble members are highly correlated?"** "The gain shrinks. Diversify the architecture, augmentation or
  data split."

### 25.7 Adversarial one-liners

- **"You just threw every trick at it."** "Each targets a specific problem in this data. I can't prove each was
  necessary without ablations, and v2 with most of them already scored 0.9348."
- **"60 GPU hours for 10 classes is wasteful."** "For a competition with no inference budget, it was worth it.
  In production I'd distil the ensemble into one model and drop most of the TTA."
- **"Your docstring says sqrt is more aggressive."** "It's wrong. Square root is milder; v2 used full inverse
  frequency."
- **"Your debug mode crashes."** "Yes: the report needs a labels argument when fewer than 10 classes are
  present."

---

## 26. Debugging Questions

#### 26.1 Training loss becomes NaN early. What do you check?

"Whether the learning rate is too high at the start (warmup should prevent it), a corrupt image, and log
of zero in the loss (prevented by the 1e-12). I'd log the gradient norm; clipping caps it at 5."

#### 26.2 One class is never predicted.

"Check the class counts and log prior passed to the loss, the label mapping, that prediction uses raw logits,
and that class's logit distribution on validation."

#### 26.3 Training accuracy is high but validation F1 is poor.

"Look at the per-class report. If it's the rare classes, they're being memorised: a milder sampler or
stronger augmentation on them."

#### 26.4 Validation improves but the leaderboard drops.

"Validation isn't shifted like the test, or the best-epoch choice overfits validation. Build a shifted
validation set and compare predicted test class counts across versions."

#### 26.5 TTA makes validation worse.

"Test each transform separately, reduce the rotation, keep the clean view, and check that the normalisation
matches."

#### 26.6 The SWA model is worse than the best epoch.

"Was update_bn run? It is in our code. Is the SWA learning rate too high, or did SWA start too early? Is the
gap within validation noise of about 0.015?"

#### 26.7 The loss jumps after resuming.

"Check that the optimiser state, the scheduler state and the SWA objects were restored, and that the epoch
count didn't change. In our final file, main never reaches the resume path at all, which is a bug."

#### 26.8 A checkpoint won't load.

"A corrupted write, a 'module.' prefix from DataParallel, or the old plain state_dict versus the new
dictionary format. Our main handles both formats."

#### 26.9 After loading a model, every prediction is the same class.

"Forgot model.eval(), wrong normalisation at inference, or a wrong or half-trained checkpoint, which is
exactly what our main would silently use after an interrupted run."

#### 26.10 The debug run crashes at its last epoch.

"classification_report is given 10 target names but no labels argument, and the debug validation subset
contains only airplanes. Pass labels equal to range 10."

#### 26.11 Two runs of the same notebook give slightly different submissions.

"Inference isn't seeded, so the random TTA views differ. Seed before inference."

#### 26.12 Out of memory at batch 512.

"WRN-28-10 stores about 22 MB of activations per image for backward, so 512 needs about 12 GB. Drop to 128 or
use mixed precision. The team tried mixed precision and reverted it."

---

## 27. ML Concepts Connected to the Project

| concept | where it appears here | takeaway |
|---|---|---|
| bias and variance | big low-bias model; SWA, ensemble, TTA cut variance | large model + regularisation + averaging |
| overfitting | 95 trucks seen about 2,750 times each | augmentation; watch tail recall on validation |
| regularisation | dropout, weight decay, smoothing, augmentation, mixing | many mild regularisers |
| class imbalance | 50:1 | fix exposure (sampler) and the decision rule (loss) |
| label (prior) shift | 50:1 train -> tail-heavy test | prior correction; ideally estimate the target prior |
| covariate shift | noise, occlusion size | augmentation; shifted validation |
| concept drift | none here | would need relabelling or retraining |
| calibration | smoothing, MixUp, Balanced Softmax all change confidence | irrelevant for argmax, important if probabilities are used |
| decision boundaries | prior pushes boundaries into tail space | adding log priors moves them back |
| model variance | seeds give different models | ensembles reduce it; correlation limits the gain |
| optimisation | warmup, cosine, Nesterov, clipping | stable start, long middle, settle at the end |
| flat minima | SWA | averaging late weights moves towards flatter regions |
| validation bias | best of 450 noisy scores | the max of noisy estimates is biased upwards |
| leakage | none found | check overlaps, duplicates and statistics explicitly |
| ablation | one bundled before/after | one change at a time, several seeds |
| reproducibility | seeds, deterministic cuDNN, unseeded inference | equivalent, not identical |
| hyperparameter choice | WRN-paper defaults, not tuned | say so |

---

## 28. AmEx Transfer Questions

**This is a conceptual transfer. ShiftGuard10 is an image project, not a payments or credit
project. Never say otherwise.**

| ShiftGuard10 | payments / credit analogue |
|---|---|
| 100 trucks in 29,400 images | fraud is a tiny fraction of transactions |
| macro F1, not accuracy | precision/recall, PR-AUC, cost-weighted metrics, not accuracy |
| Balanced Softmax / log-prior correction | correcting scores when the training base rate (often resampled) differs from the live rate |
| sampler + loss double correction | undersampling non-fraud and forgetting to correct the score's base rate, or correcting twice |
| reversed test class mix | fraud rate or customer mix changes between development and deployment |
| shifted validation | out-of-time validation: train on earlier months, validate on later ones |
| 5 trucks in validation | very few confirmed frauds in a validation window: noisy metrics, use intervals |
| per-class thresholds | approve / decline / review cut-offs set by business cost |
| calibration | probabilities feeding expected loss, pricing or limits must be calibrated |
| ensembles | common in risk if latency and explainability allow |
| reproducibility, checkpoints | model risk management: rebuildable, versioned models |

**AD-1. How would you handle extreme imbalance in fraud detection?** "Choose the metric from the
business cost first, not accuracy. Then either reweight or resample, and correct the predicted
probabilities back to the live base rate. In my image project, macro F1 and a 50:1 imbalance
needed both a sampler and a prior-corrected loss, and I learned that two corrections can stack."

**AD-2. False positives vs false negatives?** "Put a cost on each (fraud loss vs a declined good
customer and the friction that follows) and pick the threshold that minimises expected cost on data
that looks like live traffic. Revisit it when costs or fraud rates change."

**AD-3. The model was trained on 10% fraud (after undersampling), live fraud is 0.2%. What happens?**
"Scores are inflated. Multiply the predicted odds by (live fraud odds / training fraud odds), here
(0.002/0.998) / (0.10/0.90), or add the log of that ratio to the logit. That's the same log-prior
correction Balanced Softmax makes."

**AD-4. What if the live base rate is unknown?** "Estimate it from the model's predictions on live
data, corrected with the confusion matrix from validation (black-box shift estimation), or with EM on
the predicted probabilities. In ShiftGuard10 the test class mix turned out very different from
training, about twice as many rare-class images as common ones, and our strong submissions' predicted
counts were the evidence."

**AD-5. How would you detect drift?** "Monitor input distributions (population stability index), the
score distribution, approval rates, the predicted class mix, and realised precision and recall once
labels arrive. In the image project I found the shift by comparing simple statistics of train and test
images: noise level, occlusion size, and predicted class counts."

**AD-6. Labels arrive late (chargebacks). How do you validate?** Out-of-time splits with a
label-maturity window; proxy monitoring until labels mature.

**AD-7. Why does calibration matter here but not in your image project?** "There I only took the
argmax. In risk, the probability itself drives expected loss, limits and pricing."

**AD-8. Deep model or gradient boosting for tabular fraud?** "Gradient-boosted trees are the usual
strong baseline and easier to explain. The image project needed a CNN because the input was pixels;
what transfers is the evaluation, imbalance, shift and validation thinking."

**AD-9. When would you retrain?** On measured drift or decay, or on a schedule backed by monitoring,
comparing the new model with the current one out of time before switching.

**AD-10. Biggest lesson for AmEx?** "Validate on data that looks like where the model will be used,
and check the class mix you'll actually face. My validation looked like training; the test set had a
completely different class mix and more noise."

---

## 29. What I Should Say

### 29.1 Ownership line (say it early if asked about your role)

"This was a team competition. The final pipeline lives in my teammate's repository. My part was
[fill in exactly: e.g. baseline notebooks, experiments, analysis, Kaggle submission, documentation].
I know the full pipeline in detail because we worked through every decision together."

Fill in the bracket truthfully before the interview. Your files show your own Kaggle notebooks (a
simple CNN at 0.48 validation macro F1, and a WRN run with RandAugment and MixUp). If you can't
honestly say you wrote the final code, don't.

### 29.2 20 seconds

"ShiftGuard10 was a Kaggle competition in my EE708 course: 10-class, 32 by 32 images, scored by
macro F1, with a test set that was deliberately noisier, more occluded and very differently
distributed across classes than training, where trucks were only 100 of 29,400 images. Our team
trained a WideResNet-28-10 from scratch with Balanced Softmax and a square-root sampler for the
imbalance, heavy augmentation, and a three-model ensemble with test-time augmentation. We finished
first of 27 teams with 0.9488 macro F1."

### 29.3 45 seconds

"Three things made it hard. The training data was 50 to 1 imbalanced, with 100 trucks against 5,000
cats, and macro F1 counts every class equally. The test set was shifted: bigger grey occlusion
patches, much more noise, and, as far as we can tell, far more rare-class images than training. And
no pretrained models were allowed. The model was a WideResNet-28-10. For imbalance we used Balanced
Softmax, which adds the log class frequency to the logits in training and uses raw logits for
prediction, plus a sampler with weights one over the square root of the class size. Augmentation
was AutoAugment, Cutout, MixUp and CutMix. Each model trained for 450 epochs with weight averaging
over the last 90. We ensembled three seeds and averaged 31 views per test image. Our single-model
version scored 0.9348; the final one 0.9488, first of 27."

### 29.4 90 seconds

Use the 45-second version and add:

- "Our first single WRN, v2, already had Balanced Softmax and the augmentations and scored 0.9348. For
  v3 we went to three seeds, a milder square-root sampler instead of full inverse frequency, a 95/5
  split to keep more rare images for training, 450 epochs with SWA from 360, and 30 TTA views. We
  changed those together, so I can't say which one gave the extra 0.014."
- "Each seed kept either its best validation epoch or its SWA average, whichever scored higher on
  validation, and the three models' softmax outputs were averaged."
- "If I did it again, I'd build a validation set with test-like noise and occlusion, add noise
  augmentation, and estimate the test class mix explicitly instead of only removing the training prior."

### 29.5 Two minutes: pick one add-on

- **Technical:** the sampler + Balanced Softmax overlap (x7 for truck vs cat in theory) and the label
  smoothing leak (a perfectly fitted cat image gives 0.50 cat, 0.27 truck), and why on this tail-heavy
  test set that lean probably helped.
- **Data:** the measured shift: 6x6 vs 10x10 grey patches on exactly 5% of images, noise on 21% vs 75%
  of images, and the predicted-count argument that the test can't be balanced.
- **Engineering:** training ran off-Kaggle; the Kaggle notebook loaded the checkpoints and ran 93 forward
  passes per image; the final script's resume logic and seeding have bugs I'd fix.

### 29.6 Five-minute walkthrough (outline)

1. Problem and rules (40 s). 2. Data and shift, including the class-mix evidence (50 s).
3. History: CCT/LDAM and contrastive attempts, PyramidNet, then single WRN v2 at 0.9348 (30 s).
4. Imbalance: sampler numbers, Balanced Softmax, the overlap (60 s). 5. Model and training: WRN
shapes, SGD settings, warmup + cosine, augmentation stack, SWA (60 s). 6. Inference: seeds, 31 views,
probability averaging (20 s). 7. Workflow: server training, Kaggle inference (20 s). 8. Result and
honest limits (20 s).

### 29.7 Phrases to use

- "The final configuration scored 0.9488." (not "X gave us Y")
- "That's from our v2 to v3 change, which bundled six changes."
- "I believe the test set was tilted towards the rare classes; here's the evidence."
- "30 augmented views plus the original."
- "Each seed kept SWA only if it beat the best epoch on validation."

### 29.8 Never say

- "TTA / SWA / the ensemble / Balanced Softmax improved the score by X."
- "The test set was balanced."
- "The data is CIFAR-10."
- "Square-root sampling is more aggressive than inverse frequency."
- "We trained the models inside the Kaggle notebook across sessions with resume." (the winning run predates the resume code)
- "We used SWA weights for every model." (unknown per seed)
- "Batch size 512" for the trained models (training version used 128).
- "I wrote all the code" unless that is true.
- "30 teams", "31-view TTA" as a resume number, "seeds 42, 7, 13", "300 epochs" (all from the old public repo).

---

## 30. Final Cheat Sheet

**Competition.** EE708 Kaggle, 22 Jan to 11 Apr 2026; 10 classes; 32x32 RGB; macro F1; no pretrained
weights, no outside data; public/private leaderboards; team ranking.

**Data.** 29,400 train / 7,600 test. 5,000 x 4 (airplane, automobile, bird, cat), 4,000 x 2 (deer, dog),
500 frog, 500 horse, 300 ship, 100 truck. No duplicates.

**Shift.** Grey (125, 123, 114) patch on 5%: 6x6 train, 10x10 test. Noise: train 79/20/1% clean/mild/
strong, test 25/37/38%. Colour statistics unchanged. Test class mix very probably tail-heavy (about
500 per big training class, about 950 per rare one).

**Split.** Class-wise 95/5, split seed 42: 27,930 / 1,470; validation has 5 trucks, 15 ships.

**Model.** WRN-28-10, pre-activation blocks, 4 per group, 16 -> 160 -> 320 -> 640 channels, 32 -> 16 -> 8
pixels, dropout 0.3, Kaiming normal fan_out, 36,479,194 params, 5.24 GMACs per image.

**Augmentation.** RandomCrop pad 4 fill 128, flip, AutoAugment CIFAR10, Normalize (CIFAR), Cutout 16.
MixUp or CutMix (alpha 1, 50/50) on 50% of batches.

**Imbalance.** Sampler weight 1/sqrt(n): truck 0.34% -> 2.08% of draws, about 6 times per epoch (v2 used
1/n: 10%, about 29 times). Balanced Softmax with raw training counts; cat-truck log-prior gap 3.91.

**Loss.** Balanced Softmax + label smoothing 0.1 (targets 0.91 / 0.01).

**Optimiser.** SGD lr 0.1, momentum 0.9, Nesterov, wd 5e-4, clip 5.0, batch 128 (final file default 512).

**Schedule.** Warmup 5 epochs, cosine, 450 epochs, SWA from 360 (90 snapshots), SWALR to 0.005 over 10
epochs, `update_bn`, SWA kept only if better on validation.

**Ensemble and TTA.** Seeds 42, 137, 7; softmax averaged equally; per model 1 clean + 30 random views
(crop, flip, jitter 0.1, rotation 10 degrees); 93 passes per image.

**Results.** v2 0.9348 (in code). Final 0.9488, 1st of 27 (resume). First own baseline 0.4812 validation.

**History.** 12-13 Mar: CCT/WRN with LDAM-DRW, supervised contrastive attempts. 23 Mar: PyramidNet-272 +
ShakeDrop + WRN script. 24 Mar: single WRN v2 (0.9348). 24-25 Mar: v3 with 2 then 3 seeds. 27 Mar:
3-seed results. 28 Mar: checkpointing, inference-only, server-train/Kaggle-inference workflow.
29 Mar: final submission notebook.

**Three insights to offer.** (1) Predicted class counts prove the test isn't balanced. (2) Sampler +
Balanced Softmax + smoothing lean towards rare classes, which fit this tail-heavy test. (3) Cutout's
mean grey is the dataset's occluder colour, and bigger on average than the test patches.

---

## Appendix A: What the complete repository changed

| topic | previous analysis said | complete repository shows | correct interview understanding |
|---|---|---|---|
| code completeness | file cut off at the checkpoint dict; inference unknown | full `notebook.py`, 738 lines; first 499 identical to the partial copy | everything earlier verified still stands |
| TTA | 30 random views; clean view unknown | 1 clean + 30 random, softmax averaged, divided by 31 | "30 augmented views plus the original" |
| ensemble averaging | unknown | softmax probabilities, equal weights | verified |
| seeds | unknown (placeholder 42/43/44) | 42, 137, 7 | verified |
| update_bn | unknown | called on the training loader | verified |
| final weights per seed | unknown (rebuild assumed SWA) | better of best epoch and SWA by validation F1 | which won per seed is unknown |
| batch size | unknown (placeholder 128) | final default 512; training version (`18a6d54`) default 128 | trained at 128 (default of that version) |
| lr, wd, swa_lr | unknown (placeholders 0.1, 5e-4, 0.005) | 0.1, 5e-4, 0.005 | verified |
| label smoothing, mix_prob | defaults 0.1, 0.5 assumed | 0.1, 0.5 passed | verified |
| class_counts | inferred raw counts | raw counts of the 27,930 split | verified |
| DataParallel / 2x T4 | unknown | none in final file (added and removed 28 Mar); `--gpu` picks one GPU | one GPU per process |
| checkpoint/resume story | "trained across Kaggle sessions with resume" | winning run used the 25 Mar version without per-epoch checkpoints; resume added 28 Mar; resume unreachable from final `main()` | don't claim resume trained the models |
| where training ran | Kaggle (header) | commit: "server train + Kaggle inference workflow"; checkpoints from a Kaggle Model | trained off-Kaggle, inference on Kaggle |
| test class mix | probably balanced | three strong submissions predict a tail-heavy mix; balanced test caps them below their scores | tail-heavy test (inference) |
| sampler + loss overlap | weakness (over-correction) | test is tail-heavy | overlap probably helped here; still unproven |
| earlier attempts | March repo, authorship unknown | earlier commits of the same repo (abb0d29); also SupCon, PyramidNet | part of the project history |
| scores | only 0.9488 (resume) | v2 scored 0.9348 (code header) | one real before/after |
| sampler docstring | "more aggressive" is wrong | v2 used full inverse; v3's sqrt is milder | still wrong; now explained |
| old public repo | "31-view TTA" contradicts resume | code does 1 + 30 | that line was right; other errors stand |
| ownership | not known | all commits by Xavaitron / Pratyush Singh | describe your role precisely |
| bugs | `classification_report` crash in debug | also crashes the documented `--debug` run; `main()` ignores `completed`; inference unseeded | verified by running |

## Appendix B: Project timeline from the git history (`Xavaitron/Shiftguard10`)

| date | commit message (short) | meaning |
|---|---|---|
| 12 Mar | initial commit, dataset push, pipeline created | modular `src/` pipeline |
| 13 Mar | "new WLAD loss", "second run", "3rd run" (abb0d29) | CCT and WRN, LDAM-DRW, TrivialAugment, 300 epochs |
| 13 Mar | "new CL approach", "contrastive learning attempt" | supervised contrastive learning tried |
| 23 Mar | "rewrite: all-in-one notebook.py with PyramidNet+ShakeDrop, Balanced Softmax, multi-seed ensemble, TTA" | first single-file pipeline |
| 24 Mar | "simplify: single WRN-28-10 training run" | v2 (later noted as 0.9348) |
| 24 Mar | "added submission" | v2 predictions |
| 24 Mar | "harsher parameters for training" | v3 draft: 2 seeds, sqrt sampler, 95/5, 400 epochs, 30 TTA |
| 25 Mar | "another one" | 2-seed predictions |
| 25 Mar | "3 model ensemble anf 450 epochs" (18a6d54) | the version that trained the final models (batch 128) |
| 27 Mar | "3 ensemble results" | final predictions (the repo's `submission.csv`) |
| 27-28 Mar | Kaggle compatibility, skip-if-checkpoint, per-epoch checkpoints, DataParallel + batch 256, AMP (reverted), inference-only, `--gpu`, DataParallel removed, batch 512 | preparing the Kaggle inference notebook |
| 29 Mar | "Final Submission", "updated readme", "added comments", "final for sure" | checkpoint dir points to the Kaggle Model; header notes added |

## Appendix C: Source-of-truth audit

**Verified.** Everything in `notebook.py` (all values in section 5); the git history above; dataset
counts, duplicates, occlusion and noise measurements; competition rules; v2's 0.9348 (as written in the
code); your own earlier notebook results; the bugs in section 18.14 (run on CPU).

**Strong inferences.** Training batch size 128 (default of the training version; command line not
logged). The repo's `submission.csv` is the final 3-seed output. The test class mix is tail-heavy. The
v2 submission in the history is the one that scored 0.9348.

**Unknown.** Which of SWA / best epoch each seed used; per-seed validation scores; whether 0.9488 is the
public or private score; the 27-team count; team composition and roles; the exact test class counts;
the hardware of the training server.

**Resume claims to be careful about.** "Built" (ownership); "via 30-view TTA" (causal wording, and 31
predictions); "SWA" (selection caveat); "1st of 27", "0.9488" (keep proof).
