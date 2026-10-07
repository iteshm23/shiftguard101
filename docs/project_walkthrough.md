# Project walkthrough

> Section numbers refer to [interview_prep.md](../interview_prep.md), which has the full version.

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
