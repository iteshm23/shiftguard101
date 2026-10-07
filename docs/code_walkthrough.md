# Code walkthrough

> Section numbers refer to [interview_prep.md](../interview_prep.md), which has the full version.

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
