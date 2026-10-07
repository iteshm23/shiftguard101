# ShiftGuard10 competition overview

Source: the Kaggle competition page "ShiftGuard10: Robust Image Classification Challenge"
(community prediction competition, hosted by Aditya Raj for the EE708 course, IIT Kanpur).
Start 22 Jan 2026, close 11 Apr 2026.

## Task

Classify 32x32 RGB images into one of 10 classes:

airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

The page says the dataset is "intentionally constructed to include harder-than-normal
samples and a distribution shift between training and test". The short description:
"test images may include subtle perturbations and partial occlusions".

The page does not say which perturbations were used, how strong they are, or how the
test classes are distributed. What the files actually show is in
[../docs/dataset_analysis.md](../docs/dataset_analysis.md).

## Files given

| File | What it is |
|---|---|
| `train_images/` | 29,400 PNG files, 32x32 RGB |
| `train_labels.csv` | `id,label` for every training image |
| `test_images/` | 7,600 PNG files, 32x32 RGB, no labels |
| `sample_submission.csv` | 7,600 rows of `id,label`, every label set to `airplane` |
| `classes.txt` | the 10 class names, one per line |

Ids are 6-digit zero-padded strings (`000001`). Train ids and test ids both start at
`000001`, so the same id string means a different image in each folder.

## Rules (from the page)

1. Competition data only, no redistribution.
2. No pretrained models: no ImageNet weights, no self-supervised checkpoints, no external checkpoints.
3. No external datasets or external data of any kind.
4. Notebook-only submission. The Kaggle notebook must train and run inference, and the
   whole thing must be reproducible from scratch using only competition data and
   standard Kaggle packages.

## Metric

Macro F1 over the 10 classes. F1 is computed for each class as the harmonic mean of
precision and recall, then the 10 values are averaged with equal weight. Details and
worked examples: [evaluation.md](evaluation.md).

## Leaderboard

- Public leaderboard: scored on a "public" part of the test set during the competition.
- Private leaderboard: scored on the remaining "private" part, used for the final ranking.

The page does not say how the 7,600 test images are split between the two.

## Submission format

CSV with a header and exactly two columns, `id` and `label`, where label is one of the
10 class names. The page's example shows 5-digit ids (`00002`), but the real
`sample_submission.csv` uses 6-digit ids, and that is what the code writes.

```
id,label
000001,cat
000002,automobile
...
```

## How the final submission was produced

From the team repository's history: the three models were trained with the 25 March version of
the script; a later commit is titled "server train + Kaggle inference workflow". The final
Kaggle notebook (`notebook.py`) points its checkpoint folder at a Kaggle Model input, finds the
three self-trained checkpoints, skips training and runs inference. No pretrained weights or
outside data were used; the script trains from scratch if the checkpoints are removed. The
notebook header mentions about 60 hours of training on 2x T4 and a 30-hour Kaggle limit.
