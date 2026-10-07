# Evaluation: macro F1

## The pieces

For one class c, look at every image and sort it into one of four boxes:

| | model says c | model says not c |
|---|---|---|
| truly c | TP (true positive) | FN (false negative) |
| truly not c | FP (false positive) | TN (true negative) |

- Precision(c) = TP / (TP + FP): of the images I called c, how many really were c.
- Recall(c) = TP / (TP + FN): of the real c images, how many I found.
- F1(c) = 2 * P * R / (P + R), the harmonic mean of the two.

Macro F1 = (F1(airplane) + F1(automobile) + ... + F1(truck)) / 10.

Every class counts for exactly one tenth, whether it has 5,000 training images or 100.

## Why a harmonic mean

The harmonic mean is pulled towards the smaller number. Precision 1.0 and recall 0.1
gives an arithmetic mean of 0.55 but an F1 of 0.18. A class only gets a good F1 if both
precision and recall are good, so you cannot win by always predicting a class (high
recall, low precision) or by predicting it only when certain (high precision, low recall).

## Micro, weighted, accuracy

- Micro F1 pools TP, FP and FN over all classes first. In single-label multiclass
  problems micro F1 equals accuracy.
- Weighted F1 averages per-class F1 weighted by how many true images each class has, so
  big classes dominate, much like accuracy.
- Macro F1 gives every class the same weight. It is the only one of these that cares
  about the truck class as much as the cat class.

## When accuracy and macro F1 disagree (numbers computed with scikit-learn)

Training-like class mix (5,000 cats ... 100 trucks), a model that is perfect except it
calls every truck an automobile:

| metric | value |
|---|---|
| accuracy | 0.9966 |
| micro F1 | 0.9966 |
| weighted F1 | 0.9949 |
| macro F1 | 0.8990 |

One class with 0.3% of the data is completely missed. Accuracy hardly moves; macro F1
loses a full tenth.

Two models on a set with 1,000 images per class except 100 trucks:

| model | behaviour | accuracy | macro F1 |
|---|---|---|---|
| A | perfect on 9 classes, never predicts truck | 0.989 | 0.895 |
| B | 3% errors spread over every class | 0.970 | 0.960 |

A has higher accuracy but lower macro F1. This is exactly the failure macro F1 is there to
punish.

## zero_division=0 in the code

```python
f1_score(targets, preds, average="macro", zero_division=0)
```

If the model never predicts some class, TP + FP = 0 and precision is 0/0. scikit-learn
would warn and has to pick a value; `zero_division=0` says "count it as 0, quietly".
So a class that is never predicted gets F1 = 0 and costs a tenth of the macro score. That
is the honest choice: it does not hide a dead class.

## Why train with cross-entropy and not F1

- F1 depends on argmax decisions (counts), so it has zero gradient almost everywhere.
- F1 is defined over the whole dataset, not per image. A mini-batch of 128 images with
  about 2 trucks gives a very noisy estimate of truck F1.
- Cross-entropy is a proper scoring rule: it is minimised when the predicted
  probabilities match the true ones. Good probabilities plus a sensible decision rule
  give good F1.

In this project the decision rule is argmax over the raw logits. Balanced Softmax makes
those raw logits behave like a model trained on balanced classes, which is what a
macro-averaged metric rewards. No per-class thresholds were tuned.

## Thresholds in multiclass problems

There is no single 0.5 threshold. The equivalent knob is a per-class offset added to the
logits before argmax (or per-class weights on the probabilities). You could tune 10
offsets on validation to maximise macro F1. This project did not, and with only 5 truck
images in validation it would have been easy to overfit those offsets. A safer version is
one shared parameter tau in `logits - tau * log(prior)` tuned on validation.

## How noisy the validation score is

The validation split has 250 images for each big class but only 25 frog, 25 horse,
15 ship and 5 truck images. Getting one truck wrong (everything else perfect) gives:

- accuracy 0.9993
- macro F1 0.9887

One image moves macro F1 by about 0.011. In a simple simulation where each validation
image is wrong 5% of the time, validation macro F1 had a standard deviation of about
0.015 from sampling alone, against about 0.006 for accuracy.
