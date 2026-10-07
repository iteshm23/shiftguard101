# Data

The competition data is not included in this repo (competition rules: no redistribution).

Download it from the Kaggle competition page and unzip it here, so that the folder looks like:

```
data/
  shift-guard-10-robust-image-classification-challenge/
    classes.txt
    train_labels.csv
    sample_submission.csv
    train_images/   29,400 PNGs, 32x32 RGB
    test_images/    7,600 PNGs, 32x32 RGB
```

Or point the code at another folder:

```
export SG10_DATA_ROOT=/path/to/shift-guard-10-robust-image-classification-challenge
```

## Schema

`train_labels.csv`

| column | type | example |
|---|---|---|
| id | 6-digit zero-padded string | 000001 |
| label | one of the 10 class names | cat |

`sample_submission.csv` (also the submission format)

| column | type | example |
|---|---|---|
| id | 6-digit zero-padded string | 000001 |
| label | one of the 10 class names | airplane |

Train and test ids both start at `000001`; they refer to files in different folders.
