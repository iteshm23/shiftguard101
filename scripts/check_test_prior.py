"""What the submissions say about the hidden test class mix.

    python scripts/check_test_prior.py

For each submission file, prints the predicted count per class and the highest macro F1
that file could possibly score under a few guesses of the true test counts. For every
class, F1 <= 2 * min(predicted, true) / (predicted + true), so a file whose counts are
far from the truth cannot have a high macro F1.

Post-competition analysis, not part of the submitted solution.
"""

import csv
import os
from collections import Counter

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]
FILES = {
    "v2 (scored 0.9348 per the v3 code header)": "history/submissions/v2_1b98891.csv",
    "v3 2-seed": "history/submissions/v3_2seed_b90dba0.csv",
    "v3 3-seed (final)": "submission.csv",
}
GUESSES = {
    "balanced, 760 each": [760] * 10,
    "same mix as training (50:1)": [round(7600 * n / 29400) for n in
                                    [5000] * 4 + [4000] * 2 + [500, 500, 300, 100]],
    "500 x 4, 800 x 2, 1000 x 4": [500] * 4 + [800] * 2 + [1000] * 4,
}


def counts(path):
    with open(path) as f:
        c = Counter(row["label"] for row in csv.DictReader(f))
    return [c.get(k, 0) for k in CLASSES]


def max_macro_f1(pred, true):
    return sum(2 * min(p, t) / (p + t) for p, t in zip(pred, true)) / len(pred)


def main():
    rows = {name: counts(path) for name, path in FILES.items() if os.path.isfile(path)}
    print(f"{'class':12s}" + "".join(f"{n[:18]:>20s}" for n in rows))
    for i, c in enumerate(CLASSES):
        print(f"{c:12s}" + "".join(f"{v[i]:>20d}" for v in rows.values()))
    print("\nHighest possible macro F1 for each file, under each guess of the true counts:")
    for g, true in GUESSES.items():
        line = "  ".join(f"{n[:18]}: {max_macro_f1(v, true):.3f}" for n, v in rows.items())
        print(f"  {g:28s} {line}")


if __name__ == "__main__":
    main()
