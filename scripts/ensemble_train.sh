#!/bin/bash
# Train all 3 ensemble members in sequence.
# Each member is identical in architecture; only the random seed differs.
# Seeds are chosen to be well-separated in seed space.

set -e

CONFIG="configs/default.yaml"

echo "============================================"
echo "  ShiftGuard10 — 3-Seed Ensemble Training"
echo "============================================"

for SEED in 42 7 13; do
    echo ""
    echo ">>> Training seed $SEED ..."
    python src/train.py --config $CONFIG --seed $SEED
    echo ">>> Seed $SEED done."
done

echo ""
echo "All 3 seeds trained. Checkpoints in checkpoints/"
echo "Best models: best_wrn_s42.pth, best_wrn_s7.pth, best_wrn_s13.pth"
