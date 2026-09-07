#!/bin/bash
# 3-seed ensemble inference with 31-view TTA.
# Loads the best checkpoint from each seed, averages class probabilities,
# then argmax → final label.

set -e

echo "============================================"
echo "  ShiftGuard10 — 3-Seed Ensemble Inference"
echo "  31-view Test Time Augmentation"
echo "============================================"

python src/inference.py \
    --checkpoint \
        checkpoints/best_wrn_s42.pth \
        checkpoints/best_wrn_s7.pth  \
        checkpoints/best_wrn_s13.pth \
    --tta 31 \
    --output submission.csv

echo ""
echo "submission.csv generated."
