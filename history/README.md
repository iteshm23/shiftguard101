# History

Earlier versions from the team repository `Xavaitron/Shiftguard10`, kept for reference.

| file | commit | what it is |
|---|---|---|
| `notebook_v2_68c54a8.py` | `68c54a8`, 24 Mar 2026 | v2: single WRN-28-10, Balanced Softmax, full inverse-frequency sampler, 90/10 split, 300 epochs, SWA from 250, 20 + 1 TTA views. The v3 header says v2 scored 0.9348. |
| `notebook_v3_training_18a6d54.py` | `18a6d54`, 25 Mar 2026 | The version that trained the three submitted models: 3 seeds (42, 137, 7), 450 epochs, SWA from 360, sqrt-inverse sampler, 95/5 split, batch 128, saved only the best weights per seed. |
| `submissions/v2_1b98891.csv` | `1b98891`, 24 Mar | predictions committed right after v2 |
| `submissions/v3_2seed_b90dba0.csv` | `b90dba0`, 25 Mar | predictions from the 2-seed v3 draft |

The final predictions are `../submission.csv` (commit `f9638d7`, "3 ensemble results").

## Timeline (commit messages)

| date | message | meaning |
|---|---|---|
| 12 Mar | initial commit, dataset push, pipeline created | modular `src/` pipeline |
| 13 Mar | "new WLAD loss", "second run", "3rd run" | CCT and WRN with LDAM and deferred re-weighting, TrivialAugment |
| 13 Mar | "new CL approach", "contrastive learning attempt" | supervised contrastive learning tried |
| 23 Mar | "rewrite: all-in-one notebook.py with PyramidNet+ShakeDrop, Balanced Softmax, multi-seed ensemble, TTA" | first single-file script |
| 24 Mar | "simplify: single WRN-28-10 training run" | v2 |
| 24 Mar | "harsher parameters for training" | first v3: 2 seeds, 400 epochs |
| 25 Mar | "3 model ensemble anf 450 epochs" | training version |
| 27 Mar | "3 ensemble results" | final predictions |
| 27-28 Mar | Kaggle compatibility, skip training if a checkpoint exists, per-epoch checkpoints with resume, DataParallel with batch 256, AMP with 250 epochs (reverted), "--inference-only and --checkpoint-dir for server train + Kaggle inference workflow", `--gpu`, DataParallel removed, batch 512 | preparing the Kaggle inference notebook |
| 29 Mar | "Final Submission", "updated readme", "added comments", "final for sure" | checkpoint folder set to the Kaggle Model; header notes added |
