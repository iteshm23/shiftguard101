# Provenance

> Section numbers refer to [interview_prep.md](../interview_prep.md), which has the full version.

Source hierarchy used: (1) the complete team repository `Xavaitron/Shiftguard10` and its git history for the implementation, (2) the dataset for data facts, (3) the competition page for rules, (4) the resume for public claims.

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
