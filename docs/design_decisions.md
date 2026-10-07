# Design decisions

> Section numbers refer to [interview_prep.md](../interview_prep.md), which has the full version.

## 19. Why Each Decision Was Made

| problem in this data | what targets it | evidence it helped | honest status |
|---|---|---|---|
| 50:1 imbalance, macro F1, tail-heavy test | Balanced Softmax + sqrt sampler | v2 and v3 both had it, and both predict roughly 950 per rare class; earlier models without it predicted 224 or 447 trucks | strong reasoning, no ablation |
| repeated tail images | sqrt instead of full inverse sampling (v2 -> v3) | part of the v2 0.9348 -> v3 0.9488 change bundle | effect not isolated |
| overconfidence, ambiguous 32x32 images | label smoothing 0.1 | none | standard default |
| occlusion (10x10 grey in test) | Cutout 16 (same grey), CutMix | none | plausible, untested |
| perturbations in general | AutoAugment, crop, flip, MixUp | none | standard strong recipe |
| memorising 95 trucks with 36M params | dropout, weight decay, augmentation, mixing | none | standard |
| unstable steps | warmup, clipping at 5.0 | none | safety measures |
| noisy final weights | SWA (used per seed only if it beat the best epoch) | none; which won per seed is unknown | partly verified |
| run-to-run variance | 3-seed ensemble | 1 -> 2 -> 3 seeds in the history; agreement 96-98% between versions | no score per step in the files |
| framing variance | 1 clean + 30 TTA views | none (v2 already had 20 + 1) | no measurement |
| more training data for the tail | 95/5 split instead of 90/10 | part of the v2 -> v3 bundle | effect not isolated |
| compute and Kaggle limits | server training, checkpoint files, Kaggle inference-only run | git history | verified |

**The one real before/after number.** v2 scored 0.9348 (stated in the code). v3 (resume: 0.9488)
changed six things at once: 1 -> 3 seeds, full inverse -> sqrt-inverse sampling, 90/10 -> 95/5
split, 300 -> 450 epochs, SWA start 250 -> 360, TTA 20+1 -> 30+1. So "v3 was +0.014 over v2" is
true; "the ensemble added 0.014" is not something you can say.


## 20. Why Not Alternative Approaches

Things the team actually tried (from the git history) are marked "tried".

| choice | alternative | why the choice makes sense | when the alternative is better | limitation |
|---|---|---|---|---|
| WRN-28-10 | PyramidNet-272 + ShakeDrop (tried, 23 Mar) | simpler, faster, strong; the single WRN v2 already scored 0.9348 | more compute and time | 36M params |
| WRN-28-10 | Compact Convolutional Transformer (tried, March) | CNN priors matter with 29K small images, no pretraining | large data or pretraining | |
| WRN-28-10 | ResNet-18 | usually stronger on CIFAR-style data when trained long | tight compute | slower |
| from scratch | ImageNet weights | required by the rules | any real project where allowed | everything learned from 29K images |
| Balanced Softmax | LDAM + deferred re-weighting (tried, March) | no extra hyperparameters, works from epoch 1 | when margins are tuned well | targets a uniform prior |
| Balanced Softmax | weighted CE / focal loss | fixes the decision rule, normal gradient sizes | mild imbalance | |
| sqrt-inverse sampling | full inverse frequency (v2, tried) | less repetition of 95 trucks (6 vs 29 per epoch), more head data used | when tail recall is all that matters | still repeats; overlaps with Balanced Softmax |
| sampling | undersampling | keeps head data for feature learning | huge head classes | head images not all seen each epoch |
| AutoAugment | RandAugment (tried, first script), TrivialAugment (tried, March), AugMix | proven CIFAR-10 recipe | AugMix targets corruption robustness, which fits the noise | no noise op |
| Cutout + CutMix + MixUp | one of them | different kinds of regularisation; v2 already had all three | small compute budget | possible over-regularisation |
| SGD + Nesterov | Adam / AdamW | WRN's own settings; good generalisation; fits SWA | transformers, quick prototypes | LR-sensitive |
| warmup + cosine | step decay / plateau | no milestones; plateau would react to noisy validation | known good milestones | epochs fixed up front |
| SWA vs best epoch, pick by val | always SWA / always best | safety net | always SWA avoids winner's-curse selection | validation noise decides |
| 3 seeds | 1 model / more seeds | lower variance; about 20 h per model | latency-bound systems | correlated members |
| 1 + 30 TTA | none / few fixed views | averages framing randomness | latency-bound systems | 31x cost, doesn't remove noise |
| 95/5 split | 90/10 (v2), k-fold | keeps 95 trucks | when reliable validation matters more | 5 validation trucks |
| plain supervised training | supervised contrastive learning (tried, 13 Mar, `src/supcon.py`) | simpler single-stage training won out | when representation quality is the bottleneck | |

Short answers to "why not X":

- **ViT:** too little data and no pretraining allowed; CNNs have the right built-in assumptions.
- **ResNet-50:** built for 224x224; its stem shrinks 32x32 to 8x8 before the first block.
- **Adam:** fine for prototyping; SGD with the WRN settings is the reference and generalises well.
- **Focal loss:** about hard vs easy examples, not class priors.
- **Class weights:** 50x weights make single images dominate a batch and do nothing when the class is absent.
- **Transfer learning:** banned here; in any real project it would be the first thing to try.
