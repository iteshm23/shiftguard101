# Limitations

> Section numbers refer to [interview_prep.md](../interview_prep.md), which has the full version.

## 21. Limitations

1. **No controlled ablations.** One before/after number (v2 0.9348 -> v3) with six simultaneous
   changes. Answer: "We changed several things at once because each run took around 20 hours; the
   v3 bundle improved on v2 by 0.014, but I can't attribute that to one change."
2. **Small, unshifted validation.** 5 trucks, 15 ships; one truck moves macro F1 by 0.011; and it
   comes from the training distribution, while the test has bigger occlusions, more noise and a
   different class mix.
3. **Model selection noise.** Best of 450 epochs on that set is biased upwards, and the SWA model
   only replaced it if it beat that biased number.
4. **No noise augmentation.** 75% of test images are noisy; training adds no noise.
5. **Imbalance corrections overlap.** Sampler + Balanced Softmax + label smoothing tilt predictions
   towards rare classes beyond what a balanced test needs. Here the test seems tail-heavy, so this
   probably helped, but on a test set with the training class mix it would cost accuracy.
6. **Test prior never estimated explicitly.** It could have been estimated from predictions (black-box
   shift estimation or EM) and plugged in as log(test prior).
7. **Cost.** About 60 hours of training (header), 93 forward passes per test image.
8. **Workflow vs rule wording.** The rule says the notebook trains and infers; the final Kaggle
   notebook loaded self-trained checkpoints and ran inference. Be factual about it.
9. **Code issues** (section 18.14): `--debug` crash, unreachable resume, unseeded inference.
10. **Reproducibility.** Inference not seeded; resume reseeds; GPU non-determinism.
11. **Calibration not measured.** Label smoothing, MixUp and Balanced Softmax all change confidence.
    Fine for argmax F1, not for any use of the probabilities.
12. **Ownership.** The final code's commits are by Pratyush Singh (Xavaitron). Know your part.

Not real weaknesses (don't over-apologise): WRN-28-10 is right for 32x32 from scratch; CIFAR
normalisation constants match the data; TTA doesn't break rules; self-trained checkpoints are not
pretrained weights.

**Edge cases an interviewer may raise.**

| situation | what happens in this code | what to say |
|---|---|---|
| class with very few examples (95 train, 5 val trucks) | drawn about 6 times per epoch; validation F1 for it takes only a few values | risk of memorising; noisy validation; shifted and repeated validation |
| class missing from a batch | about 7% of batches of 128 have no truck; loss is per example, nothing breaks | fine |
| model predicts only big classes | macro F1 collapses; `zero_division=0` gives 0 for dead classes | that is what Balanced Softmax and the sampler prevent |
| augmentation changes the label | Cutout or CutMix can hide a small object | rare; CutMix mixes labels by area |
| CutMix box covers the key object | label still claims lambda for the base class | tolerated label noise |
| MixUp makes unrealistic images | intended | regularises between classes |
| sampler + Balanced Softmax over-correct | lean towards rare classes (x7 truck vs cat in theory) | right direction for this tail-heavy test; wrong for a training-like test |
| test prior differs from training | Balanced Softmax targets a uniform prior | add log(test prior) if it is known or estimated |
| shift stronger than expected | no noise augmentation | accuracy drops on the noisiest images |
| TTA views not label-preserving | views are mild; clean view included | keep TTA milder than training augmentation |
| ensemble members correlated | same recipe and data; 96-98% agreement between versions | gain is limited; diversify |
| SWA averages incompatible weights | late start (360) and low LR (0.005) make this unlikely | `update_bn` is run |
| scheduler resumes incorrectly | changing `--epochs` changes the cosine curve | keep arguments fixed |
| corrupted or half-trained checkpoint | `main()` uses any existing file without checking `completed` (verified) | check the flag; write to a temp file and rename |
| GPU non-determinism | small differences remain; inference unseeded | seed inference; promise equivalent, not identical |
| high train accuracy, poor validation F1 | overfitting or tail failure | per-class recall, tail first |
| validation up, leaderboard down | validation unshifted; selection noise | shifted validation; fewer leaderboard-driven decisions |
| macro F1 metric, cross-entropy loss | aligned through Balanced Softmax plus argmax | not perfectly; per-class offsets need more validation data |
| poor calibration with high F1 | F1 only uses the argmax | calibrate only if probabilities are used |

**How to design the missing ablation.** Fix the split and seeds; shorten to about 100 epochs; start
from v2; add one change at a time (sqrt sampler, 95/5, more epochs, SWA timing, seeds, TTA views),
3 seeds each, report mean and spread of validation macro F1 on both a clean and a test-like shifted
validation set. Inference-only parts (TTA views, SWA vs best, 1 vs 3 models) need no retraining:
run the existing checkpoints with different settings.
