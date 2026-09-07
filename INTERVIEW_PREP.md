# ShiftGuard10 — Interview Prep

## 30-Second Pitch

"I built a robust image classification system for an EE708 course competition where the test
distribution differs from training — a real-world challenge called distribution shift. The
approach was a 3-seed WideResNet-28-10 ensemble with Balanced Softmax loss to handle class
imbalance, AutoAugment for diverse training augmentation, MixUp and CutMix for interpolation
regularisation, SWA for flat-minima generalisation, and 31-view TTA at inference. This
combination placed 1st among 30 teams with a 0.9488 macro-F1 score."

---

## Q&A Bank

### Architecture

**Q: Why WideResNet-28-10 and not ResNet-50 or a ViT?**

WRN-28-10 is the established strong baseline for CIFAR-scale 32×32 inputs. ResNet-50 uses
a 7×7 stem and MaxPool designed for 224×224 — on 32×32 images it aggressively downsamples
in the first layer, losing spatial information. ViTs require large datasets or pretraining
to work well on 32×32 (the minimum patch size is typically 4×4 which gives only 64 tokens
— too few for effective attention). WRN-28-10 uses a 3×3 stem appropriate for small inputs
and achieves state-of-the-art on CIFAR without pretraining.

**Q: What does "28-10" mean?**

28 = total depth (number of conv layers). The depth is 4 + 6×n where n=4 (4 blocks per group,
3 groups). Factor 10 = widen factor — every group's channel count is 10× wider than standard
ResNet: [16→160, 160→320, 320→640]. ~36.5M parameters total.

**Q: What's a pre-activation residual block?**

Standard residual: Conv → BN → ReLU → Conv → Add.
Pre-activation: BN → ReLU → Conv → BN → ReLU → Conv → Add.
Pre-activation moves BN and ReLU before the convolution, which improves gradient flow and
gives slightly better performance on very deep networks. Both WRN-28-10 and original WRN
papers use pre-activation blocks.

---

### Balanced Softmax

**Q: What is Balanced Softmax and why did you choose it over LDAM?**

Standard softmax: p(c|x) = exp(f_c) / Σ_j exp(f_j). The denominator treats all classes equally,
so majority classes dominate the gradient.

Balanced Softmax incorporates class frequency n_c directly into the denominator:
```
p(c|x) = n_c · exp(f_c) / Σ_j n_j · exp(f_j)
```
Equivalently, add log(n_c) to each class logit before standard cross-entropy. Minority classes
get a positive logit boost proportional to their log frequency ratio.

Advantage over LDAM: applies from epoch 1, no DRW schedule needed. LDAM requires a two-phase
training (standard CE early, then class-weighted LDAM later) because LDAM's margin can
destabilise early training. Balanced Softmax is simpler and more stable.

**Q: Derive the Balanced Softmax gradient for a minority class.**

For class c with logit f_c:
  Loss = -log p(c|x) = -f_c - log(n_c) + log(Σ_j n_j · exp(f_j))

∂Loss/∂f_c = -1 + n_c · exp(f_c) / Σ_j n_j · exp(f_j) = p(c|x) - 1

Same form as standard CE — the loss is still -(1 - p) for the correct class. The difference
is that p(c|x) is suppressed for majority classes, so their gradients are larger earlier in
training when they would otherwise dominate.

---

### AutoAugment

**Q: What is AutoAugment and how was the CIFAR10 policy learned?**

AutoAugment (Cubuk et al., CVPR 2019) uses reinforcement learning (a controller RNN) to search
for an optimal augmentation policy on a proxy task. The CIFAR10 policy has 25 sub-policies;
each sub-policy is a sequence of two operations (e.g. Rotate then Equalize) with learned
probability and magnitude. At training time, one sub-policy is sampled per image.

The policy was learned by the original Google Brain team by training thousands of smaller models.
We use it as-is — no re-search. The CIFAR10 policy is available in `torchvision.transforms.AutoAugment`.

**Q: Why AutoAugment instead of RandAugment or TrivialAugmentWide?**

AutoAugment: learned policy specific to CIFAR10. Higher performance but slow search.
RandAugment: uniformly samples operation magnitude — simpler, nearly as good.
TrivialAugmentWide: single operation per image, widest magnitude range.

For this competition, the CIFAR10-specific policy is most appropriate since the dataset follows
the same label taxonomy. The learned sub-policy sequences are optimised for exactly this domain.

---

### MixUp and CutMix

**Q: Walk me through MixUp.**

MixUp creates a convex combination of two training images:
  x̃ = λ·x_i + (1-λ)·x_j,  ỹ = λ·y_i + (1-λ)·y_j

λ ~ Beta(α, α), α=1 gives uniform(0,1). The loss is the convex combination of cross-entropies:
  L = λ·CE(f(x̃), y_i) + (1-λ)·CE(f(x̃), y_j)

Benefit: smooths the decision boundary, reduces overconfidence, acts as a strong regulariser.

**Q: What is CutMix? How does λ differ from MixUp?**

CutMix pastes a rectangular patch from x_j into x_i. The patch size is determined by λ ~ Beta(α,α):
  patch area ratio = 1-λ, so patch dimensions are √(1-λ)·W × √(1-λ)·H

After the cut, λ is recalculated from the actual pixel area ratio (cut area may differ slightly
from target due to clamping at image boundaries). Loss is the same λ-weighted combination.

Benefit over MixUp: preserves local texture structures rather than blending them — better for
tasks where local texture (fur pattern, wing texture) is discriminative.

**Q: Why disable MixUp/CutMix during SWA?**

SWA's averaging is most effective on a stable loss surface. MixUp/CutMix introduce noisy
label combinations that change the effective loss landscape at every batch — averaging weights
over these perturbed gradients degrades SWA's ability to find flat minima. Disabling them
lets SWA accumulate weights from clean, consistent gradient directions.

---

### 3-Seed Ensemble

**Q: How does the ensemble reduce error?**

Different random seeds lead to different weight initialisations, which causes models to converge
to different local minima of the loss surface — even with the same architecture and data.
These minima differ in which test examples they classify correctly. Averaging class probabilities
(rather than hard votes) preserves confidence calibration and reduces variance:

  p_ensemble(c|x) = (1/K) Σ_k p_k(c|x)

For K=3 independent models with per-class error ε and correlation ρ between errors:
  Ensemble error ≈ ε·[1 - (1-ρ)·(K-1)/(K)] — decreases as ρ decreases.

**Q: Why seeds 42, 7, and 13?**

No special meaning — they are well-separated in seed space. Orthogonality here refers to the
diversity of learned representations, not any mathematical construction. Different seeds produce
different shuffles, different weight initialisations (Kaiming normal), and different batch order
throughout training.

---

### TTA (31-view)

**Q: Why 31 views specifically?**

1 clean view + 30 augmented views. The clean view captures the unperturbed model confidence.
Augmented views (RandomCrop, HorizontalFlip, RandomRotation(15°), ColorJitter) simulate the
distribution shift between train and test images. 31 views is the operational sweet spot:
beyond ~25 views, variance reduction from additional views is marginal, but computational cost
grows linearly. The odd number ensures no tie-breaking issues in probability averaging.

**Q: What's the mathematical basis for TTA?**

TTA approximates Monte Carlo integration over the augmentation distribution:
  p̂(y|x) ≈ (1/N) Σ_n p(y | T_n(x))

where T_n is a random augmentation. This estimates E_T[p(y|T(x))], the expected class
probability under the test augmentation distribution. As N → ∞, this converges to the
true expectation. In practice N=31 is sufficient for stable estimates.

---

### SWA

**Q: How does SWA work?**

SWA (Izmailov et al., 2018) maintains a running average of model weights during training:
  θ_SWA ← (n·θ_SWA + θ_t) / (n+1)

After collecting n weights, SWA updates the batch normalisation statistics on a full pass
through the training data (since BN stats must reflect the averaged weights, not any
individual snapshot). The averaged weights lie near the centre of a flat basin in the
loss surface, giving better generalisation than any single high-accuracy checkpoint.

We start SWA at epoch 225 (after the model has converged from the cosine LR schedule).
SWA LR is set to 0.005 (constant) to keep exploring the basin neighbourhood.

---

### Distribution Shift

**Q: What is distribution shift and how does your pipeline address it?**

Distribution shift: P_test(x,y) ≠ P_train(x,y). In this competition, test images come from
a different camera/processing pipeline — different saturation, contrast, JPEG compression.

Our countermeasures:
1. **AutoAugment**: exposes the model to diverse appearance variations during training
2. **MixUp/CutMix**: trains on interpolated inputs, teaching the model to use shape/structure
   rather than texture cues (texture bias is a known source of distribution-shift fragility)
3. **SWA**: flat loss minima generalise better across distribution changes than sharp minima
4. **31-view TTA**: average over augmented views to smooth out test-time distribution artefacts

---

## Things That Could Go Wrong in an Interview

1. **"Explain Balanced Softmax in one equation"** — be ready: `log p(c|x) = f_c + log n_c - log(Σ n_j exp(f_j))`

2. **"Why not just oversample minority classes?"** — WeightedRandomSampler oversamples but
   doesn't change the loss geometry. Balanced Softmax adjusts the geometric margin, which
   is more principled for long-tailed distributions.

3. **"What is the BN update in SWA?"** — After averaging weights, the stored BN running
   mean/variance are stale (they came from the last individual checkpoint, not the averaged
   weights). SWA must re-compute them with a forward pass through the training data.

4. **"How is AutoAugment's CIFAR policy different from the ImageNet policy?"** — The CIFAR10
   policy contains operations more relevant to small-scale textures (Equalize, Color) with
   different magnitude ranges than the ImageNet policy (which prefers Translate and Rotate
   at larger magnitudes for larger spatial images).

---

## Cheat Sheet

| Component | Key Value | Why |
|---|---|---|
| Architecture | WRN-28-10 | Best CIFAR-scale backbone without pretraining |
| Params | ~36.5M | Enough capacity; no overfitting with AutoAugment |
| Loss | Balanced Softmax | Class-frequency logit correction; stable from epoch 1 |
| Augmentation | AutoAugment CIFAR10 | Learned policy for this exact label set |
| MixUp/CutMix | α=1.0, 50% prob | Texture-agnostic regularisation |
| Cutout | 16×16 patch | Occlusion robustness |
| Optimiser | Nesterov SGD | Standard for WRN; Adam underperforms on WRN |
| LR | 0.1 → cosine → 0 | 5-epoch warmup prevents early divergence |
| Grad clip | max_norm=5.0 | Stability in the warmup phase |
| SWA start | epoch 225/300 | After full cosine cycle; explore flat basin |
| Ensemble seeds | 42, 7, 13 | Diversity across init and batch order |
| TTA views | 31 | 1 clean + 30 stochastic; diminishing returns beyond ~25 |
| Final F1 | 0.9488 | Leaderboard score; Rank 1 / 30 teams |
