# Architecture

> Section numbers refer to [interview_prep.md](../interview_prep.md), which has the full version.

## 7. Model Architecture

### 7.1 Shapes and sizes (verified by building the model)

| stage | what happens | output for one image | parameters |
|---|---|---|---|
| input | normalised image | 3 x 32 x 32 | |
| conv1 | 3x3 conv, 3 -> 16, stride 1 | 16 x 32 x 32 | 432 |
| group1 | 4 blocks, 16 -> 160, stride 1 (first block has a 1x1 shortcut) | 160 x 32 x 32 | 1,640,672 |
| group2 | 4 blocks, 160 -> 320, first block stride 2 | 320 x 16 x 16 | 6,968,000 |
| group3 | 4 blocks, 320 -> 640, first block stride 2 | 640 x 8 x 8 | 27,862,400 |
| bn + ReLU | BatchNorm(640) | 640 x 8 x 8 | 1,280 |
| pool | global average over 8 x 8 | 640 | 0 |
| fc | Linear 640 -> 10 | 10 logits | 6,410 |
| total | | | **36,479,194** |

28 conv layers (25 of size 3x3, three 1x1 shortcuts), 25 BatchNorm layers, 12 dropout layers,
one linear layer. About 5.24 billion multiply-adds per image. 139 MB of float32 weights. The
theoretical receptive field of the last stage is about 109 pixels, more than the whole image.

### 7.2 WideResNet-28-10

**WHAT.** A residual CNN for 32x32 images (Zagoruyko and Komodakis, 2016). Depth 28 gives
n = (28 - 4) / 6 = 4 blocks per group; widen factor 10 makes the group widths 160, 320, 640
instead of 16, 32, 64.

**WHY.** The model has to learn everything from 29,400 small images with no pretrained weights.
WRN-28-10 is one of the most reliable from-scratch architectures for exactly this input size.

**HOW.** `WideResNet.__init__` builds a 3x3 stem conv, three groups with `_make_group`
(the first block of each group changes channels and, for groups 2 and 3, halves the size with
stride 2), a final BatchNorm, ReLU, global average pooling and a linear layer.

**WHY THIS.** The team tried other things first (git history): a Compact Convolutional
Transformer with LDAM loss in mid-March, then PyramidNet-272 with ShakeDrop plus WRN in the
first all-in-one script on 23 March, and the next day simplified to a single WRN-28-10 ("simplify:
single WRN-28-10 training run"). ResNet-50-style ImageNet models downsample a 32x32 image far too
early; ViTs need more data or pretraining; ResNet-18 is cheaper but usually a bit weaker on
CIFAR-style data when trained long.

**TRADEOFF.** Strong and stable, but heavy: 36.5M parameters, 5.24 GMACs per image, about 60
hours to train three models (header), and 93 forward passes per test image at inference.

**LIMITATION.** With 95 training trucks seen about 6 times per epoch for 450 epochs (around 2,750
times each), a model this size can memorise them. Augmentation and regularisation are what keep
those views different.

**INTERVIEW.** "Why WRN-28-10?" "It's built for 32x32 images: the first stage keeps the full
32x32 resolution, and it's one of the strongest architectures trained from scratch on
CIFAR-sized data. We did try a transformer-style model and PyramidNet before settling on it,
and the single WRN was simpler and scored 0.9348 on its own as v2."

**CROSS QUESTION.** "Is 36M parameters too many for 28K images?" "Parameter count alone isn't
a good measure of overfitting in CNNs. The model is heavily regularised (dropout, weight decay,
AutoAugment, Cutout, MixUp, CutMix, label smoothing), and validation was checked every epoch.
The real risk is the four rare classes, which is why the augmentation matters most there."

### 7.3 Residual blocks (pre-activation)

**WHAT.** Each block computes `out = conv2(relu(bn2(dropout(conv1(relu(bn1(x))))))) + shortcut(x)`.

**WHY.** Without shortcuts, deep stacks of convs are hard to train (gradients shrink, even
training error gets worse). The shortcut lets a block learn only a change on top of its input,
and gives gradients a direct path back.

**HOW.** `WRNBlock.forward`. The shortcut is the identity (`nn.Sequential()`), except in the
first block of each group, where channels change (and in groups 2 and 3 the size halves), so it
is a 1x1 conv with the same stride to make shapes match. "Pre-activation" means BN and ReLU come
before each conv, so the shortcut path carries x completely untouched.

**WHY THIS.** Pre-activation residual blocks are the WRN design and train most easily. A 1x1
projection only where needed keeps parameters low (only three 1x1 convs in the whole network).

**TRADEOFF.** Very little; the 1x1 projections add 258,560 parameters in total.

**LIMITATION.** Residual connections make training easy; they don't by themselves make the
model robust to shift.

**INTERVIEW.** "Why does the shortcut become a 1x1 convolution?" "Because x has 16 channels at
32x32 going into group 1, or 160 at 32x32 going into group 2 whose output is 320 at 16x16. You
can't add tensors of different shapes, so a 1x1 conv with stride 2 re-maps the channels and
downsamples."

**CROSS QUESTION.** "Why does adding x help gradients?" "The derivative of F(x) + x with respect
to x is dF/dx plus the identity, so even if dF/dx is small the gradient still flows back
unchanged through the identity term."

### 7.4 BatchNorm

**WHAT.** Per channel: subtract the batch mean, divide by the batch standard deviation, then
apply a learned scale and shift. In evaluation mode it uses running averages collected during
training.

**WHY.** Keeps activations in a stable range in a 28-layer network trained with a high learning
rate (0.1) and heavily augmented, mixed batches.

**HOW.** `bn1` and `bn2` in every block, plus a final `bn` before pooling; initialised to scale
1, shift 0 in `_init_weights`. `validate` calls `model.eval()` so BN uses running statistics.
For the SWA model, `update_bn(train_loader, swa_model)` recomputes the running statistics for
the averaged weights.

**WHY THIS.** Standard in WRN; makes high learning rates and deep stacks trainable.

**TRADEOFF.** Depends on batch statistics: small batches give noisy statistics (one reason for
`drop_last=True`), and running statistics come from augmented training images, not clean test
images.

**LIMITATION.** If test images are distributed differently (noise), the running statistics no
longer match exactly. Test-time BN adaptation exists for this but was not used.

**INTERVIEW.** "What happens if you forget `model.eval()`?" "BatchNorm would normalise each test
batch with its own statistics and dropout would stay on, so predictions would depend on what else
is in the batch and be noisier. The code calls `model.eval()` in both validation and inference."

**CROSS QUESTION.** "Why was `update_bn` needed for SWA?" "The averaged weights are a new network
that never ran forward during training. Its BN running statistics would otherwise be copied from
the last training snapshot, which doesn't match the averaged weights. `update_bn` does one pass
over the training loader to recompute them."

### 7.5 ReLU

**WHAT.** max(0, x).

**WHY.** The non-linearity; without it stacked convs collapse into one linear map.

**HOW.** `F.relu(..., inplace=True)` after each BN (in-place to save memory).

**WHY THIS.** Standard for ResNets, cheap, no saturation for positive inputs, and Kaiming init is
derived for it.

**TRADEOFF.** Zero gradient for negative inputs ("dead" units are possible but BN makes this rare).

**LIMITATION.** None that matters here.

**INTERVIEW.** "Why not GELU or SiLU?" "Those are common in transformers and EfficientNets; for a
WRN trained from scratch ReLU is the standard and the init is designed for it. There was no reason to change it."

**CROSS QUESTION.** "Why in-place?" "It overwrites the BN output instead of allocating a new tensor,
saving activation memory, which matters at batch 128 with 160-channel 32x32 maps."

### 7.6 Kaiming initialisation

**WHAT.** Conv weights drawn from a normal distribution with std sqrt(2 / fan), here
fan = fan_out = C_out x 3 x 3.

**WHY.** Training from scratch, 28 layers deep: if weights start too small or too large, the
signal shrinks or explodes layer by layer before training even starts.

**HOW.** `_init_weights`: `kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")` for every
conv; BN weight 1, bias 0. The Linear layer keeps PyTorch's default init. Measured std for a
320-channel conv: 0.0263 (expected 0.0264).

**WHY THIS.** The factor 2 compensates for ReLU zeroing half its inputs. Xavier init uses about
1/fan and is designed for tanh-like activations, so with ReLU the signal shrinks every layer.
fan_out keeps the backward (gradient) variance stable; it is what torchvision's ResNets use. In
most blocks fan_in equals fan_out anyway.

**TRADEOFF.** None really; with BN everywhere, init matters less than in a plain network.

**LIMITATION.** Only the start of training; it doesn't affect the final solution much.

**INTERVIEW.** "Why does init matter if BN normalises anyway?" "BN rescales activations, but the
first steps, the shortcut paths and the gradient sizes still depend on the weight scale. Kaiming
gives sensible scales from step one, which matters with lr 0.1."

**CROSS QUESTION.** "Derive the 2." "For y = Wx with x the output of a ReLU, Var(y) = fan x Var(w) x
E[x squared], and E[x squared] is half the variance of the pre-activation. To keep Var(y) equal to
that variance you need Var(w) = 2 / fan."

### 7.7 Dropout

**WHAT.** During training, zero 30% of the activations between the two convs of each block and
scale the rest by 1/0.7. Off in `model.eval()`.

**WHY.** 36M parameters and very few tail images: the model must not depend on any single
channel.

**HOW.** `nn.Dropout(p=0.3)` after `conv1` in every `WRNBlock` (12 layers).

**WHY THIS.** That is where the WRN paper put it, and 0.3 is the paper's CIFAR value. No sweep
in the files; it is a standard default, not a tuned number.

**TRADEOFF.** Extra regularisation slows fitting; interacts a little with BN statistics.

**LIMITATION.** Combined with everything else here (AutoAugment, Cutout, mixing, smoothing,
weight decay), too much regularisation could underfit; the long 450-epoch schedule compensates.

**INTERVIEW.** "Wouldn't BatchNorm already regularise?" "Only a little, through batch noise. Dropout
is an explicit regulariser. With 95 training trucks, we wanted both."

**CROSS QUESTION.** "Why inside the block and not before the classifier?" "The WRN paper found
dropout between the convs of a residual block works well; before the classifier it would act only
on the 640 pooled features."
