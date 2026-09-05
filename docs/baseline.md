# LeWM baseline implementation and validation

The active task is baseline reproduction, before testing PATH-WM hypotheses.

Source: https://github.com/lucas-maes/le-wm/tree/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac
Paper: https://arxiv.org/abs/2603.19312
Environment source: https://github.com/galilai-group/stable-worldmodel/tree/6f1e499e9cc0c898d326112f485c1062c3d20f24

Components: random-initialized HF ViT-Tiny/14 (192, 12 layers, 3 heads), CLS
representation, 192→2048→192 BatchNorm projector, action Conv1d/MLP embedder,
6-layer causal AdaLN-zero predictor (16 heads, head dimension 64, MLP 2048,
dropout .1), BatchNorm prediction projector. Context is three frames. Each
transition consumes five consecutive actions. Images are 224px, ImageNet
normalized. Loss is next-embedding MSE + .09 SIGReg with 1024 projections and
17 quadrature knots. Both representation branches receive gradients.

The learning function takes any compatible model and objective; dataset adapters
produce pixels [B,T,C,H,W] and action blocks [B,T,K*A]. Components are plain
injected nn.Modules, without a registry. The planner accepts a rollout cost.

Validation order:
1. Data windows never cross episodes; exact observation/action timing.
2. Strict released-checkpoint loading and numerical reference parity.
3. Finite gradients through encoder, both projectors, action embedder, predictor;
   full-batch regularization and normalization semantics.
4. From-scratch learning on real PushT, held-out prediction and action controls.
5. Closed-loop PushT evaluation with declared seeds, goals, budget, CEM settings.
6. Repeat dataset/learning checks on a second dataset before broad experiments.

Published training uses batch 128, AdamW 5e-5/1e-3, bf16, gradient clip 1.
The paper states 10 epochs, while the release YAML says 100. Set 10 explicitly
for a paper-budget run. The release requests an epoch-interval scheduler whose
current dependency calculates its length from optimizer steps; this ambiguity
must not silently become a reproduction claim. Our explicit step-based linear
warmup (1% of steps) and cosine decay is a documented protocol difference.

We split by episode and compute action normalization on training episodes for
fresh training. The authors use a random window split and all-data normalization.
Released-checkpoint checks use all-data normalization to match its expected
input scale and are not evidence of held-out generalization. Small budgets or
smaller batches are labelled development validation, never paper reproduction.
Encoder recomputation/chunking is acceptable only after gradient parity checks;
microbatch SIGReg or BatchNorm is not interchangeable with a full batch.

## Additional PushT dataset for early validation

The [original Diffusion Policy data](https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip)
contains 206 demonstrations and 25,650 frames. `pusht_cchi` uses these 96px images
(resized to 224) and preserves absolute target XY actions. It is a separately
named data protocol and is never evaluated as the released LeWM training set.
The conversion receipt records its content hash. CCHI state omits velocity;
the local SWM simulator initializes that missing velocity to zero for control
checks. CCHI control is a development check, not the paper's exact benchmark.

Validation windows for the full training runs are a fixed, uniformly sampled
subset of held-out episode windows. The first 500-step development run predates
this improvement and used the first 256 validation windows; its metrics only
cover that contiguous evaluation sample.
