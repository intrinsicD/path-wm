# Mixed-supervision encoder continuation and added per-scale processing

This is an explicitly adaptive follow-up to the overnight P1/P1-T package study,
not part of its frozen six-fit protocol. The user authorized these proposed ideas
through07:00Berlin. Albedo stays excluded. The fresh simulator cohort is retained
as the primary generalization check; its cases never train or select a model.

Motivation available before this study: the first completed P1 pair passes the
geometry gate, while three completed category probes show validation AP around
0.09–0.10 for the task-trained CNN versus0.65–0.66 for the native pretrained ViT.
This does not identify architecture as the cause. Test whether updating the CNN
with mixed supervision suffices, then whether extra processing improves that
matched continuation. This adds a separately sealed study to the earlier
conditional sequence; it does not retroactively redefine P1 success.

## Three paired arms

All start from the same existing deeper/exchange-on CNN checkpoint7107 and fresh
P1 typed heads. The base encoder is now trainable. Retain its16×16 and8×8 grids,
row/column/scale embeddings, existing two residual blocks per branch and existing
simultaneous bidirectional cross-scale exchange.

* `joint`: unchanged encoder topology, mixed supervision.
* `conv`: add two residual two-convolution blocks per grid, width64,3×3kernels,
  GELU, per-token/channel LayerNorm, after the existing positional additions and
  before cross-scale exchange.
* `transformer`: same insertion point, two pre-LayerNorm transformer blocks per
  grid, width64,4heads, MLP expansion4, GELU, no dropout. Full self-attention at
  both grids. Reuse the existing positional signal; add no new positional system.

Each new residual branch has a learned scalar gate initialized to zero. This
preserves the source encoder function exactly at initialization (verified), while
branch weights initially have zero gradients until the gate opens. Conv and
transformer block/branch counts, parameter counts and costs are disclosed, not
claimed equal. Three paired seeds9107/9108/9109 vary heads, sampler and new modules;
they do not vary base pretraining. All shape-compatible source/head weights and
image draws start paired. No additional scale, third grid, or new source dataset
is silently introduced.

## Training and selection

Reuse the exact P1 frame populations,32COCO+32PushT per update,4,000updates,
full validation every100, and independent RGB/mask/pose head losses. RGB contributes
half each domain; COCO mask BCE and normalized PushT pose MSE each contribute1.
Sum a domain's losses before backpropagation because gradients now share E. Log
the generic-versus-manipulation encoder gradient cosine/norms on the first100
updates and every100 thereafter, plus separate head/encoder clipping rates.

AdamW: base encoder lr3e-5, new modules and heads lr3e-4, weight decay1e-4; no
weight decay for encoder gates or LayerNorm/GroupNorm affine parameters. Clip each
head and the full encoder separately at norm1. No schedule/dropout/augmentation.
FP32,TF32off. Log each gate and per-level extension gradient norms. Equal target
updates and matched data do not imply equal compute or parameter count.

Select minimum validation q, earliest exact tie; readiness q≤1 uses the same
8world-unit/10degree thresholds. Report all outputs at that snapshot and endpoint;
neither endpoint nor another task selects a replacement. Preserve raw losses,
source hashes, checkpoints, failures and quantitative reconstruction/foreground
retention. Test semantic accessibility afterwards with the same P1-T classifier
recipe frozen on each selected encoder; its labels do not supervise this stage.

The first comparison (`joint` versus frozen P1) tests the continuation package,
including encoder updates and optimizer. `conv`/`transformer` versus `joint`
test added modules at this insertion point. This is neither a capacity/placement
isolation nor a CNN-versus-ViT architecture/pretraining causal comparison. A trained
both-grid readout shows availability; it does not make trained fine-only/coarse-only
controls unnecessary. New renderer checks, dynamics and control remain distinct.

## Budget and development gate

Implement essential exact-initial-function, open-gate gradient and optimizer-group
checks first. Run50-update development prefixes for each arm and verify the
dashboard, then freeze code/data/config/source identities before formal work.
Initial ceiling20minutes per formal fit, nine fits≤3fitting hours, plus45minutes
evaluation/semantic readouts/reporting. Revise that ceiling prospectively only if
the development profile requires it and the global deadline permits. Do not launch
the formal study until P1 completes. Stop new training by06:00Berlin.

Claude's two public-only reviews and corrections are in the overnight collaboration
directory. We corrected claims that both-grid access proves useful utilization or
that scale-access tests cannot fail readiness. The zero-gate, explicit positional
placement and mixed-supervision controls were reconciled. The implementing agent
remains responsible for code/data verification and measured conclusions.

## Development completion and formal freeze

All three50-update development runs completed and their dashboards passed browser
verification. Measured fitting times were9.47s (`joint`),35.56s (`conv`) and19.24s
(`transformer`), with startup/runtime outliers; median update intervals after20
updates were about0.15–0.17s. Peak allocated CUDA memory was493/523/597MB.
The encoder parameter counts are390,240/685,668/590,184 respectively; the three
typed heads are identical in size and initialization across arms. The gates open
from exactly zero and their recorded gradients behave as intended.

Keep the prospective20-minute formal cap and4,000-update target. The nine-fit
coordinator seals the exact source/data/protocol identities and Git revision after
P1 completes. Source snapshots are retained under `extensions/frozen_source`.
The full13 new scientific-invariant tests pass, including the earlier frozen-head,
category, fresh-cohort and extension checks. Development data are only prefixes
and do not justify a model-quality claim.
