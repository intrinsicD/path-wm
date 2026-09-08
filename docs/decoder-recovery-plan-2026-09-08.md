# Frozen-encoder decoder recovery — 8 September 2026

The user authorized the decoder-refitting diagnostic during the voice discussion.
This slice asks whether COCO reconstruction can be recovered from the task-adapted
encoder without moving that encoder or its task readout. It does not introduce
task conditioning, skip connections or memory registers.

## Protocol fixed before training

Use the original arm B COCO warmup and selected PushT adaptation checkpoints.
Preserve source files, original split membership and RGB64 preprocessing. Three
arms each use seed 5107, 2,000 AdamW updates, batch/microbatch 128, FP32,
learning rate 0.0003, weight decay 0.0001 and gradient clipping at 1. Each arm has
a 1,800-second wall budget and 256,000 image presentations if completed.

| Arm | Frozen encoder | Decoder initialization |
|---|---|---|
| adapted_parent | Selected task-adapted E | Copy of its task-adapted D |
| adapted_fresh | Selected task-adapted E | Fresh seed-5107 D |
| warmup_fresh | COCO warmup E | Identical fresh seed-5107 D |

All arms reconstruct COCO train images only. Use identical batch draws and the
same 2,048 validation indices sampled with seed 5107. Validate every 100 updates;
select minimum validation RGB MSE with earliest-step tie breaking, including
step zero. Report final as well as selected checkpoints. No budget extension or
early efficacy stopping based on results. Decoder-only training must keep E and
H in evaluation mode with gradients disabled, save complete E/D/H state, and
verify their frozen fingerprints at checkpoint boundaries and after resume.
An H absent from the image-only parent is an unused fresh placeholder, never
represented as a pretrained task head.

The matched fresh-D comparison measures decodability under this architecture and
budget. The parent-D arm separately measures practical repair. Neither a failed
refit nor an old/new decoder swap proves information-theoretic loss in E.

## Evaluation and reporting

After selection, evaluate all three selected and final checkpoints on all 4,146
internal COCO test images and 2,506 PushT test frames; retain per-image errors.
These test populations were already examined, so results are exploratory. Use
the original warmup and adapted pairs as references on identical populations.
Compare aggregate MSE/RMSE, paired per-image error changes, and ratios to each
domain's train-only mean-image baseline. Bootstrap paired image differences as
descriptive uncertainty only, not training-seed uncertainty. Exact frozen E/H
tensors establish unchanged task readout mapping; do not rerun control or infer
that a new decoder improves predicted latent states.

Render the same six previously fixed COCO examples and six deterministically
selected PushT examples, including original, reference and repaired outputs.
Show validation learning curves and exact metrics. Record checkpoint/data/code
hashes, exposure, matching and freeze audits. Refresh and verify canonical HTML
after each completed arm and final evaluation. Preserve the previous full
curriculum dashboard and scope the new canonical view to this experiment if the
combined artifact exceeds existing payload limits; do not relax those limits.

There is no new binary quality threshold or architectural adoption gate. Report
how closely recovery approaches the original warmup and matched frozen-warmup
control. A promising one-seed result motivates confirmation; it does not replace
the original failed PushT pose-readiness gate.

## Earlier findings and user ideas retained

- Domain switch versus added pose gradients: COCO-to-CCHI reconstruction-only
  adaptation remains a separate causal control; this recovery experiment cannot
  identify the original forgetting mechanism by itself.
- Pose-readout limitation and orientation generalization: retain the frozen-E
  linear versus spatial/nonlinear H comparison. Repaired images need not improve H.
- Generic retention: replay and a protected generic E/D route remain candidates.
  Repair of a generic decoder may worsen its task rendering; report both domains.
- Task conditioning: distinguish operation tags (reconstruct/predict) from domain
  tags (COCO/PushT). E→D reconstructs current state; P→D renders imagined state.
  A shared conditional model should be compared with an equally trained
  unconditioned model, not a model deprived of generic data.
- Skips: fine/coarse and residual routes already exist. New high-resolution skips
  must not use true future frames, and static copies can misplace moving objects.
- Registers: [ViT registers](https://arxiv.org/abs/2309.16588) are extra computation
  tokens, not automatically persistent memory. Our hybrid encoder could be tested
  with encoder scratch tokens if high-norm/background-token artifacts are found;
  low across-frame coarse variation alone does not establish that pathology.
  The convolutional decoder has no token-attention interface, so decoder registers
  would require an additional architectural mechanism. Persistent recurrent tokens
  are a different intervention ([RMT](https://arxiv.org/abs/2207.06881)); our U already
  maintains 128-dimensional history. Any persistent E/D memory needs episode reset,
  independent batch state and causal rollout tests. None directly preserves old
  weights or substitutes for a generic retention objective.

## Implementation interfaces and essential tests

Extend the existing curriculum trainer with explicit `decoder_only` and
`decoder_initialization` configuration, keeping ordinary warmup/supervised behavior
unchanged. Require a parent and image-only phase. Optimize D only but save E/D/H;
resume and selected-checkpoint loading must preserve frozen state exactly. A
small runner uses the existing completion/dashboard wrapper for each arm.
Tests must fail first for frozen-state preservation, exact resumed decoder
training, correct initialization, and rejecting an invalid training phase.
