# Encoder depth, spatial scales and the general-agent design

8 September 2026. Completed design review requested by the user, including three
exchanges with Claude and independent code/source checks. The result is an agreed
proposal, not an adopted architecture or a frozen experiment protocol. No model
training ran in this review. Existing checkpoints and protocols remain unchanged.

**Recommendation:** run a bounded diagnostic/reference slice, then compare added
residual processing and cross-scale exchange in a four-arm study with the current
320×64 output contract. Additional internal or exported scales are conditional
follow-ups. We agree that extra depth is a plausible candidate; we do not claim
the present evidence proves an encoder-capacity bottleneck.

## Direct answer and distinctions

Extra spatial processing within the current two scales is a plausible and
relatively contained intervention. The current E has three strided convolutions,
one cross-scale exchange and one per-token MLP per branch. It is shallow compared
with established vision backbones. This is a reason to test more depth, not
evidence that encoder capacity caused the measured failures.

Four levels can also be useful, especially when objects/text span different
sizes. The number four is not a general requirement. Network depth, spatial
resolution and the number of exported feature maps are different quantities.

| Design axis | What changes | Main issue to control |
|---|---|---|
| More blocks at the same resolution | More feature computation before export | Optimization, parameter count, compute and generalization |
| More internal spatial stages | Additional resolutions/context pathways | Downsampling may discard needed detail; fusion and depth also change |
| More exported scales | The observation/state representation passed downstream | Token budget, decoder/predictor compatibility and rollout cost |
| Higher observed resolution | More actual input detail | Data/view changes; upsampling old pixels does not recover detail |
| Wider features | More channels per location | State size and transition difficulty |
| More fusion steps | Additional exchange between streams/scales | Distinguish routing benefit from extra computation |

The existing attention exchange already gives each branch access to every
location in the other branch. Extra coarse maps are not required merely to
obtain global access. Residual spatial blocks instead offer repeated local
processing that a per-token MLP cannot supply on its own. They can be placed
before the latent outputs without enlarging the exported state.

## What the literature supports

| Primary source | Relevant result and limit |
|---|---|
| [ResNet](https://arxiv.org/abs/1512.03385) | Residual connections make substantially deeper vision networks easier to optimize. They justify a depth experiment, not a guarantee of improvement under our small-data protocol. |
| [Swin](https://arxiv.org/abs/2103.14030) | Combines hierarchical spatial stages with repeated blocks and local-window attention. Its performance cannot be attributed to the number of scales alone. |
| [HRNet](https://arxiv.org/abs/1908.07919) | Maintains high-resolution streams and repeatedly exchanges information across resolutions for spatially precise outputs. It motivates preserving detail rather than only adding ever-coarser stages. |
| [ViTDet](https://arxiv.org/abs/2203.16527) | Competitive detection uses a plain ViT backbone and builds a feature pyramid from one output scale. A deep hierarchical backbone is therefore not a universal prerequisite for multiscale outputs. |
| [FPN](https://arxiv.org/abs/1612.03144) | Top-down and lateral fusion provide useful multiscale features with a simple mechanism. Cross-attention needs a trained comparison against alternatives. |
| [JEPA-WM design study](https://arxiv.org/html/2512.24497v4) | Section 5.2 finds capacity scaling helpful on its real-world robot data but not consistently on simulated tasks. Encoder/pretraining and predictor changes differ from adding residual blocks to our E. |

## Four exported maps can mean very different costs

At width 64, the following counts are calculated from the grid dimensions. The
last column is the relative number of attention-score entries in the current
dense predictor design, assuming its three extra memory/action tokens stay fixed:
`(N + 3)^2 / (320 + 3)^2`. It is not a measured latency or total-FLOP ratio.

| Exported grids | Visual tokens | Latent scalar count | Relative predictor attention-score count |
|---|---:|---:|---:|
| 16×16, 8×8 | 320 | 20,480 | 1.00 |
| 16×16, 8×8, 4×4, 2×2 | 340 | 21,760 | 1.13 |
| 32×32, 16×16, 8×8, 4×4 | 1,360 | 87,040 | 17.81 |

The extra 4×4/2×2 maps offer coarse context, not additional observed fine detail.
The 32×32 map preserves a finer representation of the supplied image but greatly
increases a dense predictor's sequence length. More internal computation does
not require exporting all intermediate activations. An encoder could compute
additional stages and fuse them into the existing 16×16/8×8 outputs. Such a
projection is a compression choice to evaluate, not a lossless guarantee.

## Concrete candidate agreed after discussion

Keep the present three convolution stages, learned spatial metadata and two
exported maps. Insert same-resolution residual spatial blocks after the fine
1×1 projection at 16×16 and after the third convolution at 8×8, before adding
position/scale embeddings and performing cross-scale exchange. Both residual
branches are width 64. This branch-local placement preserves the original shared
stem: the coarse convolution still reads the original 16×16 stem activations.

The proposed block is pre-activation GroupNorm → GELU → 3×3 convolution →
GroupNorm → GELU → 3×3 convolution, plus an identity skip. Use stride one, padding
one, no convolution bias, and affine GroupNorm with eight groups in this candidate.
These are explicit proposed implementation choices, not empirically selected
hyperparameters. Keep the original per-branch token MLPs in every arm.

Compare against zero added blocks. An initial deeper candidate can use two
blocks per stage, with its exact configuration/budget fixed before training.
The added four blocks contain 294,912 convolution weights plus 1,024 normalization
parameters, or 295,936 added trainable parameters. They add 47,185,920 convolution
multiply-accumulate operations per RGB64 frame. These are calculated architecture
counts excluding normalization/activation work, not measured memory or latency.
The convolution paths' receptive-field spans become 39 pixels at the fine branch
and 79 at the coarse branch. GroupNorm uses spatial statistics and attention is
global, so these spans are not total dependency bounds.

Standard residual ConvNet blocks are the initial option; ConvNeXt-style or
window-attention blocks remain alternatives if a measured limitation justifies
another comparison.

This keeps `ObservationLatent` shapes intact, so existing D/H/U/P interfaces can
remain testable. It does not make previously trained downstream weights compatible
with a differently trained E: comparisons of the complete world model still need
matching downstream training and checkpoint identity checks.

## Agreed experiment order

**A. Bounded diagnostic and reference.** Reuse existing prediction/copy baselines
and curves; add only missing controls. Make one controlled readout/optimization
check on the current frozen E, rather than an indefinite pose-head search. Compare
a frozen pretrained representation through an explicit adapter from the same
RGB64 source views, profiling memory/runtime first. Add masks/extent or another
capability before claiming general representation quality.

For DINOv2-S/14, a concrete adapter can resize the same source view to the
backbone's input resolution, project its 16×16 patch features to width 64 and
pool a matching 8×8 coarse branch. It must output the legacy structure; H/D/U/P
cannot consume arbitrary N unchanged. Fit and audit the readouts and adapter.
A native-width or wider readout is a conditional control if the projection fails;
another MLP projecting to the same width tests mapping expressiveness, not the
information cost of dimensionality reduction. Success shows another stack works;
failure does not eliminate encoder explanations. This is a reference comparator,
not a guaranteed positive control or a reproduction of DINO-WM.

**B. First custom-E comparison.** Use this four-arm design:

| Arm | Added residual blocks per branch | Cross-scale exchange | Question |
|---|---:|---|---|
| Reference | 0 | On | Current architecture |
| Deeper | 2 | On | Does extra spatial processing help with current fusion? |
| No exchange | 0 | Off | Does the exchange help the shallow encoder? |
| Deeper, no exchange | 2 | Off | Does depth change the value of exchange? |

Keep source views, stem widths, exported grids, original token MLPs, loss recipe
and checkpoint selection comparable. All arms train under their declared
architecture; do not switch off attention only for evaluation. Share compatible
initial tensors and paired data draws across seeds. This tests a residual-depth
package, including extra parameters and normalization, not depth independent of
those changes. A parameter-matched width control is a later attribution test if
the result warrants it.

Use a small one-seed development slice to verify the implementation, then at least
three paired seeds for the proposed formal comparison (12 arm-seed runs). Fix the
caps, validation selection and thresholds before training, after a development
runtime profile. Report presentations and elapsed-time curves; equal updates are
not equal compute. No numeric training cap or success threshold is claimed as
frozen by this document. Existing configuration-disjoint holdouts remain intact;
previously inspected tests remain exploratory unless a fresh evaluation is set.

Measure common output errors and retention, not just latent MSE. For dynamics,
compare each representation against its own matched copy/action baselines and
common physical/event outcomes. Train compatible U/P/readouts before comparing
complete systems. Keep the existing planner's stage gates; a failed gate is
reported as such, and a different planner needs a separate prospective protocol.
For a general encoder, no single pose threshold is a sufficient selection rule.

**C. Conditional extensions.** Extra internal 4×4/2×2 context fused back into
320 tokens, more exported levels, higher-resolution detail and alternate fusion
are separate hypotheses. A controlled perturbation study should distinguish
dependence on object size, distance, occlusion and layout before selecting a remedy.
Depth/fusion interactions can motivate one coarse self-attention or simple
resize/projection comparison. Neither is a proven winner. Preserved detail could
use a readout-only path, but that path does not establish that imagined states
retain it. Every feature used for future planning/decoding must be causally
maintained or predicted.

**D. Broader system.** Test a small software persistence task as the next distinct
action/outcome setting. Exact-symbol observations test interface and state
extensibility; a screenshot-only track is needed to test visual text perception.
The shared belief/action/outcome loop remains the direction, while driving,
learned deliberation and generated artifacts need their own data and evaluation.

Broader context remains in the [perception proposal](versatile-perception-architecture-2026-09-08.md)
and [general-agent clarification](general-agent-world-model-2026-09-08.md). Additional
image scales do not supply temporal memory, task hierarchy or trained reasoning.
Action generation and observation reconstruction remain separate functional roles.

## Collaboration record

The supplied [technical brief](../runs/architecture_cowork_2026-09-08/claude_brief.md)
includes all current questions and earlier proposals, measured findings and their
limits. The Claude agent listing had no reachable agents, so the installed CLI
was used with only public WebSearch/WebFetch tools and a supplied brief. No local
file tools, model training or architecture edits were delegated.

The [first response](../runs/architecture_cowork_2026-09-08/claude_round1.md) and
[execution receipt](../runs/architecture_cowork_2026-09-08/claude_round1.receipt.json)
are preserved unchanged. It completed in 216.95 seconds, with no permission
denials. Its main useful challenge was to prioritize a technically valid frozen
pretrained comparator and test feature mixing explicitly before a large scale
sweep. It also proposed an immediate token-interface rewrite and extra registers;
these are not adopted merely because they appeared in the review.

Codex checked the implementation and sent a [critical follow-up](../runs/architecture_cowork_2026-09-08/claude_followup.md):

- `world_model/paddle/training.py:120` encodes targets under no-gradient execution;
  `:336` trains only P at the predictor stage. `:229` and
  `world_model/paddle/rollout.py:9` implement free-running multi-step P→U rollout.
  Paddle P5 already uses five steps; target stop-gradient and multi-step training
  are not missing fixes. Fine/coarse latent losses already have fixed training
  motion-variance scaling. Physical readout losses remain diagnostic.
- H is a flattened linear/nonlinear or grid-dependent spatial head. P slices 320
  tokens and D reconstructs explicit grids. Arbitrary-token or DINO interfaces
  need actual adaptations; they are not drop-in replacements.
- A failed DINO-based comparison would not eliminate encoder explanations.
  Input resolution, a learned projection, readout/transition suitability and
  optimization are alternative explanations. Hardware fit requires profiling.
- Improving endpoint validation leaves undertraining plausible, not proved.
  One-seed results do not estimate training-seed variance. A non-centroid target
  is not proof that an expected-coordinate head cannot represent it.
- Decoder recovery does not quantify how far features moved. Frozen E alone does
  not preserve an updated D, as the measured tradeoff already shows.
- Existing planner readiness gates and initial-configuration group splits remain
  intact. A different planner can have a different prospective protocol. General
  output fidelity/retention and runtime remain relevant alongside control.

The [second response](../runs/architecture_cowork_2026-09-08/claude_round2.md) accepted
those corrections and the depth-by-exchange design. It still introduced an
incorrect hard sub-cell limit and overconfident diagnostic triggers. Codex sent
a [final reconciliation](../runs/architecture_cowork_2026-09-08/claude_settlement.md).
The [third response](../runs/architecture_cowork_2026-09-08/claude_round3.md) explicitly
withdraws those claims and agrees to A/B/C above.

A flattened linear head can recover sub-cell coordinates from continuous
features, and a soft-argmax coordinate is continuous. Error near a cell's width
does not prove quantization. Distance-correlated errors do not prove routing
failure, and training/validation gaps do not prove irreducible unpredictability
or stochasticity. Baseline errors are diagnostics, not information-theoretic
predictability bounds. These round-1/2 diagnoses are withdrawn, not adopted as
new findings.

All three CLI exchanges completed successfully with empty permission-denial
lists, taking 216.95, 106.62 and 15.47 seconds respectively; source/code checks and
discussion preparation are additional time. Round-2/3 execution receipts are
preserved beside their replies. The runner and prompts are saved under
`runs/architecture_cowork_2026-09-08/`. Reported CLI cost fields are API-equivalent
usage estimates, not a subscription billing statement.

**Remaining differences are preferences or predictions.** Claude favors coarse
self-attention over resize fusion as a possible follow-up, and favors testing a
preserved decoder per domain alongside replay. Codex leaves the fusion order to
the controlled results and keeps both decoder controls optional. Claude prefers
a generic shape inside the local pretrained adapter; both agree that the legacy
downstream contract remains fixed in the first custom-E study.

The user proposed investigating depth and level count. Claude contributed the
specific post-projection branch placement and elevated the pretrained reference;
Codex contributed the four-arm interaction design, interface constraints and
verification/corrections. Agreement is a co-designed proposal, not empirical
validation or user adoption of a new training protocol.

## How earlier proposals stay in scope

| Topic | Current design position | Evidence needed before expansion |
|---|---|---|
| Pose, shape and extent | Optional readouts from broader spatial state; preserve explicit label semantics | Multiple useful outputs and held-out configurations, not angle alone |
| Generic reconstruction | Mixed-domain decoder training and preserved reference routes | Both COCO and task reconstruction after adaptation |
| Task conditioning | Output/query or application context where it changes the requested result | Matched unconditioned baseline; reconstruction/prediction tags do not explain a domain tradeoff |
| Skips and detail | Preserve needed fine information; future outputs use predicted or causally available appearance | Foreground/motion and future-output evaluation, not background image error alone |
| Registers | Optional scratch tokens for a demonstrated within-frame limitation | Separate trained comparison; no automatic temporal-memory claim |
| Memory | Explicit caller-owned state with causal updates and branch/episode isolation | Hidden-state/history tests and improved future decisions |
| Dynamics objectives | Build on the actual frozen-target, multi-step implementation | Diagnose physical/event errors before changing several losses at once |
| General agent | Modality-appropriate observations, typed actions, belief and outcome evaluation | A distinct software/persistence task plus physical reference; exact-symbol input tests interfaces rather than proving better vision |

Static images and extra encoder layers do not supply action consequences, useful
deliberation or long-horizon skills on their own. Those capabilities need their
own data and tests, even if the representation and neural components are shared.
