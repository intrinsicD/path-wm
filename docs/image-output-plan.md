# Image detail and state-produced output interface

Active implementation slice, 12 September 2026. The user requires output from the
model even when the session contains no input of that output modality.

The existing agent already decodes state tokens, but its tiny output adapter is
separate from the trained image hierarchy. This slice connects a spatial decoder
interface to state and explicitly tests the encoder and state routes separately.

Keep the trained handwritten hierarchy as the base. Add a learnable patch-detail
feature initialized to retain each pixel's deviation from its patch mean. The RGB
head combines its base prediction's patch means with the detail feature. All base
parameters are loaded strictly and retained unchanged. This is an explicit
high-bandwidth reconstruction control, NOT learned compression or proof that the
agent remembers pixels. No claims of high-resolution generation follow from it.

A StateFeatureDecoder predicts every feature required by that same RGB head from
state tokens. It never receives images or encoder features at inference. Frozen
decoder weights still allow gradients to its generated inputs. Per-channel feature
calibration is fitted only on training targets and saved in the model. Existing
default architectures/checkpoints are unchanged; this is an opt-in composition.

Essential RED checks: high-frequency recovery and per-patch means; strict decoder
feature requirements; state-only gradients through a frozen head; no image encoder
available on the request path; exact interrupted/resumed training replay.

Predeclared local screens (single implementation seed 7601, no model selection):

* Real-image capacity control: first128 existing COCO internal test images at64px,
  reused population explicitly identified; zero training updates. Compare base,
  detail, zero-detail and shuffled-detail outputs. Success: RGB MSE improves at
  least25% vs base, and removing/shuffling detail worsens the detail result.
* Request-only development fit: four64px stripe patterns (horizontal/vertical,
  opposite phase), fixed four text requests, targets only on loss/teacher side.
  All four are training examples, NOT a generalization benchmark. Train state/text
  and spatial feature producer with frozen base/detail decoder;512 updates,
  batch4, AdamW lr0.001, gradient clip1. Pixel MSE plus0.1 times the equally weighted
  mean per-level standardized latent MSE. Evaluation every64 updates. Success:
  final image MSE<=0.01 and at least50% below request-erased and shuffled-request
  controls. Report the teacher reconstruction floor separately. No hidden image
  encoder exists in the deployed agent, and target arrays are never model inputs.
* GPU cap4GiB leaving1GiB free; total formal training wall budget300 seconds,
  preserve partial checkpoints if reached. CPU tests and read-only report/audit
  work are separate. No hyperparameter search. Run256 updates, resume to512;
  test exact replay on CPU separately. Source committed before formal execution.

For general generation the missing learned function is a conditional producer of
valid modality latents. State, recalled memories, request and imagined trajectories
can condition it; a noise seed can select alternatives where a request does not
specify every detail. A dynamics objective alone does not teach this function.
The same contract applies to text/audio/video, but their autoregressive/temporal
objectives, synchronization and practical codecs need separate training and tests.
This slice makes no claim that those capabilities are repaired.

Implementation review: Claude reviewed only generic public concepts, with no
repository access or private measurements. Its leakage and memorization cautions
are adopted above; exact brief/response/receipt are retained under
`runs/reviews/continuation_2026-09-11/output-contract-public-*`. No additional
round was needed because there was no disputed recommendation. Local review
added standalone checkpoint loading and target-aware report labels. All34
targeted image-output, hierarchy, multimodal and checkpoint tests pass, including
exact CPU interrupted/resumed model/optimizer/RNG/sampler equality. Lint passes.

## Completed screens

Source `4cda45b`, seed7601. Donor is the previously trained handwritten initialization:
`runs/hierarchy_training_v1/decoder_rate_repair/seed_7501/hand_both/weights.pt`
(SHA256 `4aea4208baa1bf1c6462355163bbbca09c7a76311f1b456afd56871984feea3a`).
All212 donor tensors compare exactly in the wrapped codec. New detail projections
start as identity matrices. The new text/state/feature producer uses ordinary
initialization; the inherited RGB decoder is frozen throughout its training.
Existing default agents and earlier checkpoints are unchanged.

| Screen/control | RGB MSE |
|---|---:|
| Real COCO, original base decoder |0.0123954175|
| Real COCO, supplied detail channel |0.0008786256|
| Real COCO, zero detail |0.0124880263|
| Real COCO, detail from the next image |0.0228668359|
| Four training requests, generated features |0.0001099732|
| Same targets, teacher encoder features |0.0000657797|
| Four training requests, erased text |0.1381580383|
| Four training requests, cyclically shuffled text |0.2662827969|

Both predeclared gates pass. The first comparison demonstrates that transporting
detail repairs the displayed blockiness; it does not establish that the compact
agent can retain real-image detail. Its new detail map alone has12288 float values
at64px, plus10752 existing hierarchy values. No information-rate/compression claim
is appropriate. Base patch means plus a zero-mean detail residual are clamped to
[0,1]; clipping can change patch means near saturated colors.

The second screen demonstrates learned production of decoder inputs without image
observations. All four prompts and targets are seen during training. Calibration
buffers are fitted to those training targets, shared across requests, saved in the
checkpoint, and not recomputed at inference. This is not held-out semantic transfer,
real-photo generation, learned physics or memory retention. Targets are used only
for teacher features and losses; the instantiated agent contains only a text encoder.
The same image head can consume generated state features or encoder features.

Training used512 updates in256+256 resumed chunks,12.118s total training-loop wall
time, peak106MiB reserved CUDA memory. Capacity control peak84MiB. No search or
budget extension. The final update is retained, even though update256 had slightly
lower fit MSE. The 300s cap applies to the training loop including its periodic
checkpoint writes; source loading, preflight, final evaluation, reports and CPU
audits are outside that recorded loop. No high-resolution dataset run was made.

All35 relevant software checks pass (34 initial checks plus the added absence-of-
matching-input check for image/audio/text/video), with no semantic-quality claim
for the other modalities. Exact CPU uninterrupted/resumed model/optimizer/RNG/
sampler equality passes. Actual GPU resume preserves the256-update ledger prefix.
Eight saved pixel metrics recompute within2.52e-8. Standalone CPU reload differs
from saved GPU outputs by at most4.12e-6; frozen decoder tensors match the control
exactly. The standalone export strict-reloads without the donor file or target
data. HTML is structurally verified; both comparison PNGs were visually inspected.

[Combined report](../runs/image_output_v1/report.html),
[raw audit](../runs/image_output_v1/verification.json),
[weights](../runs/image_output_v1/request/weights.pt).
Export SHA256 `a65528adaf3e41ad59b721125d99faf5ce4c30109916877e4660b5d9e6badf75`.
Model-state SHA256 `afc56492ba1c5e392d6e8d892648d6fd1dd536d856374f22a3bb3fc2da704906`.

### Use the implemented route

```python
import torch
from experiments.image_output import load_request_agent, request_state

agent = load_request_agent("runs/image_output_v1/request/weights.pt")
with torch.no_grad():
    state = request_state(agent, ["horizontal light"])
    rgb = agent.decode(state, modalities=["image"])["image"]  # [1,3,64,64]
```

The trained requests are `horizontal light`, `horizontal dark`, `vertical light`,
`vertical dark`. Arbitrary new prompts are unsupported by this tiny fit.

Reproduce into fresh directories (never overwrite completed runs):

```bash
.venv/bin/python -m experiments.image_output --mode capacity --weights runs/hierarchy_training_v1/decoder_rate_repair/seed_7501/hand_both/weights.pt --output runs/image_output_v1/new_capacity --device cuda
.venv/bin/python -m experiments.image_output --mode request --weights runs/hierarchy_training_v1/decoder_rate_repair/seed_7501/hand_both/weights.pt --output runs/image_output_v1/new_request --device cuda --stop-after 256
.venv/bin/python -m experiments.image_output --mode request --weights runs/hierarchy_training_v1/decoder_rate_repair/seed_7501/hand_both/weights.pt --output runs/image_output_v1/new_request --device cuda --resume
```

### What the decoder needs for general output

An image observation is optional. The required learned mapping is:

`request + current state + relevant recalled memory / imagined state → output features → decoder`.

The existing agent's `think` operation reads memory into working/reasoning state;
the decoder does not independently search the database. Its dynamics can supply
future states, but must be trained to predict meaningful changes. A planner chooses
what to render/do; it does not by itself learn a distribution of visual details.
This implementation supplies a deterministic spatial producer. A general producer
needs varied paired training and an appropriate generative objective. Optional
sampling noise selects alternatives; it supplies no missing factual evidence.
The producer may be part of the decoder or a separate module, but tensor-shape
compatibility alone is insufficient: the feature distributions must match too.

This is the motivation for separately trained autoencoding and conditional latent
generation in [Latent Diffusion](https://arxiv.org/html/2112.10752v2). Our small
deterministic fit is not a diffusion implementation. For all modalities, output
length/resolution, spatial/time positions and relevant context must be specified
or predicted. Text needs token-generation training; audio needs an appropriate
waveform/codec objective and timing; video needs temporal coherence and trajectories.
Their existing state-only interfaces work mechanically, but their useful learned
generation remains open. The next experiment should test a compact learned visual
codec and held-out paired image/request production before adding those modalities
or claiming higher-resolution generation.

## Proposed extension after the Marigold V2 discussion

13 September 2026. Proposal requested by Alex, who wants to preserve the current
design and add extensions where needed. This section specifies a candidate; it is
not an implemented generator or a predeclared training comparison. Existing run
results and the pending factual-readout optimization comparison remain separate.

The existing output boundary already supports this direction. Retain the modality
encoders, multiscale processing, latent state, memory, dynamics and planner. Extend
image output with an ordinary conditional feature-generator module, using the
same residual-transformer pattern with separate output weights. Initially use our
own encoder/decoder and existing weights. A pretrained generator/codec is an
optional later comparison through the same state-conditioning boundary, not a
replacement requirement for the agent.

### Concrete output path

The agent retrieves relevant memory into its workspace using its existing memory
operations. Select the current, recalled or imagined state to render. Context
contains those state tokens, the encoded request, masks, source/world-time metadata
and output requirements such as resolution. Context preparation must preserve
token detail; do not force everything into one pooled vector. The generator reads
this context through learned cross-attention. It does not independently search
the database or ingest target-image features during generation.

Keep StateFeatureDecoder as the deterministic reference. The new route produces
the same calibrated feature dictionary required by the chosen image head:

`state/request context + noise -> conditional feature generator -> image head -> RGB`.

Inside the generator, each noisy feature scale is projected into tokens with
position and scale embeddings, processed by k residual transformer blocks, and
conditioned on the workspace and generation progress. Cross-scale attention
jointly updates the scales before projection back to their native channel counts.
Every feature required by the head must be produced, including its detail input;
no bypass from the current image encoder is permitted on the generation route.
Start with the existing 64px feature layouts. Their present high-bandwidth detail
control is not a compact codec; record its size and compare learned compression
separately rather than claiming memory efficiency from the interface.

### Candidate learning objective

Use conditional flow matching as the first generative candidate. With the image
encoder and calibration frozen, encode training target y into standardized feature
pyramid z. Draw an independent noise pyramid epsilon and progress tau in [0,1].
Form x_tau = (1-tau)*epsilon + tau*z and train the generator's velocity prediction
v(x_tau, tau, context) toward z-epsilon, averaging normalized losses across scales.
During sampling, start at noise and numerically integrate the learned field with
context fixed, then undo feature calibration and decode. Targets enter training
loss construction only. Joint processing learns correlations across the feature
scales; independently sampled initial noise does not assert independent outputs.

This is a proposed application of [Flow Matching](https://arxiv.org/abs/2210.02747)
to our existing feature interface. The separation of image representations and
conditional generation is supported by [Latent Diffusion](https://arxiv.org/abs/2112.10752).
No single-pass guarantee follows from Marigold's pretrained task-specific result.
The generation-progress variable tau and repeated generator evaluations must not
advance world time or mutate observed memory. Sampling variation is not calibrated
belief uncertainty, and inferred texture is not recovered evidence.

### Development and comparison order

1. Establish the image reconstruction floor on held-out images using our own
   encoder/head. Continue their training if necessary before freezing a version
   for generator targets. Encoder-to-decoder reconstruction needs no workspace
   compression; this isolates codec limitations from state limitations.
2. Freeze that codec and the upstream state/memory path. Train only the generator
   and its context projections on paired states/requests/targets. Start with the
   controlled recall task and unseen attribute combinations. Add real paired data
   only where the observations and requests support the target; arbitrary COCO
   photographs are useful for codec training but are not labeled memory episodes.
3. Compare against the existing deterministic route and a matched-capacity direct
   regression control. Assess image fidelity separately from correct entity,
   attributes, arrangement and time. Hold the sampling seed fixed while changing
   state, memory or request; erase/shuffle each source and include histories with
   identical final observations but different correct recalled outputs. Where
   multiple images are valid, score specified properties across seeds rather than
   insisting on one exact pixel target. No target input may reach inference.
4. If correct content is inaccessible to diagnostic readers of the frozen state,
   investigate upstream state/memory preservation before expanding the generator.
   If it is recoverable but generation fails, investigate conditioning and output
   learning. A failed diagnostic reader alone does not prove information loss.
5. Profile actual peak memory, latency and checkpoint/disk use before increasing
   resolution or capacity. Fix data splits, numeric gates, seeds, parameter counts,
   update/sample budgets and a local-GPU headroom floor before any formal run.
   Review that bounded protocol with Claude under the existing export policy.

Reconstruction, latent prediction and state-faithful generation remain distinct
checks. A new generator does not itself repair the previous factual-readout
failure, missing visual memories or learned world dynamics. Text can retain its
autoregressive output path; audio/video can later receive their own generators
under the shared conditioning contract, with separate timing/coherence objectives.

## Conditional generator implementation comparison — 13 September

User adopted the extension and requested Claude planning/review plus implementation,
tests and iteration. Implement `ConditionalFeatureGenerator` in the ordinary model
library and `experiments/conditional_image.py` as the readable recipe. Existing
`MemoryOutput` exports gain opt-in generator settings; older checkpoints retain
their old architecture. No general trainer or renderer is introduced.

Frozen source: `runs/producer_refinement_v1/source_8502_control/weights.pt`. Keep its
own trained codec and complete observer/memory/factual path. New output parameters
use ordinary initialization; the source's handwritten-derived codec weights remain.
Train-only feature calibration, no current/target image encoder in generation.
One residual block per scale and one cross-scale fusion block, width32, four heads;
all required inherited64px feature layouts. Flow and direct controls instantiate
identical parameters; direct uses zero spatial input at progress0, flow uses the
noise-to-target field. This equal-architecture/update test does not match inference
FLOPs or effective noise/time diversity. Original source is descriptive baseline.

Training: relocation curriculum, seed41001,128 pairs/256 histories, retain only
color%2==shape (128 histories,8 of16 target triples) for new output learning. Frozen
upstream/codec were previously exposed to all target categories: held-out composition
claims apply ONLY to the new generator's training, not the whole system. Cache both
ordinary/reset working contexts and teacher targets; mixed batches contain4 of each.
Batch8,1024 updates, AdamW lr0.0003/weight_decay0.0001, clip1, same init seed41011 and
sampler/order across arms. Both arms draw identical progress/noise streams. Normalized
per-scale feature objective only; no pixel/semantic loss addition in this comparison.
Small development uses separate seeds/pairs and at most32 updates per arm; no model
selection. Validation seed41002,16 pairs, diagnostic only. Terminal checkpoints only.

Confirmations41073/41074 each32 pairs/64 histories; sample seeds13/29, Euler8 steps
fixed before results. Each adjacent selection pair shares sampling noise by explicit
pair ID. Score ordinary/reset/reset-erased/reset-swapped live memory routes; images
and old facts separately. Report seen/unseen generator triples, sample-seed stability,
weighted RGB MSE, teacher floor and swap targets. Confirm pairs have identical final
observations with differing target identity. No target enters the output context.

Adequacy: frozen teacher must reach100% template categories and weighted RGB MSE<=.01
on all targets. Capability: ordinary/reset image joint accuracy>=.95 on each seen and
unseen subset in every confirmation/seed; ordinary/reset weighted MSE<=.01; erased
reset joint accuracy drop>=.25; swapped-bank target accuracy>=.95. Flow benefit:
mean ordinary/reset weighted MSE at least5% below matched direct control, with no
joint accuracy regression in any cell. Failure is retained; no adaptive fit extension.
These deterministic synthetic targets cannot establish diverse photographic generation.

Resource caps: each formal fit300s including periodic saves, GPU reserved<=3GiB with
>=1GiB free; preflight development profiles before formal execution. Disk floor3GiB;
no model downloads. Eval wall budget300s per arm, preserving completed raw cells on
failure. CPU tests cover reference field integration, invalid/masked values, context
and frozen-head gradients, sample-local RNG, same-device resume and standalone reload.
All runs save source/settings, optimizer/RNG/progress, raw outputs and existing reports.
Browser limitations and report completion remain distinct. Source freezes during runs.

Claude's initial generic review accepts the controls and emphasizes integration
conventions, input provenance and RNG isolation. Reconciliation will clarify that
flow training samples continuous progress, not a specific inference solver schedule;
zero-input direct regression is a declared baseline, and context erasure is an
intentional causal intervention, not claimed in-distribution performance.

Pre-formal local correction: the source detail level has48 channels, so a32-wide
input projection necessarily discards some noise directions. Separate context width
from generator width and use64 for both new arms; the agent remains width32. The
original16-update width32 development result is retained (1.896s,134MiB); it was
not a quality-selection run. Repeat at most16 updates per arm at64 for corrected
profiling before formal execution. Hidden width defaults to context width when
absent, preserving the first development export's parameter layout. No formal fit
has begun. Baseline usability, fixed before formal results: direct validation seen
ordinary/reset accuracy must each reach.95, otherwise flow-superiority is reported
unassessable even if a raw difference favors flow. Capability failures remain failures.

Three actual compact Claude method reviews completed. Adopt paired counterfactual
swap targets, sample-ID noise isolation, fixed terminal checkpoints and explicit
unequal inference compute. Claude incorrectly said it had avoided the phrase
compute-controlled; its earlier receipt contains that recommendation. We retain
this wording disagreement and use equal-architecture/equal-update throughout.
Erasure is distribution-shifting sensitivity, not independent proof of grounding.
Private implementation and measurements were reviewed locally. Receipts are under
`runs/reviews/conditional_image_v1/`.

Implementation checks:57 scoped tests pass before the width correction; all7 focused
checks pass afterward (59 distinct cases across the two selections). CPU full versus
1+3 resumed updates matches model/optimizer/sampler/RNG exactly. Standalone generator
exports preserve calibration, field settings and frozen modules; source/target
encoders are forbidden during generator-only inference. Paired sampling noise is
explicitly inspected. Counterfactual swap MSE and seen/unseen grouping use swapped
targets. Corrected flow development16 updates completes at158MiB peak; direct16 also
completes. Reports structurally verified and the flow development panel inspected.
The tiny development outputs remain untrained/noisy; no quality selection follows.
Freeze library/recipe source before the formal fits and fresh confirmation.

Post-confirmation diagnostic, declared before execution: keep the completed fits
and all formal8-step results fixed. On validation41002 only, seed13, inspect reset
outputs with Euler1/8/32 and field predictions at progress0/.25/.5/.75/.9, all with
paired fixed noise. Compare predicted endpoint and unchanged noisy input against
the teacher feature target; record seen/unseen categories and feature error. No
optimization, checkpoint selection, extra confirmation fit or revised pass gate.
The intermediate-progress inputs contain target information and are diagnostic
teacher-forcing probes, not evidence of state-only generation. GPU cap3GiB and
wall60s; store all results before interpretation. This separates a wrong/noisy
initial field from accumulated sampling error without assuming either explanation.

## Pure-noise conditioning follow-up — 14 September

The first frozen comparison is complete and retained: direct fits seen generator
combinations but fails unseen combinations; flow sampling is poor even on seen
ones. Validation-only field diagnostics recover100% categories when progress.25
supplies partial target features, versus weak pure-noise endpoint prediction;
Euler32 does not repair it. This motivates a separately declared objective-weighting
comparison, not extension or selection within the earlier fixed fit.

One new fit from the same deterministic source and same generator initialization,
training data, seed41011,314,576 parameters, optimizer and1024-update/batch8 budget.
Set probability.5 of exact progress0; otherwise uniformly sample(0,1). Implement
this by remapping the existing uniform draw, preserving noise and sampler draws.
Only progress weighting changes; no target input at progress0, no pixel/semantic
loss, architecture, sampling or upstream change. Compare the retained uniform-flow
checkpoint, not an extra selected candidate. Existing loss with probability0 must
retain its original algebra/RNG. New fit cap300s,3GiB GPU reserved,1GiB GPU free,
3GiB disk free; same report/checkpoint requirements. Separate fit name flow_zero.

Fresh42073/42074 confirmations,32 pairs each, seeds13/29, Euler8. Primary repair:
seen ordinary/reset image categories>=.95 and weighted RGB MSE<=.01 in every cell;
unseen categories cannot regress versus the retained uniform-flow checkpoint.
Full capability still requires the original seen/unseen and memory-intervention
gates. This distinguishes repaired familiar generation from generalization. No
post-confirmation optimizer updates, solver selection or revised gate. The original
comparison remains failed regardless of this follow-up. Review the generic progress
weighting method with Claude; no private results are exported.

## Conditional generator results

Completed14 September. Initial comparison source98ca98f, pure-noise option7314c56.
Three terminal1024-update fits,314,576 trainable generator parameters each, identical
initial tensors and sampler/CPU/CUDA RNG ledgers. Upstream state, memory, factual
outputs and own codec remain fixed. The new64-wide generator avoids the original
48-to32 input rank bottleneck; no further architecture change or pretrained model.

Fresh41073/41074: direct seen-category accuracy100%, withheld0%, for both ordinary
and reset contexts. Uniform-flow reset accuracy averages14.0625% seen and1.5625%
withheld; ordinary15.625%/1.5625%. Neither method passes any of4 capability cells.
The direct baseline meets its predeclared seen-case usability criterion, so the
flow benefit is assessable and fails: mean ordinary/reset weighted MSE0.05551782
versus0.00828599 direct, with category regressions. The original reference passes
2/2 cells at100%; its broader prior exposure precludes treating it as the matched
new-producer training control.

Validation-only diagnostic: at pure noise, one-pass endpoint categories18.75% seen
and6.25% withheld; at progress.25 with target-containing features,100%/100%. Euler32
sampling remains12.5%/6.25%, versus Euler8 at12.5%/6.25%. No sampler was selected.
This implicates weak target-free conditioning in the tested model, without proving
an exclusive cause or establishing that more optimization/capacity cannot help.
The target-containing diagnostic is not an inference capability.

Separately declared half-mass-at-zero training preserves the default loss, metrics,
RNG and gradients exactly when disabled. It also passes exact resumed training
when enabled. On new42073/42074, mean ordinary/reset weighted pixel error falls
0.05558942→0.04130464 (25.697% reduction). However, reset seen accuracy averages12.5%
and withheld15.625%; ordinary10.9375%/16.40625%. Primary seen-case repair fails and
full capability0/4. Withheld nonregression passes but does not rescue the failed
repair. Do not adopt these experimental checkpoints or describe the image generator
as solved. No post-confirmation optimizer update or further solver selection.

60 distinct relevant tests,1026 NumPy-reproduced formal metrics,18 exact64-history
GPU confirmation reloads plus3 validation exports. Initial tensors, frozen values,
all three formal samplers/RNG and saved source snapshots verified. GPU training
28.140s/27.940s/28.653s, total84.734s; maximum reserved1116MiB. This peak includes the
first driver's full-cache teacher audit, not just isolated training updates. Disk
remains5.6GiB free. All26 reports structurally checked; overview/development figures
inspected. Browser QA remains unavailable. Four actual generic Claude reviews,
private implementation/results reviewed locally; receipts retain the compute-label
wording disagreement. Earlier preflight swap-subgroup scoring predates the correction;
its teacher adequacy and overall gates are unchanged, and the original receipt stays.

[Report](../runs/conditional_image_v1/report.html),
[figure](../runs/conditional_image_v1/overview.png),
[raw verification](../runs/conditional_image_v1/verification.json),
[follow-up](../runs/conditional_image_v1/anchor_verification.json),
[software/replay](../runs/conditional_image_v1/software.json).

### Run and load this experiment

```bash
.venv/bin/python -m experiments.conditional_image --weights runs/producer_refinement_v1/source_8502_control/weights.pt --output runs/conditional_image_new --device cuda --objective flow
.venv/bin/python -m experiments.conditional_image --weights runs/producer_refinement_v1/source_8502_control/weights.pt --output runs/conditional_image_zero_new --device cuda --zero-progress-probability .5
.venv/bin/python -m experiments.conditional_image --weights runs/conditional_image_v1/flow_zero/weights.pt --output runs/conditional_image_eval_new --device cuda --evaluate-only --test-seed 42075
```

Use a fresh output directory. `--stop-after` plus `--resume` preserves same-source,
same-device progress; repeat the original objective/development/progress options.
A mismatch is rejected. `pathwm.models.memory_output.load_model` strictly restores
the standalone export without needing its donor file. Standard model calls run
history/recall normally; for controlled sampling, call its image decoder with the
normalized working/reasoning context and explicit seed/sample IDs. The fixed request
is to render the selected object; arbitrary textual conditioning is not trained.

Next proposal: add decoded-image supervision through the frozen head while keeping
architecture/progress settings fixed, then compare fresh state-binding outcomes.
The current latent-only objective does not ensure visible content accuracy. Continue
checking withheld combinations separately from fitting familiar ones. This is an
unexecuted next comparison, not a known repair or permission to overwrite prior runs.

## Decoded-image supervision comparison — 14 September

User endorsed the next comparison. Add an optional `decoded_image_weight` to the
existing recipe, default0. For flow, decode the one-pass endpoint estimate
`x_t + (1-t)*v`; direct regression decodes its clean prediction. Unstandardize first,
then use the inherited frozen head with autograd enabled. Add coefficient10 times
existing target-foreground-weighted RGB MSE to the unchanged standardized latent
loss. This is a hybrid objective, not a guarantee of the original flow-matching
population optimum. Endpoint supervision is not an unrolled Euler trajectory loss.
The target-derived weighting is supervision only; it never enters sampling.

First RED checks: numeric endpoint/unstandardization/RGB weighting and gradient
routing for both direct/flow; coefficient0 avoids the image head and preserves
loss/metrics/RNG/gradients; finite nonnegative coefficients; exact resume/reload
with nonzero RGB loss. Keep all model architectures, codec and agent frozen as
before. Existing ordinary library/recipe/report interfaces suffice.

Formal comparison: two fresh fits from the preserved deterministic source
`runs/producer_refinement_v1/source_8502_control/weights.pt`, both init41011,
314576 trainable parameters, agent32/generator64, train41001 with128 retained
histories (8 of16 output triples), validation41002 diagnostic only. Both flow,
progress probability.5 at0 otherwise uniform, batch8,1024 updates, AdamW lr.0003,
weight decay.0001, clip1. Only decoded-image weight changes0 versus10. Repeat the
control to audit exact reproduction of the retained previous flow_zero terminal
model and optimizer; its source is preserved. Same updates and RNG, not training
FLOPs/time. One16-update development per arm on41003/41004; diagnostic profiling,
no checkpoint or coefficient selection. Freeze source before formal runs.

Fresh43073/43074 each32 pairs/64 histories, sample13/29 and fixed Euler8. Primary
repair: seen ordinary/reset image joint accuracy>=.95 and weightedMSE<=.01 in every
cell, with no withheld accuracy regression versus control. Full capability remains
all original seen/unseen, erased-memory drop>=.25 and counterfactual-swap>=.95 gates.
Separately report >=5% mean ordinary/reset weightedMSE benefit and category
nonregression on each subset/cell; a pixel benefit does not rescue failed capability.
Compare old direct/reference descriptively only, not as a matched new scientific
factor. Upstream/codec know all categories; no whole-system compositional claim.

Caps each fit300s,3GiB GPU reserved,>=1GiB GPU free,>=3GiB disk free; evaluation
300s per arm, save all results/failures and terminal checkpoints. No downloads or
original checkpoint replacement. Four-mode saved outputs, frozen tensors and factual
answers, independent numeric reproduction, exact GPU export replay and structural
reports/figure inspection. Existing browser restriction is retained as a QA limit.
Claude reviews generic public methodology only. No post-confirmation optimizer
extension, coefficient selection or threshold changes within this comparison.
