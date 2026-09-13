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
