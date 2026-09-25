# Image codes for computation, faithful inspection and generation

25 September2026. Alex requires the model to receive the features it needs and
faithful reconstruction from actual latent codes for debugging and image generation.
He asks Codex and actual Claude Opus5.5 medium to frame the complete path and plan.
This adopts the goal. Alex subsequently approves slice1 in general and adds a
model-controlled DiT/editing direction; see the final amendment. Detailed generator
choices remain proposed. No new representation or model change is made.

## Recommended contract

Use the **existing native fine export** (16×16×64) as the canonical image-code
payload. Derive coarse8×8×64 features through the same frozen hierarchy continuation
and derive the seven slots through existing slot attention. The resulting **full
`FeaturePyramid` remains the input view for consumers and the decoder**. This is a
proposed ownership/interface choice over current features, not a new representation.

Bind the payload and its geometry/validity metadata to encoder/decoder checkpoints,
input resolution/preprocessing and schema. Persist the complete existing `FeatureScale`: values, grid, valid,
times, ends and content_times, plus pyramid condition_time. Record the native
zero16-wide condition explicitly; a different condition changes the continuation.
Keep observed/generated provenance and
absolute capture/availability times explicit. Native computation uses the existing
frame-local encoding convention; provenance times must not accidentally change the
learned feature values when reconstructing the hierarchy.

The seven64-wide slots are **derived summaries**, not another name for the full
image code. A self-contained decode derives the complete pyramid and calls the
existing `SlotPerception.from_pyramid(code).recon` with its pyramid-connected decoder.
This supplies both scales and slots derived from that very code. No independently
generated coarse/slot truth and no hidden target/source-feature lookup.

The encoder and generation producer supply the same canonical payload. Generation
is incomplete if any required decoder input is available only from the target image.
Target codes/images may supervise training, but must not enter inference. A declared
pre-action image code is legitimate conditioning for image editing; it is different
from leaking the desired post-action image. Shape/time/provenance metadata is set
under caller rules, never misrepresented as a new observation.

The fine payload is16,384 floats (64KiB fp32); its coarse view adds4,096 floats
(16KiB). RGB64 is12,288 floats (48KiB). This is **not storage compression**.
A full-frame code is not automatically an object code. Derived views can be cached
under matching code/model identities without becoming separately authoritative.

**Missing interface:** expose the actual merge→stage continuation from the exported
fine state, reusing the hierarchy's operations rather than duplicating their logic.
Require exact equivalence to ordinary encoding on the same settings. This first
contract is specific to native two-scale, depth1, fusion0, no depth-readout encoding.
Reject unsupported configurations; final fused exports cannot be assumed to be
inputs to the earlier merge. No model weights or computation may silently change. Also bind cross_scale=True,
packed_merges=False and native window settings. Refactor the shared continuation
so ordinary forward and replay call the same implementation.

Standing principles: prepare the image code once and share immutable views; retain
multiple resolutions with explicit consumer access; give stored evidence and generated
content separate provenance/lifetimes; test the actual full model and each consumer.
Defer projection caching, sparse reads and quantization until correctness/fidelity
are measured. They may reduce traffic/latency but cannot silently change the decode
contract. Report code bytes separately from parameter count and runtime compute.

## Actual paths and gaps

| Path | Implemented now | What remains |
| --- | --- | --- |
| Observation → image features | `SlotPerception.pyramid`, shared with R2 belief | Persist/self-contained load and inspect the canonical payload and derive its full view with model bindings; distinguish source re-encoding |
| Image features → summaries → image | `from_pyramid`, optional native decoder connections | Trained decoder fidelity is poor; near-exact numerical input inversion only establishes available detail |
| Native core computation | `LatentCore.apply` reads machine/object summaries and concept codes; returns one next-machine token | It cannot currently read all image detail or emit a full image code; define a task-specific detail read/output connection where needed |
| Native persistent visual memory | `UnifiedAgent.visual_memory`/`render_memory` retain/decode a64-value instance slot | This is summary recall; `source_pyramid` re-encodes retained RGB and is not decoding a stored full image code |
| Image-code generation | `ConditionalFeatureGenerator` exists for another output-head/configuration | No native compatible producer/head integration or trained full-code generator is established |
| Debugging | `WorldTrace` captures bounded tensors/summaries | Exact image-code export/load/decode must expose checkpoint identity, omitted data and whether source retrieval occurred |

Evidence sources: `pathwm/models/slots.py`, `multiscale.py`, `latent_core.py`,
`conditional_image.py`, `pathwm/world_state/unified.py`, `inspection.py`.
The [actual-encoder diagnosis](reconstruction-diagnosis-plan.md) establishes recovery
on32known images/two starts, not trained decoder quality or general invertibility.

Each consumer must declare its available source, selected scales/tokens, selection
budget, masks and whether it can request more detail. Availability must be tested
at the consumer boundary. Reaching a consumer is not proof it uses the information;
use task-relevant detail changes plus removal/shuffle controls. Do not feed every
consumer every token indiscriminately or infer that a summary suffices universally.

## Measurable implementation slices, proposed

### 1. Self-contained full-code inspection

Use existing classes and existing experiment/report code. Add the smallest native
save/load/decode path for the existing fine payload, with exact hierarchy continuation, including model/schema bindings. First verify that derived coarse and slots exactly
match those from ordinary full encoding. One actual64×64 full-model command must encode an image, save its fine code, start a fresh
process, load only code plus compatible model weights, and decode it. The fresh
process receives no image filename, RGB tensor, source store or source-lookup hook.

Pass criteria: tensors/masks/metadata round-trip exactly; decoding matches direct
`from_pyramid` output on the same device/settings; reject incompatible model/shape
and incomplete codes explicitly. Record all actual decoder inputs, raw errors and
examples in the existing standalone report. Prove no RGB/source dependency by the
fresh-process API and an unavailable-source test. These are software gates; do not
call the current blurry reconstruction faithful merely because replay is exact.
Use the actual trained pyramid decoder checkpoints3601/3604 and original64-image
evaluation chunks; require equality of coarse, slots, alpha and reconstruction,
then compare scored outputs with original per-image records. Keep target pixels in
the separate scorer, never in replay. Record files opened by the replay process.
Run relevant contract checks and full CPU regression for shared-code changes; retain
default slot-only behavior and account explicitly for source-bound runtime manifests.

### 2. Faithful trained decoding of the full code

Keep the actual encoder and slot extraction fixed initially. Use full model and
all256TRAIN-kind+256validation-kind reference scenes, plus a separately registered
fresh population before qualification. Report pixel/body/gradient errors and image
panels, including worst errors, textures, small objects and lamp details. Numerical
input inversion is a diagnostic, not the operational decoder.

Proposed absolute fidelity target for this64×64 procedural slice: per-image RGB
MSE≤1e-4 on every evaluation image (PSNR≥40dB); population mean machine-body MSE and
adjacent-gradient error≤1e-4. These are proposed acceptance thresholds, not measured
success or a universal definition of perceptual quality. Preserve mask accuracy and
machine-pointer behavior using the prior proposed0.5/1percentage-point guards.

First bounded comparison remains the existing decoder's RGB+.5maskCE versus RGB-only
objective (two matched seeds,2000updates, fixed evaluation every500,10min/6GiB per
run), as detailed in the [diagnosis plan](reconstruction-diagnosis-plan.md).
Its20%relative-gain screen is **not** the final faithful-decoding gate. If it fails
the absolute goal, diagnose the remaining training/connection limitation and agree
one repair slice before adding capacity or a new synthesis path. Never loosen the
fidelity target retrospectively or substitute another codec.

### 3. Actual consumer access and generated full codes

Select one native task where detail matters; audit its real inputs first. Add only
the missing access identified there, with a baseline using the current summaries
and controlled removal/shuffling of the relevant fine information. A summary-only
core is not certified as having full-detail access merely because R2 cached it.

For the first producer integration, use a bounded conditional image-edit task:
predict a RuleWorld post-action image from the allowed pre-action image code and
explicit action/available context. Choose a deterministic task with sufficient
conditioning; do not require pixel agreement with an arbitrary outcome of an
underspecified request. The proposed scope is conditional generation/editing, not
unconditional or text-to-image generation.

Reuse an existing generation module only after its native input/output/head
compatibility is verified. The current `ConditionalFeatureGenerator` is a candidate,
not a drop-in native implementation or an adopted new training objective. Register
producer architecture/init, conditions, seeds, populations, budgets, task gates and
exact derivation and generated-code compatibility checks before implementing/training that missing path.

Freeze a qualified decoder first. Train the producer to output the canonical fine
code from allowed context; target codes/images may supervise training but are absent
at inference. Decode generated codes through the **same** code-only decode entry
point. Check fresh-process generation with targets/source-of-target unavailable,
condition/action interventions, task correctness, changed-region (including lamp) error, unchanged-region preservation,
image quality and a source-copy baseline. Code similarity or encode→decode success
alone cannot pass generation. Check generated-code distribution compatibility too.

### 4. Retained-code recall and broader qualification

If faithful detail is required after the source disappears, persist the required
code under explicit ownership/version/byte budgets. Test source withdrawal, cache
eviction and process restart; do not label re-encoding retained RGB as latent-only
recall. Keep summary recall and full-frame inspection separately labelled.

Expand to representative natural images and requested generation conditions only
after the native path works. Set those datasets, resolutions and quality targets
jointly; procedural64×64 tests do not establish that broader goal.

## Boundaries and next decision

Alex now approves slice1 in general: a self-contained save/load/decode
path around the existing fine code, with derived coarse features and slots and no hidden image
inputs. Slice2 then makes its trained decoding faithful. Slices3–4 identify real
consumer/producer/persistence gaps but do not pre-authorize speculative components.
The earlier objective-only proposal is a diagnostic inside this larger contract,
not the whole goal. No new default, training run or quality promotion in this plan.


## Claude review and reconciliation

Two actual Claude Opus5.5 calls with explicit medium effort are retained in
`runs/reviews/image_code_contract_20260925/`. We selected the fine-canonical proposal
for discussion because coarse/slots can be derived consistently. We rejected a new
standalone decoder head: the native pyramid-connected decoder exists and has been
trained, albeit with poor fidelity. Claude acknowledged its initial contrary claim,
the misclassification of the objective comparison as summary-only, the erroneous
“doubled targets” count, and an overly restrictive ban on pre-image conditioning.
The final review agrees with this reuse-first proposal. No quality result or user
adoption of the precise contract is inferred from that agreement.

## Model-controlled diffusion transformer with direct bypass

Alex's follow-up accepts the first slice in general and proposes a DiT controlled
by the model to modify latent codes for its own needs or the user's request, possibly
as a residual path that can be bypassed for direct reconstruction. This is the new
generation direction; residual parameterization is tentative. Slice1 remains useful
and approved in principle. It need not wait for generator training or gain another
codec. The following is the concrete proposed branch contract, not implemented.

Let `z` be the existing canonical fine code, `C` the model/user conditioning tokens,
and `D` the shared continuation→slots→native decoder:

| Operation | Code passed to D | Allowed generation inputs |
| --- | --- | --- |
| Reconstruct/inspect | Original `z`, unchanged | No sampler invocation |
| Edit | `z + delta_theta(noise, z, C)` as a residual candidate | Source code, request/model context, explicit noise/sample ID |
| Create | `z_theta(noise, C)` | Request/model context and noise; no source required |

The residual is the **final generated code correction**, not a single denoising
velocity/noise prediction added to the source. Its training target would be
`z_target - z_source` in a declared coordinate/normalization convention. Multiple
sampling steps produce it. Ordinary residual connections inside transformer blocks
are a separate architectural property.

Use a **hard reconstruct branch** before generator validation, normalization,
sampling or RNG use. Do not implement bypass as `z + 0 * sampled_delta`: that still
runs the sampler and can propagate nonfinite values. Bypass must preserve code,
metadata and decoder output exactly under the established same-device replay
contract, consume no sampler RNG, and work with the generator absent or raising an
error if called. It guarantees unchanged input to D, not a perfect D.

For an edit, residual addition encourages an explicit source reference but cannot
guarantee preservation of unchanged pixels: the code and decoder are learned and
may be spatially entangled. Verify requested changes and unchanged-region fidelity.
A learned gate or zero initialization is not a substitute for exact bypass. Any
edit-strength slider must be trained/evaluated; multiplying a residual does not
prove smooth semantic strength. A no-op request through the learned branch is its
own quality test, separate from the software bypass.

After either edit/create, derive coarse features and slots from the resulting fine
code with the same frozen modules. Do not mix edited fine features with stale
coarse features or slots. Never publish generated code as observed evidence.

### Model control and training

The model supplies an explicit operation plus conditioning tokens representing its
current goal/request and allowed context, with masks and provenance. The sampling
budget/seed is explicit. Diffusion/flow progress is solver time, not world/event
time. A function accepting request tokens does not establish learned instruction
following: the upstream model and conditioning interface need task supervision.

First freeze the actual encoder and a fidelity-qualified decoder. Train the
producer/editor on actual native codes and aligned source/request/target examples;
use target codes/pixels only in training losses or independent scoring. Keep
reconstruction outside that stochastic path. Do not make the encoder Gaussian or
adopt a VAE simply because the sampler starts from Gaussian noise. Statistics for
feature/residual normalization come from training data and remain checkpoint-bound;
subtract in one consistent space, never mix normalized residuals with raw codes.

Start with the already planned conditional edit whose request has unambiguous
meaning. Include copy-source, request-erased/shuffled and action-shuffled controls,
changed-region correctness, unchanged-region error, no-op cases and valid-code
checks. Separately test source-free generation and later genuine user-language
conditioning. Arbitrary natural-language requests and model-originated goals need
aligned training; the current R2/core is not a trained image-request controller.
Register seeds, data, parameterization, optimizer, steps, gates and resource bounds
before the generator implementation/training slice. No model quality is implied by
this architecture choice.

### Existing machinery and the remaining choice

`ConditionalFeatureGenerator` already has spatial tokens, transformer self/cross
attention, progress conditioning and a noise-to-feature flow sampler. It currently
belongs to another output-head path. Native fine-code output, explicit source-aware
editing, residual-target training and model/request conditioning remain missing.
Reuse its compatible machinery; do not substitute its old codec or claim that it is
already a trained native DiT.

The [original DiT paper](https://arxiv.org/abs/2212.09748) uses a transformer as the
latent diffusion backbone. [Flow matching](https://arxiv.org/abs/2210.02747) trains
vector fields along probability paths; it is a distinct training formulation.
The existing implementation is the latter. A DiT-style transformer can be used
with either direction, but choosing the existing flow objective rather than a
noise-denoising diffusion objective must be explicit. Recommendation: evaluate
reuse of the existing flow-transformer for this branch before implementing another
sampler; do not silently treat that recommendation as Alex choosing flow matching.

Ordering: approved-in-principle code replay/continuation → faithful direct decoder
→ bounded model-conditioned edit → source-free/request-driven generation. The
optional branch does not block or alter the first slice. No new code/training or
fidelity result in this design amendment.


Review reconciliation: a lamp-local pixel edit need not yield a sparse/local latent
residual because the encoder mixes information. Changed/unchanged regions are pixel
metrics; code errors are measured without assuming pixel-to-token locality. The
primary generative test must exercise the agreed diffuser/flow-transformer, not
replace it with one-pass regression; regression can be a declared control. Pixel
metrics before decoder qualification remain descriptive with the decoder limitation
explicit. They cannot establish qualified end-to-end generation. Seeds/sample IDs
may remain caller-owned; model control does not require a learned random-seed policy.
Exact diffusion versus flow training remains to be chosen explicitly.
