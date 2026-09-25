# Image codes for computation, faithful inspection and generation

25 September2026. Alex requires the model to receive the features it needs and
faithful reconstruction from actual latent codes for debugging and image generation.
He asks Codex and actual Claude Opus5.5 medium to frame the complete path and plan.
This adopts the goal. Alex subsequently approves slice1 in general and adds a
model-controlled DiT/editing direction; see the final amendment. Detailed generator
choices remain proposed; implementation progress is recorded in the execution section.

## Recommended contract

Use the **existing native fine export** (16×16×64) as the canonical image-code
payload. Derive coarse8×8×64 features through the same frozen hierarchy continuation
and derive the seven slots through existing slot attention. The resulting **full
`FeaturePyramid` remains the input view for consumers and the decoder**. This is a
proposed ownership/interface choice over current features, not a new representation.

Bind the payload and its geometry/validity metadata to the encoder checkpoint; record
slot-attention and decoder identities per decode (they are replaceable consumers), plus
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

## Memory-routed scene understanding and controllable factors

Alex extends the requirement: the model must use memory and knowledge-graph entities
and their components to route relevant data to the DiT. Image representations must
support modifying entities, scene composition, style, camera, lighting, overlays
and further aspects needed for complex understanding and editing. These are required
capabilities, not evidence that current codes already expose those controls.

### Proposed ownership and information path

`goal/request + current scene → model query/selection → memory/KG components and
relations → bound control/context tokens plus selected spatial detail → conditional
DiT → canonical fine code → derived coarse/slots → same decoder`.

Keep the faithful fine code as visual evidence, and add only the justified
**interpretable and addressable scene/component interfaces** alongside it. The
network needs learned mappings between those components and the spatial code. A
large prompt, named fields, or a graph with labels does not establish that mapping.
The goal is operational factor control and composition, not arbitrarily reserving
channels and assuming they become independently meaningful.

| Aspect | Required control and binding |
| --- | --- |
| Entities and parts | Persistent identity, instance/part ownership, appearance/shape/state, visible support, occlusion and relations; distinguish same-class instances |
| Scene composition | Placement, relative scale/pose, background, containment/support and front/back ordering; track dependencies |
| Camera | Intrinsics/viewpoint/projection where represented; coordinate frame, units and confidence must be explicit |
| Lighting and material | Separate requested illumination from object material/appearance where evidence permits; include coupled shadow/reflection effects |
| Style | Global or entity-scoped rendering appearance with explicit preservation of requested identity/content |
| Overlays | Image-space versus world-space placement, text/graphics payload, opacity and composition order |

These are scope families, not a requirement for a giant fixed component catalogue.
Start with components for which there is a concrete task and training evidence;
permit learned residual/unexplained detail instead of discarding pixels that do not
fit named factors. Unknown geometry, hidden surfaces or ambiguous lighting/material
must remain uncertain, inferred or requested—not falsely recorded as observations.

The graph owns persistent entities, component versions, relationships and evidence.
A generation/edit request is a task-local desired scene state, separate from the
observed graph. It says which entity/part/aspect to change, what to preserve, and
which coupled effects may change. For example, changing lighting should preserve
identity/material, but may change shadows across several objects; changing a camera
may alter visibility. A user-requested edit never silently rewrites remembered facts.

### Reuse and concrete missing interfaces

`ExactRetriever`, `Query`, `RetrievedContext`, `WorkingContext`, `ContextEncoder`
and `RelationEncoder` already provide bounded retrieval, component identities,
version/time restrictions, tokenization and relation endpoints. `WorldSession.think`
can pass retrieved context to its agent. Those software paths are **not a trained
memory→native DiT controller**, and the native `LatentCore.apply` still uses summaries.

The standard retrieval value budget is1024 scalars and `ContextEncoder` normally
reduces a component to one token. A full fine code contains16,384 scalars. Do not
silently squeeze the whole image through that path, increase every budget blindly,
or declare detail available merely because an entity was retrieved. Use existing
compact components for identity/state/control and explicit compatible spatial-token
reads for needed visual detail, preserving position, masks and ownership. Register
actual selected tokens/bytes, omissions and the request for additional detail.

The missing learned pieces are: task-relevant queries/selection, model/request
conditioning, mappings from component spaces into generator context, binding of
entity/part/factor controls to visual support, and the native producer/editor itself.
Reuse existing modules where compatible; learn a projection only where representation
spaces actually differ. Retrieval may be discrete; do not claim its selection is
trained merely because downstream context projections have gradients.

Each generator context needs a traceable binding from tokens to entity/component
IDs, role (source/reference/desired/preserve), representation/version, provenance,
availability and uncertainty. Unsupported/missing components and truncated reads
stay explicit. Empty retrieval is not proof of absence. Generated hypotheses do not
become observation evidence; storing a requested scene is distinct from observing it.

### How the latent representation becomes editable

Pixel retention is necessary for faithful inspection, but does not establish factor
separation. [Slot Attention](https://arxiv.org/abs/2006.15055) supplies task-dependent
object-centric representations, not automatic camera/style/light decomposition.
Unsupervised disentanglement cannot be assumed without appropriate model/data
biases or supervision ([Locatello et al.](https://arxiv.org/abs/1811.12359)).

Train the control-to-code mappings on paired changes, multiple views or temporal
identity evidence, component/property supervision where available, and preservation
objectives. Evaluate requested changes, identity consistency, unrelated-factor
preservation and combinations unseen in training. A latent axis need not correspond
to one factor, but a requested factor must have a reliable bound read/write operation.
Pure reconstruction or diffusion loss alone is not sufficient evidence of that.

Keep the frozen encoder as the initial baseline. If measured controls cannot read
or modify the needed factors, jointly plan representation-training changes with
reconstruction/identity retention checks. Do not assume the encoder must stay frozen
forever, and do not revise it before identifying that concrete gap. This refinement
preserves the native-code contract while allowing necessary learned structure.

### Measurable next increments

1. Complete the already approved code-only replay/continuation and qualify direct
   decoding. Neither requires inventing scene-factor fields in advance.
2. Choose one **existing native entity/component** and one observed property edit.
   Wire retrieved context to the agreed DiT branch and exercise the actual full
   model. Explicit-ID retrieval can verify wiring but must be labelled supplied
   selection; it does not validate learned query/selection.
3. Establish memory dependence: hide the required reference from other inputs;
   compare correct, absent, irrelevant and swapped entity/component retrieval;
   change the stored component while holding the request fixed. Verify which item
   was selected and that only the intended entity/aspect responds, allowing declared
   physical dependencies. Test stale/retracted components and source withdrawal.
4. Add learned selection with distractors and multiple same-kind entities. Measure
   retrieval precision/coverage/bytes separately from generator fidelity and factor
   correctness. Require correct reference use, not plausible imagery alone.
5. Expand independently to spatial/part controls and then camera/light/style/layers
   with appropriate data and fresh compositions. Current RuleWorld data cannot
   qualify general3D cameras, physical relighting or arbitrary overlays. Declare
   each actual counterpart, metrics, gates and budgets before adding/testing it.

The first generation test remains the actual requested diffusion/flow-transformer;
no substitute toy codec or one-pass generator. Low-level latent error, visual
plausibility, memory use, factor control and faithful reconstruction remain separate
measurements. The user's factor list guides the capability roadmap; it does not
assert that every factor is observable from one image or that a minimal factorization
has already been found. No new modules, training or capability result in this amendment.


Scene-control review clarifications: hard bypass preserves codes, not perfect pixels.
Source-derived masks/positions are inferred bindings, not observed ground truth.
Overlay rendering through existing RGB codes has not been proven impossible; do not
add a compositor/new decoder solely from the list of required factors. If an explicit
compositor later proves necessary, its controls belong to the complete image contract.
Desired-state editing (“lamp on”) and physical action prediction (“press”) require
different supervision/conditioning; pick one explicitly. Re-encoded heads/keys are
diagnostic checks, supplemented by independent target/property evidence. General
entity decomposition may need variable counts and part hierarchies; the present
seven-slot configuration is a baseline, not a permanent limit or general capability.

## Execution: approved initial slices (25September)

Alex explicitly requests establishing the capabilities with Claude. Implementation
begins with code-only replay and the already planned decoder-objective comparison.
The native replay check uses each original first64-image evaluation chunk at seeds
3602/3603 and trained decoder checkpoints3601/3604. Require exact coarse/slot/alpha/
RGB replay and original per-image MSE, compatible model/metadata validation, fresh
process reads limited to checkpoint+code, unchanged RNG/weights, and no encoder
forward. Full native model;128images per checkpoint; software gates only.

P1a registers the four previously proposed decoder runs now: seeds3601/3604,
weights0.5/0.0, native pyramid arm, parentu6000,2000updates, lr3e-4, batch32,
fixed evaluation every500,10min/6GiB per run. Existing256+256 populations and original
input streams. Control final per-image records must exactly reproduce previous
pyramid runs. Mask/pointer metrics are additional records, not changes to the RGB
metrics. Relative20% screen, absolute per-image1e-4 and meanbody/gradient1e-4 fidelity
gates, and0.5/1point mask/pointer preservation gates remain as proposed above.
No selection or gate changes after execution. Any extension is registered separately.


Implementation review reconciliation: actual Claude Opus5.5 medium independently
reviewed the code twice (`runs/reviews/scene_capabilities_20260925/`). Code meaning
is encoder-bound, not decoder-bound: retraining a consumer must not invalidate the
stored fine payload. Native slot-only perception can export the same codes. The
hierarchy continuation validates dimensions, masks, conditions and availability.
Replay exactness requires the same consumer weights/device; consumer changes are
permitted but do not inherit an exact-output claim. The first replay audit failed
because PyTorch lazily imported serialization runtime files; explicit runtime
initialization before the data-read audit repaired this without allowing image or
store reads. Preserve all earlier receipts, including this failed audit.

Next editing protocol direction, before training registration: paired desired
states for both machines on each source, retaining missing/ambiguous retrievals in
all denominators. Use the real store and ExactRetriever; compare retrieved slots
bit-for-bit with the actual encoder's slots. The existing conditional transformer
flow producer is a scoped candidate for the requested diffusion-style editing path,
not evidence of trained DiT control or broad scene decomposition. Source fine
features plus a bound entity/state request condition the generated fine residual;
reconstruction bypasses the producer entirely. Decoder fidelity remains a separate
prerequisite for claiming successful image editing. RuleWorld cannot qualify camera,
lighting, style, overlays or general hierarchical entities.

P1D diagnostic registration (before results): after P1a, evaluate the parent and all
four final decoders on the same256+256 fixed images. No optimization or checkpoint
selection. Split RGB error into4x4 patch-mean and within-patch residual energy;
verify their sum equals MSE within1e-7. Report disjoint entity-region and one-pixel
entity-boundary/interior contributions to total error, plus the fine projection's
singular values. A component exceeding50% in both populations and both seeds is
labelled dominant for this data only. Dominance does not prove a causal capacity
limit or justify a new decoder by itself; it guides a single matched repair or
longer-training comparison. Budget5min/6GiB for the entire diagnostic. Decoder
training/source remain frozen during measurement.


P0 completed: native code persistence and shared hierarchy continuation are
implemented. Final encoder-bound replay passes on both trained3601/3604 checkpoints,
128images each, with exact coarse/slots/alpha/RGB/head outputs and original MSE.
Fresh child reads only checkpoint+code; RGB encoding is forbidden; RNG/weights stay
unchanged. Reports: `runs/native_image_code_encoder_bound_3601_v1/report.html` and
`runs/native_image_code_encoder_bound_3604_v1/report.html`. This is exact replay of
the existing imperfect decoder, not faithful pixels. Earlier whole-model-bound
payload receipts are developmental formats, superseded by the encoder-bound schema.

Verification:971 CPU tests pass (734.93s). Full-native3update interrupted/resumed
training matches uninterrupted model, optimizer, sampler/global RNG, metric rows,
and mask/RGB evaluations exactly. Source/receipts:
`runs/reviews/scene_capabilities_20260925/final-software-check.json` and
`full-cpu-receipt.json`. All completed smoke/replay runs have structurally verified
standalone reports. Actual Claude's two implementation reviews reconciled the
binding and continuation validation defects. P1a quality runs are separate evidence.

P1a/P1D completed: all validity/source/frozen/resource checks pass. Both mask0.5
controls exactly reproduce prior per-image records. RGB-only gains18.36–18.84% on
train and22.46–22.63% on validation; both seeds pass mask/pointer preservation but
fail overall20% relative and absolute fidelity gates. Validation MSE0.006736–0.006740,
worst images0.01117–0.01129. Reports:
`runs/native_decoder_objective_comparison_v1/report.html` and
`runs/native_decoder_error_decomposition_v1/report.html`.
P1D's exact additive decomposition passes on all2560images. RGB-only within-patch
energy accounts for87.04–90.76% of error, and machine regions91.17–91.60%; these
are overlapping diagnostics, not an additive explanation. This locates error but
does not prove architectural impossibility or training saturation.

P1E registered before implementation/runs: test a single optional zero-initialized
fine-to-full-resolution residual connection in the EXISTING BroadcastDecoder:
1x1 projection64→16*32, PixelShuffle4, added to the final64x64 hidden map before
the existing output convolution. Retain coarse8/fine16 connections, existing RGB/
alpha mixture, encoder and slot extractor. The hypothesis is that an explicit
within-patch pixel path helps the measured fine-detail defect; not a new code or
proof that nearest upsampling cannot learn it. Enabled at zero it must exactly
preserve current output and RNG; disabled code keeps the existing branch verbatim.
Train only decoder from the sameu6000 parent, seeds3601/3604,2000updates, RGB-only,
batch32,lr3e-4,evaluation every500,10min/6GiB each. Compare against the matching P1a
RGB-only runs, with identical source batches/initial outputs. Require20% MSE gain
on both256-image populations in both seeds, unchanged absolute every-image MSE1e-4
and meanbody/gradient1e-4, and mask/pointer decline limits0.005/0.01. Record added
parameters and clipping effect. No checkpoint selection or automatic extension.
This is a narrowly justified candidate, not a default adoption; preserve failure.

P1E pre-run review clarifications: insert after finalReLU (`network[10]`) and before
existing outputconv (`network[11]`); the residual is signed. It adds33,280parameters;
capacity and changed global clipping prevent a capacity-independent causal claim.
Report clipping-active fractions from recorded gradient norms. Re-run P1D on both
candidate final checkpoints and require within-patch error to decline in every
seed/population for the scoped mechanism screen. A relative pass means improved,
not faithful. The shared residual alone adds a common alpha-logit shift (softmax
cancels it), but other trained decoder weights can still change masks; preserve
empirical guards. Boundary energy is NOT a proven error floor. The old fine
projection's singular values do not bound information in the full fine export;
no linear probe can establish that a nonlinear code lacks information. Texture
comes from full fine codes, not summary-slot edits, so generation must modify those
codes too. Actual Claude medium review retained these limitations.

P1E completed: both candidates pass the20% relative, within-patch decline, mask
and pointer preservation screens. Train MSE improves42.72–43.47%; validation
52.43–52.77% to0.003183–0.003204. All absolute fidelity gates still fail. Clipping
active fraction is0 for both controls/candidates, so clipping is not an observed
confound here; extra capacity remains one. Comparison/decomposition reports:
`runs/native_decoder_subpixel_comparison_v1/report.html` and
`runs/native_decoder_subpixel_decomposition_v1/report.html`. Keep the path optional;
this supports a useful decoder repair, not faithful reconstruction or general
factorization.

P1F preregistered budget screen: extend the same candidate training recipe to8000
updates at seed3601, without architecture/objective/LR changes. Use an isolated
frozen checkout to permit independent P2 development. Start from the sameu6000
parent, declare8000updates from the outset, pause at2000 and require its model,
optimizer, RNG/sampler and training rows to exactly match the earlier2000-update
candidate before resuming. Final8000 checkpoint only; same256+256 data/gates and
mask/pointer limits against P1aRGB-only. Budget20min totalGPU training time/6GiB;
record overlap with other jobs, so runtime is a ceiling check, not a speed claim.
This is a single-seed progress/fidelity screen. Only if absolute fidelity and
preservation pass will seed3604 run at the same8000budget for confirmation. If not,
preserve the failure and diagnose rather than spending the second-seed budget or
claiming saturation. No hyperparameter/checkpoint search.

## P2 execution registration: one real memory-controlled edit

Before quality training/evaluation, register seeds3711/3712 × objectivesflow/direct,
6000updates, AdamW3e-4,8source scenes ×2machines ×2desired states (=32requests),
TRAIN randomized textures1.0, frozen actualJ6000 perception. Existing
ConditionalFeatureGenerator uses fine64x16x16, depth2/fusion1/hidden64,16Euler steps;
ComponentRequest supplies256source tokens and one stored-slot/state request token.
This is the existing conditional transformer flow producer, not a claim to a new
classical DiT or trained core request selection. Direct regression is only its
matched control. Frozen target-code differences supervise training; target codes
or post-images never enter generation. Train-only calibration is fixed in the
recipe and checkpoint; exact CPU/CUDA resume and frozen/source checks required.
Budget20min/6GiB per training run. Use all final checkpoints, no selection.

Evaluation:128sources from train3702 and validation3703, each with all4requests;
actual UnifiedAgent observes each source, supplied structured entity binding routes
through ExactRetriever to the current validated visual_slot. Bind instance/component
IDs, revisions, evidence and model version; require store/direct payload equality.
Generator training encoder/slot/checkpoint identities must match this store before
creating the run. Refresh the actual source-bound binder by fixed-policy confirmation,
never bypass its source guard for quality evidence. Default-binder smokes remain
software-only. Store encoding staysCPU; GPU generation/decoding is declared and
compared within that evaluation. Decoder for primary P2 is fixed P1Eseed3601u2000,
not whichever later checkpoint scores best; verify matching encoder and record
consumer hashes. Poor full-image fidelity remains an independent failed requirement.

Primary sampleID0, same noise across paired requests/controls. Score correct
request, null request, swapped desired state, wrong instance binding and copy-source.
Coverage≥.95; visible-change code explanation≥.8 and normalized no-op drift≤.1;
pair lamp accuracy≥.90 (both desired states correct, other machine preserved);
null/swapped pair accuracy≤.10 and wrong-binding accuracy against its appropriately
swapped target≥.85; unchanged-region MSE against decoded true target≤1e-3. Lamp/
control conclusions require decoded-target lamp ceiling≥.98; otherwise inconclusive.
Both populations and flow seeds must pass. Full image fidelity cannot be inferred
from a successful lamp classifier. Visibility coverage is reported: zero-pixel-change
requests remain in coverage/lamp denominators; the code-explanation diagnostic
explicitly conditions on nonzero changes, with ≥.98 visibility required for the
code qualification. Failure penalties, fixed before evaluation: explanation0,
normalized no-op drift1, preservation MSE1. Report failure counts and conditional
metrics separately; failed retrievals never silently disappear from gated averages.
No post-hoc threshold changes. Four samplingIDs0–3 may be diagnosed on the first16
sources/population (full native architecture/16steps), without selecting an ID.
Quality evaluation budget15min per full run; diagnostic populations are explicitly
smaller sample counts, not smaller models.

The first implementation remains an isolated candidate until independent review,
final main-source tests and source-current store evaluation. Broader component/
relation routing, learned entity selection, create mode and editable camera/light/
style/overlays remain missing. This slice establishes only the stated entity/lamp
edit if its gates pass.

P3D preregistered optional debugging diagnostic (no new architecture): test whether
code-only pixel refinement can provide faithful slow inspection while the fast
learned decoder is still imperfect. Reuse the already validated native frozen-
encoder inversion algorithm, initialize from P1Eseed3601's decoder, optimize only
pixels against saved fine-code features (Adamlr0.03,1000cosine steps, clamp0..1).
The inverter receives only compatible checkpoint+serializedcode; sourceRGB is
available solely to a separate scorer. Re-encoding candidate pixels is intentional,
not source-image retrieval. Require encoder identity equal to the proven J6000
encoder, unchanged model hashes/parameter gradients/RNG, exact optimizer update
count, finite bounded output, and audited data reads. No generator is invoked.

StageA: first64 existing evaluation images in each population3602/3603, native
model, processing chunks16; decoder start primary and gray start control, both1000
steps. Require every primary imageMSE<=1e-4 and meanmachine-body/gradient<=1e-4.
Control success is reported independently; no claim decoder initialization helps
without its measured comparison. StageB only if primaryA passes: all256+256 same
populations, same primary recipe and gates. Budget600s/2GiB per stage; overlapped
jobs explicitly disclosed, no speed claim. Feature error and RGB error are separate:
small feature error alone never proves recovery or global invertibility. No primary
P2 decoder/gate changes, and no generated off-manifold code validity claim. A failed
1000-step screen remains failed; the earlier10000-step schedule is a possible
separately registered follow-up, not silent tuning. This establishes at most slow
code-only debugging on these populations, not faithful fast decoding or generation.


P2 software integrated: actual Claude implemented the request/flow/bypass and
real-store evaluation path in an isolated checkout; independent review corrected
failure denominators, producer compatibility, source provenance and report failure
status.982 full CPU tests pass (761.43s), including actual-store invalidation,
wrong-version rejection and target fault injection. Full native GPU3-update resume
matches weights, optimizer, all RNG/sampler state and metric rows exactly. The
fresh fixed-policy binder reproduces prior policy and metric files byte-for-byte;
`runs/native_identity_confirmation_scene_edit_v1/binding.json` is source-current.
GPU store smoke `runs/native_edit_gpu_store_smoke_v1` is software evidence only.
All256 registered evaluation scenes have both lamps visible; original strict gates
are unchanged (`runs/native_edit_visibility_audit_v1/report.html`). Quality training
is still separate. Claude's corrected reviews and raw receipts are under
`runs/reviews/scene_capabilities_20260925/`.

P1F completed8000updates within budget, with its2000prefix exactly matching the
prior candidate. Validation MSE0.002082 (worst0.005240) and train0.002021 remain
above absolute fidelity gates. Per the registered stopping rule, no second-seed
extension is run. This is improved reconstruction, not saturation or fidelity.
The run/report is `runs/native_decoder_subpixel_3601_u8000_v1/report.html`; the
prefix and summed-resource audit are `long-prefix-check.json` and
`long-result-check.json` in the review directory. Its isolated source remains intact.


External comparison: [Qwen-Image-2.1](qwen-image-21-comparison.md) distinguishes
semantic/spatial conditioning and7Bimage+8Bvision-language scale from our184,448parameter
request/producer path. This does not adopt a replacement representation or establish
that current decoder/edit failures are size-limited. General factorization remains open.


P2 first flow quality result (`runs/native_memory_edit_3711_flow_eval_v1/report.html`)
is valid but fails: train/validation paired lamp accuracy.3594/.3633, explained
visible change.1620/.0927, normalized no-op drift.3749/.3965. Decoded true-target
lamp ceiling1.0; controls do not establish useful editing. Coverage.9922/.9648
passes, with2/9 unresolved bindings. Claude's read-only diagnosis finds same-kind
twin machines competing for appearance identity; no evaluator mapping bug. Preserve
these counted failures. Binding changes affect generated outputs; similar aggregate
wrong-binding accuracy does not establish ignored conditioning. Direct-control
quality evaluation and second-seed training remain in progress.

P3D full confirmation completes: `runs/native_code_refinement_confirmation_v1/report.html`.
All512observed native images (256train,256validation) pass registered absolute
reconstruction gates using1000pixel-Adam steps initialized from P1E3601decoder.
Train/validation meanRGBMSE3.077e-6/2.699e-6; worst1.3615e-5/1.3377e-5. Mean
machine-body and gradient errors also pass1e-4. Actual frozen encoder, serialized
code-only child inputs, unchanged source/weights/RNG, finite/bounded outputs and
resource checks all pass;376.58s. Existing report renderer structurally verified.
The128image screen passed decoder starts but failed gray starts. Recipe:
`python -m experiments.image_code_refinement --checkpoint runs/native_decoder_subpixel_3601_v1/last.pt --output NEW_RUN --scenes 256 --steps 1000 --starts decoder`.
For one saved code use `--code CODE.pt --decoded OUTPUT.pt` instead of `--output`.
This establishes slow faithful debugging within these populations, not bit-exact
RGB, faithful fast decoding, global invertibility, natural images or generated-code
validity. P2 continues with its original fixed decoder and unchanged gates.

## P4: invocation-local fixed-context KV reuse (approved)

Alex approves implementing the cache comparison. All four earlier producer-training
jobs are complete before shared source edits. Reuse existing normalized/projected
context and each cross-attention layer's K/V only within one inference `features`
call. Keep the uncached reference selectable. Do not cache target self-attention,
change parameters, checkpoint formats, training gradients or reconstruction bypass.
Training/grad-enabled execution and direct regression retain their existing path.

Predeclared validation: actual trained flow3711 and3712 checkpoints, native64width,
256fine tokens plus1request,3blocks,16Euler steps, native P1E3601decoder. Saved
observed codes from the full refinement confirmation, first16train and16validation
images, both machines and both desired states; sampleIDs0and1 for numerical checks.
Compare every solver-state/velocity and final native code at atol2e-5/rtol2e-5;
decodedRGB maxabsolute difference <=2e-5. No task-quality promotion. Full-module
state/RNG unchanged; sameweights/checkpointkeys; actual full-configuration mask,
changed-context/weight freshness and training-gradient checks. CPU full regression.

Timing: RTX3050FP32, two CPU threads, synchronized wall time, batch1and8,16steps,
5warmups and30paired timed repeats with alternating order, fixedseed3711 and
sampleID0, matching native context. Report every raw pair, median latency, ratio,
peakallocated/reservedmemory and cachebytes. Include full producer+native decoder
latency. No concurrent GPU jobs. Budget15minutes/6GiB. Default inference caching
is accepted only if all equivalence checks pass and median producer latency improves
at least5% in one batch size with no >5% regression in either; otherwise keep it
opt-in or remove it based on the measured cause. Timing is local, not a general
speed guarantee. Report owns raw data, identity/source and standalone existing-renderer
HTML; no new reporting framework. Principles: prepare once, scope invariant ownership,
preserve retained information and measure actual execution separately from quality.


P4 measured result: `runs/native_context_cache_v3/report.html` passes all numerical,
frozen-state/RNG, source, resource and default-adoption gates. Both trained seeds,
32source images x4entity/state requests x2sampleIDs =256outputs per seed; all16
states/velocities, final native codes and decodedRGB are bit-identical on the tested
FP32CUDA path. This is512request outputs, not512distinct source images. Fivewarmups,
30alternating paired repeats, no concurrentGPUprocesses; CPU regression ran alongside,
so these are local developer-machine timings, not dedicated-system throughput.

| Batch | Producer uncached / cached | Producer + native decoder uncached / cached |
| --- | --- | --- |
|1|28.747 /22.187ms|37.000 /30.161ms|
|8|47.397 /42.975ms|59.274 /54.519ms|

ProjectedK/V occupy394,752bytes per sample; preparedcontext/mask add66,049bytes,
for460,801bytes of cache tensors (batch8:3,686,408bytes). Actual peakallocation and
reservation are separately in the result. No model parameters/checkpoint fields
were added. Inference eval/no-grad flow caching is now the default; explicit
`cache_context=False` keeps the reference. Training/grad/direct paths are unchanged.
The samefullnative3-update GPU training, paused at1then resumed, matches uninterrupted
weights/optimizer/rows/all RNG/sampler fields exactly (`cache-resume-check.json`).
The first report needed a missing row `split` label; its numerical results were
preserved, report repaired with an audit receipt, and v2/v3 regenerate independently.
Allthree timing trials pass; v3 adds an explicit post-timing frozen-state audit.
Full CPU regression:988passed, exit0; `cache-full-cpu.log` and
`cache-full-cpu.json` preserve the final run. Interrupted earlier attempts are not
counted. Six new native-configuration cache checks pass.
Independent actualClaude medium code review finds no blockers; its attached-head
traversal recommendation is applied. Numerical evidence is scoped to tested
FP32CUDA/CPU paths, not a guarantee of bit-exactness across all kernels/precisions.
No editing-quality, generation or representation-capability promotion follows.
