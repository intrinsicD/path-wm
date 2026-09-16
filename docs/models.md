# Models and tensor flow

[Persistent World State](world-state.md): the opt-in modular store, binding/update,
bounded retrieval, context/relation tokens and BeliefAgent session adapter. Includes
training/inference inspection and concrete optional concept/self/feedback clients;
module availability does not establish general learned capability.

[Architecture atlas](architecture-atlas.html): sixteen drawings from the complete
agent loop down to attention, belief correction, memory readers and modality
outputs. It separates the general categorical agent, the current Gaussian photo
experiment and proposed entity-graph/DAG extensions. [Mermaid and source links](architecture-atlas.md).

The modality encoders now support [final all-scale fusion](hierarchy-fusion-plan.md):
`depth` transformer blocks finish each scale before the next merge, then optional
`fusion_depth` blocks jointly process all scale tokens and return the same layouts.
Time/support masks apply in every fusion block. Default0 preserves existing model
weights and behavior. `PyramidEncoder` exposes the image hierarchy to the ordinary
spatial heads; the first two-seed comparison did not support adopting fusion by default.

[Key-box integration](key-box-integration-plan.md) is a controlled entity-store →
belief-workspace → supplied-dynamics planning path. It is separate from the default
multimodal training configuration; current capability limits are in project-state.

The current categorical model is described in [belief and memory](belief-model.md),
with the Gaussian reference and shared adapters in [multimodal architecture](multimodal.md)
and built in [the multimodal recipe](../experiments/multimodal.py). The components
below remain available as focused visual reference experiments.

These are ordinary PyTorch modules. A recipe constructs the pieces and selects
the losses. There is no model registry or hidden experiment coordinator.

## Experimental spatial image VAE

The opt-in `pathwm.models.spatial_vae_v2.HierarchicalVAE` implements the
[expanded R/P/M/C design](spatial-vae-v2-plan.md), preserving the v1 model below.
It adds a local overlapping stem, channel LayerNorm inside residual processing,
separate compression, optional coarsest self-attention with a shared loop and a
convolutional latent-only decoder. `inspect(rgb)` exposes detached stage/loop/posterior
snapshots. `LoopTransformer` generates Fourier positions dynamically and refuses
inputs above its declared token cap; zero iterations bypass position and attention
while retaining the registered weights. P/M are not guaranteed invertible.
`pathwm.evaluation.spatial_vae` provides frozen bounded probes and forward-MAC
accounting. The existing `experiments.spatial_vae` recipe trains/evaluates both
versions; `SpatialVAE.load` dispatches strict v1/v2 exports.


`pathwm.models.spatial_vae.SpatialVAE` is a separate codec implementing the
[explicit-scale design](spatial-vae-design.md). RGB B×3×H×W passes through optional
pre-processing, PixelUnshuffle, local mixing and a named1×1 channel projection at
each stage. `encode(rgb)` returns spatial `mu`, `logvar`, original/padded geometry;
`decode(z, (H,W))` needs only the latent field and requested compatible size. Padding
is bottom/right replication to a multiple of2^K; the decoder crops it away.
Default K3 and latent8 give8×8×8 latent values for64×64 RGB, growing with input area.

Channels, hierarchy depth, per-stage pre/post processing depths and latent channels
are constructor arguments. The `processing` factory isolates block choice from
the stage wiring. Variants are `base`, `attention` (coarsest queries over finer
encoder grids) and `reversible` (attention plus additive local coupling). Only
rearrangement and coupling have specified inverses; projections, attention and
the Gaussian bottleneck do not. No encoder features bypass the latent into decoding.
`vae_loss` uses per-image, original-area-normalized distortion and KL.

`save`/`load` retain strict architecture and weights. The [first comparison](spatial-vae-plan.md#results-15-september)
validates codec mechanics and variable geometry, but all three fail the photo
quality screen. This interface does not replace the current multimodal encoder or
state-conditioned generator; a learned state-to-spatial-latent producer remains open.

The [photo-detail diagnostic](photo-detail-plan.md) inspects frozen input/state
representations without changing them. `RidgeReader` fits linear or RBF kernels with
training-only coordinate statistics and float64 solves; it is an offline diagnostic.
`stage_values` exposes the real encoder, observed/stored states and recalled workspace,
and verifies memory writes are exact copies. Its RGB16 targets test spatial layout;
the codec's raw-detail branch is a separate full-image control.

## Shared spatial image codec for video

`pathwm.models.video_vae.VideoVAE(image, temporal=False)` takes the actual existing
`SpatialVAE` or `HierarchicalVAE` object; it does not copy its weights. The same
`model.image` handles still images and every video frame. `encode(Observation)`
returns a flattened B*T spatial Gaussian posterior; `forward` reconstructs
B*T*3*H*W with masks and original image geometry. Losses must exclude invalid frames.
There is a single `image.*` checkpoint path, avoiding ambiguous aliased loads.

Optional `temporal=True` adds `CausalLatentMixer`: a residual3D convolution reads
current/two preceding latent grids plus elapsed-time and validity channels.
It adjusts posterior means before sampling; log variance stays frame-local. Its
output starts at zero, exactly preserving independent-frame reconstruction. Traces
expose detached before/after means. No RGB features bypass the latent bottleneck.
No hidden state persists between calls. This is observed-clip reconstruction;
there is no learned temporal prior, streaming cache or future-frame prediction.

Alternatively inject `temporal=CausalLatentMixer(Z, spatial_kernel=5)` or set
`spatial_iterations=2` on its default3x3 input block. The repeated spatial-only3x3
block shares weights and does not extend the three-frame temporal horizon. The
default has no extra spatial-block parameters and loads old temporal checkpoints.
These are experimental options: the [frozen-codec comparison](video-context-plan.md)
does not support adopting a larger neighborhood or shared spatial loop by default.

The [recipe](../experiments/video_vae.py) loads trained image weights explicitly,
trains the shared encoder/decoder with video and direct-frame reconstruction losses,
and exports the updated image codec alongside the full video Run checkpoint.
Reconstruct the wrapper with the saved image configuration and temporal flag before
loading `last.pt["model"]`; the ordinary Run handles exact training resume.
[Protocol and results](shared-video-vae-plan.md). The existing BeliefAgent patch/token
encoder remains separate until its spatial-grid adapter is evaluated; codec tests
alone do not establish world-state video understanding.

## Controlled visual memory output

`ConditionalFeatureGenerator` is an optional replacement for the image producer,
constructed by `configure_generator` and restored through `feature_generator`
export settings. Separate context/hidden widths, per-scale residual attention,
workspace cross-attention and joint-scale fusion predict every codec feature.
The direct variant regresses clean features; the flow variant integrates a learned
field from locally seeded noise. Explicit sample IDs control reproducible noise
without entering network context. Generation progress never updates world time.
Training uses detached codec targets; generator-only inference has no encoder input.
Old exports retain StateFeatureDecoder. The [first comparison](image-output-plan.md#conditional-generator-results)
and follow-up fail capability gates; these are experimental weights, not a default
upgrade. This recipe trains a fixed selected-object request through working/reasoning
tokens, not arbitrary natural-language image requests.

The recipe can add decoded-image supervision through the frozen head. `endpoint`
uses a one-pass clean-feature estimate; `sample` differentiates through the actual
fixed Euler trajectory from noise. These change training objectives only; exported
inference uses the same generator and sampler. Coefficient0 preserves prior loss
and RNG, and no target image becomes a generation input. The [new comparison](image-output-plan.md#decoded-image-supervision-results)
improves weighted image error but still fails reliable placement and full capability.

The [real-photo continuation](real-photo-plan.md) uses this same generator with
uniform image loss on actual COCO crops. It removes the synthetic-background
`PixelMedianCentering` input wrapper and exports `input_centering=null`; other
upstream parameters and normalization stay fixed. Generator feature statistics
are fitted to training photos, then frozen before baseline evaluation and updates.
Recalled workspace tokens remain the only image-dependent generator input.
The direct codec comparison sees the original photo and carries within-patch
residual detail; its reconstruction quality is not evidence of compact memory.

`MemoryOutput` connects detached episodic snapshots to native factual and image
outputs through working/reasoning tokens. Its optional `workspace_reference` is a
frozen `TokenProbe`: fixed training-channel statistics followed by an attention
classifier. It supplies diagnostics and an optional training loss; its answers never
feed the native factual head or image feature producer. Strict exports include its
weights and buffers, so loading does not require the original probe file. The writer,
input codec and reconstruction head stay frozen during recall repair. The
[bounded supervision comparison](recall-repair-plan.md#frozen-reader-supervision-results)
did not improve native recall over equal training; this is not a default repair.

For frozen-stage diagnostics, `configure_output_readout(model, stage)` also freezes
the thinker. The optional stored route copies working/reasoning tokens from the
latest real bank snapshot, supplies zero tokens for absent memory and averages tied
latest candidates after causal validation. `output_tokens` applies identity or fixed
training-only `TokenNormalization` before the native heads. The recipe caches fixed
training tokens and teacher features, while evaluation and standalone exports still
execute the live history/bank route. Stage, statistics and cache identity are explicit;
[the comparison](recall-repair-plan.md#direct-native-output-readout-results) demonstrates
reset-only readout success in one source with ordinary-mode regression.

## Perception

```text
RGB image → encoder → named feature maps → independent output heads
                         fine/coarse          RGB, mask, pose, your own head
```

An encoder exposes `feature_spec: dict[str, FeatureSpec]`. Each entry declares its
channel count, grid size and source. `forward(rgb)` returns a dictionary of tensors
shaped `[batch, channels, height, width]`. Widths and scale counts are not universal
constants. A consumer declares which levels it needs and validates their shapes.

The retained `CNNEncoder` has two processed grids: fine at image_size/4 and coarse
at image_size/8. `width`, `depth` (residual blocks per grid), and `exchange` are
constructor arguments. Cross-scale exchange reads both original incoming grids
before either update is applied. It is a reference architecture, not a claim that
two scales are optimal. Add another encoder as an ordinary `nn.Module` with the
same dictionary interface; no other framework registration is required.

`DinoEncoder` loads explicitly supplied local official ViT-S/14 source and weights.
It provides local patch embeddings, final contextual features, and pooled final
features. Pooling does not constitute another transformer stage. The preserved
reference transform resizes the full RGB64 view to 224 and normalizes it. Keeping
a pretrained architecture does not mean a newly constructed output head is trained.

Output choices:

- `ReconstructionDecoder`: the small original equal-width fine/coarse RGB decoder.
- `DenseHead`: choose levels, native widths, output channels and activation. Each
  level gets its own projection. `retain_statistics=True` carries token mean/scale
  alongside normalized content, preserving the local/context decoder computation.
  Instantiate separate heads to keep RGB and mask parameters independent.
- `PushTPoseHead`: a task-specific example, producing pusher XY, object XY and
  sin/cos orientation. Coordinates are normalized; this does not force every
  encoder to internally represent those six values.

`Perception(encoder, heads, detached_heads=(...))` makes gradient routing explicit.
Detached RGB inputs let that decoder learn without changing E. Other heads can
still train E. Freezing parameters also needs evaluation mode to preserve buffers
and dropout behavior; the training loops handle that for fully frozen submodules.

## Memory and dynamics

```text
observed images + previous actions → encoder + MemoryUpdater → current features/memory
current features/memory + candidate action → Predictor → imagined features
imagined features + same action + previous memory → MemoryUpdater → imagined memory
```

`observe` accepts H images and exactly H-1 connecting actions, plus the action
preceding the first image. The data adapter supplies the start marker only at the
actual beginning of an episode. Short windows initialize memory to zero; this is
not full-episode memory evaluation.

`imagine` receives no future images. Future observations are encoded separately as
detached targets in the dynamics loop. Frozen U parameters still transmit gradients
through imagined activations. Candidate branches never mutate the caller's state.

The retained temporal modules require equal-width input levels; `ProjectFeatures`
is an explicit adapter for other widths. Memory size, action width and predictor
depth are constructor arguments. Predictor residual output is initialized to zero,
so the first forward copies the current features and the first gradient primarily
reaches its output projection. Memory gradients appear after that path learns.

The dynamics recipe freezes E. Joint E/dynamics training needs a separately explicit
stable-target policy; it is not silently enabled by toggling a flag. Auxiliary
physical losses, conditioning and new temporal architectures belong to future
recipes with their own tests.

`choose_sequence` evaluates explicit candidate actions using a supplied cost. The
application owns action bounds, goals and terminal behavior. This is not a migrated
claim of successful PushT/Paddle closed-loop control.
