# Encoder visualization audit and hybrid architecture review

8 September 2026. The user asked why coarse PCA looks unusual, whether attention
entropy is correct/useful, and whether convolutional stages should include deeper
transformers. This slice audits the existing frozen artifacts and develops a
proposal. It does not train a new encoder or change the adopted latent contract.

## Bounded audit plan

- Preserve the original first-seed panels, arrays, checkpoints, and training runs.
  Reuse their six fixed test frames and 256 training PCA frames in all four arms.
- Independently reconstruct all four attention heads, check their probability
  normalization and projected output against the real attention operation, and
  compare normalized per-head entropy with the previously saved mean maps.
  Include per-head means, key counts, effective attended-key counts, and the
  relative output change under uniform attention. These are local diagnostic
  measurements, not trained ablations or control evaluations.
- Compare the original joint PCA with training-fitted separate-scale PCA and
  training-fitted PCA after subtracting each image's spatial mean. Report variance
  captured, spatial-versus-image-mean variance, and original color clipping.
  Centering removes image-level information deliberately; it is an inspection
  transform, never an adopted model change or a proof of semantic quality.
- Add essential checks against independent attention outputs, entropy boundary
  cases/head averaging, and synthetic known variance decomposition before running.
- Budget: four frozen encoders, 256 training frames and six already-selected test
  frames each, CPU only, zero optimizer updates; up to ten minutes including HTML.

## Figure contract

Question: how much of the unusual coarse map is projection/global variation,
and what do the actual heads attend to? Static scientific image panels compare
the same first three declared test frames using original joint PCA, separate
scale PCA, and image-centered PCA. RGB feature colors encode three projection
coordinates, not classes, and have training-fixed percentile bounds; independent
bases are not color-aligned. Attention heatmaps use a fixed 0–1 entropy scale;
selected-query probability maps identify direction, head, and query. Raw arrays
and exact metrics accompany the panel, which is embedded in the canonical offline
dashboard and visually checked. The other fixed views remain in the raw audit.

## What the implementation actually does

The deeper/on candidate has three strided 3×3 convolutions overall, producing
two exported resolutions from RGB64. The first two convolutions produce a
16×16×32 shared map. The fine branch applies a1×1 projection to64 channels and
two residual spatial blocks. The coarse branch applies the third strided
convolution to the **unprocessed shared map**, then its own two residual blocks,
producing8×8×64. Each residual block has two3×3 convolutions with affine
GroupNorm and GELU. This is a custom residual convolution package, not a full
standard ResNet backbone.

Each branch then adds learned row/column and scale embeddings. One simultaneous
bidirectional cross-attention exchange follows, then one residual per-token MLP.
There are no stacked within-scale self-attention blocks in this encoder. The
predictor has transformer blocks, but those are a different component.

Both exchange directions consume the incoming features. Fine cannot consume
the coarse summary just produced by that same exchange. Also, deeper fine
processing is not fed directly into the coarse downsampling convolution. Those
are explicit topology choices, not consequences of having two output scales.

The decoder upsamples coarse8×8 features to16×16 with nearest interpolation,
concatenates them with fine features, and uses ordinary convolutions plus two
further nearest upsamplings to RGB64. It does **not** use transposed convolutions
or added residual blocks. Its fine-detail input already belongs to the latent
state; there is no extra raw-image shortcut around that state.

## Numerical audit results

Four frozen encoders from seed7107, the same256 training PCA frames and six
original test frames, zero optimizer updates. Checkpoints remain unchanged.
The [raw summary](../runs/encoder_study_2026-09-08/evaluation/visual_audit/summary.json)
and arrays preserve exact values. These are descriptive diagnostics on an
already-inspected population, not another trained architecture comparison.

### PCA explains only selected directions

The original PCA fits all256 fine plus64 coarse tokens jointly per image, with
one common centering vector and one set of training2nd/98th percentile color
bounds. Four times as many fine tokens weight the fit, but do not guarantee fine
dominance: scale means and within-scale variance also contribute. Projection
colors are not semantic labels, are unaligned across separately trained
encoders, and may clip outside the training bounds.

For deeper/on, the original three components capture **31.31%** of pooled
variance. Separately fitting the coarse scale captures62.14% of its variance in
three components. Across the256 training images, variation in image-level mean
features accounts for **56.66% of coarse variance**, versus **0.48% for fine**.
This helps explain the large coarse color shifts between images.

Subtracting each image's spatial mean and fitting a fresh training PCA still
leaves strong coarse spatial/boundary structure in these views; it does not
reveal a hidden crisp object map. The centered coarse top3 capture57.51% of
the remaining variance. Broad convolutional receptive fields, padding and
learned positions are plausible contributors. No blank/translated-image
intervention or trained coarse-only dense readout was performed, so the audit
does not identify their separate causes or prove absence of local information.

[Full three-view comparison](../runs/encoder_study_2026-09-08/evaluation/visual_audit/deeper_pca_comparison.png)
includes original fine/coarse, separate coarse and image-centered coarse PCA.
Centering deliberately removes potentially useful scene information; it is
only a visualization transform, not a proposed encoder normalization change.

### Attention entropy is correct, but its average hides specialization

The audit independently reconstructs Q/K/V projections, four heads of width16,
the1/sqrt(16) score scaling, key-axis softmax, weighted values and output
projection. It agrees with actual SDPA within float32 accumulation error:
maximum absolute output difference7.15e-7 and maximum relative L2 difference
5.77e-7 across the four active attention modules. Every saved per-query
head-averaged entropy map is reproduced exactly.

The displayed statistic is mean over heads of `−sum(p log p)/log(number of keys)`.
It is not entropy after averaging head distributions. It measures routing
concentration, not feature quality, uncertainty, saliency or causal importance.

| Deeper/on direction | Keys per query | Mean normalized entropy | Individual head means |
|---|---:|---:|---|
| Fine queries read coarse |64|0.999201|0.999223,0.999630,0.998843,0.999106|
| Coarse queries read fine |256|0.798106|0.952686,0.969794,0.321640,0.948304|

All fine-query heads are near uniform. Their mean effective key count
(`exp(H)` before averaging, with H in nats) is63.79 out of64. The coarse average
hides a much more selective third head. Entropy of its head-averaged distribution
would instead be0.869807, showing why the averaging order matters.

On the six frames, **99.69% of coarse attention-output variance** is variation
between image means rather than between locations within an image. This is
consistent with broadcasting an image summary across coarse locations. It does
not prove what semantics the summary carries. Changing this module's weights to
uniform probabilities while keeping projected values gives a1.03% relative
output difference for fine queries and48.31% for coarse queries. These are
module-output differences including output projection/bias, not downstream
pose/reconstruction loss changes, learned replacement results, or control gains.

The [individual-head panel](../runs/encoder_study_2026-09-08/evaluation/visual_audit/deeper_attention_heads.png)
uses the pusher-cell query in the first fixed test frame. Probability divided by
uniform probability shares the baseline1 between key counts; the display clips
at4 and exact unclipped arrays remain available. It is not a map of the entire
module's causal importance.

The existing three-seed trained comparison remains stronger evidence about
architecture utility: exchange helps the deeper package but hurts shallow
selected validation q. This audit does not replace that result with an entropy
threshold or show that both directions are necessary.

## Architecture options and recommendation

A convolution-plus-transformer hierarchy is a well-motivated candidate. It
separates local feature extraction/downsampling from content-dependent mixing
across spatial locations. It does not guarantee object-centric features,
geometry, generalization, useful dynamics or human-readable PCA.

| Option | What it changes | Main reason to test / limitation |
|---|---|---|
| Deeper or ConvNeXt-style convolution blocks | More/local wider-context processing | Strong simpler comparator; extra layers also change capacity and optimization |
| Two self-attention blocks per current branch | Repeated within-scale content-dependent interaction | Direct test of the user's proposal while retaining320×64 outputs |
| Sequential convolution/transformer pyramid | Coarse consumes processed fine features | Tests inherited computation across scales; separate it from block-type effects |
| Three exported scales, e.g.16×16,8×8,4×4 | Additional spatial compression and output state | Requires compatible decoder/readout/predictor interfaces; cannot restore detail already lost in RGB64 |
| Windowed or efficient attention at higher resolution | Reduces fine-scale attention cost | Relevant for screen text/small objects; resolution and data support remain separate interventions |

My recommended first architecture comparison is the current deeper/on reference,
a development-profiled extra-convolution control, and that same reference with
two pre-normalized transformer blocks per branch before exchange. Keep exported
sizes, width64, positions, exchange, D/H, training data and objective fixed.
Retain spatial layouts; self-attention should not pool them into one vector.
Predeclare three seeds, updates/presentations and development-selected cost
controls. Report measured wall time, memory, latency and parameter counts as
well as equal-update results; no design can generally match all costs exactly.
Matching performance is not proof that gains were caused by depth, and three
seeds do not establish statistical equivalence without an equivalence protocol.

Then test sequential coupling as a separate comparison. A sensible eventual
candidate is `conv stem →16×16 transformer stage →strided transition →8×8
transformer stage`, exporting both processed maps, with an optional4×4 stage.
Begin at2/2 transformer blocks; investigate4–6 or a third scale only if the
measured benefit and budget justify it. At16×16 there are256 fine tokens; full
self-attention is plausible for profiling. At screen resolutions it becomes
expensive and windowed/local fine attention with broader coarse interaction is
a more plausible candidate. These are proposals, not implemented architectures.

Before attributing the remaining position error to architecture, retain the
previously proposed longer-budget versus calibrated-position/angle-objective
control. All deeper/on runs selected their final allowed update, and the pose
objective's unequal readiness-scale weighting remains unresolved. Neither more
depth nor attention supplies new supervision or repairs insufficient observed
resolution. Multi-output checks should include RGB, masks/extent and geometry;
eventual adoption as a world model requires compatible fresh U/P, multi-step
action prediction and control evaluation after the existing progression gates.

Keep the existing decoder for the first architecture comparison. Later shared
spatial/query readouts can support RGB, masks, keypoints, depth or application
outputs. If decoding imagined futures, every state input must be available
causally: predicting the full fine/coarse state and decoding it is valid;
future-target image skips are unavailable. Past-detail carry-forward is a
separate causal assumption to evaluate. Skip dropout alone cannot establish
correct imagined outputs. A visual backbone also does not supply temporal
memory, action semantics or a general agent by itself.

## Literature and critical Claude collaboration

- [CvT](https://arxiv.org/abs/2103.15808): convolutional token embeddings and
  projections inside a hierarchical transformer, demonstrated on image tasks.
- [CoAtNet](https://arxiv.org/abs/2106.04803): principled combinations of
  convolution and attention; supports testing hybrid spatial computation.
- [Swin](https://arxiv.org/abs/2103.14030): hierarchical, shifted-window attention
  for efficient multiscale vision and dense prediction.
- [ConvNeXt](https://arxiv.org/abs/2201.03545): a modern pure-convolution backbone
  competes strongly with hierarchical transformers. Attention is not a default
  explanation for every architecture improvement.
- [EfficientViT](https://arxiv.org/abs/2205.14756): multiscale linear attention for
  high-resolution dense prediction; a later efficiency option, not evidence of
  improved dynamics in this project.
- [Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588): identifies
  high-norm background-token artifacts in tested ViTs and studies additional
  computation tokens. This audit has not established that failure mechanism here;
  an unusual PCA picture alone is not a reason to add registers.

These primary papers support architectural candidates, not a conclusion that a
particular hybrid will learn better action-conditioned dynamics on our data.
Relative/convolutional positional bias can coexist with absolute coordinate
outputs. Neither absolute coordinates nor relative bias determine generalization
by themselves, and absolute coordinates are necessary for some computer-use
actions. Test transformations and new configurations explicitly.

Claude completed a public-literature-only review and a critical correction round.
It withdrew claims that fine-token count necessarily dominates PCA, that global
or positional artifacts are necessarily benign, that a convolution control
identifies depth as the cause, that all skip features are invalid for imagined
states, and that successful world models necessarily use convolutional encoders.
It agreed with the bounded reference/convolution/hybrid comparison and separate
topology test. Its broad negative claim about absence of relevant literature is
not adopted as an exhaustive survey; its use of "equivalence" is not adopted as
a statistical result. Codex's proposed alternate EfficientViT identifier in the
challenge was wrong; direct primary-source verification confirmed Claude's
original2205.14756 citation. The disagreement and correction are preserved.

Exact public briefs/replies and CLI receipts are in
`runs/encoder_visual_audit_2026-09-08/collaboration/`. No private code, datasets or
local numerical results were sent. The two calls report$0.85082825 API-equivalent
usage, not a subscription bill. Agreement is not empirical validation.

## Verification, failures and delivery

The17 targeted diagnostic, encoder-study and checkpoint-contract tests pass.
The original three numerical contracts first failed before the audit module
existed; float64 attention equivalence is checked at1e-12. No model code or
training checkpoint was changed. Full historical379-test verification remains
prior evidence; it was not rerun for this additive diagnostic slice.

The first audit stopped on a2.38e-7 float32 accumulation difference near zero.
The final check uses an absolute1e-6 tolerance plus relative-vector1e-6, while
retaining the strict float64 essential check. Replaying NPZ pixels with a
different tensor memory layout then produced1.91e-6 feature differences.
Restoring the original FrameSet channel-last strides reproduces saved fine and
coarse arrays **exactly**, without relaxing the latent equality check.

Reporting first used an incorrectly global runs/ scope; historical relative
source paths require the original study's scoped reader. A subsequent sandboxed
Chromium launch failed, then the full new figure exceeded the3MB portable
payload limit. These failures are preserved in `audit.log`, `audit_verified.log`
and `audit_final.log` under `runs/encoder_visual_audit_2026-09-08/`. The final
report uses the first fixed test example as the compact embed; full three-view
and per-head panels and all numeric records remain. `report_final.log` and the
canonical dashboard receipt report passed local browser verification. This is
an explicit presentation amendment, not selection of a favorable test frame.

The exact numerical audit source is preserved at Git revision7f23d46. The final
entry point additionally invokes the compact report publisher before refresh so
future audits do not repeat the known size failure. To rebuild presentation from
completed numeric arrays without another model evaluation:

```bash
MPLCONFIGDIR=/tmp/path-wm-matplotlib .venv/bin/python -m scripts.report_encoder_visual_audit
```

The [refreshed dashboard](../runs/experiment_dashboard.html) retains all33 formal
training runs and adds this zero-update diagnostic. No hybrid/deeper-transformer
training or new world-model control run was launched.
