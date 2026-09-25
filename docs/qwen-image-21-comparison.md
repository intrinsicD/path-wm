# Qwen-Image-2.1 versus the current native PATH-WM image path

25 September 2026. Source inspection, not a head-to-head benchmark. Requested by Alex;
reviewed with actual Claude Opus 5.5 at medium effort (receipt below). Compare the
**currently trained native configuration**, not the maximum configurable PATH-WM architecture.
No architecture replacement or parameter increase is adopted by this comparison.

## Published design and source evidence

Qwen-Image-2.1 was released on20September2026. Its documented image transformer is
single-stream,32layers, approximately7B parameters. The separately named conditioning
model is Qwen3-VL8B; the7B number is not the whole pipeline. The release supports
text-to-image, reference-image editing and RGBA output in one pipeline.

Sources: [official release](https://qwen.ai/blog?id=qwen-image-2.1),
[official architecture](https://github.com/QwenLM/Qwen-Image-2.1#architecture),
[transformer configuration](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/transformer/config.json),
[conditioning configuration](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/text_encoder/config.json),
[VAE configuration](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/vae/config.json),
[official Diffusers implementation](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py).
Official repository inspected at `fb7ae1d1f9611cd91524d03c53c5246b36ac8577`.
The implementation is linked by that repository through DiffusersPR14804; do not
confuse it with the older Qwen-Image double-stream implementation.

The released configuration specifies32attention heads of dimension128, hence an
internal width4096, versus64-channel image latents. Equal latent channel counts do
not imply equal model capacity. Its64-channel RGBA VAE reduces height and width by16.
At2048square the latent lattice is128square:16,384spatial tokens before reference
images. The Qwen3-VL pathway jointly encodes instructions and reference images;
reference images are also encoded through the VAE for spatial conditioning. Its
VAE decoder reads the generated VAE latent, not every intermediate Qwen3-VL feature.
In the released transformer, expanded image positions are overwritten with VAE
latent embeddings; image-aware instruction states carry semantic context. It is
not a concatenation of every intermediate vision feature. Thus semantic
representations need not be the sole pixel reconstruction code.

The transformer applies token-causal attention to text and bidirectional attention
within image blocks under a block-causal sequence policy. The fixed reference/text
prefix can be cached across flow-matching Euler steps. This is an execution
optimization enabled by attention dependencies, not evidence of better decomposition.

## Concrete comparison

| Aspect | Qwen-Image-2.1 | Current trained native PATH-WM path |
|---|---|---|
| Image code | Learned64-channel VAE lattice,16x downsampling per spatial axis | Exported64-channel fine perception lattice,4x downsampling; coarse and slots derived from it |
| Semantic input | Separate trained Qwen3-VL8B for instruction/reference context | A retrieved64-value slot, desired-state embedding and source fine tokens; no trained language request path in this experiment |
| Image producer | Approximately7B,32layers,width4096, single-stream DiT |180,032generator parameters;184,448with request module; width64,depth2,fusion1,existing self/cross-attention output blocks |
| Objective | Flow matching; generated image latent decoded by VAE | Flow matching or matched direct regression of target-minus-source native codes;16Euler steps for flow |
| Direct reconstruction | VAE encode/decode possible; lossy architecture, no exact reconstruction guarantee | Generator bypass preserves source code exactly; native decoder remains approximate; slow code inversion tested separately |
| Pixel decoder | Dedicated RGBA VAE decoder | Slot broadcast/mixing decoder with native coarse/fine connections and optional fine-subpixel residual;87,076parameters with both additions |
| Trained visual scope | Released general image generation/editing model |64x64 synthetic RuleWorld scenes; current editing task changes one machine lamp |
| Memory and entities | Reference images/instructions in inference context; inspected pipeline exposes no persistent KG instance store | Existing explicit instance/evidence store; supplied binding retrieves current slot; learned core selection and persistent full-code detail routing remain incomplete |
| Decomposition | RGBA extraction/editing supported; no inspected interface guarantees disentangled camera/light/style/entity/part codes | Slots and masks provide tested narrow object structure; requested general scene factors are unestablished |

Parameter counts were measured on `experiments.unified_session.edit_model('flow',16)`
and `SlotPerception` with pyramid/subpixel connections. The full perception module
in that configuration contains259,448parameters. These are component counts, not
the total broader agent. Qwen's reported7B is roughly39,000times the current180k
producer; its separate conditioning model makes the capability comparison still
less like-for-like. Width/depth can be configured differently in PATH-WM.

Code-volume calculation, not an information theorem: our64x64fine export contains
16,384FP32values versus12,288RGBvalues. At equal spatial resolution, Qwen's64channels
at1/16height and width represent one scalar per16RGBA input scalars, before precision
and other overhead. Our present code is not aggressively compressed. Neither this
count nor a full-rank stem proves global invertibility.

## What size explains and what it does not

Scale and training are major differences for natural-image quality, diversity,
semantics and instruction following. Exact2.1training data composition, objective
weights and ablations were not established by the inspected release/configuration
sources; no numerical attribution of its quality advantage to parameters is justified.
Increasing our width cannot substitute for an untrained language/memory route,
missing instance binding, missing supervision or an unsuitable decoder connection.

Our own interventions already distinguish some failure modes:

- Full saved-code confirmation now recovers512observed images within all registered
  fidelity gates using1000pixel-optimization steps, worstRGBMSE1.3615e-5
  ([report](../runs/native_code_refinement_confirmation_v1/report.html)). The fast
  decoder remains approximate. Earlier fine-code inversion recovered32known images fromtwo starts under
  the longer registered budget (worstRGBMSE1.15e-10). The code retained information
  that the fast decoder did not recover on those witnesses; no global theorem.
- Adding a33,280parameter fine-to-subpixel connection reduced validationMSE52.4–52.8%
  versus the matched RGB-only decoder, but failed absolute fidelity gates. This is
  evidence for a useful decoder pathway, confounded with added capacity; it does not
  establish a need to replace the encoder or redesign multiscale representation.
- The first native flow-edit run fails requested change quality even though decoding
  the true target code gives correct lamp states100%. That local edit failure cannot
  be assigned solely to the decoder. The matched direct arm separates some learning
  and flow-objective difficulties; no causal conclusion before its evaluation.
- Eleven unresolved supplied bindings occur in same-kind twin scenes; the actual
  appearance identity binder declines conflicting candidates. This remains a counted
  failure of the current scope, not grounds to silently exclude twins or replace
  memory with a ground-truth mask. Scaling the image generator does not repair it.

Evidence: [code plan](image-code-contract-plan.md),
[decoder comparison](../runs/native_decoder_subpixel_comparison_v1/report.html),
[first flow evaluation](../runs/native_memory_edit_3711_flow_eval_v1/report.html),
[binding diagnosis](../runs/native_edit_binding_diagnosis_v1/).

## Consequences for our next slices

Keep the existing native code and separate three questions: does it retain the
image, can the pixel decoder recover it efficiently, and can the conditioned
producer make the correct change while preserving the rest? Qwen supports this
separation of responsibilities; it does not establish that our multiscale design
needs replacement.

The relevant implementation lessons are spatial detail plus semantic conditioning,
a trained image reconstruction path, and explicit generation/editing objectives.
Our requested persistent entity/component memory and controlled semantic factors
remain additional goals. Naming channels or slots does not make them disentangled;
paired interventions, binding/preservation checks and novel compositions must train
and test their behavior. Camera/lighting inference can also be ambiguous from one
image. Transparent output alone does not establish a decomposed3Dscene.

Standing principles applied: retain fine evidence alongside summaries; prepare
fixed context once and reuse valid projections; distinguish exact bypass from learned
editing and lossy decoding; train the actual routing path. Prefix caching is deferred
until useful editing is demonstrated and profiling justifies it. No Qwen component
is imported, no surrogate replaces a native test, and no broad capability is promoted.

Review receipt: `runs/reviews/scene_capabilities_20260925/qwen21-review.json`.

Review correction: the first peer response inferred ignored binding from similar
aggregate pair accuracies. This is unsupported: the independent diagnosis found
outputs change with the binding. Supervised entity-specific targets already supply
a learning signal. Under-editing is observed; absent conditioning, inadequate
capacity and optimization are not distinguished yet. See `qwen21-correction.json`
in the review directory. No blanket mostly-not-size conclusion is adopted.
