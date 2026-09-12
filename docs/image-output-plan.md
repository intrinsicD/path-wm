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
