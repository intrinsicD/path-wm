# Temporal context through frozen image features

16 September. Follow-up to the shared image/video codec and the user's spatial
neighborhood question. Test useful history for a missing current-frame region,
not general motion understanding or future generation. Keep the existing codec,
Run/checkpoint/report infrastructure and original image checkpoint.

## Plan before implementation

- Extend the optional causal mean mixer with an odd spatial kernel and optional
  repeated shared spatial-only residual block. Allow injecting a mixer into
  VideoVAE. Preserve default behavior and legacy state-dict keys.
- Essential tests: neutral initialization, future/invalid-input invariance,
  spatial gradient reach, unchanged three-frame temporal horizon, shared loop
  parameters, and restoration. Run these before training.
- Extend the existing video recipe with a separate `inpaint` task. Freeze image
  encoder AND decoder; gradients through the decoder still train the temporal
  module. Last frame has a centered16x16 RGB patch replaced by0.5 before encoding.
  Prior frames are visible. Targets never enter the model.48x48 decoded real frames,
  same six previously inspected source-disjoint development videos as before.
- Arms: spatial3x3, spatial5x5, spatial3x3 plus two iterations of one shared
  spatial-only residual convolution. All see current/two previous frames only.
  Seeds7401/7402;256 updates;batch2;AdamW lr0.001;gradient clip1. Equal exposure,
  not equal parameter count or compute. Record both. No hyperparameter selection.
- Objective: deterministic posterior-mean RGB reconstruction with4x weight on
  masked pixels and1x outside, normalized by weighted RGB pixel count. No KL or
  image fine-tuning; this is conditional repair on a fixed codec, not VAE fitting.
  Include paired clean clips in the loss to constrain regression.
- Controls: frozen occluded-frame reconstruction, clean frozen image reference,
  wrong history and zero history, plus a trained current-only3x3 control with
  past inputs zeroed. This control matches3x3 parameters but not active inputs.
- Gate for an expanded candidate: >=5% masked RGB MSE reduction versus3x3 AND
  trained current-only on the reserved development source, in BOTH seeds;
  <=2% relative regression on clean and outside-mask MSE versus3x3; >=5% masked
  MSE benefit of correct history over zero AND shuffled history. Also report
  these controls for3x3. Thresholds are a development screen, not a significance
  claim. Preserve all failed results; default unchanged unless separately adopted.
- Budget: eight fits, <=90s each, <=720s total formal training; one <=8-update
  smoke/resume check per necessary path, no extra efficacy fits. <=30MiB new
  artifacts and >=300MiB free disk; stop explicitly if that reserve cannot hold.
- Record protocol/source identities, raw metrics, small reconstruction/error
  examples, standalone reports and exact restart check. No report-renderer change.
  Compare source weights before/after. Actual Claude reviews a public generic
  methodology brief only; save receipts and reconcile material concerns.

Current-only is included before execution to distinguish ordinary spatial inpainting
from useful temporal context. The tiny single reserved source and fixed mask are
development limitations; broad motion, variable masks and scaling remain untested.
