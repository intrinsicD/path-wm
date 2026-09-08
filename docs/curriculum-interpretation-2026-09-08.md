# Curriculum interpretation and possible follow-up — 8 September 2026

This is a read-only interpretation of the completed curriculum and a proposed
experimental order. No new training, checkpoint intervention, architecture change,
adoption threshold or follow-up budget was executed or approved in this turn.
The [completed report](training-curriculum-results-2026-09-07.md) and its frozen
decision remain authoritative.

## What the COCO increase establishes

On the same 4,146 internal COCO test images, using the same RGB64 preprocessing,
the original E/D pair after warmup has RGB MSE 0.0060723247. Selected task
adaptation has MSE 0.2721955125: 44.8256 times larger, and 3.9155 times the
train-only mean-image baseline (0.0695168962). The final adapted checkpoint has
MSE 0.2650259918, so this is not confined to the selected checkpoint.
The 45-fold number refers to squared error, not a count of lost features or
45-fold pixel-distance error. The corresponding aggregate RMSE ratio is 6.695.

The [fixed image panel](../runs/curriculum_2026-09-07/generic_reconstruction/coco_retention.png)
shows coarse scene structure surviving while outputs become predominantly white
and blue. It supports an appearance shift; it does not prove that all useful
encoder information disappeared. The test measures D(E(x)), without a task head.

The code explains the leading mechanism. Warmup minimizes COCO reconstruction.
Adaptation updates E, D and a fresh H at learning rate 0.0003, minimizing
PushT reconstruction plus normalized pose MSE. COCO contributes no examples or
retention loss during adaptation. D receives reconstruction gradients; H's pose
loss reaches E but has no direct gradient path to D. Both the image distribution
and the encoder objective change. Keeping reconstruction active on PushT does
not constrain reconstruction on COCO.

Thus the combined E/D system exhibits severe forgetting after specialization.
Its decomposition into decoder specialization, encoder information loss,
encoder/decoder coordinate changes, or pose-gradient effects is not measured.
In particular, the existing CCHI-only warmup arm does not isolate the domain
switch: it starts from random weights, not from the COCO checkpoint.

Sources: [exact metrics](../runs/curriculum_2026-09-07/generic_reconstruction/metrics.json),
[objective and training](../world_model/curriculum/training.py),
[evaluation](../world_model/curriculum/generic_evaluation.py),
[shared RGB conversion](../world_model/curriculum/data.py).

## Training, task features and capacity

The architecture already demonstrates much better generic reconstruction than
the mean-image baseline before adaptation. This does not establish sufficient
capacity for every task, but the retention failure alone does not motivate a
larger encoder. Additional PushT-only updates provide no explicit reason to
recover COCO reconstruction. Additional generic warmup alone also leaves the
later forgetting mechanism unaddressed.

Reconstruction preserves appearance; control benefits from accessible geometry,
motion and task-relevant distinctions. These requirements overlap but have
different priorities. A mostly correct background can dominate pixel MSE while
a small object's pose is wrong. Good reconstruction also need not make pose
linearly accessible: the current decoder is nonlinear, whereas PushT H is one
linear map from 20,480 token features to six pose values. This is a plausible
readout/representation mismatch, not an established diagnosis.

All three selected arms reconstruct task images better than a mean-image control;
COCO-warmup B reconstructs task images better than task-only A but has worse pose
accuracy. A has training/test angle MAE 5.51/27.31 degrees, indicating an
orientation generalization gap. None passes readiness. The equal-total-update
screen favors A, while B has fewer labelled updates. It cannot establish that
pretraining is universally harmful.

External precedents support the distinction without determining this experiment's
cause: [Zhang et al.](https://arxiv.org/abs/2006.10742) learn control representations
invariant to irrelevant visual detail; [He et al.](https://arxiv.org/abs/2111.06377)
show that masked reconstruction can learn transferable visual features. Their
training methods and scale differ from this small ordinary autoencoder.

## Skip connections and application routes

The encoder already uses residual attention/MLP updates. D already receives both
16-by-16 fine and 8-by-8 coarse token maps. Earlier, higher-resolution encoder-to-
decoder skips could preserve detail, as in
[U-Net](https://arxiv.org/html/1505.04597v1). They do not protect weights or retain
generic behavior by themselves. A powerful reconstruction shortcut can also
reduce the incentive for the state used by the task to retain useful information;
that is a proposed failure mode to measure, not an observed skip experiment here.

A useful modular candidate is a frozen generic E/D route plus task-specific
residual adapters, readouts and a task decoder. A task adapter consumes generic
features and adds a learned correction, leaving the original path available to
generic reconstruction and other applications. The frozen original E/D pair
retains its old function under unchanged preprocessing and evaluation state.
Task transfer remains an empirical question; a shallow output adapter cannot
recover information its input has already discarded. The frozen-backbone adapter
principle has precedent in [Houlsby et al.](https://proceedings.mlr.press/v97/houlsby19a.html),
an NLP study, not validation of this vision proposal.

For a world model, future RGB must be decoded from predicted state or causally
available appearance memory. Skips extracted from a true future image would leak
the target. Reusing unmodified current-frame skips may preserve static appearance
but can also keep moving objects at their old positions. If skips carry dynamic
information, predict/transport it as part of the rollout, or restrict such skips
to a separately labelled current-frame reconstruction branch. A task decoder
must consume the same task-state interface for both actual and imagined states.

## Proposed order of discriminating experiments

1. Use fixed validation images for the four old/new E/D combinations. This tests
   cross-checkpoint compatibility; bad swapped pairs alone do not establish
   information loss because E and D can co-adapt their coordinates.
2. Starting from the same COCO checkpoint, compare CCHI reconstruction-only
   adaptation with CCHI reconstruction-plus-pose adaptation. Match image draws,
   updates and optimizer settings. This isolates the additional pose objective
   from the image-domain switch for generic retention.
3. Fit matched task readouts on frozen warmup features, comparing the existing
   linear H with a small spatial/nonlinear head. Use the full training population
   and grouped validation; the earlier 256-frame probe is insufficient to settle
   this. This asks whether task information is already accessible without moving E.
4. Test one retention intervention at a time. The simplest training intervention
   is COCO replay during adaptation: task pose loss + task reconstruction loss +
   a weighted COCO reconstruction loss. Compare separately with a frozen generic
   route plus task adapters/decoder. Record the extra replay presentations and
   compute; do not describe them as free or confuse matched labelled exposure
   with matched total compute. Replay is supported as a continual-learning
   baseline by [Chaudhry et al.](https://arxiv.org/abs/1902.10486).

Predeclare budgets, seeds, task-readiness criteria and a retention guardrail
before running this proposal. Use validation for choices. The existing test set
is now exposed during diagnosis, so further comparisons on it are exploratory;
a confirmatory claim needs reserved independent evaluation data. Follow-up seeds
and downstream U/P should depend on task readiness, not prettier reconstructions.
Every new evaluation must use the standing raw-ledger/dashboard workflow.

The current recommendation is to investigate the domain switch and preserve the
generic route before adding broad architectural complexity. If retention improves
but task pose stays poor, prioritize the readout, spatial inductive bias and
generalization objective; if task information cannot be recovered from frozen
features, investigate representation learning or encoder changes.
