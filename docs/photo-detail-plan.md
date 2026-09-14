# Locate inaccessible photographic detail

The user asks whether limited model size/training explains the poor photo recall,
and authorizes locating where detail becomes inaccessible. Freeze the completed
`runs/real_photo_v1/training/weights.pt`. Do not change or retrain the agent.

Two complementary checks:

1. Audit the actual image input and the reconstruction control. The agent receives
   processed 32-channel patches; the codec control also receives a separate raw
   within-patch residual. Measure the patch projection spectrum, its nullspace,
   response to zero-mean within-patch detail, and reconstruction with that residual
   removed. Rank deficiency proves a particular linear projection is non-injective;
   small singular values show poor conditioning, not automatically absent information.
2. Fit the same linear and nonlinear kernel-ridge reader families to different
   frozen stages. Predict the 16×16 RGB patch-mean grid, which measures spatial
   structure separately from sub-patch texture. Stages: raw target grid (positive
   control), patch stem, all encoder scales, first observed state, second observed
   state, its working/reasoning subset, post-recall full state and final workspace.
   Verify the stored snapshot equals the pre-write state exactly. Stored memory is
   a copy, so do not interpret later readout failure as damage from copying.

Use the previous 1,024 training and 128 validation groups, selection seed45001.
Reserve a fresh 256-photo test suffix, positions256:512 of the same shuffled test
split; prior test positions0:256 are excluded. One photo per duplicate group.
Development uses only the reserved16 training/validation suffixes, no test scoring.
Previous upstream exposure means the holdout is scoped to this diagnostic.

Readers standardize each flattened coordinate using training-only mean/std (floor
1e-4), center RGB targets with the training mean, and use kernel ridge solved in
float64. Linear kernel is dot product divided by input dimension. Nonlinear kernels
are RBF with bandwidth0.5,1,2 on dimension-normalized squared distances. Ridge
diagonal0.000001,0.0001,0.01,0.1; select by validation RGB-grid MSE independently
for each stage/family, then freeze. Closed-form fitting removes SGD duration as a
probe confound; it does not establish optimal decoding over all functions. Family,
training population, target and selection rules match; input dimensions differ, so
do not claim identical parameter counts. Include training error and one shuffled
training-target linear reader on final workspace as a negative control.

Report grid MSE, mean per-photo PSNR, spatial edge MSE and error relative to the
training-mean grid; save every raw test prediction and selected readout state.
Clamp readout RGB to[0,1] before validation selection and scoring; keep the raw
reader output available through its saved weights. Also report error after removing
each image's per-channel spatial mean, and a target-derived uniform-color oracle,
so global color alone is not mistaken for recovered spatial structure. Spatial
access additionally requires20% improvement over the training-mean spatial error.
Compare native generator output after average pooling to the same grid. Also report
full-image direct-codec and no-residual-codec metrics separately. No synthetic
class labels, no image-content inputs to downstream probes.

Diagnostic criteria fixed before fitting: raw-grid positive-control test MSE<1e-4;
useful spatial access requires at least20% lower grid MSE than the training mean.
Mark a stage transition as a readout degradation only if its selected linear and
RBF errors both exceed1.2 times the preceding stage's respective error, with a
positive paired95% bootstrap interval for each error difference (1,000 draws,
seed46199). These are accessibility checks under these probes, not irreversible
information-loss or general-capability gates. A failed probe is inconclusive alone.

Budget: cache extraction<=180s local GPU, <=3GiB reserved/free>=1GiB; all float64
reader fitting/evaluation<=600s CPU with two threads; disk reserve>=3GiB. No model
download, expansion or additional generator training. Development precedes source
freeze and formal run. Save failure/budget states, strict probe reload, independent
NumPy/primal-dual checks, frozen-source hashes, raw arrays and existing-renderer
standalone reports. Browser QA limitations remain explicit.

Consequential interpretation is reviewed with Claude using a generic public
question about regression probes and rank, without private code/data/measurements.
Run-local evidence will distinguish peer suggestions from verified findings.

## Results and verification

Plan/RED commit `ad85f85`; formal source `1eaa271`. The reserved 16-photo development
check completed in 1.18s extraction and 0.076s fitting, peak86MiB. Its raw-grid reader
did not generalize perfectly with only 16 examples; this was a workflow check, and
the declared full population and reader grid were unchanged. A report-only repair
removed an empty optimizer curve and later removed a misleading generic capability
label from the positive-control result. Receipts preserve the unchanged measurements.

Formal extraction took 18.81s with86MiB peak GPU reserve; all float64 fitting took
10.61s CPU. No model update, download, scaling or extended generator training.
The shared photo helpers now live in `pathwm/data/photo_recall.py`; prior selection
identities match exactly. Old fits retain their frozen source snapshots; a changed
code identity must not be silently accepted when resuming them.

| Readout input | Scalars | Linear grid MSE | RBF grid MSE |
| --- | ---: | ---: | ---: |
| Raw RGB16 positive control | 768 | <0.00000001 | 0.000908 |
| Patch stem | 8,192 | 0.00000242 | 0.001444 |
| All encoder scales | 10,752 | 0.00000706 | 0.001575 |
| First observed state | 960 | 0.033545 | 0.034050 |
| Second observed/stored state | 960 | 0.034603 | 0.034969 |
| Stored working/reasoning subset | 256 | 0.036563 | 0.036473 |
| Full reset/recalled state | 960 | 0.044153 | 0.042649 |
| Recalled workspace | 256 | 0.043975 | 0.042678 |

The training-mean grid has MSE0.056046; native recall after pooling to the same grid
has0.046570. Shuffled-target readout0.056341 confirms it does not recover the correct
image. The raw-grid linear positive control passes. These metrics compare the same
target, train/validation/test populations and reader families, with differing input
dimensions. They are not equal-parameter neural-network comparisons.

The encoder→first-state degradation passes both predeclared reader criteria:
linear difference0.033538, paired bootstrap95% interval[0.031415,0.036076]; RBF
difference0.032475, interval[0.030368,0.034961]. Full stored→reset/recalled state
also degrades in both families: linear27.60%, RBF21.96%. Other transitions do not
pass the composite flag. These intervals condition on this fitted model and reader
selection; they are not independent model-training replications.

Removing per-image channel means reveals that native recall's spatial MSE0.041310
is close to the fixed-mean baseline0.041892. The recalled workspace readers reach
0.040701/0.040733, not the required20% spatial improvement. First-state linear
readout0.033344 barely passes that spatial screen; RBF0.033907 does not. Global
color explains much of the apparent recall improvement.

All writes store the full state exactly. Two snapshots exist and both are selected
for retrieval. After reset, the thinker only updates eight working/reasoning tokens;
other state groups remain from the blank observation. Thus the later drop concerns
reset/recall/readout processing, not a corrupted memory copy or an omitted snapshot.

The separate linear patch audit has48 inputs/32 outputs and16 null directions.
Top-three squared singular-value energy is0.999999992605; squared weight energy
on zero-mean within-patch directions is3.1123e-8 of the total. Float64 algebraic
rank32 differs from rank5 under the usual float32 tolerance. The saved null direction
has maximum float64 response2.40e-14. This proves non-injectivity of the patch layer;
it does not identify all dataset-relevant detail or equate tiny singular values with zero.

Direct codec reconstruction full-image MSE0.000860/31.18dB becomes0.012097/19.74dB
with its extra raw-detail channel zeroed. Native recall is0.058086/12.84dB. These
64px-image measurements must not be compared numerically with the RGB16 grid MSE
as though the targets were identical. The first eight fixed examples visibly retain
layout at the encoder and lose much of it at the first state and recalled workspace.

Six focused tests pass, including primal/dual regression agreement, training-only
normalization, causal frozen-stage extraction and the earlier real-photo checks.
All17 exported readers reproduce saved full test outputs exactly; independent
NumPy normal-equation residuals are below4e-9. 157 numeric values/interval bounds,
52 exact GPU tensors over32 fresh photos, split exclusion, original source weights
and63 snapshots verified. The only subsequent recipe change removes a report label;
the fitted source snapshot remains intact. Three reports pass structural validation;
scientific figure inspected, unchanged renderer browser QA unavailable.

One actual Claude review of generic public probe/rank methods reinforced the
limits of failure-based claims. Normalization cannot remove cross-stage differences
in probe difficulty; no private architecture, code, photos or results were sent.
Receipt: `runs/reviews/photo_detail_v1/`; API-equivalent usage$0.007993, not a charge.

Next proposed repair: train observation/state updating and recall on photographic
spatial supervision, keeping intermediate readouts and held-out controls. Evaluate
the fine-detail input interface separately. This experiment neither retrains those
modules nor proves that model scaling or a stronger decoder cannot help.
