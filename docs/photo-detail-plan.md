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
