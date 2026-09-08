# Read-only feature and cross-scale diagnostics

Follow-up to the completed frozen P1 package comparison, within the overnight
authorization. No optimization, source checkpoint modification or new dataset.
Produce reproducible scientific PNG panels and source arrays, then integrate the
panels through the existing analysis/dashboard mechanism and inspect the images.

Fit each PCA basis on128 training images:64 privately sampled COCO and64 PushT,
seed92011, per frozen encoder and scale. Show ordinary and per-image-centered
coarse PCA with deterministic signs and training2nd/98th percentile color limits.
Report variance split between image means and within-image spatial changes, top3
variance fraction and clipping. Colors are unrelated across encoder/scale/basis;
interpretability or strangeness of colors does not establish feature usefulness.
Display the first3 COCO and first3 PushT test images, declared before inspection.

Show fixed first6 COCO test RGB/reconstruction/mask panels. For fresh geometry,
show cases0,1,2,3 plus each frozen package's worst seed9107 case by declared case q
(deduplicate, explicitly label the error-selected rows). Display predicted and
true pusher/body-origin coordinates and orientation, output-location probability
maps and the separate orientation-pooling map. These are decoder distributions,
not encoder attention. Quantitative results always use all512 fresh cases.

Audit the existing CNN cross-scale attention with the previously tested explicit
Q/K/V matmul reference. Compute normalized entropy along keys separately for each
head before averaging; compare its output against the actual SDPA output. Report
querywise/headwise entropy, distance from uniform, branch output magnitude and
the effect of two evaluation-only interventions on the same frozen encoder/heads:
uniform attention weights (retaining value and output projections), and zeroing
the entire attention branch output. Apply both directions simultaneously.

Run all512 fresh cases and all three frozen P1 CNN head seeds. Keep per-case raw
predictions, errors, checksums and unchanged-parameter fingerprints. Baseline must
match the earlier untouched fresh evaluation within0.02world-unit coordinate MAE
and0.02degree angular MAE to allow documented CPU/GPU low-order differences;
otherwise investigate before interpreting interventions. Interventions are
distribution shifts applied after training: they measure reliance of this fitted
system, not the best possible model trained without attention. Head adaptation,
attention maps and trained no-exchange controls answer different questions.

Use the existing entropy/reference/variance unit tests plus a new scoped-hook
cleanup test (including exception exit), then a16-case development evaluation,
before the formal read-only512-case unit. No optimizer or target fitting. Budget
15minutes CPU total,4threads; can overlap the single GPU training queue. Preserve
development/formal outputs separately. Add useful panels to the canonical HTML,
with source checks and actual browser verification.
