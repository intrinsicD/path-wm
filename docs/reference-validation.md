# Cached PushT learning and full-source reference validation

The released checkpoint reached 45/50 goals (90%) using the verified full PushT
source and complete source normalization. The cached PushT learning check
completed all 400 updates and its untouched checkpoint shows strong prediction
on the small related subset. Learned control remains
weak: it reached 1 of 5 held-out goals, compared with 5/5 for recorded actions
and 0/5 for stationary actions. This establishes limited control success, not
broad generalization or a successful training reproduction.

The user approved this follow-up after the [first diagnostic report](diagnostic-results.md).
Training, prediction diagnostics and five control cases completed before the
computer crash. Recovery verified the checkpoint against its pre-crash SHA256;
those completed runs were not repeated. Recovery work covers source verification and the released-checkpoint evaluation;
their current status is recorded below.

## Completed 400-update learning check

The run uses the same random initialization, seed 3072, seven training episodes,
one held-out episode, batch 128, architecture, loss, and 400-update learning-rate
schedule as the earlier capped attempt. Caching and zero data-loader workers
allow the schedule to finish in 665.98 seconds (11.1 minutes), within the
840-second cap. The training manifest records clean commit `79ddc55`.

Diagnostics use 512 fixed training windows and all 90 held-out windows. These
come from eight closely related early PushT trajectory variants, so their
predictive advantage cannot establish generalization to diverse trajectories.

| Held-out metric, float32 | Untouched checkpoint | Discarded clone with training-only BatchNorm recalibration |
| --- | ---: | ---: |
| Prediction MSE | 0.19154 | 0.05698 |
| Copy-current-embedding MSE | 1.53825 | 1.58392 |
| Shuffled-action MSE | 0.60674 | 0.51459 |
| Zero-normalized-action MSE | 0.42288 | 0.31528 |
| Three-transition open-loop MSE | 0.29783 | 0.10348 |
| Mean embedding standard deviation | 0.98044 | 0.99298 |

The untouched checkpoint beats copying by 87.5% and shuffled actions by 68.4%.
The severe saved-checkpoint failure at 120 updates is absent after this
completed schedule. Recalibrating BatchNorm still lowers prediction error by
about 3.36 times at unchanged weights; the mismatch has diminished but has not
disappeared. The clone uses at most 512 training windows, with no held-out
calibration or optimizer update. Recalibration remains a diagnostic, not an
adopted change to the baseline.

On exactly the same held-out batch, the saved checkpoint scores 0.07648 in bf16
and 0.19154 in float32. Both the target embeddings and predictions change with
precision. The smaller bf16 error is therefore not an independent model
improvement; the float32 table is the comparison used here, matching control
evaluation precision.

Evidence: [manifest](../runs/diagnostics/pusht_cached_learning/manifest.json),
[completion record](../runs/diagnostics/pusht_cached_learning/status.json),
[diagnostics](../runs/diagnostics/pusht_cached_learning/diagnostics.json),
[same-batch precision probe](../runs/diagnostics/pusht_cached_learning/diagnostics_precision.json).

## Held-out control and action baselines

Five starts were sampled with seed 42 from held-out episode 0, at source steps
7, 35, 53, 61 and 82, without filtering by goal difficulty or model outcome.
All goals are 25 source steps ahead and each rollout has a 50-step budget.
The learned planner uses 300 candidates, 30 iterations and 30 elites, with
per-case reset and solver seeds 1234–1238 and training-only action statistics.

| Controller | Successes | Already successful at reset |
| --- | ---: | ---: |
| Untouched trained checkpoint | 1/5 | 0/5 |
| Recorded-action replay | 5/5 | 0/5 |
| Stationary actions | 0/5 | 0/5 |

The replay results show that these goals are reachable with this simulator and
reset protocol. The single learned success is evidence of limited control,
while four failures show that good predictive error on this narrow subset is
insufficient to establish reliable planning. The five starts share one episode
and are not independent samples of broad task performance.

Evidence: [case manifest](../runs/diagnostics/pusht_cached_control/manifest.json),
[control summary](../runs/diagnostics/pusht_cached_control/summary.json),
[stationary and replay baselines](../runs/diagnostics/pusht_cached_control/action_baselines.json).
The checkpoint hash is
`b11151d1a30f971fff7d86cddf381970094069e443e1c0ae5ca2868bc55db4a6`;
it matches the original control manifest after the crash.

## Full-source reference evaluation

TwoRoom has passed its pinned archive SHA256 check and complete extraction.
Its source contains 10,000 episodes and 920,809 frames; episode offsets are
contiguous and the first and last source pixel/action records are readable.
The decoded HDF5 file is 12,775,849,984 bytes. Evidence:
[extraction receipt](../data/tworoom/extraction.json),
[source check](../runs/diagnostics/tworoom_full_source_check.json).
This does not enlarge or rerun the earlier 32-episode learning experiment.

PushT is fully verified and extracted: 18,685 episodes, 2,336,736 frames, and
46,300,921,856 decoded bytes. Episode offsets and pixel/action row counts agree;
endpoint frames are readable. Evidence: [recovery receipt](../data/pusht/recovery.json),
[extraction receipt](../data/pusht/extraction.json),
[source check](../runs/diagnostics/pusht_full_source_check.json).

The released checkpoint reached **45/50 goals (90%)**, sampled from 50 distinct
episodes using the authors' valid-window rule, with complete source normalization.
The evaluation phase took 171.4 seconds; download, extraction, model loading and
scaler fitting are excluded from that time.

Its startup recorder needed a compatibility correction for
Gymnasium 0.29.1: record reset seed arguments rather than querying a missing
environment property. The failed startup manifest and log remain under
`runs/diagnostics/reference_full_source_startup_failure*`; it produced no
control score. The corrected runner is committed as `513c83c`. This source has
no seed column, so the upstream defaults pass 50 null seed arguments. Sampling
and CEM use seed 42; simulator entropy remains unseeded, as in the upstream
default protocol. The retry preserves the same 50 starts/goals and normalization
as the failed startup; no cases were filtered by outcome.

This provides a successful released-weight control check on the full-source
protocol. It is one 50-case sample under the pinned current SWM implementation
and thin API adapter, not a reproduction of the paper's historical dependency
environment or a training reproduction. It is not directly comparable with our
trained checkpoint's five closely related prefix starts.

Evidence: [case and normalization manifest](../runs/diagnostics/reference_full_source/manifest.json),
[results](../runs/diagnostics/reference_full_source/summary.json),
[reset seed arguments](../runs/diagnostics/reference_full_source/reset_seeds.json),
[reference status](../runs/diagnostics/reference_full_source_status.json).

## Validation and scope

All 11 CPU tests pass after recovery, including causal action timing,
normalization exclusion of held-out episodes, reference computation and
gradients, cached-window identity, source-image injection, and goal sampling
within episode boundaries. The reset-recorder correction was validated by the
completed 50-goal integration run and checking all 50 recorded seed arguments.
[Test record](../runs/diagnostics/post_crash_tests.json). The saved training
status retains its historical
`pending_closed_loop_evaluation` field; the separate control summary above is
the later result.

No additional training was launched during recovery. Long training still
requires the user's next instruction, and component research remains deferred.


## Next validation decision

The released-model control check works on verified full-source data, and the
completed short schedule removes the earlier severe saved-checkpoint predictive
failure. Reliable control from our own training is still unresolved. The next
recommended step is a capped learning check on a broader, explicitly separated
training/held-out episode split from the verified PushT source, with float32
prediction and held-out control evaluated together. Keep the architecture and
objective fixed; the current evidence does not establish a need for a
normalization-method change. No such additional run has been launched.
