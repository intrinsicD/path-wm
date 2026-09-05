# Bounded baseline diagnostics — 2026-09-05

This is the first diagnostic round. See the [follow-up validation report](reference-validation.md)
for the completed cached PushT schedule, held-out control cases, and source
recovery after the crash.

The local control implementation now matches the upstream evaluator on all ten
paired cases. The short PushT learning check exposed a large BatchNorm
training/evaluation mismatch: estimating its statistics again from training
windows alone restored predictive and action-sensitive behavior on the narrow
subset. TwoRoom completed 400 updates and modestly beat both predictive
controls with its saved checkpoint. Useful control by a model trained here and
full benchmark reproduction remain unestablished. Long training and research extensions remain stopped.

## Scope and data

These are the four checks the user authorized after requesting tests before a
long run. See [the protocol](diagnostic-protocol.md) for pinned sources,
commands and deliberate differences from the published evaluation.

The official full PushT and TwoRoom archives are not yet completely downloaded
or checksum-verified. These checks use standalone HDF5 subsets extracted only
from completely received source chunks:

| Dataset | Complete episodes / frames | Learning split | Limitation |
| --- | --- | --- | --- |
| Official LeWM PushT prefix | 8 / 872 | 7 training, 1 held out | Closely related early trajectory variants |
| Official LeWM TwoRoom prefix | 32 / 2,875 | 28 training, 4 held out | Small, early source subset |

Receipts: [PushT](../data/pusht_prefix/episodes.json),
[TwoRoom](../data/tworoom_prefix/receipt.json). Training normalization uses only
training episodes. Paired released-checkpoint evaluation uses shared statistics
from the supplied subset; full reference statistics are still pending.
TAU Urban AV source media remain retained and are not used in these
action-conditioned learning checks.

## 1. Recorded-action replay

Four official PushT episodes were reset at frame 25 and replayed for ten steps.
Position RMSE was 0.555–0.558 simulator pixels at reset and 0.555–0.559 at
step ten; it did not accumulate across this short replay. The largest
intermediate error was 0.583. Source/render image MAE was 0.486–0.552 on the
0–255 scale. These cases support the outgoing-action alignment and reset
conventions, with small residual physics/render differences.

A separate CCHI source check also showed close reset/render agreement. Missing
initial velocity produced short replay transients, so neither check establishes
universal simulator parity.

Evidence: [official replay](../runs/diagnostics/alignment_official/alignment.json),
[CCHI replay](../runs/diagnostics/alignment_cchi/alignment.json), and image panels
in the same directories.

## 2. Upstream versus local control

The released checkpoint was evaluated on the same ten starts/goals using the
unmodified upstream `World.evaluate`, `WorldModelPolicy` and `CEMSolver` through
a thin runner, and the normal local evaluator's shared control function.

| Measure | Result |
| --- | --- |
| Upstream successes | 5 / 10 |
| Local successes | 5 / 10 |
| Success/failure agreement | 10 / 10 |
| Maximum absolute action difference | 5.96 × 10⁻⁸ |
| Executed step-count agreement | 10 / 10 |

The integration fixes were recorded initial/goal images, reset order and seeds,
persistent CEM randomness, and matching float32 action denormalization. The
normal and paired local paths now share one control loop. A CPU behavioral test
covers source-image injection.

This validates these integrations. The 50% result is not a paper-performance
estimate: it uses a tiny related subset, subset-fitted statistics and ten
single-environment cases. An earlier exploratory selection requiring larger
goal displacement gave 2/10 on both sides; those artifacts remain available.
The final comparison uses the authors' valid-window sampling rule without that
filter, so the two rates should not be interpreted as a model improvement.

Evidence: [final summary](../runs/diagnostics/evaluator_final/summary.json),
[manifest](../runs/diagnostics/evaluator_final/manifest.json), per-case records
and action arrays in the same directory; exploratory results under
`runs/diagnostics/evaluator_pair_seeded/`.

## 3. Capped PushT learning and BatchNorm diagnosis

The full baseline, initialized randomly, trained at batch 128 with unchanged
architecture and objective. It reached 120 of the allowed 400 updates before
the wall-clock stop, finishing in 848.6 seconds. The configured limit was 840
seconds; fetching the next batch delayed the stop check. The training objective
fell from 3.982 to 0.739, but ordinary checkpoint evaluation failed badly.

| Held-out prediction metric | Saved checkpoint | Diagnostic clone with training-only BatchNorm statistics |
| --- | ---: | ---: |
| Prediction MSE | 38.2377 | 0.2200 |
| Copy-current-embedding MSE | 1.0698 | 1.0560 |
| Shuffled-action MSE | 38.2707 | 0.4276 |
| Zero-normalized-action MSE | 38.2726 | 0.3352 |
| Three-transition open-loop MSE | 41.0304 | 0.3026 |
| Mean embedding standard deviation | 0.8847 | 0.9041 |

The clone reset and estimated BatchNorm buffers using 512 fixed training
windows at frozen weights, with dropout disabled. Its held-out evaluation used
90 windows from the held-out episode. No optimizer step or held-out
calibration was used, and the saved checkpoint was not modified. A separate
same-batch-statistics probe scored 0.2217 on those held-out windows; this probe
uses the evaluation batch's statistics and is only a diagnostic.

The roughly 174-fold error reduction at unchanged weights and similar embedding
spread identifies BatchNorm statistics as a major cause of this checkpoint's
evaluation failure. The calibrated clone beats copying and shuffled actions,
which supports learned prediction and action sensitivity on these related
trajectories. It does not establish broad generalization, successful control,
or that recalibration should become part of the reference method.

Evidence: [checkpoint diagnostics](../runs/diagnostics/pusht_prefix_learning/diagnostics.json),
[stop record](../runs/diagnostics/pusht_prefix_learning/status.json), training
manifest and metrics in the same directory.

## 4. TwoRoom learning

The unchanged full baseline completed 400 updates in 638.2 seconds (10.6
minutes), using batch 128 and the bounded data cache. The training objective
fell from 4.131 to 1.000. Final diagnostics cover 512 fixed training windows and
all 285 windows from four held-out episodes.

| Held-out prediction metric | Saved checkpoint | Diagnostic clone with training-only BatchNorm statistics |
| --- | ---: | ---: |
| Prediction MSE | 0.7673 | 0.6038 |
| Copy-current-embedding MSE | 0.7931 | 1.0842 |
| Shuffled-action MSE | 0.8067 | 0.6560 |
| Zero-normalized-action MSE | 0.7786 | 0.6392 |
| Three-transition open-loop MSE | 0.8590 | 0.6720 |
| Mean embedding standard deviation | 0.7286 | 0.8505 |

The ordinary saved checkpoint beats copying by 3.2% and shuffled actions by
4.9% on these held-out windows. This is a modest predictive learning signal
without modifying checkpoint statistics. Training-only recalibration also
helps here, but its effect is much smaller than at the 120-update PushT
checkpoint. Both calibrated and ordinary results remain subset diagnostics;
no TwoRoom control evaluation was performed.

The training runner's 128-window, bf16 validation reported a larger advantage
(0.7108 prediction versus 0.9933 copying). The table above is the final float32
fixed-window diagnostic over all 285 held-out windows. Different sampling,
batch grouping and precision mean the two measurements are not interchangeable;
use the final table for conclusions.

Data validation found 91.4% of recorded position transitions matching
unobstructed clipped-action motion at speed 5. The simulator implements wall
collisions, consistent with the remaining deviations; those residuals were not
all independently replay-verified. Dataset labels do not enter the objective.

Evidence: [checkpoint diagnostics](../runs/diagnostics/tworoom_prefix_learning/diagnostics.json),
[completion record](../runs/diagnostics/tworoom_prefix_learning/status.json), and
[source extraction receipt](../data/tworoom_prefix/receipt.json).

## Data loading and validation

The PushT test lost substantial wall time restarting workers on very short
epochs. Iterating eight batches across two epochs took 46.54 seconds with two
disk-backed workers, versus 0.111 seconds from an already cached dataset with
zero workers. These are iteration times, not end-to-end training speedups or
cache-construction timings. TwoRoom uses the bounded in-memory cache, which
changes I/O only. Cached and uncached windows are checked for identity.

Ten CPU tests pass, including source-image injection, action timing,
recomputation-gradient parity and cached-window equivalence. The baseline
architecture, loss and checkpoint weights were not changed by the diagnostics.

Evidence: [loading measurements](../runs/diagnostics/data_loading.json).


## Decision and next bounded steps

These results support continuing with the published baseline. They identify
integration corrections and a concrete normalization issue; they do not justify
calling the method excellent, rejecting it, or beginning component research.

1. Complete one cached, capped PushT learning check so that the intended 400
   updates and learning-rate schedule finish. Evaluate the untouched checkpoint,
   the normalization diagnostic and a few control goals separately. This will
   test whether the early statistics mismatch settles as training stabilizes.
2. Finish and verify the official archives, then repeat released-checkpoint
   evaluation with the complete reference normalization and benchmark sampling.
   The ten subset cases cannot settle published-performance reproduction.
3. If the saved checkpoint still fails despite good training-mode predictions,
   trace BatchNorm buffers and training/evaluation conventions against the pinned
   upstream implementation before adopting any normalization change.

These are recommendations for the next step, not additional runs launched by
this report. Both capped training processes have finished. Source downloads
continue; no long training or research experiment is scheduled.

## Provenance

PushT training records clean code commit `78f6b58`; TwoRoom records clean commit
`d733959`. Each run manifest records the complete configuration, seed 3072,
episode IDs, action statistics and window counts. The replay, paired evaluator,
learning configurations and diagnostic scripts are versioned in the repository.
Raw measurements and checkpoints are local under `runs/diagnostics/` and are
excluded from Git; the research artifact retains a compact evidence snapshot.

## Broader follow-up

The approved 128/32-configuration PushT pilot completed 1,000 updates and paired
control evaluation. See [the full report](pusht-broader-pilot.md): prediction
modestly improves over controls, but all trained checkpoints reach 0/20 goals
versus 17/20 for released weights. The earlier prefix results above remain
separate and do not establish broader learned control.
