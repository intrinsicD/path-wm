# TwoRoom calibration diagnosis and one-epoch PushT continuation

Completed 2026-09-06. The accepted bounded follow-up and all queued evaluations
are complete. **TwoRoom's calibrated diagnostic clone reaches 48/50 primary goals
versus 14/50 originally. PushT continuation improves 17/50 to 30/50.** Calibration
does not improve PushT control. The original trained checkpoints remain immutable.

[Open the verified experiment dashboard](../runs/experiment_dashboard.html).
The [preserved plan/execution log](tworoom-followup-2026-09-06-log.md) records
predeclared protocols, conditional decisions, failures and repairs. All exact
comparisons, sample checks and source hashes are in the
[reconciled result ledger](../runs/diagnostics/tworoom_followup_2026-09-06/followup_summary.json); its
[collector](../runs/diagnostics/tworoom_followup_2026-09-06/collect_results.py) rejects missing final stages.

## Matched closed-loop results

| Checkpoint and protocol | Saved buffers | Training-only calibrated clone | Released reference |
|---|---:|---:|---:|
| TwoRoom 4,074 · goal 25/budget 50 · CEM 30 | 14/50 | 48/50 | 42/50 |
| TwoRoom 4,074 · goal 100/budget 150 · CEM 10 | 0/50 | 13/50 | 5/50 |
| PushT 8,404 · goal 25/budget 50 · CEM 30 | 17/50 | 11/50 | 45/50 |
| PushT 13,933 · goal 25/budget 50 · CEM 30 | 30/50 | 29/50 | 45/50 |

All rows use 50 frozen cases, reset/CEM seeds 1234–1283, history 3, 300 CEM samples,
30 elites, five-block planning horizon, action block 5 and receding horizon 5.
Solver iterations and goal/budget differ only as labeled. Replay reaches 50/50 on
each population. Primary TwoRoom includes four initially satisfied goals:
saved 10/46, calibrated 44/46, released 38/46 among initially unsolved cases.
There are no initial successes in the other rows. These are descriptive outcomes;
no scientific pass threshold was introduced.

TwoRoom calibration gains 34 primary cases and loses none. PushT continuation gains
17 and loses 4 versus 8,404. PushT 8,404 calibration gains 2 and loses 8; final 13,933
calibration gains 5 and loses 6 relative to the saved-buffer checkpoint. Exact case
identities remain in the raw records. The one-case final calibration difference
is not evidence of a general disadvantage; neither tested PushT checkpoint shows
a control benefit from the calibration procedure.

## Calibration diagnosis

Claude implemented [the layerwise calibration utility](../scripts/bn_calibration.py)
and three essential tests through the registered MCP tool. Codex integrated it
into [the mode diagnostic](../scripts/check_training_modes.py), preserving source
checkpoints, parameters, split identities and Torch RNG. Only six buffer tensors
change in each diagnostic clone: mean, variance and batch count in the projector
and prediction projector. Their original BN counts equal their optimizer steps.

Each clone uses 512 fixed **training-window** indices, seed 103072, batch 128 and no
optimizer updates. Each BN layer gets its own pass, with upstream BN and dropout
in evaluation mode. Previously calibrated upstream layers therefore supply the
inputs that the downstream layer will see at inference. These are cumulative
averages of equal-sized per-batch means and unbiased variances, not a pooled
variance over all activations. Static layer execution order and one use per BN
are checked. Calibration indices are disjoint from the 512 saved validation-window
indices; random-window splits can still share episodes and individual frames.

The table reports validation prediction/copy MSE ratios (lower is better):

| Checkpoint | Saved FP32 | Calibrated FP32 | Saved BF16 | Calibrated BF16 |
|---|---:|---:|---:|---:|
| TwoRoom 4,074 | 3.94798 | 0.02965 | 5.01199 | 0.11336 |
| PushT 8,404 | 0.22739 | 0.18511 | 0.22930 | 0.18540 |
| PushT 13,933 | 0.14300 | 0.11349 | 0.14250 | 0.11348 |

TwoRoom FP32 prediction MSE changes 2.63049→0.0279162 while copy MSE changes
0.666288→0.941393. Changing normalization also changes the encoder/target scale;
the absolute MSE reduction alone cannot justify adoption. Its paired 48/50 control
result establishes a substantial buffer intervention effect on these cases.
It does not isolate which of the two BN layers matters or why training produced
the mismatch. Precision drift, changing upstream features and train/eval
activation distributions remain possible explanations.

The separate current-batch BN probe disables dropout but couples examples through
batch statistics. It is diagnostic evidence, not a deployable single-state
controller. Full per-mode prediction, copy, zero, shuffle and short-rollout values
are indexed as native ledger records; current-batch results are labeled separately.

The motivation follows the training/inference distinction in
[PyTorch BatchNorm1d](https://docs.pytorch.org/docs/2.14/generated/torch.nn.BatchNorm1d.html)
and the fixed-weight statistic recomputation rationale in
[fvcore precise BN](https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/precise_bn.py).
Our layerwise method is not fvcore's pooled-population estimator. These sources
motivate the diagnostic; they do not establish this checkpoint's failure mechanism.

## Simulator-grounded TwoRoom action ranking

On the first eight frozen primary cases, each unchanged model scores the same20
physical action candidates: replay, stationary,16 fixed random sequences and both
models' CEM plans. This yields320 records. The25-action open-loop simulation runs
to its full horizon even after a success; it is distinct from 50-action closed-loop
control. The local model's predicted-cost selection succeeds2/8 versus 8/8 for the
released model, with mean physical regret48.88px versus 1.71px.

Selection by measured simulator-image latent cost succeeds8/8 for both, but the
candidate set includes replay of the exact source goal, which can have zero latent
cost. This supports a ranking problem on these candidates without establishing
reliable latent geometry generally. No calibrated ranking rerun was performed.
See the [frozen ranking manifest](../runs/diagnostics/tworoom_followup_2026-09-06/action_ranking/manifest.json) and
[raw ranking result](../runs/diagnostics/tworoom_followup_2026-09-06/action_ranking/ranking/ranking.json).

## PushT continuation and internals

The [continuation config](../configs/reproduction/pusht_epoch1_continuation.yaml)
forks the immutable8,404 checkpoint into
`runs/diagnostics/pusht_epoch1_2026-09-06`. It restores model, optimizer, sampler
position and RNG, keeps the full139,330-update learning-rate schedule, and uses an
operational stop at 13,933. A CPU continuation test verifies exact dropout/RNG
continuation and rejects scientific configuration changes.

The additional5,529 updates took**6,507.798486 training-loop seconds (108m27.8s)**,
including validation/checkpoint overhead, within the 7,800-second additional cap.
Logged update time averaged 1.16764s; setup and final HTML add wall time. It processes
707,712 additional windows, for1,783,424 total: one full-batch epoch, with 124 of
1,783,548 training windows dropped by the last incomplete batch. Training and
validation use the original full source and random-window split. BF16 training,
FP32 validation, batch 128 and full-batch activation checkpointing are preserved.

Final checkpoint, validation and status all reference13,933. Final SHA256:
`151b356addea1a9bc7c939fcd102986ed1e7dfca693212463308b8456b3ca4f0`.
The original 139,330-update ten-epoch reproduction is still incomplete and is not
queued. The bounded prefix ends with explicit `step_limit` status.

Full internals use the same 512 validation windows,1, 024 probe windows and 256
rollout windows as the saved 8,404 and released references. Data/split/window hashes,
history, image size and seeds match exactly. The representation width is 192.

| Checkpoint | Effective rank | Mean state-probe R² | Eight-step rollout/copy |
|---|---:|---:|---:|
| PushT 8,404 | 38.71492 | 0.54617 | 0.29060 |
| PushT 13,933 | 48.42076 | 0.60055 | 0.21061 |
| Released PushT | 88.71905 | 0.72003 | 0.11380 |

These measurements improve with continuation but remain behind the reference.
The short validation ratio at 13,933 is 0.1430, versus 0.2274 at 8,404. Full-window
internals are separate from the noisier32-window scalar snapshots during training.

## Qualitative and dashboard verification

All four fixed-first-case rollout panels were inspected. Calibrated TwoRoom crosses
the doorway and succeeds in 18 actions, versus 21 for released weights. Both final
PushT variants still fail the first case at 50 actions; released weights succeed
in 16. Keeping this preselected failure makes the 30/50 aggregate's limits visible.
Frames come from the simulator executing saved actions; there is no image decoder.

- [TwoRoom calibrated rollout](../runs/diagnostics/tworoom_followup_2026-09-06/calibrated_qualitative/case_0_rollouts.png)
- [PushT 8,404 calibrated rollout](../runs/diagnostics/tworoom_followup_2026-09-06/calibrated_pusht_qualitative/case_0_rollouts.png)
- [PushT 13,933 saved-buffer rollout](../runs/diagnostics/tworoom_followup_2026-09-06/pusht_final_qualitative/case_0_rollouts.png)
- [PushT 13,933 calibrated rollout](../runs/diagnostics/tworoom_followup_2026-09-06/pusht_final_calibrated_qualitative/case_0_rollouts.png)

The final four internals panels were also inspected: attention maps emphasize
object/target regions, the spectrum retains concentrated variance, and nearest
frame retrieval can mismatch agent position despite a similar block pose. These
are qualitative observations on the displayed frames, not generalization claims.

**76 tests pass, including three browser checks.** Mobile chart labels now use
compact chart keys with full identities retained in hover/exact rows. A regression
requires a visible nonzero bar within 390px; actual mobile and desktop captures
were inspected. A navigation-start screenshot race was repaired. Forked image
comparisons now match recorded sampling/preprocessing identities rather than run
directories, excluding mismatched/incomplete references against complete records.
Legacy records without complete identity retain the original same-manifest rule.

The final canonical receipt passes package, data and browser verification:
31 charts, 8 tables and 4 image blocks with 8 images (local/released). Every numeric
record remains indexed. Each completed experiment refreshed the HTML. The final
[QA receipt](../runs/diagnostics/tworoom_followup_2026-09-06/final_qa.json) records exact artifact/screenshot hashes.

## Co-work, deviations and next decision

Claude's completed implementation task used restricted mode with no repository,
file tools or project context. Reported API-equivalent usage was **$0.478356** for
this follow-up; the earlier bare-mode attempt could not authenticate and spent $0.
Broader source-access implementation tasks were blocked before launch by automatic
approval review because repository content would be sent to an unverified Claude
destination. The specific transfer-permission request remains pending. Codex
completed the repository integration, ranking, continuation and dashboard work.
Actual remaining account quotas are not exposed.

The initial queue supervisor exited while its trainer/wrapper stayed active. A
replacement supervisor adopted monitoring of those same processes and ran only
the remaining evaluations. No update was restarted. The original wrapper exit
code was unavailable; final status/checkpoint identity and successful wrapper HTML
verification establish completion, with the unavailable exit code recorded as null.
Both the failed browser capture and the supervisor recovery remain in the log.

Next, isolate the TwoRoom projector and prediction-projector normalization effects
using preserved clones and a predeclared paired comparison; do not infer a cause
from a marginal Gaussianity or position-probe result. Keep PushT's saved-buffer
13,933 checkpoint as the continuation reference. A further training budget and a
group-held-out generalization protocol are separate future decisions. All current
jobs are finished; further training and architecture/objective extensions are not
queued. The history 3 TwoRoom 100/150 check is not full reproduction of the paper's
history 1 protocol. These results measure source interpolation and control on the
frozen source cases, not unseen-configuration generalization.
