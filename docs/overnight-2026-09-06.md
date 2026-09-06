# Overnight results, 6 September 2026

The longer PushT prefix reaches **17/50 frozen goals**, compared with **0/50** for
the preserved ten-minute run and **45/50** for released weights. TwoRoom reaches
**14/50 primary goals**, including four already satisfied at reset, but **0/50**
on the separate longer-goal test. Both implementations learn some short-goal
control; the released-model gap and longer-goal failures remain substantial.
This is a bounded, single-seed study, not a completed ten-epoch reproduction.

[Verified offline dashboard](../runs/experiment_dashboard.html) ·
[Reconciled numeric evidence](../runs/overnight_2026-09-06/overnight_summary.json) ·
[Predeclared plan and execution history](overnight-2026-09-06-log.md)

## Training and matched control

| Dataset | Updates | Training windows processed | Fraction of one epoch | Recorded seconds | Full schedule |
| --- | ---: | ---: | ---: | ---: | ---: |
| PushT | 8,404 | 1,075,712 | 60.31% | 10,625.41 | 139,330 updates |
| TwoRoom | 4,074 | 521,472 | 79.28% | 4,804.89 | 51,380 updates |

Both runs stop with `time_limit`; their saved checkpoint, numbered final snapshot
and final validation identify the same step. The full learning-rate schedules
were not compressed to fit the execution ceilings. PushT conservatively charged
an additional 180 seconds for unsaved work discarded during its documented loader
recovery. Final validation accounts for the small ceiling overrun; restart idle
time is separate from recorded training time. Estimates were approximately 8,400
and 4,100 updates after measuring throughput.

All control rows below have 50 independently recorded case outcomes. Local and
released models use identical cases, reset/CEM seeds and solver settings within
each row. Released PushT evidence was reused only after its checkpoint and frozen
case-manifest hashes were rechecked.

| Dataset and goal offset / action budget | Local | Released | Recorded replay | Stationary | Initially satisfied |
| --- | ---: | ---: | ---: | ---: | ---: |
| PushT 25 / 50 | **17/50** | 45/50 | 50/50 | 0/50 | 0 |
| TwoRoom primary 25 / 50 | **14/50** | 42/50 | 50/50 | 4/50 | 4 |
| TwoRoom longer goals 100 / 150, 30 CEM iterations | **0/50** | 5/50 | 50/50 | 0/50 | 0 |

For primary TwoRoom, success among initially unsolved goals is **10/46 local**
versus **38/46 released**, compared with 0/46 for stationary actions. All four
initially satisfied cases also succeed under both models. PushT has 16 cases
both models solve, one solved only locally, 29 solved only by released weights,
and four solved by neither. The earlier step-375 result remains 0/50.

These are source-population compatibility evaluations. Training and validation
use random windows sharing source episodes/configurations. There is one training
seed (3072); no unseen-configuration generalization or multi-seed confidence is
established. The different short and full-length learning-rate schedules prevent
attributing PushT's gain to update count alone. No numerical success threshold
was introduced, and no passing scientific gate is inferred.

## Prediction, internals and visual evidence

Final in-training prediction/copy and prediction/shuffled-action ratios are
**0.2274 / 0.1825 for PushT**, and **3.9480 / 0.9355 for TwoRoom**. TwoRoom also
predicts worse with recorded actions than with zero actions in this check
(MSE 2.6305 versus 2.3173). Falling training loss alone therefore does not establish
useful dynamics. Absolute latent MSE is not comparable across separate encoders.

Full inspections use the same 512 validation windows per local/reference pair,
1,024 training windows for ridge probes and 256 source windows for eight-step
rollouts, all in float32 with saved BatchNorm buffers. Matched window hashes and
checkpoint identities reconcile. Reinspection differs from in-training
validation by at most 8.23e-6 absolute; it does not change model/checkpoint state.

| Diagnostic | PushT local | PushT released | TwoRoom local | TwoRoom released |
| --- | ---: | ---: | ---: | ---: |
| Effective rank / 192 dimensions | 38.715 | 88.719 | 14.682 | 91.478 |
| Mean probe R² | 0.5462 | 0.7200 | 0.9939 | 0.9900 |
| Eight-step prediction / copy error | 0.2906 | 0.1138 | 3.2835 | 0.1084 |

Probe means have different targets across tasks: PushT averages eight pose and
velocity targets; TwoRoom averages agent x/y. Compare within each dataset.
PushT position readouts reach R² 0.84–0.91, while angle sine/cosine are only
0.42–0.43 versus 0.90–0.92 for released weights. Velocity probes from a single
frame remain near zero for both. TwoRoom position is strongly recoverable, yet
this does not translate into reliable dynamics or longer-goal control. The
32-window live rank measurements are not the full 512-window inspection values.

Saved actions reproduce the recorded outcome, action count and final distance;
TwoRoom also checks final coordinates. Case zero was fixed before outcomes:

- [PushT rollout](../runs/overnight_2026-09-06/pusht_control/qualitative/case_0_rollouts.png): the new model contacts and rotates the block but still misses this goal; released weights and replay succeed.
- [TwoRoom primary rollout](../runs/overnight_2026-09-06/tworoom_primary_control/qualitative/case_0_rollouts.png): the local model remains on the wrong side of the wall; released weights succeed in 21 actions and replay in 20.
- [TwoRoom longer-goal rollout](../runs/overnight_2026-09-06/tworoom_paper_control/qualitative/case_0_rollouts.png): both models fail this fixed case; replay succeeds.

All panels contain actual source/simulator frames, with the last frame repeated
after termination. No image decoder or selected successful showcase is used.
Full quantitative panels and their raw paths are retained in the dashboard.

## Protocol fidelity and literature check

TwoRoom's pinned simulator matches source reset pixels and all 6,250 recorded
transitions across the two prepared 50-case sets exactly. The released checkpoint
has a history-three positional embedding. The pinned released evaluation uses
25/50 goals/budget and 30 CEM iterations. By contrast, the paper specifies history
one, 10 CEM iterations for TwoRoom, and 100/150 goals/budget. Horizon, action block
and replanning block are five in both. Therefore our 100/150 results retain
released history/solver settings and do not reproduce the full paper protocol.
[Paper, Appendices D–F](https://arxiv.org/html/2603.19312v1),
[pinned CEM configuration](https://github.com/lucas-maes/le-wm/blob/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac/config/eval/solver/cem.yaml).

Existing `tworoom_paper_*` path names remain as provenance labels; no frozen case
file or result was rewritten. [Protocol clarification](../runs/overnight_2026-09-06/protocol_clarification.json)
records the discrepancy. The model/objective, .09 SIGReg weight and mixed-precision
recipe remain pinned; numerical suspicions were not treated as proven bugs.

The case sets also differ geometrically: median displacement is **36.35 vs
110.37 pixels**, with **12 vs 41** goals across the divider. All longer-goal cases
come from 101-frame source episodes. This post hoc description prevents treating
the score difference as an isolated action-budget effect; it does not identify
a failure mechanism. [Case geometry](../runs/overnight_2026-09-06/control_geometry.json).

## Exploratory solver check

The matched check completed at 07:42:35. Reducing CEM iterations from 30 to
10 leaves success unchanged: **0/50 local and 5/50 released**, with exactly the
same successful cases. Local case execution fell from 729.37 to
245.60 seconds (2.97×); released execution fell from
680.57 to 230.08 seconds (2.96×).
These timings exclude loading and dashboard publication. All case identities,
seeds, other solver settings and checkpoint bytes match. Ten iterations is a
promising compute-saving setting on this case set; it does not resolve the
control failure or establish equivalence across seeds/tasks. No default changed.
History remains three, so this is not full paper reproduction.
[Reconciled solver evidence](../runs/overnight_2026-09-06/solver_iteration_comparison.json).

The first scheduling attempt checked launch time instead of the committed
condition that the main queue finish by 07:32. The main queue finished at
07:29:51; the corrected check started at 07:33:42 with an estimated ten-minute
pair budget and a fifteen-minute final-report reserve. Both attempts remain in
`solver_iteration_driver.jsonl`.

## Correctness, performance and dashboard repairs

- Validation gets its own CPU random generator; diagnostics no longer perturb optimization. Versioned resume fingerprints allow only declared operational overrides, preserve legacy contracts and reject scientific changes. Resume receipts preserve the original manifest and record overrides/code identity.
- Timed and requested stops validate the actual saved step and keep numbered snapshots. `run/STOP` provides a graceful boundary; interval timing now includes every update rather than one sampled step.
- Chunked HDF5 frame reads avoid expensive strided selections while returning identical pixels, action blocks, episode IDs and starts. Both real sources passed paired complete-item checks on 48 fixed windows, including unused terminal NaNs.
- TwoRoom reuses the pinned licensed simulator, shared CEM interface, source-goal injection and frozen replay/stationary controls. Dataset-aware inspection preserves physical-label row order, actual history/image size and reference normalization.
- Dashboard case identities include dataset/source and goal offset; initial-goal denominators remain separate. Oversized native datasets are partitioned without dropping rows or splitting chart series. Failed publication preserves the previous HTML, artifact and receipt together. Supported 10/5/25-second browser budgets retain every canonical QA check.

| Loader measurement, paired source items | Original median | Optimized median |
| --- | ---: | ---: |
| PushT | 55.94 ms | 3.58 ms |
| TwoRoom | 58.02 ms | 4.83 ms |

These are warm-cache loader measurements under shared-machine load, not 12–16×
training gains. Across all measured updates after recovery, PushT averages
**1.1628 seconds update processing + 0.00327 seconds loader wait** (7,404 updates),
versus roughly 1.9 seconds/update before recovery. TwoRoom averages **1.1687 +
0.00151 seconds** across 4,074 updates. Native batch-128 bf16 failed before its
first update with GPU OOM; full-batch activation checkpointing fits at roughly
3.28 GB peak allocated and keeps the objective/batch semantics. All benchmark
updates were discarded. [Raw loader benchmark](../runs/overnight_2026-09-06/data_loading_benchmark.json).

## Verification, collaboration and remaining work

**64 tests pass**, including both opt-in browser checks. CPU tests cover exact
optimization with diagnostics, stopped/resumed dropout weights, legacy and
operational fingerprints, final validation, data integrity, simulator causality,
control denominators, inspection targets/history and failed dashboard publication.
The main queue completed every training/evaluation/visualization/inspection with
canonical HTML verification. All 2,304 spectrum values are retained in bounded
parts; no dataset exceeds 2,000 rows. The final post-ablation receipt passes packaging, validation, desktop/mobile
verification and source interaction: 31 charts, six tables and four image-panel
blocks. Final desktop and 390-pixel screenshots were inspected locally. On narrow
screens the long-label control chart requires horizontal navigation; desktop is
the clearest view for comparing those bars. Exact tables retain all values.
[Desktop capture](../runs/overnight_2026-09-06/dashboard_final_desktop.png),
[mobile capture](../runs/overnight_2026-09-06/dashboard_final_mobile.png).

Claude completed an independent implementation review and contributed three
essential TwoRoom tests through MCP; those were verified failing before the
implementation and passing afterward. The two completed calls report **$4.164104
in API-equivalent usage**; actual remaining account quotas are not exposed. A
later additional review of code and aggregate results was rejected before launch
by automatic approval review, which treated the transfer to Claude as sensitive
egress to an unverified destination. Specific user approval is pending. No
additional Claude review is claimed; local evaluation and review continued.
[Collaboration status](../runs/overnight_2026-09-06/claude/primary_review_status.json).

The full ten-epoch baseline and unseen-configuration generalization remain
unestablished. Preserve these checkpoints and the earlier references. The next
research step should distinguish dynamics/evaluation calibration and planner
behavior using matched controls before attributing TwoRoom's failure to the
representation regularizer or changing the architecture. No further training is
scheduled beyond this bounded work. Passive TAU/Charades extensions remain deferred.
