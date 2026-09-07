# Paddle world model — implementation and measured evidence

Status: the fixed reference stopped at its failed one-step quality gate. Software
execution is verified; learned-control success has not yet been established.
A separately declared predictor continuation passed the gate and five-step
training is active. This report is
updated from immutable stage/evaluation records as they finish.

The supplied [implementation brief](../world_model_codex_implementation_brief.md)
is the contract. [Design notes](../world_model_design_notes.md) retain its history.
See the [plan](paddle-world-model-plan.md), [usage](paddle-world-model-usage.md),
and [verified offline dashboard](../runs/experiment_dashboard.html).

## Implemented system

`world_model/paddle` supplies continuous event-based paddle physics, area-coverage
RGB rendering, deterministic datasets, exact E/U/P/D/H/R modules, four staged
trainers, exhaustive243-candidate five-action planning, and matched evaluation.
Actual observations update real memory exactly once. Each imagined branch runs
P then U with its own memory; later losses differentiate through frozen U.
The deployed planner receives image features, memory and candidate actions.
Simulator labels remain training targets or explicitly privileged diagnostics.

Checkpoints retain weights, optimizer, sampler/RNG state, configuration, source
and dependency identities. Stages resume atomically and keep the selected best
snapshot separate from the most recent optimizer state. Five-step P retains the
selected one-step weights and training-only latent scale statistics. Full runs
stop if one-step prediction fails to improve both latent loss and each H position
coordinate over matched copy-S. Only the labelled smoke bypasses this gate.
Control cases and diagnostic exports also resume using their saved identities.

The previous LeWM code, data and25 development runs remain available. The shared
dashboard includes the new ledgers without mixing their metric definitions.

## Environment and commands

Measured local execution: RTX4090,24564MiB, driver580.159.04; Python3.14.7,
PyTorch2.14.0+cu130, NumPy2.5.2, PyYAML6.0.3, Pillow12.3.0,
matplotlib3.11.1. FP32, TF32 off, four CPU threads. CUDA requires running outside
the tool filesystem sandbox. The documented CLI works from the repository root:

```bash
.venv/bin/python -m world_model doctor
.venv/bin/python -m world_model run-all --config configs/paddle/smoke.yaml --data data/paddle/smoke_v2 --run runs/paddle/smoke_v2
.venv/bin/python -m world_model generate --config configs/paddle/baseline.yaml --output data/paddle/baseline
.venv/bin/python -m world_model verify-data --data data/paddle/baseline
.venv/bin/python -m world_model run-all --config configs/paddle/baseline.yaml --data data/paddle/baseline --run runs/paddle/baseline
```

Executed commands set `MPLCONFIGDIR=/tmp/path-wm-mpl` where appropriate.
`run-all` refreshes and browser-verifies the canonical dashboard after stages and
evaluation. Its repeat invocation resumes compatible work. Completed references
reject changed data/configuration. Use a separate directory for a new experiment.

## Data and correctness

| Split | Episodes | Frames |
|---|---:|---:|
| Train |5000|184808|
| Validation |500|18115|
| Test |500|17831|

There are214754 transitions,2039 catches and20.254% ascending frames;5979 episodes
terminate and21 reach the cap. Compressed NPZ data occupy25949103 bytes.
Generation took71.05s and exact replay verification19.66s. The dataset identity
is `c6d255dc9919b1bb9ce38182f9d180047f964754a32c616db072bc965a3be54f`.
Counts and identities are in `runs/paddle/data_baseline/coverage.json`.

Claude's adversarial physics review exposed an inclusive-contact roundoff case.
The corrected comparator uses the physical8-unit threshold plus1e-10 numerical
tolerance; an8+1e-8 offset still misses. All150 prescribed validation/test pairs
retain continuously sampled centers and byte-identical final images. Exact
enumeration verifies precisely the two expected catching three-action sequences
for all300 members. Old-fingerprint preliminary smoke data remain preserved.

The final integration suite currently has181 passing tests and3 opt-in browser
tests skipped in the ordinary CPU invocation. Those three browser tests passed
when explicitly enabled. Raw logs are `runs/paddle/tests_final_integration.log`
and `runs/paddle/tests_browser.log`. Tests cover causal alignment, masking,
frozen-U derivatives, exact simulation, collision boundaries, checkpoint recovery,
cache corruption, compatible resumption, and raw/report reconciliation.

## Execution smoke

The corrected20/4/4-episode smoke completed20 updates in every stage and generated
all five controller comparisons, prediction/copy/reset diagnostics, velocity
probes, PNG/GIF panels and an inference bundle. Bundle module fingerprints and
CPU planning score/action match its source checkpoints exactly.

| Controller | Ordinary first interception | Paired first interception |
|---|---:|---:|
| Learned |1/2|0/4|
| Memory reset |1/2|2/4|
| Random |0/2|1/4|
| Current-frame tracker |1/2|2/4|
| Privileged simulator planner |2/2|4/4|

All engineering targets fail in this untrained smoke. Its inspected images
mostly omit both objects. Results are execution evidence only; tiny denominators
and incidental catches do not show useful learning. The measured learned decision
median/p95 is33.5/35.7ms, excluding rendering and real E/U assimilation; this does
not establish end-to-end20Hz control.

## Full reference: current evidence

Perception completed its10000-update budget, selecting update9750 from4096 fixed
validation frames. H x/y/paddle MAE is0.071877/0.076952/0.094528 pixels, with image
MSE1.694906e-5. The selected validation reconstruction panel was visually inspected
and reconstructs both objects accurately. These validation position errors meet
the one-pixel target; full held-out evaluation is pending.

Memory completed10000 updates, selected10000, after80000 sampled episodes and
740.08s training/validation. Ordinary validation R x/y/vx/vy/paddle MAE is
2.78267/3.04628/0.80920/0.54549/5.02473. Both velocity errors miss the0.5 target.
Paired-validation R vx MAE is4.31146 versus6.0 after reset; y MAE is26.8911.
These failures remain visible in the reference.

P1 completed10000 updates and selected10000. Its validation latent loss0.273366
beats copy1.077386, and H ball-x/paddle-x3.70127/1.79505 beats copy3.83335/2.52450.
Ball-y error3.59783 is worse than copy2.38499. The one-step gate therefore fails;
five-step training and500-start/100-pair control were correctly not launched.
The failed stage and canonical dashboard are preserved. The new
[continuation protocol](paddle-predictor-continuation-plan.md) tests at most10000
additional updates with the same optimizer state, data, modules and losses.

The continuation completed20000 cumulative updates, selecting20000; exactly10000
updates/640000 sampled windows were added. Selected latent loss0.190358 and H
x/y/paddle MAE2.42536/2.22583/1.65811 beat all matched copy values above. The
original gate passes without a smoke bypass or changed threshold. Additional
training/validation took90.08s, excluding observer-cache preparation. The fixed
10000-update reference remains failed; this result used twice its P1 update
budget. Five-step P training is now active in
`runs/paddle/continuation_v1/predictor_5`, initialized from that selected P1 with
a fresh optimizer, identical latent statistics and the original10000-update cap.

Independent128-window validation diagnostics from the failed P1 reference showed
that shuffling memories increases loss0.2720→0.8366: history is materially used.
Error projected into H's three readout directions increased while overall latent
error decreased. Predicted balls were often faded or absent; conditional decoded
centroids were more accurate than H in many visible cases, but22/128 lacked a
detectable ball at the diagnostic threshold. See
`runs/paddle/collaboration/predictor_diagnosis/analysis.md` and its inspected PNG.
This explains why image appearance and latent improvement did not replace the
readout gate, and does not establish unique failure causation.

The paired diagnostic starts zero memory at ball y40; ordinary training starts
at y8–16. An independent CPU check on the immutable4750-update snapshot compared
the same23 ordinary validation frames at y42–48: full-history R y/vx MAE is
3.58/0.89, versus24.24/3.59 when starting from only the last three frames.
Forty paired members give H MAE0.047/0.059/0.076 pixels but R y MAE27.40.
This supports a cold-start distribution problem, distinct from perception error.
U attention is also diffuse (effective278–310 of320 tokens per head), while
logged gradients do not show pervasive clipping. These diagnostics do not prove
an architectural impossibility or guarantee that a particular follow-up works.
The user has been asked whether to introduce explicitly documented training
starts throughout existing trajectories or retain the exact start recipe.

An exact raw RGB/state/action disk cache removed a measured input bottleneck.
Matched128-frame loader median improved from360.37ms to0.997ms; sustained
perception updates after resumption take about8.6ms. The cache occupies2.73GB
decimal across all splits, built in10.67s, and matches sampled source arrays and
normalized tensors bitwise. The trainer resumed its update250 optimizer/RNG state.
This is an input-storage change, with original data, sampling and loss preserved.
Original interruption and resumed output remain in `runs/paddle/baseline.log` and
`runs/paddle/baseline_resume.log`.

## Review and reporting

Independent Codex agents implemented simulation/data, models/planning, and
evaluation/reporting. Independent Claude tasks reviewed physics, learning
semantics, integrated code, and the measured memory weakness. Review
prompts/results live under `runs/paddle/collaboration`. Completed original Claude
reviews plus the memory diagnosis report total API-equivalent cost$11.97402150;
this is receipt metadata,
not a claim about subscription billing.

The canonical HTML passes desktop1440px/mobile390px and source interaction
verification. Isolated synthetic capacity checks preserve all3500 potential
controller outcomes and all failed-case IDs, with maximum artifact1.27MB below
the3MB limit. Synthetic fixtures were not added to the experiment ledger.

Visual review subsequently found that the portable reader's categorical line
axis placed appended validation step0 midway through training. Sorting alone
introduced missing-series gaps. Training and validation/copy now use separate,
chronological charts; all338 previously sampled objective values are preserved
without interpolation. Both validation/copy SVG paths show all41 checkpoints in
one continuous segment on desktop/mobile. The prior artifact was replaced and
browser/source verification repeated successfully. The reader's ordered-sample
spacing is explicitly described in the chart captions.

Historical P1 validation rows recorded `windows:64` as mean batch size; the
manifest actually contains1024 validation windows and all objective/readout
means used that full population. Those raw rows remain unchanged. Count
aggregation is corrected to sum batches in future rows, protected by a ragged
batch regression. This metadata defect did not cause the failed gate.

## Full selected-continuation evaluation (3500 cases)

Raw evaluation finished2026-09-07 after500ordinary starts and100paired histories
(two members each), allfive controllers:3500case outcomes, no prior errors.
The selected P5 checkpoint completed10000 updates. This is full held-out
evidence for the declared continuation, not a passing engineering result.

| Controller | Ordinary first catch /500 | Paired first catch /200 | Paired correct first action /200 |
| --- | ---: | ---: | ---: |
| Learned |181 (36.2%)|41 (20.5%)|79|
| Memory reset |136 (27.2%)|40 (20.0%)|84|
| Random |115 (23.0%)|14 (7.0%)|62|
| Image tracker |369 (73.8%)|0|0|
| Privileged five-step planner |444 (88.8%)|200 (100%)|200|

All17831testframes have H x/y/paddle MAE0.070746/0.076114/0.092347pixels,
meeting the1pixel target. On16831post-warm-up actual observations, R vx/vy
MAE0.819868/0.552002 misses0.5. At horizon5 on1024matched windows, H errors
are4.581881/4.042754/4.671391 versus copy16.167912/11.078113/5.544553.
Latent error0.708690 versus copy1.852585 improves prediction but does not meet
the2pixel target or establish useful control.32/35terminal predictions fall
below the exact y61 threshold; even actual H does so342/496times. Preserve
this declared threshold in the reference; a margin would be a separate change.

Learned ordinary decision latency median35.585ms,p9540.202ms over18929decisions,
including allcandidate evaluations, excluding real E/U assimilation and rendering.
Privileged median2.870ms,p953.815ms. No learned planning failures or invalid
candidates were recorded. This timing is not an end-to-end20Hz claim.

Inspected success `ordinary_1_learned` and failure `ordinary_0_learned` panels
show accurate reconstructions but fading imagined balls, underestimated paddle
motion in the success case, and incorrect imagined vertical motion in the
failure. Actual and imagined futures use identical executed actions. The
canonical dashboard capacity repair is complete and browser-verified with both
full paddle and new PushT evidence. All raw JSON/CSV and PNG/GIF evidence remains under
`runs/paddle/continuation_v1/evaluation`.
