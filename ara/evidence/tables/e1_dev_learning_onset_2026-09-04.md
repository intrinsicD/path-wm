# E1 Learning-Onset and Curriculum Audit — 2026-09-04

All values below are development diagnostics, not experiment results. Every condition used seed 0,
the same 20,480 ordinary transitions, the same 2,048 paired intervention groups, the same fixed
256-group K=4 counterfactual probe, 2,000 optimizer steps, CUDA bf16 on one RTX 4090, and the paired
objective selected in the preceding audit (`kappa=0.002`, contrastive weight 0.04, positive weight 6).
No checkpoint is promoted and ABI thresholds remain unset.

## Question and causal alternatives

The observed branch ranking appeared only near the end of earlier runs. The audit distinguishes:

1. **Scheduled absence:** the default stage-2 gate supplies no counterfactual updates before step 1,000.
2. **Gradient starvation:** early representations or predictor paths may make the loss locally inactive.
3. **Representation bootstrap:** a random encoder/adapter may emit normalized but scene-insensitive states.
4. **Residual-scale bootstrap:** a random delta readout may dwarf the true transition before learning identity.
5. **Curriculum choice:** encoder-only pretraining may help, hurt, or merely move the delay.

Diagnostic checkpoints at steps `{0,50,200,500,750,1000,1250,1500,1750,2000}` were evaluated on
the same held-out probes. The replay control has the same learning configuration as the previous
paired-MSE run but also saves snapshots. Its first loss differs slightly from the historical run because
the extra CUDA synchronization changes the numerical trajectory; all conditions in this audit have the
same snapshot schedule and are compared only with this replay.

## Initialization is not empty, but both state and predictor are badly scaled

At seed-0 initialization, per-token LayerNorm makes state RMS `0.999984`, so W is not numerically zero.
Across-example variance is only `0.007058`, however, versus approximately `0.98` in trained models:
most initial variation is shared positional/model structure rather than observation-dependent content.
The default predictor's paired positive MSE is `0.310718`, while simply copying the current state gives
`0.001732`; the random residual begins 179.4 times above the identity transition error.

The loss is not gradient-starved. On the initial paired batch, raw InfoNCE component norms are
`2.034/0.383/0.392` for encoder/adapter/predictor, and paired-MSE norms are
`2.865/1.122/1.395`. The problem is the geometry and relative scale of those gradients.

An initialization-only calibration scaled the predictor readout weights and bias while holding every
other initial tensor fixed:

| Readout scale | Initial paired positive MSE | Ratio to identity | Predicted branch separation |
|---:|---:|---:|---:|
| 1.00 | 0.310717881 | 179.39x | 0.000011921 |
| 0.10 | 0.006129730 | 3.54x | 0.000001172 |
| 0.05 | 0.002842336 | 1.64x | 0.000000599 |
| **0.03** | **0.002131115** | **1.23x** | **0.000000389** |
| 0.01 | 0.001776555 | 1.03x | 0.000000093 |
| 0.00 | 0.001732096 | 1.00x | 0 (disconnected on the first update) |

Scale 0.03 was selected before training because it is close to transition scale while retaining a
nonzero action-to-output gradient path. This is an initialization calibration, not a post-hoc weight fit.

## Tested schedules

| Condition | P / CF onset | Readout | CF accuracy | One-step | Identity | Zero action | H=4 |
|---|---|---:|---:|---:|---:|---:|---:|
| Late replay control | step 0 / 1,000 | 1.00 | 0.318359 | 0.000672073 | 0.000494709 | 0.000685718 | 0.003893960 |
| CF from step 0 | step 0 / 0 | 1.00 | 0.250977 | 0.001223361 | 0.000708938 | 0.001223918 | 0.008006779 |
| CF from step 0, scaled residual | step 0 / 0 | 0.03 | 0.525391 | **0.000435704** | 0.000329761 | 0.000383512 | **0.001988869** |
| Encoder-only first 500 | step 500 / 500 | 1.00 | 0.281250 | 0.000796414 | 0.000575614 | 0.000812639 | 0.004924343 |
| Encoder-only first 200, scaled | step 200 / 200 | 0.03 | 0.492188 | 0.000511749 | 0.000461907 | 0.000495942 | 0.001999207 |
| Joint first 200, scaled; then CF | step 0 / 200 | 0.03 | **0.535156** | 0.000708962 | 0.000675157 | 0.000689616 | 0.002567180 |

Raw latent errors across separately learned encoders are not directly commensurate, so each candidate
must also be read against its own controls. The short joint warm-up has the smallest correct-vs-identity
gap (5.01%) and a 43.99% gain over shuffled action, but correct action is still 2.81% worse than zero.
The direct scaled-residual model has the lowest raw one-/four-step errors and a 38.46% shuffled-action
gain, but correct action is 13.61% worse than zero. The replay control alone beats zero (1.99%) but has
weak discrimination and remains 35.85% above identity. None passes all action-correctness controls.

## When held-out discrimination emerges

Chance is 0.25 over 1,024 decisions; one binomial standard error is 0.01353. Using chance plus three
standard errors (`0.29059`) as an onset diagnostic:

| Condition | step 750 | step 1,000 | step 1,250 | step 1,500 | step 2,000 | First >3 SE |
|---|---:|---:|---:|---:|---:|---:|
| Late replay | 0.250000 | 0.250977 | 0.248047 | 0.250000 | 0.318359 | 2,000 |
| CF from step 0, default residual | 0.250000 | 0.249023 | 0.250000 | 0.250000 | 0.250977 | never |
| CF from step 0, scaled residual | 0.250977 | 0.273438 | 0.361328 | 0.437500 | 0.525391 | 1,250 |
| Encoder-only first 500 | 0.249023 | 0.250000 | 0.250000 | 0.250977 | 0.281250 | never |
| Encoder-only first 200, scaled | 0.250000 | 0.249023 | 0.298828 | 0.415039 | 0.492188 | 1,250 |
| **Joint first 200, scaled; then CF** | 0.252930 | **0.397461** | **0.471680** | **0.538086** | **0.535156** | **1,000** |

The training minibatch for the short joint warm-up first rises above chance at step 600 and reaches
0.40625 at step 950; the larger fixed probe confirms strong onset at step 1,000. Thus semantic learning
can be moved roughly 1,000 wall-clock optimizer steps earlier without extra data or total updates.

## Why encoder-only pretraining is the wrong current curriculum

SIGReg plus inverse dynamics does make W vary across scenes: in the 500-step representation-first run,
across-example variance rises from 0.0071 to 0.9518 by step 200. But it does not make a temporally
predictive geometry. Paired identity MSE simultaneously rises from 0.00173 to 0.16111 at step 200 and
0.20627 at step 500. The inverse objective is rewarded for exposing frame differences; without the
dynamics term, nothing requires nearby physical states to remain nearby. The fresh predictor then spends
hundreds of updates learning that distorted coordinate system before branch ranking can begin.

This rejects the tested recipe “SIGReg + inverse encoder pretraining, predictor later.” Encoder
pretraining could still be useful with a temporal-consistency or passive predictive objective (ideally
with a stable/EMA target), but that is a different experiment, not evidence for the current curriculum.

## Decision

The late onset has three demonstrated causes: a literal stage-2 delay, an initially scene-poor latent,
and an O(100)-scale residual mismatch. It is not caused by absent gradients. The best tested onset recipe
is a transition-scale predictor initialization plus a short joint dynamics warm-up, followed by the
counterfactual objective at stage 1. It is retained as the E1-b training-schedule candidate but not
promoted as E1-a because zero/identity controls still fail.

Further E1-a schedule or scalar tuning cannot repair the exact hidden-velocity alias proved in the
preceding audit. The next architectural test remains E1-b's predict-then-correct updater, with history in
W and the Markov predictor contract unchanged. Planning remains off until that reference passes the panel.

## Artifacts and validation

- Configs: `configs/dev/first_slice_onset_*.yaml`
- Runs: `runs/dev/first_slice_onset_*/0/`
- Fixed curves: each run's `learning_curve.jsonl`
- Curve evaluator: `evaluation/learning_curve.py`
- Scoped model suite: 123 passed, 2 threshold tests deselected; full current workspace: 127 passed
- Threshold suite on the short-joint checkpoint: 2 skipped with `threshold_unset`, 127 deselected
- Device: NVIDIA RTX 4090, CUDA bf16
