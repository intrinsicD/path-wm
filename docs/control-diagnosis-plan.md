# Saved-checkpoint control diagnosis

Authorized on 2026-09-05: finish browser QA first, evaluate the existing broader
pilot on training goals, inspect rollout/action ranking, and prepare a larger
reproduction protocol. No completed training is repeated. A long training launch
is outside this slice.

## 1. Reporting repair

Problem: the canonical portable builder's dump-DOM virtual clock expires while
the reader is waiting for animation-frame startup. Full Chrome also subtracts
window chrome from the requested viewport. The artifact and raw ledgers pass
structural validation; browser verification remains required.

Interface: a repository-owned Chromium command adapter accepts the canonical
builder's browser arguments, launches an installed browser with a disposable
profile and CDP pipe, sets its viewport/media explicitly, and waits in real time
for the original canonical probe's terminal marker. Return the actual DOM,
including failed probe results, unchanged. Do not alter the artifact, probes,
assertions or success receipts. Missing markers and browser errors fail boundedly.
The shared builder remains responsible for packaging and verification.

Essential checks: a real animation-frame probe observes the requested viewport;
negative probe results survive the transport; missing results time out. Run these
as explicit browser integration checks, plus existing CPU harness tests. Refresh
the current dashboard and require canonical desktop/mobile and source checks to
pass before launching diagnostics.

## 2. Training-goal control

Freeze 20 distinct training episodes and starts without outcome filtering, using
sampling seed 42 and reset/CEM seeds 1234–1253. Use the existing 1,000-update
checkpoint, saved normalization and BatchNorm buffers, float32, and the held-out
pilot's exact planning settings: 300 samples, 30 iterations, 30 elites, horizon
5, five actions per block, target offset 25 and environment budget 50. Report
initial successes and paired stationary/replay controls. Preserve checkpoint
hashes. This is development diagnosis, with no new passing threshold.

## 3. Rollout and action ranking

Use explicit, matched training and held-out cases. Compare replay, stationary,
seeded random and CEM-planned action sequences on the same simulator resets.
Record predicted terminal goal costs and simulator outcomes, together with
multi-step latent prediction errors against simulator-rendered observations.
Include released weights as a positive control and distinguish each model's
latent scale. Freeze cases, sequence counts and budgets before measurement.
Qualitative panels must show actual simulator images and make clear that the
latent model has no image decoder.

## 4. Reproduction preparation

Audit pinned author code against the local trainer and prepare explicit source,
split, normalization, optimizer-update/scheduler and evaluation contracts. State
paper/release ambiguities. Keep authors-style evaluation separate from stricter
unseen-configuration transfer. Estimate the local budget; do not launch it.

## Frozen rollout diagnostic details

Use the first four frozen training-control cases and first four existing held-out
pilot cases, retaining their original order and reset seeds. Every case has
20 raw 25-action sequences: recorded replay, stationary, 16 independent Gaussian
sequences (seed 9000 + case index, in the pilot's normalized action units), and
the pilot/released model's initial CEM plan. Both models score all 20 identical raw
sequences using their own saved normalizers. Plans use 300 samples, 30 iterations,
30 elites and five five-action blocks; no tuning after seeing outcomes.

Execute each candidate from the same simulator reset, continuing all 25 actions
for terminal comparisons even if success occurs earlier. Record initial, any-step
and terminal success, the upstream state-distance diagnostic, combined agent/block
position error and circular block-angle error. The primary ranking comparison is
within each case: Spearman correlation of predicted goal cost with terminal
position error, selected-candidate position error and regret versus the best of
these 20 candidates. Position error alone is not the success predicate; report
angle and exact success separately. Tied/constant ranks have null correlation.

At steps 5/10/15/20/25 record autoregressive latent MSE against simulator-rendered
frames, copy error and actual-versus-predicted latent distance to the source goal.
Save candidate actions and simulator states; retain image strips for replay,
stationary and both planned sequences. Compare errors within each model's latent
space. No success threshold or causal attribution is introduced after the fact.
Eight cases are a bounded development diagnostic, not population estimation.
