# Saved-checkpoint PushT control diagnosis

The pilot fails control on familiar goals as well as held-out goals. On matched
simulator trials it prefers action plans that move away from the desired state.
A numerical check also found bf16 gradient differences from encoder batch slicing;
full-batch activation checkpointing avoids that discrepancy in the measured
native comparison and fits a batch-128 backward on the local GPU. These findings
do not establish which training difference caused the pilot's poor dynamics.

The approved diagnostics and reproduction preparation are complete. The pilot's
saved checkpoints remain unchanged. No long training was launched and the
learned-control baseline gate remains unmet.

## Training-goal control

The untouched 1,000-update checkpoint was evaluated on 20 frozen, distinct
training episodes using the same planning/reset budgets as the earlier held-out
pilot. Cases were sampled without outcome filtering; none succeeded at reset.

| Method | Training goals | Previous held-out goals |
|---|---:|---:|
| Pilot, 1,000 updates | 0/20 | 0/20 |
| Recorded replay | 20/20 | 19/20 |
| Stationary | 0/20 | 0/20 |
| Released weights | Not rerun on these 20 | 17/20 |

The new learned-control evaluation took 104.35 seconds of recorded control time.
Every case used its full 50-step budget. Replay demonstrates that these training
goals are reachable under the reset protocol. Failure is therefore not explained
solely by unfamiliar held-out initial configurations. It does not separate
insufficient training from all remaining numerical/recipe effects.

## Matched action rankings and multi-step predictions

The frozen diagnostic uses four training and four held-out cases, selected before
measurement. Each case has 20 identical raw 25-action sequences scored by both
models: replay, stationary, 16 seeded random sequences, and each model's CEM plan.
Each sequence is simulated from the same seeded reset for all 25 actions.
There are 160 simulator trials and 320 model/candidate records.

| Population | Model | Mean within-case rank correlation¹ | Mean selected position error | Selected successes |
|---|---|---:|---:|---:|
| Training (4 cases) | Pilot | 0.180 | 175.37 px | 0/4 |
| Training (4 cases) | Released | 0.654 | 8.29 px | 4/4 |
| Held out (4 cases) | Pilot | −0.114 | 257.12 px | 0/4 |
| Held out (4 cases) | Released | 0.653 | 8.39 px | 3/4 |

¹ Spearman correlation between predicted terminal latent cost and combined
agent/block position error, calculated separately within each model/case and
then averaged over four cases. Positive correlation is desirable because both
cost and physical error should be small. These are descriptive fixed-case means,
not population estimates. Success additionally requires the angular threshold.

The pilot selects its own plan in all eight cases. Those plans end an average
216.24 px from the goal and succeed in 0/8. Recorded replay succeeds in 8/8,
with mean terminal position error 1.31 px. Released-model selection succeeds in
7/8; the released model's own plans succeed in 6/8 (in one case it instead selects
replay). Candidate-set selection is not the earlier 50-step closed-loop test.

For the pilot's own plans, its mean final-step latent prediction MSE is 0.8937
versus 0.9049 for copying the start embedding: multi-step prediction barely beats
copying on those optimized plans. On replay, prediction MSE is 0.4474 versus
copy 0.9727. Thus prediction improvement on recorded actions does not imply a
planning cost that remains useful on optimized actions. The dashboard includes
all five prediction/copy and predicted/measured goal-distance points per case.

The pilot's measured terminal-image latent distances also correlate only weakly
with position error (mean 0.386 on training cases and 0.261 on held-out cases).
Its prediction-to-measured-latent ranking correlations are 0.264 and 0.026.
Both representation geometry and rollout prediction merit attention; these
observations do not isolate a single causal mechanism. Latent MSE values across
separately learned encoders are not directly comparable.

### Qualitative evidence

These panels show actual simulator observations. LeWM has no image decoder;
predicted latent trajectories appear as curves in the dashboard. The preselected
training and held-out examples show the pilot moving the agent/block away from
the desired state, while replay and released plans remain directed toward it.

![Training case actual simulator rollouts](../runs/diagnostics/pusht_control_diagnosis/ranking/case_0_rollouts.png)

![Held-out case actual simulator rollouts](../runs/diagnostics/pusht_control_diagnosis/ranking/case_4_rollouts.png)

## Numerical training parity and memory

All comparisons use discarded clones of the saved checkpoint, identical real
training windows, seeds, native components/loss and full projector/SIGReg batches.
The reference is pinned native JEPA plus the authors' `lejepa_forward` loss.
The diagnostic batch is four sequences/16 images, not the original pilot's
batch 128. Relative gradient L2 is normalized by the native gradient norm.

| bf16 backward variant | Relative gradient L2 difference |
|---|---:|
| Ordinary local full batch | 1.36e-7 |
| Encoder recomputation in chunks of 8 images | 0.06124 |
| Recompute all 16 images together | 1.82e-11 |
| Full-batch nonreentrant activation checkpointing | 1.36e-7 |

Losses match in the bf16 comparisons and BatchNorm buffers match exactly.
Float32 activation checkpointing has relative gradient L2 difference 2.45e-5,
with buffer maximum difference 5.96e-8. The chunk-size comparison isolates a
numerical effect of changing encoder batch computation, without proving it caused
the pilot failure or measuring the exact drift of the original batch-128/chunk-16
recipe. Previous small float32 tests did not establish bf16 batch-size parity.

The source-scale candidate therefore uses full-batch activation checkpointing and
`encoder_chunk: 0`. A separate single backward with batch 128, bf16, the complete
architecture and SIGReg completed in 1.60 seconds, with finite gradients, peak
allocated memory 3,137,145,344 bytes and reserved memory 3,858,759,680 bytes.
It took zero optimizer steps. This probe excludes optimizer state, full-source
I/O and long-run behavior; it is not a throughput guarantee.

## Prepared reproduction and next step

[The configuration](../configs/reproduction/pusht_source_scale.yaml) and
[protocol](pusht-reproduction.md) now have an executable data-only preparation
and matched checkpoint evaluator. Preparation verified 1,981,721 full-source
windows, froze 1,783,548 training / 198,173 validation windows and full-source
unbiased action statistics, and saved 50 fixed source control goals. Ten epochs
produce 139,330 optimizer updates with 1,393 warmup updates.

The next research step is a separately scheduled source-scale run with the
prepared full-batch checkpointing configuration. The pilot-rate extrapolation is
58.3 hours; full-source I/O, optimizer cost and changed recomputation behavior
make the actual duration uncertain. No run has been scheduled or launched.
Scheduler ambiguity, unknown released-checkpoint training history and the
interpolation-only random-window evaluation remain explicit protocol limits.
Research extensions remain deferred until learned control works.

## Evidence and verification

- [Verified interactive dashboard](../runs/experiment_dashboard.html) and its
  [canonical browser receipt](../runs/experiment_dashboard.receipt.json).
- [Frozen diagnostic protocol](control-diagnosis-plan.md).
- [Source-hashed diagnostic summary](../runs/diagnostics/pusht_control_diagnosis/diagnosis_summary.json).
- `runs/diagnostics/pusht_control_diagnosis/`: frozen manifest; training-control
  per-case outcomes/action arrays/baselines; 320 ranking records and simulator
  action/state arrays; original, expanded and checkpointed parity measurements;
  memory preflight. Intermediate negative results are preserved.
- [Prepared reproduction manifest](../runs/reproduction/pusht_source_scale_preparation/manifest.json),
  frozen window indices and paired control cases.

The full essential suite passes 32 checks, including both data protocols' tiny
CPU training/resume slices, native gradient comparisons and two explicit browser
transport integration checks. Canonical desktop/mobile rendering, source
interaction and exact artifact embedding pass. All original pilot checkpoint
hashes match the completed-pilot report; no weights or saved buffers were changed.
