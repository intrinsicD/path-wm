# Baseline checks — 2026-09-05

Long training is stopped at the user's request. No automatic training restart is
scheduled. The new implementation passes its essential checks and shows an
early predictive learning signal, but useful closed-loop control is unproven.
Research extensions remain deferred.

## What was reset

Previous model code, tests, configs, generated datasets, experiment outputs and
ARA empirical records were removed from the active workspace. The two clean
old temporary worktrees were removed. Git history was not rewritten.
Ideas are preserved in [ideas.md](ideas.md). TAU source media, metadata and
archives remain; old processed shards and manifests were removed. New baseline
code reuses pinned authors' components, with licenses and source attribution.

## Short implementation checks

Nine CPU tests pass. They cover episode boundaries, observation/action timing,
train-only normalization, preprocessing layout, target-encoder gradients,
causality, numerical agreement with the reference implementation, encoder
recomputation gradients/BatchNorm state, CEM execution cost verification, and
the pinned PushT environment API. Two upstream dependency warnings occurred
(pygame/pkg_resources deprecation and Gymnasium array conversion).

The complete released LeWM PushT checkpoint strictly loads all keys in the fresh
18,034,478-parameter model. Float32 encoder and predictor outputs on the same
inputs match the authors' implementation exactly (maximum differences:
0.0, 0.0). This validates
implementation compatibility; it does not measure task success.

Evidence: [reference check](../runs/checks/reference.json), [tests](../tests/).

## Short real-data learning check

Data: original Diffusion Policy PushT release, 206 episodes / 25,650 frames.
Train/test split: 185 training episodes and 21 held-out episodes (seed 3072).
Architecture: full LeWM, random initialization, 224px input, context 3, action
blocks of 5, MSE + .09 SIGReg, AdamW. Development budget: 500 updates, batch 32,
bf16; about 236 seconds including periodic validation. This is a reduced-budget
check on a separately named dataset, not a reproduction of the paper's score.

The training objective fell from 1.153 on the first batch to 0.350 on the final
batch. Different minibatches and a changing latent space make this descriptive,
not a stand-alone quality measure. The initial validation curve used contiguous
windows; the following final check uses 512 fixed, randomly sampled windows
from held-out episodes, float32 inference:

| Measure | Value |
|---|---:|
| Next-state prediction MSE | 0.111139 |
| Copy-current-state MSE | 0.119291 |
| Shuffled-action prediction MSE | 0.115151 |
| Zero-normalized-action prediction MSE | 0.114012 |
| Open-loop rollout MSE | 0.226588 |
| Mean embedding standard deviation | 0.776770 |

Prediction error is 6.8% lower than copying
the current embedding, and 3.5% lower
than with shuffled actions. This is a small positive signal, not a statistical
significance or generalization claim. Windows overlap and only one seed was run.
Zero normalized actions mean the dataset mean absolute target, not a stationary
physical action.

Evidence: [run manifest](../runs/pusht_cchi_dev/manifest.json),
[metrics](../runs/pusht_cchi_dev/metrics.jsonl),
[held-out check](../runs/pusht_cchi_dev/heldout_check.json),
[data receipt](../data/pusht_cchi/conversion.json).

## Short control check

Two fixed held-out episode goals, 25-step goal offset, 50 environment steps,
5 action blocks per plan, 5 raw actions per block. CEM: 300 candidates, 30
iterations, 30 elites; the executed mean receives an additional rollout check.
Result: **0/2 successes**. A preceding cheaper CEM smoke check also scored 0/2.

The CCHI observations/actions and missing recorded velocity differ from the
LeWM release's exact training/evaluation population. The local SWM simulator
initializes missing velocity to zero. These are development control checks;
they neither reproduce nor refute the paper's benchmark result. This model
is not ready to serve as the research baseline.

Evidence: [control settings](../runs/pusht_cchi_dev/control_check/config.json),
[per-goal outcomes](../runs/pusht_cchi_dev/control_check/episodes.jsonl),
[summary](../runs/pusht_cchi_dev/control_check/summary.json).

## Stopped work and next bounded check

A longer batch-128 CCHI run was started before the user's latest instruction and
interrupted immediately afterward. Its last logged update was 140; its saved
checkpoint is still step 0. It is not included in the completed learning result.
No long training is currently running or scheduled.

The official LeWM PushT and TwoRoom archives are still downloading. Their
complete datasets have not yet been extracted, trained on, or evaluated.
TwoRoom's extracted schema/path and action semantics still need verification.
TAU is retained passive audio/video source data and is not training this
control model.

The next bounded checks should be the released checkpoint's control behavior on
its exact PushT data protocol, and a small real-data alignment/learning check on
TwoRoom. Only after those checks and user review should a long training run
start. The LeWM PushT archive's HDF5 header reports 18,685 episodes and 2,336,736
frames; a full ten-epoch local run is likely to take several days at the measured
~1.7 seconds per batch of 128. This remains a throughput estimate, not a completed
run measurement.
