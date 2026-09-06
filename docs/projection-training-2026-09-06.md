# Paired projection-count training results — interim

The screen is still running. The first PushT seed gives mixed control results:4096 projections solve fewer cases at750 updates and more calibrated cases at1500, while saved-buffer scores favor1024 at both checkpoints. One pair does not establish faster learning. All remaining pairs stay scheduled under the unchanged [frozen plan](projection-training-plan-2026-09-06.md).

## Question and evidence boundaries

The [preceding investigation](sample-efficiency-2026-09-06.md) found roughly4× lower conditional SIGReg gradient variance with4096 rather than1024 directions on fixed trained weights and data. The current experiment tests whether that intervention improves actual control learning from scratch. Gradient-estimator variance, total gradient norm, prediction quality and control success are different measurements.

Both arms receive identical full-source random-window populations, initialization, optimizer-window order and global model/dropout RNG within each seed. A separate checkpointed sketch generator prevents the larger draw from shifting dropout or data randomness. Configuration differences are restricted to projection count and output directory. Three paired seeds per dataset are independent training draws; the frozen50 control cases are shared regression populations. They do not measure unseen-configuration generalization.

Training uses latent192, batch128, lambda0.09, AdamW5e-5, bf16 and a1500-update cosine schedule with15 warmup updates. This short schedule is not a prefix of the full reproduction schedule. The750/1500 checkpoints represent96,000/192,000 optimizer-window presentations. Each calibrated clone also accesses512 windows drawn from the full training split; full-source action normalization accesses the existing source population. This experiment cannot establish training from a smaller independent dataset.

Exact metadata accounting of the planned prefixes is saved in [window_exposure.json](../runs/projection_training_2026-09-06/window_exposure.json), with its executable companion beside it. Every prefix contains distinct window starts. At 1500 updates, PushT covers 18,660–18,664 of 18,685 episodes and 671,146–671,678 unique encoded source-frame rows; TwoRoom covers 9,999–10,000 of 10,000 episodes and 540,753–541,370 frame rows. Frame presentations per unique row are about 1.14× and 1.42× respectively. These are source identities, not independent configurations or visually distinct observations. They describe the planned prefix for pending runs.

The 512 calibration windows add 452–459 PushT / 344–370 TwoRoom window starts outside the 1500-update optimization prefix, depending on seed. The existing first-pair calibration row lists match this exact reconstruction. Window reuse within this short schedule is therefore not a plausible explanation for the observed slow learning by itself; broad episode exposure occurs well before useful control is established.

## Recorded control outcomes

Snapshot after8 of48 expected outcomes. Scores are successes out of50; all PushT initial-success counts are zero. Differences below are4096 minus1024. Saved and calibrated policies remain separate; calibrated control was predeclared primary.

| Dataset | Seed | Updates | BN policy |1024|4096| Difference |
|---|---:|---:|---|---:|---:|---:|
|PushT|3072|750|Calibrated|3|2|−1|
|PushT|3072|1500|Calibrated|2|4|+2|
|PushT|3072|750|Saved|3|1|−2|
|PushT|3072|1500|Saved|4|2|−2|

Seeds3073/3074 and TwoRoom are pending. No statistical threshold was frozen; no passing gate or significance claim follows from these values. The [live derivative](../runs/projection_training_2026-09-06/projection_comparison.json) explicitly lists every missing outcome and recomputes paired summaries from native reconciled case records. The [verified dashboard](../runs/experiment_dashboard.html) refreshes after every completed stage.

## Optimization and prediction context

Both first PushT arms completed1500 updates within the2100-second cumulative loop cap:2023.70 seconds for1024 and1909.43 for4096. These serial observations share the GPU with existing desktop activity and are not a causal speed comparison. The predeclared20-update preflights measured1.3265/1.3368 seconds per update for PushT and1.3198/1.3354 for TwoRoom (1024/4096); allocated peaks were3.2822/3.2830GB.

At1500 updates, saved-buffer one-step prediction/copy ratios are approximately0.747 for1024 and0.727 for4096 on matched512-window validation populations. Raw latent MSEs have different learned coordinate scales and must not be interpreted as directly comparable task quality. Quick32-window effective ranks are13.58/14.31; these are not the earlier full-window inspection ranks. The first PushT baseline exceeds the clip threshold in all61 logged gradient samples, not a recorded claim about every1500 update.

An inherited reporting label was corrected: training validation rollout_mse / identity_mse compares autoregressive error with an adjacent one-step copying error. It is now labelled “rollout / one-step copy,” with the different horizons disclosed. It is not matched multi-step prediction skill. Raw metrics and training/evaluation numerics were preserved.

## Preselected qualitative evidence

The first frozen case was selected before scores:seed3072, calibrated1500, each arm versus the existing released model on exactly the same protocol. PushT row150187, episode1235, start36 and reset seed1234 is reproduced from saved model actions. Both local arms fail after50 steps; the released model succeeds in16 and the recorded-action replay in14. In both local panels, the block eventually moves away from the goal. This is one fixed case, not an outcome-selected illustration or a representative rate estimate.

- [PushT1024 rollout](../runs/projection_training_2026-09-06/qualitative/pusht_s3072_m1024/case_0_rollouts.png)
- [PushT4096 rollout](../runs/projection_training_2026-09-06/qualitative/pusht_s3072_m4096/case_0_rollouts.png)

Both model action replays exactly reproduce recorded control outcomes. TwoRoom qualitative panels are pending its first pair.

## Implementation, co-work and verification

Claude supplied a private-RNG implementation/test design and the paired-statistics implementation through two restricted neutral MCP tasks. Reported API-equivalent usage totals$1.132410 against the predeclared$4 ceiling; this is not an account-balance statement. Integration uses the pinned LeWM numerical expression, quadrature and precision rather than generic substitutions. Absence of sigreg_seed preserves legacy behavior; opt-in checkpoints require and restore the private stream.

Native FP32/CPU-bf16 loss and input gradients match, and the tiny CUDA-bf16 check confirms bitwise parity plus exact private-stream recovery without changing global CUDA RNG. Within the first actual PushT pair, initial parameters, validation indices and global CPU/CUDA RNG agree at750 and1500. Calibrated clones preserve trained states and record training-only source rows; calibration has disjoint validation window starts, with source episode/frame overlap still possible under this protocol.

The implementation suite passed99 tests including3 browser checks before two subsequent guards were added. Targeted checks also pass for actual frozen case ordering and duplicate-coordinator refusal without overwriting active progress. A final full suite is pending. Browser fixtures verify one-checkpoint grouped bars and two-checkpoint markers at390/1440 pixels; synthetic values are excluded from the scientific ledger. Dashboard curves disclose adaptive axes; final standalone figures will use fixed0–100% scales.

The [first-pair integrity audit](../runs/projection_training_2026-09-06/integrity_audit_first_pair.json) confirms all 13 protected scientific-code/reference/case inputs are unchanged, both paired global RNG states match at both checkpoints, and every exported calibrated clone changes only BN buffers. Full source HDF5 bytes were not rehashed in this screen; declared dataset revisions and hashes remain recorded. Final all-pair integrity verification, completed-results plots, dashboard inspection and interpretation await the remaining runs.
