# Paired projection-count training results — interim

Two of the three paired seeds are complete on both datasets (32/48 control outcomes). Increasing projections has not shown a consistent learning advantage: early calibrated TwoRoom differences reverse sign across seeds, and final calibrated control ties at 46/50 and 45/50. PushT remains weak in both arms. The final seed continues under the unchanged [frozen plan](projection-training-plan-2026-09-06.md).

## Question and evidence boundaries

The [preceding investigation](sample-efficiency-2026-09-06.md) found roughly4× lower conditional SIGReg gradient variance with4096 rather than1024 directions on fixed trained weights and data. The current experiment tests whether that intervention improves actual control learning from scratch. Gradient-estimator variance, total gradient norm, prediction quality and control success are different measurements.

Both arms receive identical full-source random-window populations, initialization, optimizer-window order and global model/dropout RNG within each seed. A separate checkpointed sketch generator prevents the larger draw from shifting dropout or data randomness. Configuration differences are restricted to projection count and output directory. Three paired seeds per dataset are independent training draws; the frozen50 control cases are shared regression populations. They do not measure unseen-configuration generalization.

Training uses latent192, batch128, lambda0.09, AdamW5e-5, bf16 and a1500-update cosine schedule with15 warmup updates. This short schedule is not a prefix of the full reproduction schedule. The750/1500 checkpoints represent96,000/192,000 optimizer-window presentations. Each calibrated clone also accesses512 windows drawn from the full training split; full-source action normalization accesses the existing source population. This experiment cannot establish training from a smaller independent dataset.

Exact metadata accounting of the planned prefixes is saved in [window_exposure.json](../runs/projection_training_2026-09-06/window_exposure.json), with its executable companion beside it. Every prefix contains distinct window starts. At 1500 updates, PushT covers 18,660–18,664 of 18,685 episodes and 671,146–671,678 unique encoded source-frame rows; TwoRoom covers 9,999–10,000 of 10,000 episodes and 540,753–541,370 frame rows. Frame presentations per unique row are about 1.14× and 1.42× respectively. These are source identities, not independent configurations or visually distinct observations. They describe the planned prefix for pending runs.

The 512 calibration windows add 452–459 PushT / 344–370 TwoRoom window starts outside the 1500-update optimization prefix, depending on seed. The existing first-pair calibration row lists match this exact reconstruction. Window reuse within this short schedule is therefore not a plausible explanation for the observed slow learning by itself; broad episode exposure occurs well before useful control is established.

## Recorded control outcomes

Snapshot after 32 of 48 expected outcomes. Scores are successes out of50; all PushT initial-success counts are zero. Differences below are4096 minus1024. Saved and calibrated policies remain separate; calibrated control was predeclared primary.

| Dataset | Seed | Updates | BN policy | 1024 | 4096 | Difference |
|---|---:|---:|---|---:|---:|---:|
|pusht|3072|750|calibrated|3|2|-1|
|pusht|3073|750|calibrated|1|1|+0|
|pusht|3072|750|saved|3|1|-2|
|pusht|3073|750|saved|1|2|+1|
|pusht|3072|1500|calibrated|2|4|+2|
|pusht|3073|1500|calibrated|1|1|+0|
|pusht|3072|1500|saved|4|2|-2|
|pusht|3073|1500|saved|2|1|-1|
|tworoom|3072|750|calibrated|41|37|-4|
|tworoom|3073|750|calibrated|35|41|+6|
|tworoom|3072|750|saved|18|11|-7|
|tworoom|3073|750|saved|22|11|-11|
|tworoom|3072|1500|calibrated|46|46|+0|
|tworoom|3073|1500|calibrated|45|45|+0|
|tworoom|3072|1500|saved|43|47|+4|
|tworoom|3073|1500|saved|44|44|+0|

Seed 3074 is pending. The [first-seed control figure](../runs/projection_training_2026-09-06/figures/outcomes16/projection_learning_curves.png) preserves a snapshot of the first seed’s 16 outcomes on fixed0–100% axes. No statistical threshold was frozen; no passing gate or significance claim follows from these values. The [live derivative](../runs/projection_training_2026-09-06/projection_comparison.json) explicitly lists every missing outcome and recomputes paired summaries from native reconciled case records. The [verified dashboard](../runs/experiment_dashboard.html) refreshes after every completed stage.

## Optimization and prediction context

Both first PushT arms completed1500 updates within the2100-second cumulative loop cap:2023.70 seconds for1024 and1909.43 for4096. These serial observations share the GPU with existing desktop activity and are not a causal speed comparison. The predeclared20-update preflights measured1.3265/1.3368 seconds per update for PushT and1.3198/1.3354 for TwoRoom (1024/4096); allocated peaks were3.2822/3.2830GB.

At1500 updates, saved-buffer one-step prediction/copy ratios are approximately0.747 for1024 and0.727 for4096 on matched512-window validation populations. Raw latent MSEs have different learned coordinate scales and must not be interpreted as directly comparable task quality. Quick32-window effective ranks are13.58/14.31; these are not the earlier full-window inspection ranks. The first PushT baseline exceeds the clip threshold in all61 logged gradient samples, not a recorded claim about every1500 update.

The source-bound [interim gradient figure](../runs/projection_training_2026-09-06/figures/diagnostics_interim_v2/gradient_norms.png) and [prediction/copy figure](../runs/projection_training_2026-09-06/figures/diagnostics_interim_v2/prediction_copy.png) preserve all metric rows in an immutable snapshot. Both completed PushT arms exceed the clipping threshold in all 61 logged samples. The discrete prediction comparisons show nonmonotonic intermediate behavior; no smoothing or seed pooling is applied. Pending facets remain explicit.

An inherited reporting label was corrected: training validation rollout_mse / identity_mse compares autoregressive error with an adjacent one-step copying error. It is now labelled “rollout / one-step copy,” with the different horizons disclosed. It is not matched multi-step prediction skill. Raw metrics and training/evaluation numerics were preserved.

## Early TwoRoom result and targeted normalization literature

The first TwoRoom arm (seed3072,4096 directions) has completed1500 updates. At750, saved buffers solve11/50 cases versus37/50 after the fixed calibration procedure. The corresponding initially-unsolved counts are8/46 and33/46; one initially solved case is not retained by the saved-buffer controller, so subtracting four from total successes would be incorrect. The1024 arm reaches18/41 at750 and43/46 at1500 (saved/calibrated). Its initially-unsolved counts are14/37 and39/42 out of46. Thus4096 changes calibrated control by−4 cases at750 and zero at1500 for this seed. Final saved-buffer one-step prediction/copy is0.213 at1500. At1500, saved/calibrated control reaches47/46 out of50 (43/42 of46 initially unsolved), so calibration decreases the final score by one case. Recorded elapsed time is2106.36 seconds: the last update began before the2100-second cap, then final validation and checkpointing completed. The1024 arm completes1500 in1960.7 seconds and has final saved-buffer prediction/copy0.364. Initial weights and final global CPU/CUDA RNG states match between arms.

Wu and Johnson (2021) show that EMA normalization statistics can lag an evolving model and destabilize early inference. Their PreciseBN discussion separates SGD and normalization batch sizes, and Appendix A.3 describes sequential layerwise statistics with earlier layers in inference mode. This supports investigating inference-state mismatch before interpreting poor control as insufficient learned weights. Our implementation averages per-batch unbiased variances; it is not their exact aggregate population-moment estimator. Their image-recognition results do not prove an improvement in this world model. [Rethinking “Batch” in BatchNorm, §3 and Appendix A.2–A.4](https://arxiv.org/html/2105.07576v1)

Ioffe (2017) proposes Batch Renormalization to reduce training/inference discrepancies while retaining derivatives through batch moments. Its correction schedule is material; simply using moving statistics throughout training is not the proposed method. This is a possible later training-time ablation, not an intervention adopted into the current screen. [Batch Renormalization, §3–4](https://proceedings.neurips.cc/paper/2017/file/c54e7837e0cd0ced286cb5995327d1ab-Paper.pdf)

Our provisional inference is that part of TwoRoom's apparent iteration requirement can be an inference-buffer problem: at750, the same learned parameters already support much stronger control after a512-window buffer update. This does not identify the responsible layer, remove the source-population limitation, or show that4096 directions improve learning. The remaining paired results and the earlier PushT calibration failures must constrain any general recommendation.

## Preselected qualitative evidence

The first frozen case was selected before scores:seed3072, calibrated1500, each arm versus the existing released model on exactly the same protocol. PushT row150187, episode1235, start36 and reset seed1234 is reproduced from saved model actions. Both local arms fail after50 steps; the released model succeeds in16 and the recorded-action replay in14. In both local panels, the block eventually moves away from the goal. This is one fixed case, not an outcome-selected illustration or a representative rate estimate.

- [PushT1024 rollout](../runs/projection_training_2026-09-06/qualitative/pusht_s3072_m1024/case_0_rollouts.png)
- [PushT4096 rollout](../runs/projection_training_2026-09-06/qualitative/pusht_s3072_m4096/case_0_rollouts.png)

Both PushT model action replays exactly reproduce recorded control outcomes. The [TwoRoom4096 panel](../runs/projection_training_2026-09-06/qualitative/tworoom_s3072_m4096/case_0_rollouts.png) also passes replay verification: row58908, episode644, start42, reset seed1234. Calibrated local and released controllers both succeed in21 steps; recorded-action replay succeeds in20. The first-case local trajectory crosses the doorway and reaches the goal neighborhood. The [TwoRoom1024 panel](../runs/projection_training_2026-09-06/qualitative/tworoom_s3072_m1024/case_0_rollouts.png) is also verified and visually inspected: calibrated local succeeds in21 steps on that same case. All four preselected qualitative comparisons are complete.

## Implementation, co-work and verification

Claude supplied a private-RNG implementation/test design and the paired-statistics implementation through two restricted neutral MCP tasks. Reported API-equivalent usage totals$1.132410 against the predeclared$4 ceiling; this is not an account-balance statement. Integration uses the pinned LeWM numerical expression, quadrature and precision rather than generic substitutions. Absence of sigreg_seed preserves legacy behavior; opt-in checkpoints require and restore the private stream.

Native FP32/CPU-bf16 loss and input gradients match, and the tiny CUDA-bf16 check confirms bitwise parity plus exact private-stream recovery without changing global CUDA RNG. Within the first actual PushT pair, initial parameters, validation indices and global CPU/CUDA RNG agree at750 and1500. Calibrated clones preserve trained states and record training-only source rows; calibration has disjoint validation window starts, with source episode/frame overlap still possible under this protocol.

The implementation suite passed99 tests including3 browser checks before two subsequent guards were added. Targeted checks also pass for actual frozen case ordering and duplicate-coordinator refusal without overwriting active progress. A final full suite is pending. Browser fixtures verify one-checkpoint grouped bars and two-checkpoint markers at390/1440 pixels; synthetic values are excluded from the scientific ledger. Dashboard curves disclose adaptive axes; final standalone figures will use fixed0–100% scales.

The [first-pair integrity audit](../runs/projection_training_2026-09-06/integrity_audit_first_pair.json) confirms all 13 protected scientific-code/reference/case inputs are unchanged, both paired global RNG states match at both checkpoints, and every exported calibrated clone changes only BN buffers. Full source HDF5 bytes were not rehashed in this screen; declared dataset revisions and hashes remain recorded. Final all-pair integrity verification, completed-results plots, dashboard inspection and interpretation await the remaining runs.

## Second-seed results and reporting recovery

Seed 3073 PushT calibrated scores tie at 1/50 at both checkpoints. Saved-buffer scores are 1/2 at 750 and 2/1 at 1500 (1024/4096). TwoRoom calibrated scores are 35/41 at 750 and 45/45 at 1500; saved scores are 22/11 and 44/44. At 750, calibration therefore adds 13 and 30 successes to the TwoRoom arms without changing learned parameters. Candidate saved-buffer control retains only three of the four initially successful cases at 750: its 11 total successes include eight newly solved cases, so subtracting four would be incorrect. The final seed remains necessary for the predeclared comparison.

After outcome 22, canonical static-chart extraction exceeded its 16 MB output ceiling because fallback numeric-cell tooltips repeated the entire global source-file list. A reporting-only repair uses stable record keys for repeated exact metric/context rows, preserves full identities and paths in the inventory, and stores the complete input-file union once in source metadata. Source tooltips name the actual reconciled SQL input. Every exact value remains indexed, and all size limits and browser probes are unchanged. The actual dashboard and a separately labelled full-size synthetic capacity fixture pass desktop/mobile/source checks. Twenty-four targeted reporting/coordinator tests pass; final full-suite verification remains queued. Desktop inventory and mobile horizontal value scrolling were visually inspected.

The coordinator reused the completed scientific evaluation and repaired only reporting. Its first restart failed during imports because resolving the interpreter symlink selected system Python; retrying the workspace virtual-environment path succeeded before further scientific work. The preserved failure records and recovery logs remain under the experiment root. A stale dashboard receipt can no longer mark a failed current stage verified. No training or control result was repeated to obtain a different score.
