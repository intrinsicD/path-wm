# Paired SIGReg projection-count training screen

User instruction: cowork with Claude on the proposed sample-efficiency experiment. This implements and tests the first intervention only; original checkpoints and reproduction configs remain immutable.

## Frozen screening design

- Datasets: existing full-source PushT and TwoRoom, source-compatible random-window split and full-source action normalization. This is a source-population learning-efficiency screen, not independent-data generalization.
- Paired seeds: 3072, 3073, 3074. Within each seed, use identical initialization, train/validation populations, exact window order, model/dropout RNG and independent sketch RNG seed=seed+700000. Other pairs are independent initialization/data-shuffle draws.
- Arms: 1024 versus 4096 resampled SIGReg directions. Both use the new independent, checkpointed sketch RNG; no frozen direction set. All other scientific choices remain identical: latent192, batch128, lambda.09, source model/dropout, bf16 training, FP32 evaluation, AdamW lr5e-5, wd.001, clip1, full-batch encoder activation checkpointing.
- Train from scratch for1500 updates per arm. Explicit short-budget cosine schedule, warmup15 updates. This is not a continuation of the139330-update reference schedule. Save750/1500-step checkpoints, prediction checks every250 and training logs every25.
- Twelve runs total: 2 datasets ×3 seeds ×2 arms. Counterbalance projection-arm order across paired seeds/datasets; execute serially on the available RTX3050. Estimate6h training from1.17s/update; preflight measures actual arm overhead and memory. Cap2100 cumulative training-loop seconds per run, preserving partial checkpoints if hit. Account separately for final validation, setup and control evaluation.
- Primary descriptive outcomes: paired calibrated control successes at750 and1500 processed-window checkpoints, with saved-buffer control reported separately; prediction/control learning curves, samples per update, gradient norms, objective components, update/data-loading time and GPU memory. Existing frozen50-case sets are regression populations shared across arms, not a generalization test.
- Apply the same training-only512-window BN calibration policy per checkpoint/arm. Do not mix saved and calibrated scores or choose the policy after seeing outcomes. Preserve original trained states and label clones diagnostic_only.
- Compare within pairs. Report each seed, paired differences and aggregate uncertainty; three pairs are a screen, not a universal claim. No statistical pass threshold is fixed for this exploratory screen. A loss/noise improvement cannot imply a control gain or4-fold training acceleration.
- Reproduction guards: raw dataset/source and case-set hashes, initial-weight hash equality per pair, restored independent sketch stream and global RNG across resume, final checkpoint/validation-step agreement, all outcomes including stopped or failed runs.

## Implementation and co-work

Claude implements SeededSIGReg and essential RNG/parity/serialization tests in a neutral MCP task using the public LeWM algorithm and generic requirements. No private source/data/result transfer. Codex integrates the opt-in regularizer, checkpoint restoration, preflight, configs, serial driver, measured result collection and existing dashboard. Claude also contributes experiment assessment through a second bounded generic/public task if useful. Cap$4 reported API-equivalent across two calls; no claim of available account balance.

Scientific config extension: optional sigreg_seed selects an independent RNG implementation; absence preserves the original source recipe and global RNG behavior. Store regularizer state only for the new opt-in mode and require it when resuming that mode. Legacy source checkpoints remain readable and unchanged.

Essential tests precede implementation: fixed-direction loss and input-gradient parity, sketch RNG isolation, exact next-sample serialization, incompatible-state rejection, and exact training recovery with sketch state. Then preflight a bounded actual update benchmark for both projection counts before launching the full serial screen.

Every completed run/evaluation goes through run.py and must produce verified runs/experiment_dashboard.html. Keep the raw ledger authoritative and report publication failures separately. Freeze any necessary protocol amendment before affected experiments; record budget stops or incomplete control evaluations explicitly.

## Evaluation implementation detail (before affected runs)

Reuse check_training_modes with an optional calibration_only flag requiring save_calibrated: preserve the identical layerwise512-training-window calibration and clone provenance, skip its auxiliary four precision/split probes. Saved-buffer prediction is already logged during training; the frozen control comparison still evaluates both variants at750/1500. A tiny-data integrity test must establish that only BN buffers change, source hashes remain unchanged, and exported clones retain the exact training-step and calibration population. Historical50-case control commands cost about3–4min each, so48 evaluations add about3h plus calibration/reporting. Total expected duration is9–10h.
