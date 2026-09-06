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

## Comparison reporting contract

A projection_comparison.json derivative binds per-arm control rows to the native reconciled cases, exact case-manifest hash, trained initialization hash and calibration row hash. It records all48 expected outcomes and missing arms explicitly; paired statistics come from Claude's summarize_pairs. The dashboard shows separate dataset/BN-policy learning curves and an exact per-seed table, with source file hashes in the derivative. It must reject a tampered derivative whose group statistics disagree with its arm rows. No seed pooling across datasets or calibration policies.

## Preflight and reporting verification

All four 20-update preflights passed, with equal paired initialization and global CPU/CUDA RNG states. Update times for 1024/4096 were 1.3265/1.3368 seconds on PushT and 1.3198/1.3354 seconds on TwoRoom; peaks were 3.2822/3.2830 GB. The 2100-second cumulative training-loop cap remains in force. A separate tiny CUDA check confirms bitwise native bf16 loss/gradient parity and exact private-stream resume. Its first launch failed before computation because the script import path was missing; the preserved rerun passed with the repository import path set.

The full implementation suite passed 99 tests including three browser checks. A synthetic paired-chart browser probe then exposed that the canonical renderer refuses one-x line charts even with visible-point settings. The repaired view uses grouped seed bars until two distinct checkpoints exist, then marked learning curves. All six synthetic bars render within both 390- and 1440-pixel viewports; captures were visually inspected. Synthetic fixture values are not included in the experiment ledger. The relevant semantic reporting suite passes after the repair.

Qualitative inspection is fixed before control results: first frozen case, seed 3072, calibrated 1500-update checkpoints for each arm/dataset, each compared with the existing released reference on exactly that protocol. Replay the saved actions and verify outcomes with render_prepared_control; do not select a successful case after seeing scores. Calibration windows have disjoint starts from recorded validation windows, but the random-window source protocol shares episodes and can share frames; this is not a frame-disjoint generalization test.

Reporting correction found while interpreting the first completed arm: the inherited validation ratio rollout_mse / identity_mse compares an autoregressive rollout with an adjacent one-step copying error. It is not a matched multi-step skill score. Keep raw metrics and this frozen training recipe unchanged; label that ratio as rollout / one-step copy and disclose the different horizons. One-step prediction/control ratios and the standalone matched-horizon inspector remain valid as defined.

Sample accounting: 96,000/192,000 denotes optimizer-training window presentations at750/1500 updates. Each calibrated checkpoint additionally uses the same512-window set drawn from the full training split; some may lie outside its already-optimized prefix. Full-source action normalization also uses the existing source population. Both arms receive identical access within a pair. This screen tests optimization efficiency at matched exposure and source recipe; it does not demonstrate training from a smaller independent dataset.

Coordinator recovery guard: a duplicate invocation must reject the launch without overwriting the active coordinator's progress or stage receipts. Validate this with an isolated locked-directory test; do not restart or interfere with the active trainer.

## Final diagnostic figure contract

Source-backed PNG/SVG companions will compare the recorded training gradients and saved-buffer one-step prediction controls within each seed, alongside the primary control figure. Use six facets (dataset × seed), at most two color roots (1024 blue,4096 orange) plus distinct markers. Preserve the complete source metric rows and their hashes in each export receipt. Gradient panels show all logged pre-clip L2 norms as discrete observations against updates, with the configured clipping threshold; these samples do not estimate gradient variance or every-update clipping frequency. Prediction panels show discrete scheduled trained-checkpoint pred_mse / identity_mse comparisons (250–1500), with a ratio-one reference and common logarithmic scale; do not connect six sparse checkpoints into a smooth learning trend. Initialization metrics remain in the snapshot but are explicitly outside this trained-checkpoint view. No extra evaluation points or changes to the frozen schedule are needed. Missing arms/facets remain visibly missing. Inspect the actual exported figures at readable size before delivery; the existing canonical HTML retains its native chart runtime and exact ledger tables.
