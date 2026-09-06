# TwoRoom diagnostic follow-up and conditional PushT continuation

User accepted the proposed bounded diagnostic sequence and explicitly requested
Claude implementation co-work via MCP on 2026-09-06. Work begins at 11:24 local.
The previous overnight study remains frozen. No numerical scientific gate is
introduced. Target reporting artifact remains runs/experiment_dashboard.html.

## Plan and interfaces

1. Claude implements split-safe training-mode diagnostics in
   scripts/check_training_modes.py plus tests/test_training_modes.py. Restore the
   saved random-window/episode split using inspection_datasets; preserve exact
   validation indices, history, image size, statistics and checkpoint identity.
   Compare saved float32/bf16, current-batch BatchNorm with dropout disabled, and
   training-only calibrated disposable clones. Use 512 saved validation windows,
   512 fixed training calibration windows, batch128 and seed103072. Calibration
   uses no validation windows or optimizer updates. Record per-mode copy, zero,
   shuffle, prediction and rollout controls with actual sample identities; model
   and original checkpoint hashes must remain unchanged. Layerwise calibration
   freezes previously calibrated BatchNorm layers, avoiding inconsistent inputs
   to downstream prediction normalization. Write ledger-native prediction.json
   and manifest.json per usable eval-mode variant; batch-statistic diagnostics
   remain separately labeled and are not deployable single-state evaluators.
2. Codex implements simulator-grounded TwoRoom ranking, using the first eight
   primary25/50 frozen cases, both unchanged checkpoints, source resets/seeds and
   a five-block (25-action) open-loop comparison. Candidates: replay, zero raw
   action, sixteen seeded random sequences, local CEM plan, released CEM plan.
   Score exactly the same physical actions with each model's own normalization.
   Use released-compatible CEM30 iterations,300 samples,30 elites; distinguish
   terminal and any-step success, predicted/actual latent goal costs, physical
   distance and rollout/copy errors. This is action-ranking diagnosis, not a new
   50-action closed-loop score. Freeze a manifest before computing outcomes.
3. Claude repairs mobile control-chart layout, preserving canonical delivery and
   all exact source labels/data. Require visible nonzero bars within390px rather
   than merely passing document-overflow checks. Keep desktop behavior and all
   canonical verification. Read the existing installed reader before CSS changes.
4. If calibration yields a substantial prediction improvement, run its disposable
   clone on the same50 primary control cases alongside the unchanged14/50 result;
   treat improved MSE alone as insufficient to adopt a baseline change.
5. If implementation checks reveal no unresolved training-integrity error,
   continue PushT from8404 to13933 updates in a distinct directory, preserving
   parent bytes, optimizer/RNG state and the139330-update learning-rate schedule.
   Add an explicit operational stop_at_step, separate from max_steps/schedule,
   and an auditable fork interface rather than mutating the completed parent.
   Estimated5529 updates at1.166s/update is107.4min plus preparation/evaluation;
   bound continuation training to2h10 with final validation extra. Recheck frozen
   50-case control and matched internals, then compare against8404/released.

Each substantive slice gets essential failing tests committed before its fix.
Claude owns only named diagnostic or mobile files; Codex integrates and runs GPU
work serially. Initial Claude caps are$6 diagnostic implementation and$3 mobile
implementation in reported API-equivalent usage; account quota balances are not
exposed. Review follow-ups are conditional and bounded. No unrelated files,
credentials or dataset images need transfer to Claude.

The initial diagnostic compute budget is about one hour, excluding implementation
and reporting repairs. Every experiment uses run.py and must publish verified
HTML. Preserve failed results and protocol deviations. The decision record will
separate measured calibration/dynamics effects from unproven causal explanations.

## Results

Pending execution.
