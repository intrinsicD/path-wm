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

The first eight-case ranking completed: local predicted-cost selection succeeds
2/8, released8/8; selection by measured simulator-image latent costs succeeds8/8
for both. Local mean physical regret48.88px versus1.71px released. This diagnoses
this candidate population; it is not a new closed-loop control score.

Claude source-access tasks were blocked before launch by automatic review;
specific permission is pending. A safer task succeeded through the same MCP
using --safe-mode --restricted, no file tools or project context: Claude generated
the layerwise calibration utility and three essential tests for$0.478356 reported
usage. Codex inspected, integrated and verified that implementation. The --bare
attempt could not use existing authentication and spent$0.

The integrated suite passes74 tests including browser checks. TwoRoom validation
prediction/copy changes from3.948 to0.02965 on a training-only calibrated clone;
absolute prediction MSE2.63049 to0.0279162, with copy error0.666288 to0.941393.
The encoder/target scale changes, so the matched control ratios and subsequent
closed-loop test matter. Saved float32/bf16 and calibrated float32/bf16 are
recorded separately. Original checkpoint/model bytes and training/validation
window separation pass. Calibrated control is the next committed conditional
step. This new evidence also warrants the same non-training check on PushT before
launching the accepted continuation; no architecture or objective change.


TwoRoom calibrated primary control completed48/50 (44/46 initially unsolved),
versus original14/50 (10/46) and released42/50 (38/46), on identical cases,
reset/CEM seeds and30-iteration solver. Parent and clone hashes unchanged.
Predeclare the next bounded follow-ups before launching: apply the same512-window
mode/calibration protocol to PushT8404; test its calibrated copy on the existing50
PushT cases. Also evaluate the TwoRoom calibrated clone on the existing frozen
100/150 CEM10 case file, directly paired with the completed0/50 local and5/50
released CEM10 references. No solver parameters or thresholds are tuned. These
short checks fit within the planned diagnostic compute allowance; implementation
and HTML work are separate.
