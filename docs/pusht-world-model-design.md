# PushT E/U/P adaptation — proposed first experiment

This design adds a separate PushT experiment using the new visual encoder,
recurrent observer and latent predictor. It does not change the fixed paddle
baseline or the preserved LeWM implementation. No PushT model or training result
is established by this document. The coordinator acquired the compact official
CCHI dataset and source compatibility checks are underway. The coordinator has
stated the working interpretation **new E/U/P** and is proceeding after the
user's architecture clarification remained unanswered; later steering can
select a separate LeWM task without changing these experiment identities.

The implementation contract for shared concepts is the
[paddle brief](../world_model_codex_implementation_brief.md). PushT-specific
choices below are a separate declared experiment, not amendments to paddle.
The [data audit](pusht-world-model-data-audit.md) owns acquisition/schema facts
and source verification. Historical dataset counts are planning inputs until
the new acquisition receipt confirms them.

## Default scope and source

Use `data/pusht_cchi/pusht_cchi.h5`, converted from the official CCHI archive by
the existing [converter](../scripts/convert_pusht_cchi.py). The acquired HDF5
audit reports 206 episodes, 25,650 RGB96 observations, absolute target-XY actions,
and five pose labels: pusher XY, block XY, block angle. Images/actions/poses are
stored as float32; image values use 0..255 units, not 0..1. Episodes contain
49..246 observations and yield 25,444 valid transitions. Preserve the archive hash, raw
Zarr arrays, exact conversion receipt and source episode identities.

The configured LeWM relative-action archive is a different population and does
not fit the currently measured free disk when fully extracted. Do not mix its
action statistics, checkpoints, split or results with CCHI. The existing
[CCHI dataset configuration](../configs/datasets/pusht_cchi.yaml) is source
metadata; its `frameskip: 5` belongs to LeWM and is **not inherited** here.

Implement concrete modules under `world_model/pusht/`, new configurations under
`configs/pusht_world_model/`, derived data under `data/pusht_world_model/`, and
runs under `runs/pusht_world_model/`. Keep the current `world_model` paddle CLI
dispatch unchanged; the proposed separate entry point is
`python -m world_model.pusht <command>`. These paths/commands are planned, not
currently runnable interfaces.

## Time, images and executable actions

One model transition is one source interval: **0.1 seconds, 10 Hz, one 2D
absolute target command**. The checked simulator is
[third_party/swm/pusht.py](../third_party/swm/pusht.py), not the absent
`world_model/environment.py`. In `relative=False` mode its PD controller holds
the supplied world-coordinate target through ten 0.01-second physics steps.
An action is a target, not a realized displacement or velocity.

Canonical data records for an episode of L observed frames:

| Field | Shape / convention |
| --- | --- |
| `frames` | `[L,64,64,3]` canonical uint8; original float32 RGB96 retained as source |
| `actions_world` | `[L-1,2]`, actual absolute target XY in 512-unit world |
| `actions` | `[L-1,2]` float32, `actions_world / 512` |
| `poses_world` | `[L,5]`, pusher XY, block XY, angle in radians |
| `pose_targets` | `[L,6]`, four positions divided by 512, then sin/cos angle |
| `motion_targets`, `motion_mask` | `[L,5]`, defined below; false before index 2 |
| `source_episode`, `source_rows`, `group_id` | Exact source/split identity |
| `timestamps` | `0.1 * arange(L)`, explicitly inferred from declared source frequency |
| `end_reason` | Source episode boundary; termination/truncation unknown unless supplied |

Discard the final stored action from transition supervision because no successor
is observed inside the episode. Never infer a terminal failure from file end.
Require finite actions for all used transitions and audit bounds. The acquired
source action minima are `[12,25]`, maxima `[511,511]`; all final action rows are
finite but remain excluded. The default
executable normalized action domain is `[0,1]^2`; any out-of-domain source action
requires a recorded domain decision, not silent clipping of training data.

The completed full pixel scan found finite integer-valued float32 RGB scalars
only, range 65..255 and zero fractional values. Canonical preprocessing is fixed:

```
rgb96_uint8 = source_rgb96.astype(numpy.uint8)  # verified lossless source cast
rgb64_uint8 = cv2.resize(rgb96_uint8, (64,64), interpolation=cv2.INTER_AREA)
x = rgb64_uint8_tensor.permute(0,3,1,2).to(torch.float32) / 255
```

Apply the same uint8 INTER_AREA resize to **live simulator renders at resolution
96**, preserving RGB order. Store RGB64 uint8; model normalization is float32
`/255` only. Record OpenCV/version/resize identity in data manifests, caches and
checkpoints. Do not render directly at 64 for one population, use the earlier
superseded bilinear proposal, or inherit the old ImageNet-normalized 224-pixel
path. Future fractional source pixels require a different explicit preprocessing
identity rather than silent quantization.

The simulator's declared Box still has relative-action bounds even in absolute
mode. The new wrapper validates absolute XY itself and calls
`env.step(512 * normalized_action)` with `relative=False`; do not use that Box
to normalize CCHI commands. A zero vector means “target the origin,” not “stay.”
A hold-position baseline repeatedly targets the pusher's initial position; its
privileged initialization is labeled and never supplied to the learned planner.

## Observer, predictor and readouts

Reuse `Encoder`, `Decoder`, `Attention`, `TransformerBlock`, and the observation
and planning-state containers from the concrete paddle modules where they are
task independent. Reuse code/architecture, **not trained paddle weights**.
Implement small PushT U/P/H/R definitions with their own schema/version. Avoid
changing the fixed paddle constructors or introducing a general model registry.

| Module/value | Concrete interface |
| --- | --- |
| E | RGB `[B,3,64,64]` → fine `[B,256,64]`, coarse `[B,64,64]` |
| U | `(memory[B,128], S, previous_action[B,2])` → `[B,128]`; existing attention read plus GRU input width **66** |
| P | `(S, updated_memory[B,128], candidate_action[B,2])` → next S; action projection **2→64**, two existing transformer blocks, zero residual output head |
| D | S → RGB `[B,3,64,64]`, unchanged architecture |
| H | Flattened S `[B,20480]` → six pose targets; one linear layer |
| R | Updated memory `[B,128]` → six pose plus five observable-motion targets; one linear **128→11** layer |

Use **`(-1,-1)` only as the initial previous-action marker**. It lies outside
the executable normalized absolute-action domain and avoids adding a validity
bit or confusing start with an origin command. P never receives the marker.
Fresh-memory ablations assimilate the current frame with zero memory and this
same marker. Padding uses a valid executable vector plus a mask; padding never
updates memory or contributes a loss.

Real and imagined timing remains exactly the brief:

```
S_i = E(real_frame_i)
I_after_i = U(I_before_i, S_i, marker if i == 0 else action_(i-1))
S_hat_next = P(S_hat, I_hat_after, candidate_action)
I_hat_next = U(I_hat_after, S_hat_next, candidate_action)
```

Every candidate owns an independent memory. Execute one primitive action,
preserve the actual pre-branch `I_after_i`, and assimilate the next real image
once. Never install the selected imagined memory into the real observer.

CCHI has no measured pusher/block velocity labels. Define R's final five
coordinates as **causal frame-to-frame observable motion**, not instantaneous
Pymunk velocity:

```
delta_xy_i = pose_xy_i - pose_xy_(i-1)             # four positions
delta_angle_i = atan2(sin(theta_i-theta_(i-1)),
                     cos(theta_i-theta_(i-1)))
motion_target_i = [delta_xy_i / 512, delta_angle_i / pi]
```

Supervise pose at every real frame and motion only for `i >= 2`, preserving the
three-observation warm-up convention. No centered differences or future frames
enter targets, observer inputs, caches, or probes. Report motion errors as
world units per 0.1-second interval and radians per interval; optional rates per
second multiply by ten and must be labeled. These are motion-estimation
diagnostics, not proof that R identifies the full simulator state. Keep unusually
large wrapped angle changes visible as a temporal-aliasing diagnostic.

Train raw H/R sin/cos coordinates against unit-circle targets. Report orientation
error with `abs(atan2(sin(theta_hat-theta), cos(theta_hat-theta)))`, not subtraction
of raw angles across 0/2π. Also report predicted sin/cos norms. For planning,
normalize the predicted/goal pair before `atan2`; reject nonfinite values and
norms below `1e-6` explicitly. No extra angle or unit-norm loss is introduced.

## Splits, training and bounded budgets

Audit exact/near initial configurations using the existing circular-angle
grouping rule: pusher and block distance ≤5 world units and angle distance ≤0.05
radians. Freeze groups before selecting frames/windows, keep every member of a
group in one split, and publish exact episode/group lists. Default seed 3107,
80% train / 10% validation / remainder test by group count, with nonempty splits.
The acquired CCHI audit found 206 distinct exact/near initial configurations;
the proposed split therefore has **164 training / 20 validation / 22 test
groups/episodes**. The historical LeWM count of 185 groups does not apply to CCHI.
Audit duplicate trajectories/frames across splits and describe any remaining
configuration-group limitation. Do not reuse the old random-window split.

Use the brief's constant AdamW 3e-4, weight decay 1e-4, matrix/kernel decay only,
gradient clip 1, FP32, seed 3107. No SIGReg, pretrained LeWM encoder, augmented
images, extra auxiliary predictor objective, or architecture search in this path.

| Stage | Objective and frozen components | Initial GPU ceiling |
| --- | --- | --- |
| E/D/H | RGB MSE + mean squared error over the six normalized pose coordinates | 1,000 updates, 128 frames/batch |
| U/R | Masked mean squared error over valid 11-coordinate targets; E/D/H frozen | 1,000 updates, 2 complete padded episodes/batch |
| Statistics | Train-only per-coordinate across-frame latent population variances, averaged within each scale; float64 streaming, floor 1e-6 | One streaming pass, no optimizer |
| P K=1 | Equal fine/coarse variance-normalized latent MSE; observer frozen | 2,000 updates, 64 windows/batch |
| P K=5 | Mean same latent loss at all five imagined P→U steps; initialize from selected K=1 P | 2,000 updates, 16 windows/batch |

These are initial development ceilings, not convergence or “working PushT”
claims. First profile at most 100 GPU updates within those ceilings and record
time/examples/peak memory; schedule GPU work after the active paddle GPU task.
Do not contend with the expensive paddle evaluation or silently extend budgets.
Changing/continuing a ceiling uses a separate recorded continuation and parent
checkpoint hash, preserving cumulative and additional work.

Validate initially and every 100 updates on fixed validation populations:
up to 2,048 frames, up to 16 complete episodes, and up to 256 windows per P
horizon. Reduce counts to available data and record actual denominators. Select
lowest validation objective; retain last/resume and selected checkpoints.
Before P K=5, require selected K=1 to improve on matched copy in latent loss,
pusher-position MSE, block-position MSE and circular orientation MSE. The
pusher diagnostic remains in this predictive gate even though the control goal
is block-only. Publish
strict ratios/counts and failures; a failed gate stops five-step training and
starts diagnosis. Copy uses the exact same source/windows/actions/targets.

The acquired episodes are at most 246 frames. Start complete-episode memory
batches at two and reduce if needed; do not reset memory at
arbitrary chunks. Any later truncated BPTT change must carry detached memory
across contiguous chunks and be declared. Preserve gradients through frozen U
during imagined P training, as in paddle.

For the first CPU smoke, use two source training episodes and one episode from
each held-out split, deterministic from the frozen lists. Take two optimizer
updates per stage with frame/window batches 2/2/1 and memory batch one. Use
complete episodes; if CPU cost is excessive, an explicitly named ≤64-frame
episode-prefix smoke may exercise wiring, with full-episode behavior checked
separately. Smoke P K=5 can bypass the predictive gate only with a recorded
`smoke` override, never a passing scientific gate. Check six prediction windows
and two short control cases with CEM 32 candidates, 2 iterations, 8 elites,
horizon 5, at most 10 real intervals. Refresh the canonical HTML after each
completed stage/evaluation, including failures.

Estimate caches before allocation. The historical entire CCHI population needs
about 315 MB for canonical RGB64 uint8, 2.10 GB for full float32 S, and 13 MB
for memory128; original source storage is accounted for separately.
A bounded disk cache is feasible if the current free-space audit confirms it;
avoid loading all S onto GPU. Every cache includes source/split, resize,
action/label/tensor schema, and exact frozen E/U fingerprints. Preserve the
source and any failed cache builds.

## Goal planning and leakage-resistant control evaluation

Keep desired goal image G separate from predicted futures. Compute `H(E(G))`
once; candidate ranking uses only current S/memory, executable candidate actions,
and this learned image-goal readout. True source poses may supervise H/R and
initialize/evaluate a simulator, but never enter the learned planner or its cost.
No source future action sequence, future actual frame, simulator clone, contacts,
or block/agent state is available to learned CEM.

Default CEM: 64 candidates, 4 iterations, 8 elites, horizon **5 primitive actions
= 0.5 seconds**. Optimize absolute normalized XY in `[0,1]^2`, including the
current mean as a candidate and clipping only sampled planner candidates before
scoring/execution. Initialize mean at H's estimated current pusher XY, clipped
to domain, and std 0.2. Use a recorded RNG per case/controller. Score terminal
H-predicted pose against H-goal pose with

```
J = sum((predicted_block_xy - goal_block_xy)^2) / 20^2
    + wrapped_angle_error^2 / (pi/9)^2
```

Both XY vectors contain **block XY only**. The fixed task tolerances are
20 world units and π/9 radians. Pusher goal distance has default weight zero;
report it diagnostically and declare any future nonzero auxiliary cost. Evaluate the final CEM mean itself and execute only its first 2D action.
Reject invalid candidates, count them, and record explicit all-invalid failure.
Report total predictor transitions and synchronized decision latency including
all candidate evaluations, excluding rendering and next-real-frame encoding.

The first control experiment uses reachable **five-interval block-pose goals**,
with at most 50 executed intervals. Build goal images in the **current simulator**
from held-out source actions, so a slight source/current-physics mismatch cannot
make the desired goal unreachable by construction. This is a new local PushT
protocol, not the LeWM 25-interval task or a standard coverage benchmark.

Prepare cases before any controller outcomes are measured:

1. For each test episode, reset from its initial source pose using a fixed seed
   and the audited CCHI rendering configuration. Replay the entire actual source
   action prefix, retaining actual RGB96, actions and simulator pose after every
   interval. A valid decision index is `2 <= i` and `i+5 < L`.
2. At each eligible index, derive a goal by executing the next five held-out
   source actions on an independent exact copy of the current simulator state.
   A default exact-copy implementation is deterministic reconstruction from the
   identical reset and entire action prefix; verify body positions, angles,
   linear/angular velocities before branching. A seven-value observation passed
   to `_set_state` is **not** an exact clone. Do not share mutable Pymunk bodies.
3. Keep windows whose current-versus-goal **block** position distance is at least
   20 world units or whose wrapped orientation difference is at least π/9. These
   are initially unsolved under the declared strict success predicate and
   generally require contact-driven block motion. Actual contact counts are a
   diagnostic, not a hidden additional selection rule. Publish all eligible and
   rejected window/group counts and threshold definitions.
4. Choose up to 20 distinct held-out groups using seed 3107 and the **earliest
   eligible index** in each chosen group. If fewer groups are eligible,
   publish the smaller denominator; do not silently lengthen the horizon or
   relax the task. A lack of eligible cases prompts a separately recorded
   longer-horizon design, not arbitrary replacement by almost-static goals.
5. The public harness case contains initial physical reset state, observed
   prefix actions, goal RGB, case identity and budget. The harness uses reset
   labels only to reconstruct the simulator; it passes **only actual rendered
   prefix RGB/actions and goal RGB** to the learned controller. Source future
   actions stay in a separate oracle record. No reset/state labels enter the
   learned observer, predictor or planner. Every controller starts from the
   same reconstructed physical state and observer history.

The prefix collector may reuse a continuous episode replay to enumerate all
candidate indices; it need not replay every prefix independently merely to
count eligibility. Materialize/replay the selected exact branches and check
consistency before freezing final cases. Native SWM full-pose termination is not
this block-only task: the wrapper applies the declared block predicate during
control, while oracle prefix collection advances physical dynamics without
stopping at unrelated native goal flags.

Evaluate learned memory, fresh-memory reset, hold-position, seeded uniform
absolute commands, and recorded-action replay on these identical cases.
Replay is a privileged reachability check, never a planner input, and must reach
its five-interval constructed goal; a failure is an implementation/environment
reconstruction defect to retain and resolve before scientific control claims.
Report every case, initial status, replay error, final block position/angle
error, auxiliary pusher distance, success denominator, action count, and latency.
Use similarly constructed **validation** cases for development choices; reserve
the frozen test cases for the selected final comparison.

Important compatibility limit: `_set_state` advances physics by 0.01 seconds and
the five source pose values omit all velocities. The local simulator may not
reproduce CCHI trajectories exactly. Audit actual reset poses/pixels and complete
recorded prefixes, report position/angle/pixel residuals, and inspect examples
before interpreting control results. Never combine a source RGB frame with a
mismatching post-reset simulator state as if they were one observation. Drift
from CCHI is a training-versus-control domain discrepancy to quantify; it does
not invalidate reachability of goals constructed in the current simulator.
Failure to reconstruct those current-simulator branches is a separate defect
that must be resolved before learned-control claims.

Include an action/frame timing check in that audit: compare the source's declared
`action_i` transition with explicitly shifted controls on a fixed prefix. A
lower shifted residual alone is not permission to relabel the source; resolve
any discrepancy against source collection semantics and record the decision.

The initial acquisition audit compared episode starts 0, 1, 102 and 205. The
first three reset poses were exact and ten-step position errors were below
`1e-5`; contacting episode 205 had reset position RMSE 0.20564 world units and
angle error 0.00042765 radians, with ten-step position RMSE 0.009376 and angle
error 0.001089 radians. Initial image MAE was 0.04865..0.11382 in 0..255 units
and 0.217..0.445% of pixels differed. These four cases support a practical
compatibility path but do **not** establish exact whole-source replay. The raw
[acquisition audit](../runs/pusht_cchi_data_audit_2026-09-07.json) remains
authoritative; perform the per-control-case checks above.

The SWM environment's numeric goal-distance implementation expects a seven-value
goal. Append two explicit zero pusher-velocity defaults to source pose5 for reset
and numeric-goal compatibility. These are reset conventions, never invented
source velocity labels or R targets. The task success predicate itself uses
only block XY and block angle.

Source and live goal/background rendering also require inspection.
`with_target=True` draws a green shape from `goal_pose`; `_set_goal_state` changes
numeric goal labels without changing that drawing. Match the source renderer
configuration; do not remove or relocate its green overlay silently. Additional
goal colors, arrows, state labels and future actions belong only in diagnostic
panels, not raw training/controller frames.

The new task success predicate is **block position L2 <20 world units and
wrapped block angle error <π/9**. It deliberately excludes pusher goal position,
which remains an auxiliary readout/error. Implement this task-specific predicate
without modifying SWM's preserved full-pose evaluator. Do not mistake its unused
`success_threshold=0.95` attribute for a measured 95% block-overlap criterion,
and do not use its mixed-coordinate seven-dimensional `state_dist` as the
primary metric. A geometric coverage benchmark is a distinct protocol.

## Prediction evidence and reporting

On fixed held-out windows with `i >= 2`, report K=1..5 latent error, decoded RGB
MSE, H pose errors, and R pose/motion errors for learned prediction, copy S, and
fresh-memory reset. All methods share the same starts, actions and targets.
Include shuffled-action and fixed-current-target action diagnostics separately
to distinguish motion prediction from action sensitivity. Do not call a zero
absolute action “no action.” Report all physical-coordinate mean/p95/max and
exact counts, with small source-motion versus large-motion subsets fixed using
training-defined thresholds. Source data lacks contact labels; do not infer
“collision” merely from an error spike.

Evaluate H on all held-out actual frames; evaluate R motion only after its
warm-up mask. A train-only linear S→motion probe may diagnose current-frame
correlation, but CCHI has no certified identical-current-image/opposite-motion
pairs: ordinary probe error cannot prove memory identifies hidden motion.
Do not manufacture paddle-style pairs or claim the paddle diagnostic transfers.

Save raw per-window/per-case JSON/CSV, actual-versus-reconstruction-versus-imagined
RGB panels, desired goal images, and separately labeled real/imagined R motion.
Show both preselected successes and failures. Regions for reconstruction
diagnostics must be separately validated masks, never hidden model inputs.

Use task-specific PushT manifest/result/evaluation schema tags and a read-only
native dashboard adapter. Preserve LeWM and paddle curves with separate metric
labels/populations. Exact source paths, episode/group split, counts, selected
versus latest updates, cache/dependency hashes, compute budgets and negative
gates must survive into the canonical artifact. Reuse the tested separation of
training versus validation/copy curves so the portable reader cannot reorder
steps or break sparse curves. Browser success alone does not replace visual QA.

## Essential implementation checks and direction choices

The following concrete interfaces let data/model/planner work proceed independently
without changing the agreed semantics. Function names are proposed contracts for
the new package, not claims that code is present:

| Owner/module | Interface to implement |
| --- | --- |
| `data.py` | `prepare(source, output, seed=3107)`; `verify_dataset(output, source=None)`; `EpisodeDataset(output, split)` exposes `.entries`, `.lengths`, `.manifest`, `.fingerprint` and the canonical episode dictionaries; `.window_indices(horizon, min_history=2)` returns valid causal windows; `make_targets(poses_world[L,5]) -> (targets[L,11], mask[L,11])`; `canonical_frame(rgb96) -> RGB64 uint8` follows the single frozen contract above |
| `models.py` | `build_models() -> dict(E,D,H,U,R,P)` with shapes above; PushT schema constants distinct from paddle; shared E/D/attention/block code imported without weight reuse |
| `rollout.py` | `observe(memory_before, rgb, previous_action, models) -> PlanningState`; `rollout_step(state, action, models) -> PlanningState`; no simulator or privileged-label arguments |
| `planner.py` | `encode_goal(goal_rgb, models) -> learned_pose[B,6]`; `plan(state, learned_goal_pose, models, config, generator) -> PlanResult(action[2], sequence[5,2], cost, selected_states, invalid_candidates)` |
| `evaluation.py` | `prepare_cases(dataset, split, count, seed, output)`: public history/goal records plus separate oracle records; `evaluate(system, public_cases, oracle_cases, config, output)`: same-case controllers, physical outcomes, matched prediction diagnostics |
| `training.py` / `checkpoints.py` | `train_stage(stage, config, data, run, dependencies, horizon=1, initialize_from=None, resume=False)`: explicit staged optimizer/freeze/dependency contracts; `load_system(...) -> dict(E,D,H,U,R,P,statistics)` |
| `__main__.py` / `cli.py` | Separate `prepare-data`, `train-perception`, `train-memory`, `train-predictor --horizon 1|5`, `evaluate`, `demo`, `run-all`; every completed stage/evaluation invokes canonical reporting |

Use independent `pusht-eup-observer-v1`, `pusht-absolute-xy512-start-negative-v1`
and label/preprocessing schema tags in checkpoints and caches. Run manifests
record task `pusht_eup`, stage, horizon, exact source/split/config, dependencies,
versions, and resolved counters. Evaluation records distinguish
`current_simulator_replay_goal_k5_block_pose_v1` from any legacy/source-image
goal protocol. Do not make a PushT record masquerade as `paddle-evaluation-v1`
merely to obtain existing dashboard charts.

Before implementation, add informative failing checks for: L frames/L−1 aligned
actions; no cross-episode windows; disjoint groups/train-only statistics; RGB
resize parity; absolute-action scaling/start marker; causal wrapped finite
differences; no padded memory updates; P→U branch isolation and frozen-U gradient
flow; goal/candidate interfaces that receive no privileged state; execution of
exactly one primitive action; and checkpoint/cache schema rejection. Reuse the
already meaningful paddle tests where possible without weakening their contract.

Choices that materially change the experiment:

| Choice | Current default | Alternative requiring a declared protocol change |
| --- | --- | --- |
| Which model to train | New E/U/P using CCHI; coordinator's stated working interpretation | Resume/reproduce LeWM, requiring its compatible data/weights and separate recipe |
| Meaning of PushT goal | New block XY/orientation predicate; pusher distance auxiliary | Standard geometric coverage, or the old full pusher+block pose task |
| Planning timescale | Primitive 10 Hz model, K5 training/planning, short five-interval goals | K25 primitive rollout for 2.5-second plans, or five-command macro actions; these are different training/horizon interfaces |
| Data population | Verified compact CCHI, absolute XY, group-held-out split | Relative-action LeWM source/subset with its own provenance/domain/normalization |

The architecture clarification remains open to user steering. The coordinator's
stated interpretation and agreed defaults allow the thin path to proceed
without redundant permission.
Confirm a harder task/horizon before calling short-goal success “working PushT”
for the old goal protocol. Extending primitive planning to 25 steps while only
training K5 is an extrapolation experiment; do not hide it by relabeling steps.
The existing LeWM loader packs five primitive actions per prediction, and its
current evaluator executes all 25 flattened actions per plan despite a
`receding_horizon: 5` metadata field. Reuse the CEM idea with the new explicit
one-action execution loop, not that legacy control loop or its timing labels.
