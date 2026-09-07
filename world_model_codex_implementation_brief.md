# Multimodal world-model research prototype: Codex implementation brief

Version 1.0 · 7 September 2026 · Prepared for Alex

## 1. Task for the implementing Codex agent

Implement a complete, runnable first world-model experiment on Alex's computer, using the specification below. Start with a deterministic paddle-and-ball simulator, then implement perception, recurrent state estimation, latent prediction, debugging readouts, and planning. Train in the specified stages and evaluate actual control performance.

This brief is self-contained. You do not need the original conversation or its earlier architecture alternatives. The design is agreed; implement it before suggesting replacements. This document describes work to do: no simulator, dataset, implementation, trained model, or benchmark result has yet been produced by this design session.

The long-term project is a multimodal world model with learned local filters, multiresolution representations, attention across learned features, persistent latent memory, and a separate predictor used for planning. The first experiment deliberately uses RGB frames and discrete controls. Add modalities after the complete initial path works. Preserve module boundaries so that future changes remain possible, but implement concrete modules rather than a general plugin framework.

### Decision status

**Agreed:** the E/U/P architecture, tensor sizes, visual decoder, supervised observer objectives, freeze order, predictor loss, one-then-five-step predictor training, exhaustive five-step planner, task score, environment geometry/timing, dataset counts, and evaluation targets.

**Implementation defaults supplied here:** Python/PyTorch, exact positional embedding construction, attention projection details where previously unspecified, initialization conventions, file formats, optimizer settings, batch sizes, validation frequency, run budgets, command names, and repository organization. Use these defaults and record any justified adjustment. A batch-size change is an implementation adjustment; replacing U with another state model is a research change.

**Research targets, not guaranteed results:** the numerical accuracy and interception thresholds. Do not fabricate success or redefine metrics when a run fails them. Debug implementation errors, then report specific model/objective limitations and the smallest follow-up experiment.

### Working approach

1. Inspect the selected local directory, its instructions, available Python environment, GPU, and installed packages. If an existing project is supplied, integrate with it carefully. Otherwise create a dedicated `world_model_prototype/` project in the working directory.
2. Use Python and PyTorch for this first experiment. Target Alex's Ubuntu computer and RTX 4090 with 24 GB VRAM, but verify the actual machine before choosing an installation command or batch size.
3. Work in a project virtual environment. Use a compatible stable PyTorch CUDA build from the [official installation selector](https://pytorch.org/get-started/locally/), preserving any working installation. Record the actual Python, PyTorch, CUDA-runtime, driver, and dependency versions. Do not assume this document's creation date fixes the installed version.
4. Build and test the whole runnable path, including a small training run. Provide resumable full-run commands. Run the configured full experiment when local execution is authorized and practical; do not substitute an endless hyperparameter search for the specified baseline.
5. Keep implementation simple: NumPy for CPU simulation, PyTorch for models, standard configuration files, local logs, and a small CLI. Suitable supporting packages are pytest, Pillow, matplotlib, and PyYAML. No engine integration, distributed training, external experiment service, or custom CUDA kernel is needed initially.
6. Honor local instructions and preserve unrelated files. Report genuine blockers with the completed work and an exact recovery command. Make routine implementation choices without repeatedly reopening accepted decisions.

## 2. The fixed architecture and time convention

There are three core learned modules:

| Module | Role | Persistent state owned by the module |
|---|---|---|
| E | Encode the current observation into multiscale features S | None |
| U | Assimilate current features and the previously executed action into memory | Its caller explicitly carries the 128-value memory |
| P | Predict next observation features under a candidate action | None; it receives memory from U |

For a real observation at step i:

```text
S_i  = E(x_i)
I'_i = U(I_i, S_i, a_(i-1))
B_i  = (S_i, I'_i)
```

- `x_i`: actual RGB frame at decision step i.
- `S_i`: features of the current frame, retaining fine and coarse spatial grids.
- `I_i`: memory entering observation step i, before assimilating that frame.
- `I'_i`: memory after assimilating the frame.
- `a_(i-1)`: action executed during the preceding interval.
- `a_i`: candidate or selected action for the next interval.
- `B_i`: structured planning state, not an extra learned embedding.
- `G`: goal/evaluator configuration, distinct from a predicted future.

One imagined step, starting from a copied planning state:

```text
predicted_S_next      = P(S, updated_memory, candidate_action)
predicted_memory_next = U(updated_memory, predicted_S_next, candidate_action)
next_B               = (predicted_S_next, predicted_memory_next)
```

Repeat these exact P-then-U operations for a longer imagined rollout. Every branch owns its memory copy. Actual future observations never enter an imagined branch.

When an action is executed in the real environment, carry `I_(i+1) := I'_i`. After the actual next frame arrives, compute:

```text
S_(i+1)  = E(x_(i+1))
I'_(i+1) = U(I'_i, S_(i+1), a_i)
```

Do not perform an additional action-only update of real memory between these operations. The interval must not be advanced twice. Do not replace real memory with the state of the chosen imagined branch.

The input features S and internal memory I have different dimensions and meanings. No loss requires their coordinates to match. For same-space next-step prediction, the actual target is `T_i = S_(i+1)` and the prediction is `T'_i = predicted_S_next`.

## 3. Exact data and tensor interfaces

Use channels-first tensors for convolutions and batch-first tensors for attention. The document's earlier image-grid descriptions correspond to these implementation shapes:

| Value | Shape | Meaning |
|---|---|---|
| RGB batch | `[B, 3, 64, 64]` | Float image values in [0,1] |
| `S.fine` | `[B, 256, 64]` | 16 x 16 spatial positions, 64 channels |
| `S.coarse` | `[B, 64, 64]` | 8 x 8 spatial positions, 64 channels |
| `S.tokens()` | `[B, 320, 64]` | Fine positions followed by coarse positions |
| Memory entering/updating U | `[B, 128]` | Explicit recurrent state |
| Action ID | `[B]` | Integer in {0,1,2} |
| Action one-hot | `[B, 3]` | Shared executable action semantics |
| Position readout H | `[B, 3]` | `(ball_x/64, ball_y/64, paddle_x/64)` |
| State readout R | `[B, 5]` | `(ball_x/64, ball_y/64, ball_vx/6, ball_vy/6, paddle_x/64)` |
| Decoded image | `[B, 3, 64, 64]` | Float values in [0,1] |

Flatten grids in row-major order: token index is `row * width + column`. Each token stores channels contiguously. To flatten S for H, concatenate fine/coarse tokens first, then flatten the token and channel dimensions to 20,480 values. Use the identical conversion everywhere.

Use a small typed container such as `ObservationLatent(fine, coarse)` and `PlanningState(observation, memory)`. Supply operations to concatenate, split, detach, clone, index, and batch these values. Avoid a universal latent registry.

Unless specified otherwise below, use PyTorch's default initialization for linear, convolutional, and GRU parameters. Use affine LayerNorm with epsilon 1e-5 and zero attention/MLP dropout throughout the baseline. The explicit positional/role embedding initialization and zero predictor delta head override those defaults.

### Action mapping

| ID | Executable action | One-hot | Paddle displacement command per interval |
|---|---|---|---|
| 0 | left | `[1,0,0]` | -4 |
| 1 | stay | `[0,1,0]` | 0 |
| 2 | right | `[0,0,1]` | +4 |

Only the initial observation uses the all-zero previous-action vector `[0,0,0]`. It means no previous action; it is not stay. Do not record it as a fourth executable action.

U receives the previous action as this one-hot vector. P receives the candidate one-hot vector and learns its own projection. Keep the executable action ID for the simulator. No shared learned action encoder, inferred latent action, or action decoder is part of the baseline.

### Minimal public interfaces

```text
E.forward(rgb_batch) -> ObservationLatent
U.forward(memory_before, observation_latent, previous_action_onehot) -> memory_after
P.forward(observation_latent, memory_after, candidate_action_onehot) -> ObservationLatent
D.forward(observation_latent) -> reconstructed_rgb
H.forward(observation_latent) -> normalized_positions
R.forward(memory_after) -> normalized_state
rollout_step(planning_state, action_onehot, P, U) -> PlanningState
planner.plan(planning_state, goal_config) -> PlanResult
```

`PlanResult` should contain the chosen action ID, five-action sequence, score components, and optional selected imagined states for inspection. The learned planner receives no simulator state or environment object. A separate privileged simulator planner is an evaluation diagnostic only.

## 4. Deterministic environment

### Geometry and units

Use float64 for the CPU simulator. World units equal pixel widths; object centers remain continuous.

| Property | Value |
|---|---|
| Arena | `[0,64] x [0,64]`; y increases downward |
| Observation | 64 x 64 RGB |
| Decision/observation interval | 0.05 seconds |
| Ball shape | Axis-aligned square with side 4 and half-size 2 |
| Paddle rectangle | `[p_x-6,p_x+6] x [56,59]` |
| Paddle center limits | `[6,58]` |
| Ball side-wall center limits | x = 2 and x = 62 |
| Ball ceiling center limit | y = 2 |
| Paddle interception center plane | y = 54 |
| Loss center line | y = 61 |
| Initial horizontal ball velocity | Uniform choice from `{-6,-4,-2,2,4,6}` units per interval |
| Initial vertical ball velocity | Uniform choice from `{2,3}` units per interval |
| Episode cap | 200 action intervals |

Use time measured in decision intervals inside the integrator: one call to `step` advances time by 1. Record observation timestamp `step_index * 0.05` seconds. A horizontal velocity of 6 therefore means 120 world units/second; do not multiply or divide by the interval again inside the interval-unit integrator.

### Initial state and warm-up

Sample independently:

```text
ball_x   ~ Uniform(8,56)
ball_y   ~ Uniform(8,16)
paddle_x ~ Uniform(16,48)
ball_vx  ~ UniformChoice(-6,-4,-2,2,4,6)
ball_vy  ~ UniformChoice(2,3)
```

Observe x_0, execute stay, observe x_1, execute stay, observe x_2. Assimilate all three frames before learned control begins at step 2. These two stay actions are part of the recorded trajectory and the 200-action cap. Use the same warm-up for all control baselines. Initialize memory to zero only at the episode's first observation.

### Within-interval evolution

The paddle moves at the commanded constant velocity -4, 0, or +4 until it reaches its horizontal bound, then remains there for the rest of that interval. The ball moves with constant velocity between events.

Implement event-based integration within each interval. Candidate events include a ball side-wall hit, ceiling hit, downward crossing of the paddle contact plane, loss-line crossing, paddle reaching its bound, and the interval end. Advance all moving coordinates to the earliest event, apply the event, then integrate the remaining time.

Rules:

1. Side walls reverse ball vx; the ceiling reverses ball vy.
2. When a descending ball crosses center y=54, compute the paddle's position at that exact time. A catch occurs if `abs(ball_x - paddle_x) <= 8`. Reverse ball vy on a catch; preserve vx and speed. Increment the hit count and continue the episode.
3. A downward crossing without horizontal overlap is a missed contact opportunity. Continue toward the loss line. Do not permit a later side/underside contact with the paddle.
4. At center y=61, enter a terminal state. Freeze ball and paddle for the remainder of the current interval and set terminal velocities to zero. Return the visible terminal frame at the normal interval boundary. No reset occurs inside that trajectory.
5. A catch is not terminal. This ensures training contains post-catch motion and prediction can continue after a catch.
6. At 200 actions without a miss, set `truncated=true` and `terminated=false`. If a miss also occurs on the last action, termination takes precedence. Do not label a time limit as a missed ball.

Handle simultaneous events consistently at their shared time. Snap an event coordinate to its exact boundary. Use small numerical tolerances, such as 1e-10 in interval time, only for numerical comparisons. Handle a ball already on a wall and moving outward as an immediate reflection. Avoid repeated zero-time events: after processing a missed paddle-plane crossing, the same crossing must not be reconsidered at the next loop iteration or interval. A descending ball already at/below the plane has passed that opportunity. Add a defensive event-iteration limit that raises a diagnostic error instead of silently changing the physics.

Do not implement endpoint-overlap-only collisions: the fastest ball moves 6 pixels during an interval. Collision timing, including paddle movement during that interval, is part of the task.

### Rendering

Render black background, cyan paddle `(0,255,255)`, and white ball `(255,255,255)`. No overlays, velocity trails, motion blur, labels, action markers, seed indicators, or time-dependent colors enter the model image.

Use deterministic pixel-area coverage for the axis-aligned rectangles. For a rectangle `[x0,x1] x [y0,y1]`, coverage of pixel `(column,row)` is the product of its horizontal and vertical intersection lengths with `[column,column+1] x [row,row+1]`. This preserves subpixel changes. Ball color takes precedence where shapes overlap; compute the rectangle-intersection coverage to combine colors exactly, then round to uint8. The terminal ball at center y=61 has bottom y=63 and remains in the image.

The diagnostic display may have overlays around or on a separate copy of the raw frame. Save raw inputs independently to prevent accidental leakage.

### Simulator API

Provide reset from a seed, reset from an explicit state for tests, step(action_id), render(), and an exact clone/state-copy method for diagnostic planning. After termination/truncation, require reset rather than recording more transitions. Return observation, terminated/truncated status, and an info object containing privileged labels/events. Learning and deployed planning must receive these through separate input/label paths.

## 5. Dataset and replay

### Split and collection policy

| Split | Episodes | Suggested seed IDs |
|---|---:|---|
| Train | 5,000 | 0 through 4,999 |
| Validation | 500 | 5,000 through 5,499 |
| Test | 500 | 5,500 through 5,999 |

After the two stay warm-up actions, choose each action independently and uniformly from the three IDs. Record until miss or truncation. Use separate random streams for initial conditions and random actions. Save actual initial states and executed actions so replay does not depend solely on a future RNG implementation.

A dataset episode with T actions has T+1 observations. Never split adjacent frames of one trajectory across training, validation, and test. Do not mix later test-controller trajectories back into training. Start with a small smoke dataset before producing the full split; keep smoke and full manifests distinct.

### Storage default

Use a manifest JSON and one compressed NPZ per episode. This is a default for simplicity, not a required storage research choice. Load with pickle disabled. Include a schema version and environment/configuration fingerprint. Write files atomically and make collection resumable by verifying existing metadata before skipping an episode.

Each episode contains:

| Field | Shape/type | Definition |
|---|---|---|
| `frames` | `[T+1,64,64,3]`, uint8 | Raw rendered observations |
| `actions` | `[T]`, int64 | a_i leads from frame i to frame i+1 |
| `states` | `[T+1,5]`, float64 | `(ball_x, ball_y, ball_vx, ball_vy, paddle_x)` |
| `timestamps` | `[T+1]`, float64 | Seconds at observation boundaries |
| `terminated` | `[T+1]`, bool | True only at a terminal observation |
| `truncated` | `[T+1]`, bool | True only at a nonterminal time-limit observation |
| `hit_counts` | `[T+1]`, integer | Cumulative paddle hits |
| `metadata_json` | UTF-8 JSON string | Initial state, seeds/streams, schema and config fingerprints |
| `events_json` | UTF-8 JSON string | Event type, transition index, fractional interval time, contact coordinates |

State labels describe the state at the observation boundary, after any within-interval reflections or termination. Velocities are signed world units per interval. Only selected numeric labels are training targets; events/hit counts are evaluation metadata and a possible later audio source.

Estimate storage from a small sample before full generation and report actual frame counts. RGB frames require 12,288 uncompressed bytes each; episode duration and compression determine final size. Stream data rather than requiring the complete dataset in RAM.

### Sampling

- Perception: sample training frames with a reproducible frame index.
- Memory: start with complete episodes, padded within a batch and masked. This preserves exact initial-memory semantics without a complicated truncated-history scheme.
- Predictor: sample valid source indices i >= 2, with K recorded future actions/targets available. For K=5, require `i+5 <= T`; the last target may be terminal. Never cross a reset or fabricate missing targets.
- Validation examples are fixed and separate from training sampling. Test episodes are reserved for final reporting.

The frozen encoder outputs 20,480 floats per frame. A float32 feature cache is 81,920 bytes/frame, larger than raw RGB. Do not automatically cache all features without a disk/memory estimate. For U training, compute frozen E in bounded frame batches. After U is frozen, it is useful to cache only `I'_i` (128 float32 values per frame) after replaying each complete episode; encode source/target frames as needed for P. Every cache must include matching dataset, E, U, tensor-ordering, and numerical-mode fingerprints. Recompute a stale cache.

## 6. Visual encoder E

### Learned local features

All convolutions have bias. Use GELU after each of the three strided convolutions.

| Layer | PyTorch-style operation | Output |
|---|---|---|
| Input | RGB | `[B,3,64,64]` |
| Conv 1 | Conv2d(3,16,3,stride=2,padding=1), GELU | `[B,16,32,32]` |
| Conv 2 | Conv2d(16,32,3,stride=2,padding=1), GELU | `[B,32,16,16]` |
| Conv 3 | Conv2d(32,64,3,stride=2,padding=1), GELU | `[B,64,8,8]` |
| Fine projection | Conv2d(32,64,1) applied to Conv 2 output | `[B,64,16,16]` |

Fine and coarse branches then become token arrays `[B,256,64]` and `[B,64,64]`. There is no discrete codebook and no global pooling of S.

### Position/scale implementation default

For each grid, learn row embeddings of width 32 and column embeddings of width 32. Concatenate the corresponding row/column embeddings to obtain a 64-value position embedding for each token. Use separate row/column tables for the 16 x 16 and 8 x 8 grids, plus two learned scale embeddings of width 64. Initialize these embedding parameters with normal standard deviation 0.02.

Add position and scale information once to the incoming branch tokens. They are subsequently part of S. U and P receive these features; they must not repeatedly add E's positional embeddings during an imagined rollout.

### One bidirectional cross-scale exchange

Use four attention heads, width 64, dropout 0, affine LayerNorm with epsilon 1e-5, and separate parameters for the two directions. Use pre-attention LayerNorm and a separate pre-MLP LayerNorm on each branch.

```text
F0 = projected_fine_tokens + fine_position + fine_scale
C0 = coarse_tokens        + coarse_position + coarse_scale

NF = LN_f_attention(F0)
NC = LN_c_attention(C0)

F1 = F0 + attention_f_from_c(query=NF, key=NC, value=NC)
C1 = C0 + attention_c_from_f(query=NC, key=NF, value=NF)

F  = F1 + MLP_f(LN_f_mlp(F1))
C  = C1 + MLP_c(LN_c_mlp(C1))
```

Both cross-attention updates use the incoming branch values F0/C0; do not update coarse using the already updated F1. Each attention includes learned Q/K/V and output projections. Each per-token MLP is Linear(64,128), GELU, Linear(128,64). Return F and C in the fixed ordering.

## 7. Memory updater U

U contains a memory-dependent attention read and one GRUCell. Its hidden size is 128. This is a nonlinear recurrent baseline, not a claim that GRUs use the structured SSM formulation of Mamba or S4.

Let `Z = concatenate(S.fine,S.coarse)` with shape `[B,320,64]`:

```text
Q = Linear(128,64,bias=True)(memory_before)     # one query
K = Linear(64,64,bias=True)(Z)
V = Linear(64,64,bias=True)(Z)
r = Linear(64,64,bias=True)(four_head_attention(Q,K,V))
memory_after = GRUCell(input_size=67, hidden_size=128)(
    concatenate(r, previous_action_onehot), memory_before
)
```

Use head dimension 16, attention scaling `1/sqrt(16)`, dropout 0, and no causal mask across spatial positions. The biased query projection permits a learned initial query when memory is zero. Do not introduce a second recurrent state in the attention read. The formula specifies the projections once; avoid accidentally adding another query projection by wrapping it in a differently configured attention layer.

The GRU and all U attention projections are trained together during memory training. Initialize/reset the activation state explicitly in the caller. Module weights are shared between real observation updates and every imagined branch.

## 8. Predictor P

P is deterministic and stateless between calls. It has its own parameters; no weights are shared with E or U.

### Input assembly

1. Take the 320 current observation tokens of width 64.
2. Split the 128 memory coordinates into two consecutive 64-value tokens. This is a reshape, not a learned compression to one token.
3. Project the candidate one-hot action with Linear(3,64) into one action token.
4. Add separate learned slot/role embeddings to the two memory tokens and the action token; initialize them with normal standard deviation 0.02. Observation tokens already contain E's positions/scales.
5. Concatenate in order `[fine, coarse, memory_0, memory_1, action]`, yielding `[B,323,64]`.

### Processing/output

Apply two distinct pre-LayerNorm Transformer blocks. Each block has four-head self-attention, dropout 0, and a residual per-token MLP 64 -> 128 -> 64 with GELU. Use full attention within this single-step input. All input tokens are current information or a candidate action; there are no actual future tokens requiring a temporal mask.

Apply a shared Linear(64,64) output projection to only the first 320 output tokens. Initialize this final projection's weight and bias to zero. The resulting delta is added to the original S:

```text
delta = output_projection(processed_tokens[:, :320, :])
predicted_tokens = original_S_tokens + delta
predicted_S = split_fine_and_coarse(predicted_tokens)
```

Do not emit updated memory tokens from P as the next internal memory. Reuse U to compute that memory. Do not add P's role embeddings to returned S or apply an extra latent normalization/clipping step. Zero initialization makes the initial predictor copy S; training must improve on this explicit baseline. At the very first backward pass, the zero output projection can prevent upstream P parameters from receiving a nonzero gradient; this is expected until the head changes.

## 9. Image decoder D and numeric readouts H/R

### Image decoder

Reshape the two scales back to channels-first grids. Upsample coarse from 8 x 8 to 16 x 16 using nearest neighbor, then concatenate with fine along channels.

| Operation | Output |
|---|---|
| Concatenate fine and resized coarse | `[B,128,16,16]` |
| Conv2d(128,64,3,padding=1), GELU | `[B,64,16,16]` |
| Nearest resize x2; Conv2d(64,32,3,padding=1), GELU | `[B,32,32,32]` |
| Nearest resize x2; Conv2d(32,16,3,padding=1), GELU | `[B,16,64,64]` |
| Conv2d(16,3,1), sigmoid | `[B,3,64,64]` |

All decoder convolutions use stride 1 and bias. D receives only the retained S; no raw-image or discarded-feature skip connections. Train D with E, then freeze it. Reuse the identical D for encoded real observations and predicted S.

### Numeric readouts

- H: Linear(20480,3) on flattened, ordered S. Predict normalized ball x/y and paddle x.
- R: Linear(128,5) on updated memory I' alone. Predict normalized ball x/y, ball vx/vy, and paddle x.

Both have bias and no final activation. H is trained with E and later also serves as the planner's frozen goal readout. R is trained with U and remains a diagnostic of internal memory. Do not feed true state labels into these heads as inputs.

There is no separate memory-to-image decoder initially. For visual debugging, R's estimates may be overlaid on a separate diagnostic rendering. Such a schematic is not a decoded photographic observation.

## 10. Staged training and losses

### Common implementation defaults

Use AdamW with learning rate 3e-4, betas (0.9,0.999), epsilon 1e-8, and weight decay 1e-4. Apply weight decay to matrix/kernel parameters, excluding biases and LayerNorm parameters. Clip trainable-parameter gradient norm to 1.0. Keep the learning rate constant initially.

| Stage | Default batch | Maximum optimizer updates |
|---|---|---:|
| Perception E/D/H | 128 frames | 10,000 |
| Memory U/R | 8 complete padded episodes | 10,000 |
| Predictor K=1 | 64 source windows | 10,000 |
| Predictor K=5 | 16 source windows | 10,000 |

These are bounded starting budgets, not duration or convergence promises. Profile a short run and report the observed rate. Reduce batch size if needed before changing the architecture. Use FP32 for the correctness baseline. Mixed precision can be added as a measured optimization after matching FP32 behavior; compute statistics and loss reductions in FP32. Consult the [PyTorch AMP documentation](https://docs.pytorch.org/docs/stable/amp.html) for the installed version rather than using deprecated API spellings.

Validate every 250 updates on fixed validation subsets, e.g. 4,096 frames for perception, 64 full episodes for U, and 1,024 valid windows per P horizon. Use all available examples if a smoke split is smaller. Save the lowest-validation-objective checkpoint and periodic resumable checkpoints. Optional early stopping after at least 1,000 updates and eight validations without a 0.1% relative objective improvement is an implementation default; preserve the hard maximum budget. A low training objective is not sufficient to claim the physical-accuracy targets passed.

Use reproducible seeds, log actual examples processed, and make validation deterministic. Report all target metrics on the held-out test set after selecting checkpoints using validation. Do not tune repeatedly against test results.

### Stage A: perception

Train E, D, and H together from scratch. U and P are not used.

```text
S = E(image)
image_hat = D(S)
position_hat = H(S)
position_target = (ball_x/64, ball_y/64, paddle_x/64)

L_image = mean((image_hat - image)^2)
L_position = mean((position_hat - position_target)^2)
L_A = L_image + L_position
```

Use means within both terms and initial coefficient 1 for each. Targets are fixed observations and simulator labels, not a second trainable representation. Inspect ball/paddle region reconstruction as well as global image MSE. Region masks are evaluation aids only in this baseline; do not silently introduce an object-weighted reconstruction loss.

Select and save E/D/H, then freeze all their weights. Record a fingerprint for their exact parameters, architecture, and tensor ordering.

### Stage B: memory

Train U and R; E/D/H remain frozen. Encode each real frame with E. Replay every episode in order using the previous actual action, with a zero action marker at frame 0. Initialize memory to zero at the episode start.

```text
memory = zeros(B,128)
for t in observation_indices:
    S_t = frozen_E(frame_t)
    prev_action = zero_marker if t == 0 else one_hot(action_(t-1))
    memory = U(memory, S_t, prev_action)
    estimate = R(memory)
    target = (ball_x/64, ball_y/64, ball_vx/6, ball_vy/6, paddle_x/64)
```

Use a mask for actual observations versus padding. Mask the two velocity coordinates at frames 0 and 1, then supervise all five coordinates from frame 2 onward. Average squared errors over valid supervised scalar entries. Do not update memory for padded frames. Backpropagate through the complete episode in the initial implementation; a maximum of 201 observations keeps the semantics simple. Compute frozen E outputs in bounded batches without retaining E's graph.

Use a valid stay action ID for padded action slots, then mask those slots; do not pass an invalid negative ID to one-hot conversion. Padded RGB/labels may be zero because they are excluded from state updates and the loss. A frozen-E frame batch size of 128 is an initial default independent of the number of episodes in a U batch.

If complete-episode training is a measured memory problem, implement truncated backpropagation while carrying the correct detached memory across contiguous chunks. Document that adjustment. Do not silently reset memory at arbitrary training frames with unknown motion.

Select and save U/R with the E fingerprint. Freeze U/R for predictor training. Replay each episode with the final frozen E/U to obtain correctly initialized I'_i; cached states must belong to this exact observer.

### Stage C: latent scale statistics

On the training split only, using the frozen E, compute a mean and population variance across frames for every fixed spatial/channel coordinate separately. Average those variances within each scale:

```text
v_fine   = mean_over_position_and_channel(Var_over_training_frames(S.fine))
v_coarse = mean_over_position_and_channel(Var_over_training_frames(S.coarse))
```

Use a streaming algorithm with float64 accumulation and save the resulting scalar variances, sample count, and encoder fingerprint. For loss division, floor each variance at 1e-6. Constant spatial embeddings should not be counted as across-frame variation. Do not compute this statistic by pooling spatial coordinates into the frame axis.

This changes only loss scaling. P, U, D, H, and saved S continue to use original encoder coordinates.

### Stage D: one-step P training

Train P only. For each source i >= 2, obtain S_i and correctly replayed/cached I'_i, then predict with recorded a_i. The target is frozen E(x_(i+1)).

Define the per-step loss:

```text
ell(pred,target) = 0.5 * (
    mean((pred.fine   - target.fine)^2)   / max(v_fine,   1e-6)
  + mean((pred.coarse - target.coarse)^2) / max(v_coarse, 1e-6)
)
```

Mean over positions and channels within each scale, then average the batch. Fine and coarse have equal weight despite their different numbers of positions. Frozen targets prevent a moving target space; these losses do not guarantee that every task-relevant feature is learned.

Track the explicit copy-S baseline on the same validation examples. Continue to five-step training after next-step predictions improve on copying, including moving-object readouts. If the stage reaches its budget without improvement, preserve its checkpoint and diagnose the specific failure; do not present a failed stage as successful pretraining.

### Stage E: five-step P training

Continue training the same P from the selected one-step checkpoint with K=5. For each source i, initialize from the real B_i once. Use recorded actions a_i through a_(i+4), then repeatedly apply P and U to the imagined state.

```text
S_hat = detach(S_i)
I_hat = detach(I_prime_i)
loss = 0

for j in range(5):
    action = one_hot(recorded_actions[i+j])
    S_hat = P(S_hat, I_hat, action)
    loss += ell(S_hat, detached_target_S[i+j+1]) / 5
    I_hat = U(I_hat, S_hat, action)

loss.backward()
optimizer_for_P.step()
```

Targets may be computed in a separate no-gradient pass. Do not feed any actual future S or real future I' into the imagined recurrence. Do not detach S_hat or I_hat between imagined steps.

**Critical autograd contract:** set U's parameters to `requires_grad_(False)` and exclude them from the optimizer, but retain the computation graph through U for its predicted inputs. Wrapping imagined U in `torch.no_grad()` would sever a path needed for earlier P outputs to learn from later errors. Frozen real-history initialization and target encodings can use no-gradient evaluation. `.eval()` and disabling parameter gradients are separate operations. This distinction is documented in [PyTorch autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html).

The P objective contains only the latent loss. Pixel errors, H position errors, and R state errors remain diagnostics. Do not add direct regression of arbitrary internal-memory coordinates. Loss changes are follow-up experiments, not unrecorded fixes to this baseline.

## 11. The first planner

### Candidate generation and imagined transitions

At every real decision step after warm-up, enumerate all five-action sequences: 3^5 = 243 candidates. Use the existing executable action IDs, not freely optimized latent vectors.

Copy B_i into a candidate batch. Apply five sequential P-then-U steps, batching candidates at each step. Model parameters and readouts are fixed; inference can disable gradients. Respect independent candidate memory even if broadcasting creates shared storage views. A naive implementation evaluates 1,215 candidate transitions; prefix sharing is optional later.

Use frozen H on predicted S at each step to estimate ball/paddle positions. Multiply H's position outputs by 64 to obtain world units. Do not decode pixels for every candidate and do not use simulator labels to score a learned-model candidate.

### Goal score

A predicted miss occurs at the first imagined step whose estimated ball y is >= 61. Include that step in the score, then stop accumulating that candidate. If batched computation continues for convenience, mask inactive candidates and preserve their recorded terminal prefix; post-terminal predictions cannot improve the score.

For every candidate, compute:

```text
miss_flag = 0 or 1
first_miss_step = integer 1..5, if a miss was predicted
evaluated_steps = first_miss_step if miss_flag else 5
C_align = mean over evaluated_steps of ((ball_x_hat - paddle_x_hat)/64)^2
movement_count = number of non-stay actions in the evaluated prefix
```

Minimize this lexicographic tuple:

```text
(
  miss_flag,
  miss_flag * (5 - first_miss_step),  # define as zero for no-miss candidates
  C_align,
  movement_count,
  sequence_tie_order
)
```

The tie order uses stay, then left, then right at each sequence position; this tie order differs intentionally from the numeric action-ID order. It is only for reproducibility. Among candidates that all miss, prefer the latest first miss before comparing alignment. Reject non-finite predictions as invalid candidates and report the issue; if all candidates are invalid, return an explicit planning failure rather than inventing a successful plan.

Select the best sequence, execute its first action only, assimilate the actual next frame, and plan again. Preserve the exact real-memory convention in section 2.

The score is a hand-written interception heuristic based on learned readouts. It is not a calibrated interception probability. A five-step search finds the best of these 243 candidates under this model and score, not an optimal real-world policy.

### Known limits to measure

- H is trained on real encoded features and may be inaccurate on predicted features.
- A terminal target is exactly y=61; H underestimating that position can cause a false-negative miss detection. Preserve the baseline threshold initially, log the effect, and evaluate any later margin or classifier as a separate change.
- Five steps may not cover a delayed consequence. Compare with a privileged simulator using the same horizon and score before blaming model capacity.
- Initial planning uses visible-ball observations. An occlusion extension needs a validated memory-based goal readout; H should not be assumed to recover hidden positions from an occluded current frame.

## 12. Paired-history diagnostic

Create 50 validation pairs and 100 test pairs, with disjoint seeds and centers c sampled in [26,38]. Each pair contains two histories with opposite horizontal velocity signs. This is a dedicated near-interception diagnostic distribution, separate from the ordinary high-ball initial-state test population.

For sign d in {-1,+1}, initialize the diagnostic ball at `(c - 12*d, 40)`, velocity `(6*d,3)`, and paddle center c. Execute two stay actions to obtain these three observations:

```text
time 0: ball (c - 12*d, 40), paddle c
time 1: ball (c -  6*d, 43), paddle c
time 2: ball (c,        46), paddle c
```

The last raw frame must be byte-identical within each pair. The two preceding frames disclose motion. Feed each history separately into a zero-initialized observer and start control after its third observation.

Why the first action matters:

- Contact is `(54-46)/3 = 8/3` intervals away.
- The ball moves 16 units horizontally before contact.
- Moving in the correct direction immediately moves the paddle 32/3 units, leaving offset 16/3, about 5.33, within the catch tolerance of 8.
- Staying for the first interval leaves only 20/3 units of corrective movement, producing offset 28/3, about 9.33, outside the tolerance.
- No side-wall reflection occurs before contact for the specified center range.

A fixed current-image-only first-action choice cannot solve both members. Include an exact simulator test for this construction before using it as evidence about learned memory. Record both first-action accuracy and eventual first-interception success.

## 13. Evaluation, diagnostics, and reporting

### Numerical targets

| Quantity | Initial target |
|---|---|
| H ball/paddle position MAE on actual S | Less than 1 world unit/pixel |
| R ball velocity MAE after warm-up | Less than 0.5 world units per interval |
| H ball/paddle position MAE on five-step predicted S | Less than 2 world units/pixels |
| First-interception success on ordinary held-out starts | At least 90% |
| First-interception success on paired-history tests | At least 90% |

Report each object/coordinate, each prediction horizon 1..5, and collision cases separately. Report R position estimates too, although the explicit position target above is for H. Include distributions or high-error examples so an average does not hide systematic failures. Give success counts and denominators, not just percentages; uncertainty intervals are useful for comparisons.

These are engineering targets. They are not assertions that the fixed architecture or five-step goal score can attain them without further iteration.

### Required baselines and controls

1. Copy-S prediction: output the initial S at every imagined step and advance copied U with the same actions. Compare latent/object errors using identical source windows.
2. Memory reset: replace history-derived memory with the state obtained by a fresh U assimilation of the current frame with a zero initial memory/start marker. This is a current-observation-only comparison; keep its definition explicit.
3. Current-frame velocity probe: train a separate small diagnostic linear head from frozen S to vx/vy using training labels. It does not join the deployed architecture. Evaluate on matched-image pairs to check that memory's advantage is history-dependent.
4. Random controller, with the same three-frame warm-up.
5. Current-frame tracker using the same frozen H: stay if abs(ball_x_hat - paddle_x_hat) <= 2, otherwise move toward ball_x_hat.
6. When needed for diagnosis, the same exhaustive planner with true simulator rollouts/positions. Label this privileged. It is an evaluation reference, never the deployed learned planner or a source of model inputs.

Initialize controllers from the same 500 held-out initial states and warm-up, then let each generate its own real trajectory. Record first hit before miss, total hits, episode length, failure cases, and decision latency. Ordinary starts and paired-history cases have separate result tables.

### Visual/debug outputs

For selected held-out rollouts, produce aligned panels at each time:

1. Actual raw frame.
2. `D(E(actual_frame))`, showing the reconstruction baseline.
3. `D(predicted_S)`, showing the imagined frame.

Place H positions, R velocities, executed/candidate action, horizon, and errors in accompanying overlays or text. Include both successful and failed cases. For memory readouts, distinguish estimates after real observations from estimates after imagined updates. Decoder realism alone does not establish correct state information.

Save PNG grids and short animations/videos plus machine-readable metrics. At minimum, save JSON/CSV metrics and local plots; no external dashboard account is required.

### Performance

Measure actual decision latency after warm-up, with proper GPU synchronization around timed work. Report median and p95, candidate batch size, precision, GPU model, and whether rendering/readouts are included. Simulation time is independent of wall-clock speed. Do not claim 20 Hz control without measuring it.

## 14. Verification that protects the design

Write focused tests for these concrete risks. They matter more than a broad coverage-percentage target.

### Environment/data tests

- Deterministic replay from saved initial state and actions reproduces states and frames.
- Left/stay/right displacement and boundary clipping are correct.
- Side, ceiling, and paddle reflections preserve the required speed components.
- A moving paddle's overlap is evaluated at contact time, including a case where endpoint-only collision testing gives the wrong result.
- A missed top-plane crossing cannot become a later side/underside catch or be processed repeatedly.
- Simultaneous boundary events and a ball initially on a wall do not loop or drift.
- A terminal ball stays visible, velocities become zero, and no reset is recorded as a next frame.
- T actions have T+1 correctly aligned observations/labels; terminated and truncated semantics are distinct.
- Warm-up actions, initial previous-action marker, and split membership are correct.
- The paired-history diagnostic has identical final frames and requires the correct first action, checked by simulator action enumeration.

### Model/rollout tests

- Tensor packing/unpacking preserves fine/coarse ordering and all values.
- E outputs the specified shapes; D reconstructs from only those outputs.
- Untrained P with a zero delta head copies S exactly within numeric tolerance.
- U is the same module/parameters in real and imagined updates.
- Candidate evaluation leaves the caller's actual S/memory unchanged.
- A two-/five-step rollout passes its own predicted S and updated memory to the following step.
- Changing only future target observations changes the loss but not the forward imagined predictions.
- During P training, E/U/D/H/R parameter values are unchanged after an optimizer step.
- A later loss has a differentiable path through imagined U back to an earlier predicted S. Use a nonzero predictor output head for this test, since zero initialization initially suppresses memory dependence.
- Batched exhaustive search agrees with a simple scalar candidate reference on selected deterministic cases.
- Save/load reconstructs the same model outputs and planning action in a fixed numeric mode.

Use CPU tests where practical. Add a small CUDA forward/backward smoke test for the actual machine. Do not run a full training experiment inside ordinary unit tests.

## 15. Repository, commands, and checkpoints

### Suggested layout

Keep a compact package, for example:

```text
world_model_prototype/
  README.md
  pyproject.toml
  configs/{smoke,baseline}.yaml
  src/world_model/
    env.py
    data.py
    types.py
    models.py
    rollout.py
    planner.py
    training.py
    evaluation.py
    cli.py
    __main__.py
  tests/
  docs/implementation_brief.md
```

Split a module further only when it becomes difficult to read. Store generated data, caches, checkpoints, and reports outside tracked source files, with clear configurable paths. Add standard ignore rules. Use a small versioned configuration schema and save the resolved configuration in every run.

### Command contract to implement

These are proposed CLI commands to build, not commands already available. Provide help, clear errors, reproducible seeds, explicit output paths, and resume support.

```bash
python -m world_model doctor
python -m world_model generate --config configs/smoke.yaml --output data/smoke
python -m world_model verify-data --data data/smoke
python -m world_model test-history-cases --output runs/history_checks
python -m world_model train-perception --config configs/smoke.yaml --data data/smoke --run runs/smoke/perception
python -m world_model train-memory --config configs/smoke.yaml --data data/smoke --perception runs/smoke/perception/best.pt --run runs/smoke/memory
python -m world_model train-predictor --config configs/smoke.yaml --data data/smoke --perception runs/smoke/perception/best.pt --memory runs/smoke/memory/best.pt --horizon 1 --run runs/smoke/predictor_1
python -m world_model train-predictor --config configs/smoke.yaml --data data/smoke --perception runs/smoke/perception/best.pt --memory runs/smoke/memory/best.pt --initialize-from runs/smoke/predictor_1/best.pt --horizon 5 --run runs/smoke/predictor_5
python -m world_model evaluate --config configs/smoke.yaml --data data/smoke --perception runs/smoke/perception/best.pt --memory runs/smoke/memory/best.pt --predictor runs/smoke/predictor_5/best.pt --output runs/smoke/evaluation
python -m world_model demo --perception runs/smoke/perception/best.pt --memory runs/smoke/memory/best.pt --predictor runs/smoke/predictor_5/best.pt --output runs/smoke/demo
```

The smoke configuration can use 20/4/4 episodes and 20 optimizer updates per stage, with small batches. It tests execution, shapes, gradients, serialization, and report generation; it does not claim trained performance. It may explicitly bypass the one-step quality gate to exercise the five-step code, and must label that result untrained/smoke.

Implement `run-all --config configs/baseline.yaml --data data/baseline --run runs/baseline` as a convenience to collect/resume data, execute the accepted stages, and evaluate, with stage checkpoints and documented quality-gate behavior. Its default full counts and budgets are those in this brief. A full run must not silently bypass the one-step improvement check; on failure, retain diagnostic artifacts and report the failed stage.

### Checkpoint contract

Every checkpoint records stage, global update, actual model state dictionaries, optimizer state, seeds/RNG states needed for resume, resolved configuration, package versions, dataset manifest fingerprint, tensor/action schema version, training metrics, and best-validation criterion. Record sampler state or enough information to resume deterministically where supported.

- Perception checkpoint: E, D, H.
- Memory checkpoint: U, R, and exact perception dependency fingerprint.
- Predictor checkpoint: P, horizon, exact E/U dependency fingerprints, and the two latent-variance scalars/statistics metadata.
- Final inference bundle: all necessary model weights/configuration/statistics in one loadable package; no fragile absolute paths to another machine.

A memory or predictor checkpoint must refuse incompatible encoder ordering/weights. Do not reuse a state cache just because tensor dimensions match. Save `best.pt` and `last.pt` atomically. Keep interrupted work resumable without overwriting unrelated runs.

## 16. Implementation milestones and completion report

Implement in this order:

1. Project environment, configuration, action/tensor contracts, and CPU simulator.
2. Deterministic rendering, recorder, replay validation, and paired-history checks.
3. E/D/H and perception training/debug images.
4. U/R and memory training/history diagnostics.
5. P, fixed scale statistics, one-step then five-step training, and gradient/freeze tests.
6. Exhaustive planner, real observation loop, baseline controllers, and evaluation reports.
7. Full configuration, resumable runs, final inference bundle, and documented commands.

Before full training, make the simulator and full smoke pipeline pass. Before claiming learning success, run held-out evaluation and show real measurements. Before changing architecture, identify whether failure is caused by data/timing, perception, memory, prediction, goal score/horizon, or implementation.

The final report should state what was implemented; exact commands run; CPU/GPU and package versions; actual dataset size; checkpoint locations; achieved metrics versus targets; baseline comparisons; representative visual successes/failures; decision latency; and any remaining limitation. Separate software completion from empirical success. If a stage did not converge within its budget, say so and leave a reproducible checkpoint/report rather than masking the result.

## 17. Later research, after the initial path works

The following are future experiments, not initial implementation requirements: collision audio and audiovisual fusion; brief visual occlusions; a memory-based goal readout for hidden objects; text goals/instruction grounding; other encoder architectures; structured SSM replacements for U; stochastic/uncertainty-aware prediction; longer-horizon or more efficient planning; inferred latent actions; self-supervised observer training without simulator labels; and additional modality decoders.

Preserve the distinction between visible observation features S, persistent internal memory I, candidate executable actions, predicted targets, and desired goals when adding any of these. A new modality need not share raw tokenization or force all information through text.

## 18. Sources and scope

The exact combined prototype is an engineering hypothesis, not a published architecture with established performance. These sources support individual ideas:

- [Cho et al., Learning Phrase Representations using RNN Encoder-Decoder](https://arxiv.org/abs/1406.1078): GRU origin; not evidence for our exact memory size/interface.
- [World Models](https://worldmodels.github.io/): random rollout collection and staged visual/recurrent training; its recurrent model differs from our separate U/P design.
- [DINO-WM](https://arxiv.org/html/2411.04983v2): frozen visual features, action-conditioned latent squared-error prediction, optional image decoding, and model predictive control. Its history representation, decoder training, goal, and search differ from this prototype.
- [PlaNet](https://arxiv.org/abs/1811.04551): probabilistic latent dynamics and a multi-step latent objective; not our exact frozen-U loss.
- [Deconvolution and Checkerboard Artifacts](https://distill.pub/2016/deconv-checkerboard/): resize followed by convolution as an image-generation building block.
- [PyTorch scaled dot-product attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html): implementation reference for Q/K/V attention. Explicitly set attention dropout to zero in this baseline and use noncausal spatial attention; use the installed version's supported API.

Implement the contract above first. Keep all later changes traceable to an observed problem or a stated research question.
