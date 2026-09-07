# Multimodal world model — working research notes

Updated 7 September 2026 · Revision 0.12

**Architecture status: fixed by Alex.** The module connections and state semantics below are the agreed architecture. The initial visual encoder, GRU state updater, attention predictor, partly supervised observer training recipe, image decoder, predictor loss/rollout recipe, and first planner/goal evaluator are accepted. The exact environment, data collection, and numerical evaluation targets are also accepted. A self-contained implementation handoff is provided in world_model_codex_implementation_brief.md; its explicitly labeled implementation defaults fill gaps without changing this architecture. Later modalities and action interfaces remain extensions. Suggestions marked as recommendations are not additional accepted commitments. No implementation or experiment has yet validated the complete system.

## Current decision log

**Handoff status:** Alex requested a detailed description for Codex to implement on his computer. This session produces the implementation brief only; no simulator, dataset, model training, or local installation has been performed. Treat the handoff as the current implementation contract and this file as the decision/research history.

**Accepted — first environment:** a small 2D world with a moving ball and a horizontally movable paddle. Observe RGB images at fixed simulation steps. The executable actions are left, stay, and right. The goal is to intercept the ball. Record observations and executed actions for training, and simulator positions/velocities as training targets and evaluation labels under the accepted observer recipe below. Brief occlusion is a later memory diagnostic. Collision audio is a later modality extension, not part of the initial working path.

**Accepted — development approach:** work through decisions one at a time, use a simple working baseline, and change it in response to an observed limitation. Keep the agreed E/U/P architecture and copied-memory rollout.

**Accepted — visual encoder baseline:**

| Operation | Output shape, excluding batch |
|---|---|
| RGB input | 64 x 64 x 3 |
| 3 x 3 convolution, stride 2, padding 1, GELU | 32 x 32 x 16 |
| 3 x 3 convolution, stride 2, padding 1, GELU | 16 x 16 x 32 |
| 3 x 3 convolution, stride 2, padding 1, GELU | 8 x 8 x 64 |
| 1 x 1 projection of the 16 x 16 features | 16 x 16 x 64 |
| One residual cross-scale attention update in each direction | Fine: 16 x 16 x 64; coarse: 8 x 8 x 64 |

Use 2D position and scale information, four attention heads, pre-attention LayerNorm, and a small per-position residual MLP (hidden width 128, GELU) on each branch. Compute both cross-scale attention updates from the incoming branch features. Retain both resulting scales as S_i: 256 fine feature vectors and 64 coarse feature vectors, each of width 64. These are continuous learned features; no discrete codebook or single pooled vector is required. The persistent temporal state remains in U.

These dimensions are accepted as modest starting values rather than research-established optima. Select ball/paddle rendering sizes and data variation that allow the spatial resolution to be tested meaningfully. Check held-out ball/paddle position readouts and reconstructions, including small spatial shifts; do not rely on global image error alone, which could be dominated by background. Training objectives are specified in the accepted observer recipe below; this module has not been implemented.

**Accepted — state updater U:**

Use one GRU cell with a 128-value hidden state as the first nonlinear recurrent state update. This is a gated recurrent baseline; it does not assert that a GRU has the structured SSM formulation used by S4/Mamba. The broad state-estimation role and public U interface remain fixed, and a structured SSM implementation can be considered later only in response to a concrete limitation or research question.

Read the current S_i using one learned attention query obtained by projecting I_i from 128 to 64 dimensions. Keys and values come from all 320 current encoder features, preserving their existing spatial/scale information. Use a four-head attention read with a 64-value output r_i. A biased query projection permits a learned initial query when memory is zero. This is an observation-to-memory interface inside U; the fine/coarse encoder outputs remain available directly to P.

For this three-action environment, represent left, stay, and right with a three-component one-hot vector u(a). Use the all-zero vector only as a start marker when there is no previous action. A standalone learned action embedding or decoder is unnecessary for this initial interface. The same action meaning must be used by U and P. The more general action-adapter alternatives later in this notebook remain future options; they do not override this accepted initial interface.

    r_i  = Attention(query(I_i), keys(S_i), values(S_i))   # 64 values
    I'_i = GRUCell(concat(r_i, u(a_(i-1))), I_i)          # 67 inputs, 128 memory values

The attention projections and GRU weights are learned during observer training. Initialize I to zeros for a fresh episode. Carry memory across real frames without resetting it; clone it for candidate planning branches. In imagination, exactly the same U operation receives the copied current memory, predicted S_(i+1), and candidate a_i. No new raw frame is needed for that hypothetical update. Fixed-duration simulation steps require no extra elapsed-time input in the initial implementation.

Validation after training: use paired histories with different ball velocities but the same final image; read out velocity/direction from I'_i on held-out histories and compare with a current-frame-only baseline. Test brief ball occlusion, and check that resetting memory removes history-dependent information. These are learning/evaluation criteria, not claims that gating alone guarantees useful memory. The observer objectives and freeze order are accepted below; exact optimization settings remain to be specified before training.

Reference: [Cho et al., Learning Phrase Representations using RNN Encoder-Decoder](https://arxiv.org/abs/1406.1078), which introduced the gated recurrent unit. The 128-value memory and attention interface are our small-prototype choices, not results established by that paper.

**Accepted — predictor P:**

Use a small attention network to predict a residual change to the current observation features. It has no additional persistent memory: I'_i remains the persistent context supplied by U.

| Input or operation | Initial specification |
|---|---|
| Current observation features | 256 fine + 64 coarse vectors, each width 64 |
| Updated memory | Split the 128 values into two vectors of width 64, preserving all coordinates |
| Candidate action | Project the 3-component one-hot action to one vector of width 64 |
| Combined sequence | 323 vectors, each width 64 |
| Processing | Two pre-LayerNorm attention blocks; four heads; per-vector residual MLP with hidden width 128 and GELU |
| Output head | Shared linear 64-to-64 projection on the 320 observation positions |
| Predicted output | Add the predicted changes to the original S_i and restore the fine/coarse grid shapes |

Keep the observation positions and scale identities. Give the two memory inputs and action input distinct learned role/slot identifiers. Input-role embeddings belong to P's internal processing; do not add them again to the returned S during repeated rollouts. Split the memory vector into two inputs instead of first forcing it through a single 64-dimensional bottleneck.

All attention inputs are available at the current decision time or are supplied candidate controls. Full attention within this single-step predictor is causal with respect to environment time; there are no actual future observation inputs to mask. Future features are predicted jointly, not one spatial location at a time. The memory/action positions provide context but are not emitted as predicted observation features.

    Delta(S_i) = f_theta(S_i, I'_i, u(a_i))
    hat(S_(i+1)) = S_i + Delta(S_i)

The addition is separately shape-matched for the fine and coarse arrays. It represents learned changes in feature values, not a claim that physical actions are additive latent displacements. Initialize the final delta projection to zero so the initial model copies the current features; train it to improve on that explicit no-change baseline. The two attention blocks and output projections have their own parameters; they do not share weights with E or U.

This is a deterministic first predictor for the controlled toy environment. It does not supply a calibrated distribution over ambiguous futures. Use sufficient observation history when evaluating deterministic motion, and retain unknown-history/occlusion failures as evidence for possible later uncertainty modeling.

After prediction, the already accepted U advances the copied memory:

    hat(I'_(i+1)) = U(I'_i, hat(S_(i+1)), a_i)

Checks after training: predict paddle effects for actions with different actual consequences; distinguish future ball motion for identical current images with different preceding motion; outperform copying S_i on held-out moving trajectories; and run several complete P-then-U steps without actual future observations as inputs. Compare object/motion readouts as well as latent error so unchanged background does not dominate the result.

Reference: [DINO-WM](https://arxiv.org/html/2411.04983v2), section 3.1.2, uses an action-conditioned attention predictor over visual latent features. Its history representation differs from ours. Our two-block size, memory inputs, multiscale output, and residual head are prototype choices rather than results established by that paper. The image decoder and predictor training recipe are accepted in their own decision entries below.

**Accepted — observer training:**

For this controlled prototype, use simulator positions and velocities as supervised training targets as well as evaluation labels. Alex accepted this expansion of the earlier evaluation-only use of simulator state. These labels never enter E, U, or P as inputs. The first experiment is intended to establish a partly supervised working system; learning equally useful representations without simulator labels remains a later research step.

Train the observer in two sub-stages, then freeze it before training P:

| Stage | Trainable parameters | Inputs and targets |
|---|---|---|
| Visual perception | E, image decoder D, position readout H | Input current RGB image; reconstruct that image from S and read out ball x/y and paddle x |
| Temporal state estimation | U, state readout R; E frozen | Input encoded observations in temporal order and previously executed actions; read out current ball x/y, ball vx/vy, and paddle x from I' alone |
| Subsequent prediction | P; E and U frozen | Predict next S conditioned on S, I', and candidate action; use the accepted scale-normalized latent MSE and one-then-five-step rollout |

Visual objective:

    L_E = mean_squared_error(D(E(x_i)), x_i)
          + mean_squared_error(H(E(x_i)), normalized_position_i)

Use RGB values in [0, 1], position targets normalized by arena dimensions, and a mean within each loss term. Start with unit coefficients as a baseline, and inspect the component errors independently. The position term makes ball/paddle location an explicit training target even when most pixels are static background. It does not guarantee that reconstruction preserves each object: evaluate object-region reconstruction errors and small spatial shifts separately. Any simulator object masks used for these region metrics are evaluation aids, not additional encoder inputs. D receives only the retained S features, without raw-image or discarded-feature skips. Use a simple linear H over the flattened, spatially ordered S as the initial position readout. The accepted image-decoder layers are specified below.

Temporal objective:

    I'_i = U(I_i, E(x_i), a_(i-1))
    z_i = (ball_x, ball_y, ball_vx, ball_vy, paddle_x)_i
    L_U = mean_squared_error(R(I'_i), normalized_z_i)

Use a linear 128-to-5 R as the initial state readout. Normalize velocities by a fixed simulator speed scale specified with the dynamics. R gets I' alone, so the loss must be served through the recurrent memory rather than a direct S-to-readout bypass. Gradients flow through the unrolled U, including its attention read, but do not update E in this stage. Initialize memory at episode starts and preserve history within each training sequence. If batching clips, provide preceding observations as a warm-up rather than pretending an arbitrary clip begins with known hidden motion. Mask velocity supervision at episode start until enough observations make velocity identifiable; align frame, action, position, and velocity timestamps, including collision conventions. Future observations and simulator state are loss targets or labels only, never observer inputs.

Use varied initial positions and velocities, together with left/stay/right action sequences. Split by complete trajectories and independent initializations, not neighboring frames. First train on visible trajectories. Add the already planned short-occlusion curriculum only after basic state estimation works, then repeat memory validation before freezing the observer for a new predictor-training run.

The recurrent task estimates the current state after assimilating observations. It is not a second future predictor, and it does not require that the 128 internal coordinates equal the five simulator coordinates. H, R, and D can remain available as debugging readouts. Once frozen-observer predictor training starts, these diagnostic heads must not silently change the observer weights. Freezing U freezes parameters, not the evolving memory activations; the same U still updates copied memory during imagined rollouts.

Observer acceptance checks: held-out image reconstructions retain the small objects; H locates them accurately; R distinguishes velocities for paired histories ending in identical current images; a current-frame-only baseline and a reset-memory baseline lose that velocity advantage. Test short occlusions separately once introduced. Low average reconstruction error alone is insufficient. Final suitability for planning still depends on the later P-then-U rollout and closed-loop tests; current-state readout accuracy does not guarantee robust imagined updates.

Reference: [World Models](https://worldmodels.github.io/), Car Racing procedure, provides precedent for training visual reconstruction before a recurrent model. Its recurrent model directly predicts future latents, unlike our separate U and P. The supervised state readouts and exact two-sub-stage observer recipe above are our accepted prototype choices, not a result established by that paper.

**Accepted — image decoder D:**

Use a small deterministic convolutional decoder that consumes the two retained scales in S and returns one RGB frame. Upsample with nearest-neighbor interpolation, then learn local filters with ordinary convolutions.

| Operation | Output shape, excluding batch |
|---|---|
| Upsample coarse features from 8 x 8 to 16 x 16 | 16 x 16 x 64 |
| Concatenate these with fine features along the channel dimension | 16 x 16 x 128 |
| 3 x 3 convolution, 128 to 64 channels, GELU | 16 x 16 x 64 |
| Upsample to 32 x 32; 3 x 3 convolution, 64 to 32 channels, GELU | 32 x 32 x 32 |
| Upsample to 64 x 64; 3 x 3 convolution, 32 to 16 channels, GELU | 64 x 64 x 16 |
| 1 x 1 convolution, 16 to 3 channels, sigmoid | 64 x 64 x 3 |

Use stride 1 and padding 1 for the 3 x 3 convolutions, and stride 1 with no padding for the final 1 x 1 convolution. All resize operations are fixed 2x nearest-neighbor upsampling. The sigmoid matches the accepted [0, 1] RGB targets. The exact widths and interpolation choice are baseline settings to validate on our small rendered objects.

D receives only the retained fine/coarse features, in their original spatial order. Its parameters are separate from E. During visual training, the already accepted reconstruction loss updates both E and D, while the position loss updates E and H. Once visual training finishes, freeze D together with E and H; train U and R in the next stage as agreed. Later decoding predicted features is diagnostic and does not introduce a pixel loss into P training by default.

Reuse the same D for D(S_i), the reconstruction of an actual encoded observation, and D(hat(S_(i+k))), the visualization of an imagined observation. For a recorded future step, compare the actual RGB image, D(E(actual RGB)), and D(predicted S). This exposes the reconstruction baseline alongside prediction errors; it does not guarantee a unique attribution of every error, particularly for predicted features outside the training distribution. Check ball/paddle visibility, position and shape, small motions, and wall boundaries. Keep the numeric H and R readouts alongside images. The accepted R supplies an initial way to inspect memory I'; no separate memory-to-image decoder is selected yet.

Pixel decoding is not part of the P-then-U transition and need not run while scoring every candidate plan. The accepted predictor loss and rollout training horizon are specified below.

Reference: [Odena et al., Deconvolution and Checkerboard Artifacts](https://distill.pub/2016/deconv-checkerboard/) studies upsampling followed by convolution as a way to reduce characteristic checkerboard artifacts. It supports the general operation, not the optimality of our particular multiscale decoder or a guarantee of artifact-free reconstructions.

**Accepted — predictor loss and rollout training:**

Train P with a normalized latent mean-squared error, first for one future step and then for five successive imagined steps. Keep E, U, D, H, and R weights frozen throughout this stage. The parameters of P, including its own action projection and role embeddings, are the trainable parameters. Freezing the observer keeps the target representation stable; this is not a claim that the chosen observer is sufficient until rollout and planning tests succeed.

For each training example, process a real observation/action history through E and U up to decision time i. Detach and copy the resulting starting state (S_i, I'_i). Encode recorded future frames with the same frozen E to obtain target features S_(i+1) through S_(i+K). These future features are targets only; they never enter the imagined branch. Use the recorded executable action sequence a_i through a_(i+K-1) to condition that branch. Each example stays within one episode, with sufficient preceding history to infer velocity; do not cross an episode reset or supervise nonexistent post-terminal frames.

Define fixed scalar feature variances for the two scales using only the training split after E is frozen:

    v_l = mean over spatial positions p and channels c:
              Var over training frames [S_l[p,c]]
    d_l(A, B) = mean_squared_error(A, B) / max(v_l, 1e-6)
    l is fine or coarse

Compute variance across frames at each fixed spatial/channel coordinate before averaging, so a static positional offset does not count as temporal feature variation. The variance floor is a numerical starting value, not a tuned optimum. Store the two variances with the frozen encoder checkpoint and recompute them only if that encoder changes. This scaling is used only in the loss calculation: P, U, and D continue to exchange the original S coordinates. Mean within each scale and weight fine/coarse equally, so the fine grid's fourfold number of spatial positions does not automatically give it fourfold weight. This scaling does not by itself solve object/background imbalance.

At predicted step k:

    ell_k = 0.5 * [d_fine(hat(S_(i+k))_fine, S_(i+k)_fine)
                   + d_coarse(hat(S_(i+k))_coarse, S_(i+k)_coarse)]
    L_P = (1 / K) * sum from k=1 to K of ell_k

Initially use K=1. Once held-out next-step predictions improve on copying S_i, including moving-object diagnostics, continue training the same P with K=5. Five is a modest initial training horizon, not an established optimum or a commitment to the planner's horizon. Average all five step losses equally, including the first; do not score only the endpoint.

The imagined branch starts at hat(S_i)=S_i and hat(I'_i)=I'_i, then repeats exactly the accepted transition for j=0 through K-1:

    hat(S_(i+j+1))  = P(hat(S_(i+j)), hat(I'_(i+j)), a_(i+j))
    hat(I'_(i+j+1)) = U(hat(I'_(i+j)), hat(S_(i+j+1)), a_(i+j))

After initialization, each input S and I' on this branch comes from the preceding imagined step. In particular, replacing predicted features or memory with their actual-future counterparts would not be this multi-step training objective. Later losses backpropagate through the unrolled P operations and frozen U computations to earlier P outputs. Keep U's parameters frozen while retaining derivatives with respect to its imagined inputs; disabling all gradient tracking around imagined U or detaching between imagined steps would lose those paths. Computing the real-history initialization and fixed target encodings without gradient tracking is appropriate.

The initial predictor objective contains only the latent loss above. D, H, and R provide diagnostic images and physical-quantity readouts; their pixel/position/velocity errors are not additional predictor training losses at this point. Do not regress arbitrary I coordinates as an extra loss. If the measured small-object errors remain poor despite low latent error, that is a concrete reason to revisit this minimal objective in a later iteration.

Evaluate on held-out trajectories and report errors separately for steps 1 through 5, each scale, ball/paddle positions, and memory-derived velocity. Compare against a copy-S rollout that outputs the initial S at every step and advances a copied U under the same actions. Include action sequences with different actual effects, paired histories ending in identical images but different velocities, collisions, and the memory-reset diagnostic. Use the shared image decoder to compare actual frames, reconstructions of actual encoded frames, and imagined frames. These readouts are evidence about rollout quality, not a guarantee that all predicted features are on the real-feature distribution.

Shorter or longer useful planning horizons depend on the eventual simulation time step, motion speeds, and measured prediction accuracy. Validate any chosen planning horizon explicitly. The next design decision is the planner's goal score and discrete action-sequence search; optimizer settings, dataset quantity, exact dynamics, and timing still need concrete implementation values.

References: [DINO-WM](https://arxiv.org/html/2411.04983v2), section 3.1.2, trains action-conditioned prediction with a latent squared-error target from a frozen encoder. Its teacher-forced history architecture differs from our free-running P-then-U branch. [PlaNet](https://arxiv.org/abs/1811.04551) introduces a multi-step latent objective, latent overshooting, for a probabilistic world model. These support the respective general ideas; our equal-scale variance normalization, frozen-U gradient flow, and one-then-five-step recipe are accepted prototype choices rather than results established by those papers.

**Accepted — first planner and goal evaluator:**

Use receding-horizon model predictive control with exhaustive discrete action-sequence search. Start with a planning horizon of five fixed simulation steps, matching the longest initially trained rollout. The action alphabet remains left/stay/right: enumerate all 3^5 = 243 sequences. The planner is an ordinary search procedure; no additional policy, reward, or value network is introduced. All model and readout weights are fixed during planning.

At each real decision time, initialize every candidate branch from its own copy of B_i=(S_i,I'_i). For each of its five candidate actions, run exactly P followed by U, carrying the predicted S and updated copied memory to the next step. The action projection remains internal to P; U receives the same action's one-hot representation. Candidate sequences specify executable controls, so no latent-action decoder is required. Evaluation can be batched across candidates; five dependent time steps remain sequential. A naive implementation evaluates 243 x 5 candidate transitions. Sharing common prefixes is an optional later optimization, not needed to define the baseline. No real-time performance is claimed before measurement.

For this prototype, explicitly extend the accepted position readout H from its training/debugging role to the planner's goal evaluator. H stays frozen. At every imagined step, H(predicted S) estimates ball x/y and paddle x. Convert the normalized outputs back to consistent arena units as needed. The goal G is a hand-written interception objective using those estimates and fixed arena geometry, not an encoded desired full image or access to the simulator's hidden state. The actual simulator positions, future frames, and collision outcomes are unavailable to candidate evaluation; they remain training labels or external evaluation ground truth.

Rank sequences as follows:

1. Prefer sequences with no predicted miss within the horizon. A miss is detected when the estimated ball center passes the environment's fixed loss line below the paddle. The exact loss line must agree with the simulator's terminal and rendering convention in the upcoming environment specification.
2. If every candidate predicts a miss, prefer the one whose first predicted miss occurs latest.
3. Among the remaining candidates, minimize the mean squared horizontal ball/paddle center offset along the imagined sequence:

       C_align = mean over evaluated steps k:
                     ((predicted_ball_x_k - predicted_paddle_x_k) / arena_width)^2

4. Break equal-score ties by fewer non-stay actions, then a fixed lexicographic action order (stay, left, right) for reproducibility.

For a candidate with a predicted miss, score only up to and including its first missed step and treat the branch as terminal. Do not use post-terminal model hallucinations to improve the score or require training data beyond a terminal observation. A precise equivalent ordering is the lexicographic tuple (miss_flag, miss_flag * (5 - first_miss_step), C_align, non_stay_count, sequence_order), with the second component zero for no-miss candidates. Count non-stay actions over the evaluated prefix. Sequence order makes ties deterministic, not physically preferable.

Horizontal alignment is a deliberately simple shaping objective that supplies direction even when the ball has not yet reached the paddle. It is not identical to maximizing interception probability, and this first deterministic planner does not provide calibrated success probabilities. The goal score should be assessed against real interception success. The first planner assumes the visible-ball phase of the prototype; any later occluded-observation planning needs a validated memory-based goal readout rather than assuming H can recover hidden positions from a current image.

Select the best complete sequence under this model-based score, execute only its first action in the real environment, and observe the actual next frame. Carry real memory I_(i+1):=I'_i and update it with E(x_(i+1)) and the action just executed. Then run planning again from the corrected real state. Never replace real memory with a candidate's imagined memory. This is search over actions, not an update to the model parameters.

All 243 possibilities are considered only for the chosen five-step horizon and learned-model score; this is not a guarantee of optimal control in the real environment. Validate that five steps cover enough physical time to be useful under the chosen frame/action interval and speeds. Longer approaches or delayed consequences may require a longer validated model rollout and a different search budget later.

Initial evaluation: actual interception rate on held-out initial conditions/action histories, predicted versus realized object motion, and decision latency. Compare with random controls and a current-frame tracking controller that moves toward the currently estimated ball x without forecasting. For diagnosis, the same search/score can be evaluated offline with true simulator rollouts; label that as a privileged reference, not the deployed planner. This helps distinguish a weak short-horizon objective from learned-model prediction errors. Include approaching balls with equal current images but different motion histories and wall-bounce cases. Keep true held-out test episodes separate from settings chosen on validation data.

The next decision is a concrete environment and data specification: action duration/frame interval, paddle and ball sizes/speeds, collisions, visible terminal observations, initial-state sampling, dataset quantity and trajectory splits, and numerical success criteria. No code or training run has yet been started.

Reference: [DINO-WM](https://arxiv.org/html/2411.04983v2), section 3.2, uses model predictive control with a latent world model. Its search uses CEM and its goal is image-feature matching. Our exhaustive three-action search, five-step horizon, and explicit interception score are accepted prototype choices, not results established by that paper.

**Accepted — exact environment, data, and evaluation:**

Use a deterministic arcade-style environment with continuous positions and fixed-duration controls. These numerical values are engineering starting points, not published optimal settings.

| Quantity | Accepted initial value |
|---|---|
| Arena and observation | [0,64] x [0,64] world units; 64 x 64 RGB pixels; y increases downward |
| Fixed simulation interval | 1/20 second; one observation and one action per interval |
| Ball | Axis-aligned 4 x 4 square, described as the ball; half-size 2 |
| Paddle | 12 x 3 rectangle; horizontal center p_x; top y=56, bottom y=59 |
| Paddle center bounds | 6 <= p_x <= 58 |
| Paddle controls | Left/stay/right command -4/0/+4 world units per full interval; constant velocity within an interval, clamped at arena bounds |
| Ball horizontal velocity | Uniform choice from {-6,-4,-2,+2,+4,+6} world units per interval |
| Initial ball vertical velocity | Uniform choice from {+2,+3} world units per interval |
| Ball wall reflections | Ball-center x=2 and x=62; top boundary at center y=2 |
| Paddle interception plane | A descending ball center crosses y=54; catch if horizontal ball and paddle intervals overlap then |
| Loss line | Ball center reaches y=61 without a paddle reflection |
| Maximum trajectory length | 200 action intervals (10 simulated seconds); a time limit is a truncation, not a miss |

The square ball keeps the first collision and rasterization rules simple while still testing learned visual features. Render a white ball and cyan paddle on a black background with fixed colors. Use deterministic pixel-area coverage for rectangle edges so fractional positions remain observable. Do not encode velocity, actions, seeds, labels, or time in colors or overlays. Raw labels never enter the image input. Spatial units coincide with pixel widths, but object centers remain continuous rather than rounded before rendering.

The game uses top-face paddle interception only: at the crossing of ball-center y=54, a catch occurs when abs(ball_x - paddle_x) <= 8. On a catch, reverse only vertical velocity, record a hit, and continue the episode. Ignore paddle side/underside contacts; there is no gravity, spin, or speed change. Wall/top contacts reverse the corresponding velocity component. Resolve crossings within the fixed interval, including the paddle position at the contact time and any wall reflections, instead of testing endpoint overlap alone. This prevents a fast ball from skipping a collision.

On a miss, stop at the first loss-line crossing, set an absorbing terminal state, and freeze ball and paddle for the remainder of that interval. Set terminal velocities to zero. Render the terminal frame with ball center y=61, so its bottom is y=63 and the object is still visible. Return the terminal observation at the ordinary interval boundary; do not reset within the recorded trajectory. The planner's estimated miss predicate for these proposed units is ball_y >= 61. A catch is not terminal, so post-catch frames are available for P/U training and imagined rollouts. Do not generate training targets after an actual terminal frame or across a time-limit reset.

Sample initial ball x uniformly in [8,56], ball y in [8,16], and paddle x in [16,48], independently of the discrete velocity choices. Begin every episode with zero memory and no previous action. Observe the initial frame, execute two stay actions, and assimilate the two subsequent frames before learned control begins. This supplies three observations for motion estimation. Apply the same warm-up to the control baselines. For U training, mask initial velocity supervision before the third observation; report wall/contact cases separately because history and timestamp conventions matter. Positions are normalized by 64 and both velocity coordinates by 6 world units per interval, consistent with the accepted supervised observer loss.

These high initial ball positions leave at least (54-16)/3 = 12.67 intervals before the first possible paddle contact, including the two warm-up intervals. The initial range therefore avoids an immediate catch before motion can be observed. After warm-up, the fastest descent still gives the paddle more than 10 intervals to act. The five-step planner looks ahead 0.25 simulated seconds and can move the paddle up to 20 units; its actual success over longer approaches remains an experiment. Simulation time is fixed independently of wall-clock execution speed; measure decision latency rather than assuming the 243-candidate search already meets a 50 ms deadline.

Collect random-action trajectories after the initial two stay actions, choosing each of the three actions independently with equal probability. Use 5,000 training episodes, 500 validation episodes, and 500 test episodes, with disjoint seed sets and complete trajectories confined to their assigned split. An episode contains up to 201 observations and 200 actions, but actual frame counts depend on when misses occur. Store the actual initial state and actions for deterministic replay; use separate initialization and action random streams so policy comparisons preserve the same initial conditions. Model-selection and normalization statistics use only the appropriate training/validation data, not the held-out test set.

Each episode records RGB observations x_0 through x_T, actions a_0 through a_(T-1), simulation timestamps, ball/paddle positions, signed ball velocities at each observation, collision events/hit counts, and separate terminated/truncated flags. The record order is observe x_i, execute a_i for one interval, observe x_(i+1). State labels describe the actual state at that observation after any within-interval collision. Model inputs remain images plus executed/candidate actions; position/velocity labels supply the already accepted observer losses and evaluation, and collision events support evaluation or the later audio extension.

For end-to-end control evaluation, initialize each policy on the same 500 test initial states, with the same warm-up, and allow its own actions to determine its actual trajectory. Report first-interception success as the headline measure and hits before miss/truncation as a secondary measure. Compare the learned planner with random controls and the accepted current-frame tracker using the same frozen H. Give the tracker a half-action deadband of 2 units: stay if abs(estimated_ball_x - estimated_paddle_x) <= 2, otherwise move toward the estimated ball x. This makes the baseline definition reproducible.

Add a separate paired-history diagnostic, reported separately from the ordinary initial-state test population. Use 50 validation pairs and 100 test pairs with balanced left/right motion. In each pair, the last image has ball center (c,46), paddle center c, and c in [26,38]; ball velocity is either (-6,+3) or (+6,+3). The preceding two observed ball heights are 40 and 43, with horizontal positions chosen from that direction of motion and the paddle held at c. Thus the final raw image is identical within a pair, while the visible histories differ. These are explicitly near-interception diagnostic starts rather than the ordinary high-ball initial distribution.

At that diagnostic decision, contact is 8/3 intervals away and horizontal ball displacement at contact is 16 units. Immediately moving toward the correct future side gives a contact offset of 16 - 4*(8/3) = 5.33, within the catch tolerance of 8. Staying on the first interval leaves only 5/3 intervals of corrective movement, giving an offset of 16 - 4*(5/3) = 9.33, outside the tolerance. The first action must therefore use motion history; a deterministic current-image-only choice cannot succeed on both members of a pair. There is no side-wall reflection before contact for the specified c range. These arithmetic checks validate the diagnostic construction, not the neural model's performance.

Accepted numerical targets for the first trained version: observed ball/paddle position MAE below 1 unit, ball velocity MAE below 0.5 units per interval after motion warm-up, and five-step predicted ball/paddle position MAE below 2 units. Report each object/coordinate and each rollout step separately, plus collision errors; pooled background or easy-case averages are insufficient. Aim for at least 90% first-interception success on ordinary held-out starts and at least 90% on the paired-history diagnostic, with comparison to both control baselines. These are engineering targets to test, not expected results established by calculation or publications. Inspect the true-simulator version of the same planner/score if the control target fails, as already agreed, to separate horizon/score limitations from learned-model errors.

Random rollout collection before representation/dynamics training has precedent in [World Models](https://worldmodels.github.io/), Car Racing procedure. The exact geometry, timing, episode counts, paired-history construction, and numerical thresholds here are our accepted prototype choices. No simulator implementation, dataset, or trained model has been produced in this design step.

After this specification, the next useful work is implementation: first the deterministic simulator, recorder, and diagnostic cases; then E/D/H, U/R, P, and finally the planner in the accepted training order. Framework and optimizer settings can be chosen as routine implementation defaults and recorded with the runs.

## 1. Fixed architecture and notation

The observation encoder E includes modality-specific learned local processing, multiresolution feature hierarchies, and attention across modalities and scales. A state-space module U maintains persistent internal memory. A distinct predictor P forecasts the next observation features. Imagined memory is advanced by reusing U on those predicted features.

    Real observation update:
        S_i  = E(x_i)
        I'_i = U(I_i, S_i, a_(i-1))

    One imagined step, on a copied state:
        hat(S_(i+1))  = P(S_i, I'_i, a_i)
        hat(I'_(i+1)) = U(I'_i, hat(S_(i+1)), a_i)

    Complete planning state:
        B_i = (S_i, I'_i)

    Planner:
        Plan(B_i, goal G, callable rollout P+U) -> action sequence

The next imagined step uses both newly imagined components. Candidate plans never overwrite real memory. Execute a selected real action through an environment/executor, then encode the actual next observation and update real memory. Elapsed time and action duration must be included wherever the selected time convention requires them.

| Symbol | Meaning |
|---|---|
| x_i | Actual multimodal observations available at decision step i |
| S_i | Encoded/fused observation features, retaining the selected modality and scale structure |
| I_i | Memory entering step i, before assimilating S_i |
| I'_i | Memory after assimilating S_i |
| a_(i-1) | Previously executed action that led toward the current observation |
| a_i | Candidate or selected next action |
| T_i | Encoded actual target; T_i = S_(i+1) for next-step prediction in the same representation |
| T'_i | Predicted target features; T'_i = hat(S_(i+1)) in that case |
| G | Desired goal/objective, distinct from a predicted outcome |

S and I have different roles and may have different dimensions and structures. Adapters connect them; matching dimensions does not establish matching semantics. Preserve both S_i and I'_i as predictor context. The older suggestion that a predictor could directly replace the entire P+U rollout is outside this fixed baseline.

The fixed timing convention carries I_(i+1) := I'_i after execution and incorporates a_i when the next actual observation arrives. Do not silently reinterpret I_i as a state already advanced under the same previous action and advance that interval again.

## 2. User preferences retained

- Modality-neutral integration: images, video, audio, text, and extensible later inputs; no mandatory language hub.
- Learned local filters and hierarchical/multiresolution perception before broad latent interaction.
- Cross-modal and cross-scale attention in learned feature space.
- Persistent recurrence/state-space processing in internal latent memory.
- Staged development and training: establish the observer first, then train prediction for planning.
- Debug outputs should make observed representations, accumulated memory, and predicted targets inspectable.
- Keep the agreed core architecture while deciding its concrete implementations.

## 3. Remaining decision register

| Area | Decisions still required | Proposed starting direction |
|---|---|---|
| First environment and data | Exact geometry, 20 Hz timing, dynamics, episode counts/splits and numerical targets accepted | 64 x 64 arcade world; 5,000/500/500 random-action episodes; three-observation warm-up; separate paired-history diagnostics |
| Encoder E | Compact visual baseline and observer training recipe accepted; optimization settings and later modality extensions remain | Three strided convolutions, 16 x 16 and 8 x 8 features of width 64, bidirectional cross-scale attention |
| Memory U | GRU/attention baseline and state-supervision recipe accepted; optimization settings remain | Attention read to 64 values, previous action as one-hot, 128-value GRU memory |
| Predictor P | Two-block attention/residual baseline and latent loss/rollout recipe accepted | Predict next fine/coarse S from S, I' and candidate action, then reuse U for imagined memory |
| Action interface | Prototype one-hot left/stay/right at fixed steps accepted; wider action schemas and instruction grounding remain future extensions | U takes the previous one-hot action; P projects the candidate one-hot action; no action decoder |
| Training | Observer objectives, freeze order, and P loss/one-then-five-step rollout accepted; exact optimizer settings remain | Frozen-observer prediction with fixed-variance-scaled MSE, equal scale/step weights, and gradients through copied U computations |
| Debug readouts | Image decoder and numeric position/memory readouts accepted; later output modalities remain extensions | Upsample/concatenate both S scales, then resize and convolve to RGB; same decoder for real and predicted S; separate numeric heads from S and I' |
| Planner | Exhaustive five-step MPC and goal score accepted; fixed timing implies 0.25 simulated seconds; control performance and latency remain to validate | Enumerate 243 executable sequences through P/U; score predicted misses and alignment through frozen H; execute first action and replan |

These rows are specifications inside or around the agreed modules; they do not change their connections. No exact layer counts or loss weights are claimed to be optimal before selecting a first task and budget.

## 4. Executable actions versus action embeddings

For the accepted three-action prototype, U takes one-hot actions directly and P learns its own projection, as specified in the decision log. No separate shared action encoder or action decoder is required. The remainder of this section records possible later generalizations.

General action-adapter convention:

    executable action a_i -> learned action encoder A -> z_i = A(a_i)

The action inputs of P and U use z internally. The public action remains a_i, which the executor understands. This adapter is part of implementing the existing action inputs, not another dynamics module. A may share a semantic embedding across P/U with separate input projections. Freeze a shared A with U once their meaning is established, or explicitly coordinate any later adapter training so that U never receives a silently changed action coordinate system.

Typical implementations:

- Discrete commands: an ID and embedding lookup.
- Continuous controls: normalized parameters and an MLP/projection; preserve units, ranges, and duration in the action specification.
- Structured commands: operation/type embedding plus encoded arguments, including entity references and coordinate frames where needed.

Record executed actions with transitions. Such data can train the action adapter through the observer's action-dependent objectives and subsequent prediction objectives, subject to the chosen freeze policy. Observation-only videos with unknown actions are not examples of intentional no-op commands.

The learned embedding z does not need the same coordinates as S or I. It is not assumed to be a displacement that can be added to either. Retain a alongside z, so an action decoder and invertible A are unnecessary in the initial design. Prediction-sensitive learning should preserve distinctions among actions whose effects differ.

### Instructions

- A concrete command such as 'rotate joint 2 by 15 degrees' maps to a validated structured action a and then A(a).
- A goal such as 'put the cup on the shelf' maps to G and any constraints; the planner chooses the action sequence.
- A report such as 'the person opened the door' is observational evidence, with provenance, rather than an instruction to execute that action.

An unrestricted language embedding is not automatically an action code. Use a command schema/parser for explicit commands or learn a mapping from command/action demonstrations. Task-level language also needs grounding to a goal evaluator. Text describing intended effects can be ambiguous and does not by itself identify one executable motor command.

### If we later choose inferred/free latent actions

An inferred latent action, z=L(S_i,S_(i+1)), explains a transition without requiring recorded action labels. This is different from embedding a known executable action. [Genie](https://arxiv.org/abs/2402.15391) is a precedent for learning controllable latent action codes from video; it does not automatically map our future commands to robot/tool controls.

Planning freely over latent codes requires a grounded executor/decoder, potentially state-dependent:

    a_k = D_action(z_k, B_k)

Constrain codes to realizable, learned actions. Evaluate the action that will actually execute: when using A as the predictor's action interface, decode the candidate and re-encode A(a_k) during rollout, or enforce an equivalent consistency constraint. Otherwise a planner can exploit latent codes that look favorable to P but decode to different effects or no valid action.

No such free-latent-action optimization is yet selected. Latent state-space planning already occurs in the fixed architecture even if action search uses ordinary executable command coordinates.

## 5. Debugging the two latent spaces

Add trained output readouts without changing E, U, P, or the rollout connections.

| Input | Readout | Meaning |
|---|---|---|
| Observed S_i | D_m(S_i, q) | Reconstruct requested modality m from current encoded evidence |
| Predicted T'_i = hat(S_(i+1)) | The same D_m(T'_i, q) | Render predicted next evidence with the same output interpretation |
| Internal I'_i | R_m(I'_i, q), possibly an adapter into D_m | Read what accumulated memory supports |
| S_i or I'_i | Small supervised probe | Read measurable quantities and test information accessibility |

The query q can specify modality, time interval, spatial region, or viewpoint. A memory state may represent many observations and events; a query makes the requested output unambiguous. I_i and I'_i share the internal-state interface, so their readouts can also be compared before and after an observation update, with timing made explicit.

Potential output implementations:

- Image: spatial latent queries plus learned upsampling/pixel decoder.
- Video: decode a sequence of predicted observation features, with temporal decoding where necessary for coherence.
- Audio: waveform/codec decoder, or spectral prediction followed by waveform synthesis; preserve required timing, phase/channel cues or represent their ambiguity.
- Text: a trained latent-to-text adapter and text decoder. Prefer structured factual outputs for the initial diagnostic interface.
- Numeric sensors/structured data: heads producing defined values and units; uncertainty where meaningful.

Exact reproduction of arbitrary raw data is not guaranteed by a compressed feature representation. A decoder cannot recover discarded evidence uniquely. A powerful generative decoder may supply plausible details from its own learned prior, so attractive images or fluent verbal reports alone are not proof of correct world state.

Recommended diagnostic views for a recorded transition:

1. Actual target raw data x_(i+1).
2. D_m(E(x_(i+1)), q), showing the observation representation plus decoder's reconstruction limit.
3. D_m(hat(S_(i+1)), q), showing predicted features through the same decoder.
4. R_m(I'_i, q), showing memory readout for a defined query.
5. Independent held-out probes for position, velocity, identity, events, remembered properties, and uncertainty where ground truth is available.

Train diagnostic decoders/readouts on detached features after the relevant representation is stable. A reconstruction head used to train E/U is a separate, explicit training choice. Do not silently let debugging losses alter the fixed observer during predictor training.

[DINO-WM](https://arxiv.org/html/2411.04983v2), section 3.1.3, trains an optional image decoder independently from its frozen visual features. Pixel decoding is not required during latent prediction or planning. This is the closest precedent for the requested debugging role.

## 6. Training objectives within the fixed stages

There is no single universal loss that specifies all desired behavior. Decide which information each stage must preserve and test it.

### Observer stage: E and U

E needs local information-retention objectives and cross-modal grounding, including missing-modality examples and complementary clues. U additionally needs objectives that reward useful history: delayed recall, currently hidden but previously observed information, or supervised state quantities in a controlled environment. Reconstructing only the visible current input can let U ignore memory.

Use a complete non-collapsing representation-learning recipe. Merely matching two jointly trained latent vectors permits constant representations. Fixed informative targets, suitable reconstruction/supervision, or explicit regularization can prevent that particular solution; stop-gradient or EMA alone is not a universal guarantee. [VICReg](https://arxiv.org/abs/2105.04906) is one established variance/covariance regularization example, not a mandated extra architecture component.

Specify the action-adapter training and freeze boundary before this stage finishes. If U is expected to use executed controls, train that input on suitable action-linked data. Entirely passive pretraining leaves action grounding as a distinct later training requirement.

### Predictor stage: stable E and U

Train P with recorded source histories and action sequences. Actual future encoder outputs provide targets. Start with one-step prediction, then train the same P+U rollout used in planning:

    L_P = sum over future steps k, modalities m, and scales l:
          w_(k,m,l) * d(hat(S_(i+k)^(m,l)), stopgrad(S_(i+k)^(m,l)))

Distances, feature normalization, scale/modality weights, and horizon weights remain to be selected. Mean-squared or Huber error can be a deterministic baseline in a suitable environment; ambiguous futures may require a probabilistic output and likelihood-based objective. Normalize so that numerical units and the sheer number of features do not accidentally determine modality importance.

Rollout uses predicted S as input to frozen U, exactly as fixed above. Frozen U parameters do not require detaching gradients through U's input/output computation: gradients can pass through that operation to train earlier P predictions. It is essential to assess how real-feature-trained U behaves on predicted features. Do not supply actual future S as an input to the imagined multi-step rollout.

Matching arbitrary internal memory coordinates is not automatically a meaningful objective. If memory-consistency supervision is added, validate that it improves retained information and future outcomes rather than just numerical agreement.

### Debug readout stage

Train D and R on fixed representations and corresponding raw/structured targets. Use appropriate per-modality reconstruction, sequence, or supervised losses, with held-out evaluation. Their training does not require reopening joint E/U/P training.

[V-JEPA 2](https://arxiv.org/html/2506.09985v1) demonstrates video representation pretraining followed by a frozen encoder and separately trained action-conditioned predictor. It does not establish the best loss for this additional recurrent memory architecture.

## 7. How the planner evaluates a sequence

For an executable sequence (a_i, ..., a_(i+H-1)):

    Copy:       (s, h) := (S_i, I'_i)
    For each action a in the candidate sequence:
        z := A(a)
        next_s := P(s, h, z)
        next_h := U(h, next_s, z)
        (s, h) := (next_s, next_h)
        accumulate task/action costs and constraint violations
    evaluate the final state against goal G

Here z is simply the embedded representation supplied through the agreed action inputs. State evolution remains P followed by U. Each candidate has its own memory copy; stochastic models can evaluate multiple outcome samples per action sequence.

A conceptual objective is:

    J = terminal_goal_cost(B_H, G)
        + sum_k running_cost(B_k, a_k, G)
        + constraint or uncertainty penalties when defined

Define a grounded task evaluator first. Euclidean distance between arbitrary multimodal latents is not automatically a valid goal metric. Image-goal feature distance is one possible tested special case; language goals need an evaluator grounded in actual outcomes. Uncertainty estimates and their penalties must be meaningful and validated, not arbitrary added scores.

For the current three-action prototype, the decision log accepts exhaustive five-step search and a frozen position readout for goal scoring. The generic action adapter in the pseudocode above is internal to each module in this prototype: P projects the candidate action and U takes it as one-hot. The following CEM discussion is a possible later continuous-control extension, not the accepted first planner.

For a future small continuous-control baseline, receding-horizon planning can use the Cross-Entropy Method (CEM). Sample valid action sequences, evaluate each through the complete rollout, retain the best, refit the sampling distribution, and repeat. Execute the first action and replan after the next real observation. Discrete controls require categorical proposals, enumeration, or another discrete search rather than a continuous Gaussian by default.

CEM changes candidate action values during planning, not E/U/P weights. Both [PlaNet](https://planetrl.github.io/) and [V-JEPA 2](https://arxiv.org/html/2506.09985v1) use CEM with latent world models and receding-horizon execution. The search algorithm can later change without changing this architecture.

Planner decisions still open: executable versus free latent search coordinates; horizon and time step; valid action ranges/durations; goal metric; optimization budget; optimizer; outcome uncertainty; and how to detect model errors that planning might exploit.

## 8. Recommended decision order

1. Specify the first environment, available modalities, executable action interface, time convention, and a measurable goal. These define training data and meaningful losses.
2. Fix tensor/interface shapes for S, I, and action features without treating all spaces as identical.
3. Specify E and U internals together with observer objectives and history-sensitive validation.
4. Specify P internals and one-/multi-step objectives; establish the observer and action-adapter freeze boundary.
5. Attach and validate observation, prediction, and memory readouts for debugging.
6. Implement a transparent planner/evaluator and measure goal achievement, prediction error, memory use, and compute.

This order keeps the high-level architecture fixed while turning open questions into concrete, testable implementation decisions.

## Archived v0.1 proposal and research references

The material below records earlier exploration. Its alternate state transitions, fixed modality token groups, 224-token prototype, Transformer-heavy frontends, and training preferences are not current commitments. The fixed architecture and current staged-training contract above take precedence.

---


# Multimodal encoder architecture — research and design v0.1

Prepared for Alex · 6 September 2026

Status: a research-backed architecture proposal, not an implemented or experimentally validated system. Published results below support individual components; their combination remains a hypothesis. The goal is a modality-neutral representation for prediction and planning, initially covering images, video, audio and text, with an interface for additional modalities.

**1. The design decision**

Use modality-specific input stems, a common token interface, and a shared fusion/state-update network. Represent the world with a set of latent tokens that can preserve both common context and local detail. Do not require every modality to collapse to the same pooled embedding or pass through language.

Three things must be distinguished:

| Object | Meaning | What it does not establish |
|---|---|---|
| Common interface | Different front ends produce vectors with compatible dimensions and metadata | Equal dimensions do not establish equal semantics |
| Aligned semantic projections | Related observations can be compared across modalities | Retrieval similarity does not establish retention of all useful details |
| Fused predictive state | Evidence over time is combined into a state used for future prediction | Its sufficiency must be tested on prediction and planning |

For example, video might reveal a ball's position, audio its material through an impact, and text an instruction about which ball matters. Preserve the first two as evidence, and treat the instruction as a goal. Changing the goal should not by itself change the inferred physical position.

No finite latent can preserve every input detail for all unspecified future tasks. Our measurable objective is better retention, accessibility and predictive usefulness at a stated token, compute and data budget.

**2. The publications most directly shaping the design**

| Publication | Verified contribution | Design implication and scope |
|---|---|---|
| [CoMM: What to align in multimodal contrastive learning?](https://arxiv.org/abs/2409.07402), ICLR 2025 | Studies redundant, modality-unique and synergistic information; uses multimodal fusion and contrastive learning over augmented multimodal representations | Evaluate information available only in one input and information requiring their combination. Its experiments do not establish a general physical world model |
| [Factorized Contrastive Learning: Going Beyond Multi-view Redundancy](https://arxiv.org/abs/2306.05268), NeurIPS 2023 | Explicitly addresses shared and unique information | Supports retaining complementary information. Factorization alone is not a solution to all synergistic reasoning |
| [ImageBind: One Embedding Space To Bind Them All](https://arxiv.org/abs/2305.05665), CVPR 2023 | Aligns images, text, audio, depth, thermal and IMU data using image-paired training rather than every possible pairing | Demonstrates cross-modal grounding without all-to-all paired datasets. Its image-anchored semantic space is not proof of a dynamics-sufficient state |
| [Perceiver IO](https://arxiv.org/abs/2107.14795), ICLR 2022 | Uses cross-attention into a latent array and queries for structured outputs; includes audiovisual autoencoding | A useful shared-workspace template. A fixed array is still a compression bottleneck; causal recurrence is an additional design choice |
| [Attention Bottlenecks for Multimodal Fusion](https://arxiv.org/abs/2107.00135), NeurIPS 2021 | Exchanges audiovisual information through a small group of bottleneck tokens | Supports efficient intermediate fusion while maintaining local streams. Fusion tokens are not automatically temporal memory |
| [MultiMAE](https://arxiv.org/abs/2204.01678), ECCV 2022 | Masks and predicts multiple image-related modalities; supports different available input combinations | Useful precedent for missing-input training. Its main scope is image-centric geometry and semantics |
| [4M](https://arxiv.org/abs/2312.06647), NeurIPS 2023, and [4M-21](https://arxiv.org/abs/2406.09406), 2024 | Modality-specific tokenization, a common Transformer, and randomized input/target subsets support many tasks and modalities | Adopt extensibility and subset training. Many of the 21 modalities are image-derived representations; this is not a demonstration of 21 independent live sensors. We need not inherit discrete quantization |
| [MJEPA](https://arxiv.org/abs/2606.25225), 2026 | Uses separate audio/video tokenization with a shared encoder; combines within-modality masked prediction with cross-modal prediction | Particularly relevant shared-core precedent. Naive weight sharing hurt its unimodal baselines; cross-modal objectives and joint encoding improved results. Scope: audiovisual representations, not text or demonstrated planning |
| [data2vec](https://arxiv.org/abs/2202.03555), ICML 2022 | Applies a common contextual teacher-target learning algorithm to speech, vision and language | Useful training mechanics across stems. The modalities were trained individually, so this does not establish an aligned joint space |
| [VL-JEPA](https://arxiv.org/abs/2512.10942), first released 2025 | Predicts text embeddings from visual/query context, with text decoding when required | Closely matches the optional text-readout idea. Its target is language semantics, not a complete modality-neutral physical state |

Pairwise alignment is not mathematically guaranteed to erase every private detail. The more precise concern is that alignment alone need not reward retaining a detail that is missing from the paired modality. Architecture capacity, data and additional objectives affect the result.

**3. Modality-specific front ends**

The logical interface should support both a pretrained reference implementation and a new jointly trained implementation. Do not make compatibility depend on choosing one pretrained model family forever.

| Input | Proposed front end | Information to preserve | References and cautions |
|---|---|---|---|
| Images | Patch embedding or convolutional stem; spatial Transformer blocks; optional registers | Patch positions, boundaries, small objects, pose and relative layout | [Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588) and [DINOv3](https://arxiv.org/abs/2508.10104). Use dense features, not only a classification token |
| Video | Share the visual spatial path; add temporal processing over timestamped frames or tubelets | Motion, event order, identity and elapsed time | [V-JEPA 2.1](https://arxiv.org/abs/2603.14482) improves dense representations through local and intermediate-layer supervision. A still image is a one-time observation; using a particular pretrained video tokenizer may require separate handling |
| Audio | Time-frequency or waveform stem; local/contextual blocks; retain time-resolved tokens | Events, phonetic content, pitch/prosody, timing and any task-relevant channel information | Compare [BEATs](https://arxiv.org/abs/2212.09058), [Audio-MAE](https://arxiv.org/abs/2207.06405), and [Audio-JEPA](https://arxiv.org/abs/2507.02915). Semantic event recognition is not evidence of waveform or phonetic fidelity |
| Text | Reversible tokenization; embeddings and contextual Transformer blocks; preserve token sequence | Negation, order, numbers, identifiers, entities and relations | [BERT](https://arxiv.org/abs/1810.04805) supplies a token-level baseline; [ByT5](https://arxiv.org/abs/2105.13626) motivates bytes for unusual strings and spelling sensitivity, at a sequence-length cost |
| Later sensors or structured inputs | Appropriate point, set, graph, numerical-series or document-layout stem | Coordinates, units, source identity, adjacency and uncertainty where available | Add a front end and grounding examples. A PDF may contain text, images and layout rather than require a new physical modality |

Registers are learned working tokens used inside a forward pass. Their activations are not persistent memory in the original method, and their output is normally discarded. They can improve local representations but are independent of how the world state is carried across time.

DINOv3 shows that local features can still degrade with registers present and uses a Gram-based objective to preserve spatial relationships. V-JEPA 2.1 supplies another reason to consider training objectives before stacking architecture components: its dense and intermediate-layer supervision changes the quality of the resulting features. Neither paper establishes that the same objective is automatically optimal for every modality.

For audio, a normalized mono mel-spectrogram has already discarded some waveform, loudness or channel information. If spatial hearing or faithful audio output becomes a goal, preserve or explicitly encode the necessary cues. Keep a waveform/codec route as a separate future option rather than assuming a semantic audio latent can reconstruct everything. [CLAP](https://arxiv.org/abs/2206.04769) is useful for audio-language semantic alignment; [wav2vec 2.0](https://arxiv.org/abs/2006.11477) is a speech-specialist reference if the general audio branch fails phonetic tests.

U-Net or feature-pyramid processing is an optional response to missing fine detail, not a required common backbone. Good reconstructions through encoder-decoder skip paths cannot demonstrate that the latent actually used for prediction contains the same information. [U-Net](https://arxiv.org/abs/1505.04597)

**4. A common token contract**

Each front end emits a variable-length set or sequence of token records:

| Field | Meaning |
|---|---|
| `features` | Learned continuous vector, projected to a common width |
| `modality`, `source_id` | Input type and actual sensor/document/source identity |
| `position` and its schema | Image coordinate, time-frequency location, sequence offset, 3D coordinate, or another meaningful address |
| `time_start`, `time_end` | Interval described by this token, when applicable |
| `available_at` | Earliest time the complete token could actually have been computed |
| `valid_mask` | Missing or invalid values; absence must remain distinguishable from a measured zero |
| `scale`, `units`, `calibration` | Only where meaningful and known; explicitly mark unavailable metadata |
| `role` | Observation, reported claim, instruction/goal, or hypothetical content |
| `encoder_version` | Identity of the feature coordinate system used to produce the token |

Use normalization and a learned projection per modality. Add or otherwise encode positional and type metadata without pretending that image coordinates, text offsets and frequencies are identical kinds of position.

Token width is a software/model interface choice, not proof of cross-modal meaning. The modalities must still learn correspondences from paired episodes, shared tasks, calibration or another source of grounding.

Maintain native time resolution through the front ends. Cross-attention can combine irregular streams without resampling every modality to the video frame rate. [MulT](https://aclanthology.org/P19-1656/) is a precedent for cross-modal attention over unaligned sequences; strict causal streaming is an additional requirement here.

Completed bidirectional chunks are valid inputs only after the whole chunk is available. This matters for video tubelets, audio windows/STFT lookahead and text encoders. Restricting only the final fusion attention cannot remove future information already encoded by a front end.

**5. Shared fusion and the latent we will actually evaluate**

Represent the predictive state as one token set with several addressable groups:

\[
B_t=[G_t;L_t^{visual};L_t^{audio};L_t^{text};\ldots].
\]

`G` supplies global integration capacity. The `L` groups retain local/detail capacity and appropriate addresses. They are not required to be statistically independent or exclusively unimodal after fusion. A visual detail token can incorporate audio evidence about the observed object. Nor do individual tokens have to represent discrete objects.

The state-update network cross-attends from its state queries to available observation tokens, then applies shared attention among all state groups. Gated residual updates provide a way to preserve prior evidence when the current observations add little. Whether this actually learns robust memory is an experimental question.

There are two related operations:

\[
B_t^- = F_\psi(B_{t-1},a_{t-1},\Delta t),\qquad
B_t = U_\phi(B_t^-,\{E_m(o_t^m)\}_{m\in\mathcal M_t}).
\]

`E` encodes available observations; `U` updates the estimated state; `F` advances the state under an action and elapsed time. The initial encoder study can use a small predictor as a diagnostic without committing to the final planning architecture.

During imagination, repeatedly apply `F` without future observations. Every state group needed by planning must be evolved or maintained by that transition. A cache of real observation patches cannot supply unobserved future detail. The planner and evaluation readouts must use the same full state that is available during imagined rollout.

Uncertainty should be represented and evaluated when evidence is insufficient. An arbitrary confidence scalar does not make a model calibrated; use probabilistic outcome/state prediction where needed and test its behavior under ambiguous or missing inputs.

Textual goals condition planning separately from factual state updates. Textual evidence may update a belief, but should retain its provenance as reported information. A hypothetical statement must not silently become an observation.

**6. A concrete small prototype configuration**

These are proposed starting hyperparameters, not values established as optimal by the cited papers:

| Component | Initial choice | First alternative |
|---|---|---|
| Common feature width | 384 | 256 or 512 if capacity/compute is limiting |
| Local processing | Two small modality-specific blocks after the input stems, or adapters over frozen reference encoders | More local depth only when detail probes justify it |
| Fusion | Two cross-attention update blocks and four shared self-attention blocks, six heads | A simple concatenation Transformer at a comparable budget |
| Global state entries | 32 | 16 or 64 |
| Detail entries | 64 per supported modality group; visual images/video share a group | 32 and 128 per group |
| Internal ViT registers | Four where the backbone is designed/trained for them | Comparable model without registers |
| Numeric latent | Continuous vectors | Discrete codes only if later compression/generation needs justify them |
| Additional modules | No required object grouping, U-Net decoder or 3D scene graph in v0.1 | Add one based on a demonstrated failure |

With visual, audio and text groups, the starting state has 224 tokens. This is an explicit compression hypothesis. Modality absence does not automatically erase its previous state entries; their evidence becomes older and uncertain as appropriate. Adding a modality can add another token group without changing the attention operator, but integration training is still necessary.

This configuration is not a memory/throughput guarantee for a particular GPU. Input lengths, selected reference backbones, batch size and training unroll length must be measured during implementation.

A new model can train the stems, fusion and predictor jointly from random initialization. A pretrained reference implementation helps distinguish weaknesses of the proposed fusion from the cost of learning basic perception. Compare both under clearly reported data and compute assumptions.

**7. Training the properties we want**

Use a small set of objectives, introducing additional terms only for identified deficiencies:

1. **Preserve within-modality information.** Predict masked local targets or stable teacher features. Text token/span recovery, acoustic detail targets and spatial readouts can diagnose particular failures. Auxiliary decoders can be training/inspection tools even if they are not used during planning.
2. **Learn cross-modal grounding.** Align or predict appropriate semantic projections and temporally corresponding events. Do not force every local feature to equal its counterpart in another modality. Cross-modal regression should not require an image to determine an inaudible speaker's timbre or random acoustic phase.
3. **Train fusion on complete and incomplete subsets.** Include visual-only, audio-only, text-only and genuinely joint examples. Preserve complementary evidence when present; express uncertainty when the required input is missing.
4. **Add predictive pressure early.** Train on future outcomes/features and action-labelled sequences where available. Ordinary video co-occurrence alone does not identify every action's causal effect.

An optional conceptual objective is

\[
\mathcal L=\lambda_{local}\mathcal L_{local}
+\lambda_{ground}\mathcal L_{ground}
+\lambda_{future}\mathcal L_{future}.
\]

This is an accounting framework, not a validated new loss recipe. Use a complete established self-supervised recipe for any trainable target branches, then measure changes. Stop-gradient or an EMA teacher alone is not a universal anti-collapse guarantee. Monitor feature diversity and external readouts; do not interpret low self-prediction error as sufficient evidence.

Normalize losses by modality and valid target count so a dense stream does not dominate solely because it produces more tokens. No modality is a mandatory semantic hub. Equal status does not mean equal weight when reliability or relevance differs.

Augmentations encode assumptions about what may be ignored. Lighting changes may be irrelevant in one task and signal a state in another. Audio gain, time stretching, rotations and reordered text should not be declared harmless without checking the target capabilities.

**8. Experiments that can reject this design**

Use a small controlled audiovisual environment with multiple moving objects, collisions, occlusions and materials. Generate synchronized audio and optional factual text, plus separate instructions. Record ground-truth state and actions for evaluation. Synthetic labels need not all be training targets. Include held-out real recordings later, since success on procedural audio/rendering may exploit generator shortcuts.

| Question | Test | Interpretation |
|---|---|---|
| Is information retained? | Compare expressive readouts for pose, velocity, identity, contact/material, pitch, event timing, numbers and negation | A failed small probe alone is not proof of information destruction |
| Is it accessible? | Linear/tiny readout accuracy and learning curves with limited labels | Measures how much downstream work the representation requires |
| Is fusion useful? | Redundant-input tasks, single-modality-only clues, and tasks requiring complementary inputs | Separates semantic matching from actual evidence combination |
| Does it preserve temporal state? | Occlusion and delayed cues with increasing gaps | Compare single-frame, short-history and recurrent variants |
| Does it respect causality? | Remove future chunks and enforce `available_at` at every front end | Exposes offline feature leakage |
| Does it handle missing evidence? | Missing, corrupted, delayed, irrelevant and contradictory inputs | Evaluate uncertainty and negative transfer against unimodal baselines |
| Is the actual state useful? | Multi-step outcomes and goal-reaching using only imagined state transitions | Prevents observation-only side paths from hiding an inadequate state |
| Is extension practical? | Add one new modality, such as depth or a contact sensor, and measure adaptation/data needs and old-task regression | Tests modularity rather than assuming it |

A concrete synergy task: two visually similar objects have different materials, identifiable through impact sound; visual position identifies the target object and its predicted collision. Vary layouts and independently randomize textures so audio cannot be ignored through a visual shortcut. Use a simple two-bit/XOR task only as a supplementary sanity check, not as the entire physical benchmark.

Run the first ablations in this order:

1. Pooled embedding versus token state; then two or three token budgets. Report differences in parameters/compute rather than claiming perfect control where it is absent.
2. Global semantic alignment alone versus alignment plus local retention and joint fusion objectives.
3. Per-observation representation versus short causal context versus recurrent state.
4. Frozen reference front ends versus learned/adapted front ends; then depth sharing versus separate local encoders.
5. Only afterward prioritize registers, multiple scales, object slots or geometry-specific equivariance.

Independently pretrained checkpoints are a practical comparison, not a causally isolated test of one architectural change. Likewise, a larger state may improve results because it retains more information rather than because its organization is better; report the budget explicitly.

**9. What remains deliberately open**

The common token contract, modality-neutral fusion and explicit predictive-state boundary are the proposed stable interfaces. Local backbone depth, state size, token allocation and the degree of weight sharing remain experimental choices.

Object structure is a promising later branch when grouping and interactions become a central question; [SlotFormer](https://arxiv.org/abs/2210.05861) is an appropriate reference. It is distinct from adding scratch registers or general memory slots.

Adding an unknown modality is possible architecturally, but understanding it requires grounding data or other constraints. Exact reproduction of arbitrary files is a different requirement from a compact world-model state; retain reversible source data separately when exactness is required, and be explicit about when a task relies on that external memory.

The first decision to validate is whether the shared token state preserves useful complementary detail and makes it easier to predict. The research contribution should be judged by retention, accessibility and held-out dynamics/planning at a stated budget, rather than by attractive embedding visualizations alone.
