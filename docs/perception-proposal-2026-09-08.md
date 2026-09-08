# PATH-WM encoder, decoder and experiment proposal

8 September 2026 · Technical design proposal · Prepared with a critical Claude review.

**Recommended direction:** retain spatial evidence in a replaceable encoder, keep temporal memory explicit, and attach decoders suited to their outputs. Start by comparing the existing deeper convolutional encoder with a frozen pretrained ViT using fresh, capable readouts. Then diagnose memory and choose the next experiment from the measured failure. A hybrid, conditioning, extra scales and registers are options with entry conditions.

This sheet consolidates the previous discussions. It specifies proposed work; it does not launch training, select a replacement architecture, or claim that the existing control targets have passed. The companion [draft experiment manifest](proposals/perception-program-2026-09-08.yaml) records budgets and decisions. It is documentation, not a runnable trainer configuration.

## 1. The first decision is a bounded package comparison

Compare **two frozen encoders, each with three fresh readout seeds**. Keep the observed images, sample order, decoder families and selection procedure matched. Preserve each encoder's native feature width; any decoder projection is explicit. Select checkpoints using PushT validation readiness, then report every output at that same checkpoint.

The question is whether a practical perception package exposes useful geometry while supporting other outputs at an acceptable cost. It does not isolate the causal effect of transformer architecture from pretraining. The initial recommendation covers P0 and P1 only. Afterwards, publish the result and select one next branch; the complete matrix below is not an automatic queue.

| Decision | Proposed starting point | Change it when |
|---|---|---|
| Encoder | Existing deeper CNN versus pinned pretrained DINOv2-S/14 | Required outputs or system cost distinguish the packages |
| Spatial hierarchy | Keep two CNN grids; use native ViT patch grid | A measured resolution or aggregation limit justifies another level |
| RGB decoder | Deterministic convolutional decoder reading all declared spatial levels | A required output still fails after a fair fit |
| Additional outputs | Independent spatial mask and geometry readouts | Sharing or variable queries have a demonstrated consumer |
| Memory | Separate recurrent state with explicit time and reset semantics | Readouts establish a temporal-state limitation |
| Conditioning | Goal/output context at the consumer; executed action in state update; candidate action in prediction | A matched experiment establishes a benefit from moving context earlier |

The design is asymmetric by intent: a ViT encoder may use convolutional decoders, and different outputs may use different decoders. Two, three or four scales are not measures of intelligence. Spatial sampling, processing depth and temporal abstraction are separate choices.

## 2. Existing results justify stronger tests, not a universal winner

These are completed measurements, not results of the proposed experiments. Exact values and populations remain in the linked reports and the existing experiment dashboard.

| Completed evidence | Interpretation for this proposal | What remains unproved |
|---|---|---|
| Deeper/on validation q: 1.396 / 1.412 / 1.422; shallow/on: 2.874 / 2.541 / 2.968 across paired seeds 7107–7109 | Added residual processing helped this training recipe substantially | All validation readiness gates still fail; more depth alone is not a demonstrated solution |
| Exchange helps the deeper stack and hurts the shallow stack | Keep the tested deeper/on reference; compare fusion only with trained controls | Cross-scale attention is not universally beneficial |
| DINO adapter foreground IoU about 0.583 versus always-foreground 0.324; custom variants about 0.31–0.34 | Pose and RGB do not exhaust representation quality; test native pretrained features with adequate readouts | This is foreground union coverage, not instance understanding or control |
| Frozen adapted encoder: COCO test MSE 0.272195 → 0.006105 after decoder refit; original warmup 0.006072 | Much of the reconstruction failure was recoverable through the decoder | The encoder is not proved unchanged or lossless |
| The same COCO decoder refit worsened PushT reconstruction by 8.72× | Evaluate retention on both domains; mixed training is the first control for domain interference | Task conditioning is not yet shown to fix the tradeoff |
| Deeper coarse-head normalized entropies 0.953 / 0.970 / 0.322 / 0.948 in the audit | A mean hides selective heads; inspect heads and perturbation effects | Attractive PCA colors or low entropy are not evidence of better prediction |

Sources: [encoder study](encoder-study-results-2026-09-08.md), [decoder recovery](decoder-recovery-results-2026-09-08.md), [visual audit](encoder-visual-audit-2026-09-08.md). The completed encoder study comprised 33 formal runs and 110,000 updates. No compatible new U/P/control training was triggered by that study. This proposal does not turn perception results into world-model results.

## 3. Design backward from the outputs and decisions

| Consumer | Evidence it needs | Appropriate output and evaluation |
|---|---|---|
| Manipulation | Location, orientation when defined, boundaries and contact | Coordinate-aware geometry and masks; configuration-held-out errors and action outcomes |
| Reconstruction or image output | Observed appearance, small marks, correct colors and layout | RGB reconstruction with per-domain errors and detail-region checks |
| Motion and occlusion | History, elapsed time, executed actions, uncertainty | Causal temporal state; velocity and persistence probes, multi-step predictions |
| Software interaction | Exact controls/text, layout, hidden interaction history | Adequate-resolution visual features plus optional available structured/text input; correct action outcomes |
| New downstream output | Evidence retained beyond the original task | Freeze the encoder, train a new output, measure data/compute needs and recheck prior outputs if shared weights change |

Position, orientation and extent should usually be **readouts**, not the definition of the entire latent. A visible bounding box can be computed from an instance mask; a foreground-union mask cannot tell us each object's extent. Hidden shape or metric 3D size needs further evidence. Text generation and clicking are action-generation problems; predicting what executing them changes belongs to the transition model.

Convolutions supply a strong local spatial bias; attention supplies content-dependent interactions. Both can learn spatial and semantic features. Neither supplies missing pixels or missing history. A finite compressed representation also cannot promise to preserve every unknown future property.

## 4. Architecture: spatial evidence, temporal state, typed outputs

<!-- architecture-diagram -->

The proposed contract is:

```text
z_t       = E(image_t)                         # spatial observation evidence
m_t       = U(m_(t-1), z_t, executed_action_(t-1), dt)
s_t       = (z_t, m_t)                        # current planning/belief state
z_hat_t+1 = P(s_t, candidate_action_t, dt)
m_hat_t+1 = U(m_t, z_hat_t+1, candidate_action_t, dt)
y         = D_output(z_at_target_time, optional aligned memory, output query)
```

These equations make timing explicit. The current implementation uses the corresponding E/U/P roles, with a fixed step convention rather than a new explicit `dt` input. The proposed general interface records the interval; making it a learned input is a separate change when intervals vary. Future decoding reads predicted spatial features and, if requested, predicted memory at the same future time. Actual future frames or labels never enter imagined branches.

### Encoder alternatives

| Package | Concrete feature computation | Status and constraint |
|---|---|---|
| E-CNN reference | RGB64 → two stride-2 convolutions → 16×16 shared stem. Fine projection to 64 channels; coarse stride-2 branch to 8×8×64. Two residual convolutional blocks per branch, learned spatial/scale positions, one simultaneous bidirectional cross-attention exchange, then per-token MLPs | Implemented deeper/on candidate. Coarse branch reads the shared stem before fine residual processing. No within-scale transformer stack |
| E-ViT alternative | Same RGB64 source, bicubic resize to 224 with antialiasing, no crop, ImageNet normalization; frozen pinned DINOv2-S/14 → 16×16×384 normalized patch features; exclude class token | Reuse existing source/weights. Do not repeat the old universal 384→64 adapter as the new comparison. A pooled 8×8 map is derived from the fine map, not another learned hierarchy stage |
| E-Hybrid conditional | Reference stem and residual processing, then two pre-norm within-scale transformer blocks per branch: width 64, four heads, MLP ratio 4, existing position and cross-scale exchange conventions | Test against the reference and extra-convolution control. Extra blocks run before cross-scale exchange; positions are added once. Keep the branch topology unchanged for this comparison |

Preserve legacy `ObservationLatent` checkpoints (256 fine + 64 coarse tokens, width 64) and `PlanningState` memory width 128. A new package declares a separate feature schema: each level's grid, channels, coordinates, valid image region, preprocessing, normalization, backbone identity and observation time. Decoder projections may reduce channels, but they do not silently redefine the stored representation or let old U/P weights consume incompatible features.

Start with available two-level computation. If source detail is lost, first compare **genuinely higher-resolution observations**; upsampling RGB64 to 224 cannot recover lost information. If information survives but spatial interaction is weak, test processing depth. A sequential pyramid in which coarse processing reads refined fine features is a later topology intervention, separate from adding attention blocks. Three/four levels require an explicit resolution/latency reason. A six-layer transformer sweep is not the first experiment.

### Decoder and readout specification for P1

Each output owns its weights. Convert each input grid to width 128 through its own LayerNorm/channel projection. For CNN, read both existing levels; for ViT, read its native fine grid and a declared 2×2 average-pooled grid. Resize coarse features to the fine grid and concatenate. The projections are trained decoder components, and their parameter counts are reported.

| Output | Proposed initial module | Training signal and meaning |
|---|---|---|
| RGB | Fused 256 channels → 3×3 convolution to 128 → one two-convolution residual block; nearest-neighbor upsample to 32×32 with 64 channels, then 64×64 with 32 channels; final 3-channel sigmoid. GELU and GroupNorm with 8 groups in hidden blocks | Pixel MSE on both COCO and PushT, scored separately as well as in the fixed training mixture |
| Foreground mask | Independent trunk of the same spatial family; final one-channel logit | BCE on available COCO union masks with crowd/invalid pixels ignored; report IoU, Dice and constant baselines |
| PushT geometry | Independent fused width-128 spatial trunk with two residual blocks; two location heatmaps and expected coordinates; a separate learned orientation pooling/readout with 128-unit MLP → two values | Existing normalized pose MSE initially; preserve prepared target order: pusher XY, simulator body-origin XY, sin θ, cos θ. Orientation is not obtained by subtracting the location predictions |

The orientation branch has its own spatial weighting, followed by a pooled feature MLP. Normalize its two output values for angle evaluation, handle near-zero norms explicitly, and retain raw-vector norm diagnostics. Supervise the original coordinate convention: simulator body origin is not necessarily the silhouette centroid. No new keypoint labels are required for P1. Width 128 and these block counts are engineering starting choices, subject to a tiny development feasibility check before the protocol is frozen.

Current D already reads both fine and coarse features and uses ordinary convolutions plus nearest-neighbor upsampling. “Adding both levels” is not a new intervention. New decoders need not mirror E, and mask/RGB output heads must not be mistaken for a policy. Shared query decoding is deferred until variable objects, regions or output types justify its cost.

## 5. Conditioning is feasible when its timing and purpose are explicit

| Context | Default destination | Conditional extension and fair control |
|---|---|---|
| Requested output type or object/region query | Output selector or decoder | Compare separate outputs, shared decoding, then FiLM/query decoding where sharing loses or a consumer needs queries |
| Goal/task | Planner or task readout | Compare a late context-aware adapter with conditioning before encoder compression; both receive the same context |
| Prior internal state | U; optionally a contextual encoder adapter | Identity-initialized FiLM or cross-attention from prior memory, while retaining a stable observation route; test ambiguous histories |
| Previous executed action and elapsed time | U/state estimation | Earlier conditioning is possible, but must beat equally informed late processing |
| Candidate future action | P or an explicit affordance/action-query branch | Keep the canonical observed state stable across candidate actions unless a separately specified action-conditioned representation is justified |
| Sensor/view identity | Input metadata, positions or normalization | Useful when observations have genuinely different acquisition geometry |

A small initial conditioning mechanism is FiLM: context produces channel scale/offset around an identity transformation. Cross-attention is an alternative for structured queries; do not combine both in the first test. A goal can direct attention to the requested button, but it must not cause the model to invent that the button is present. Test counterfactual queries, context-only baselines and unseen query/scene combinations.

Encoder conditioning can change which information survives compression; decoder conditioning only changes how available information is read. In the first bounded encoder-conditioning experiment, freeze the base observation route and train the contextual branch. If later joint training changes shared weights, repeat generic and prior-task retention tests. Merely using a `reconstruct` versus `predict` token cannot replace action-conditioned dynamics.

Direct decoder access to memory is useful only for an output that needs history or a belief about hidden state. Compare visual-only, memory-only and combined input with suitable trained controls. Standard image reconstruction should target visible pixels; hidden-object completion is a different supervised or probabilistic task. Evaluate a future decoder on the predicted-state inputs it will actually consume. Transfer from encoded-state training is a valid baseline; a measured deployment mismatch can motivate fitting on predicted or mixed states.

Skip connections are allowed from available observation features for current reconstruction. Future skips must come from predicted features or explicitly retained past appearance; include moving-object and stale-copy tests. They cannot pass the target future frame around P. ViT registers are within-frame computational tokens in the cited work; they are not persistent temporal memory. A register-trained checkpoint is a separate pretrained package, not a silent repair to the first comparator.

## 6. Experiment matrix: choose the next branch from its result

All budgets below are proposed ceilings. Each stage freezes its primary metric, arms, selected checkpoint rule and complete resource envelope before formal execution. Reaching a cap without the required evidence yields “inconclusive at this budget,” not permission for an automatic extension.

| Stage | Question and maximum comparison | Primary selection metric | Entry, exit and resulting decision |
|---|---|---|---|
| P0: contracts and development | Data/label audit, frozen-feature contracts, tiny forward/backward/profile and resume checks | Correctness and feasibility; no scientific winner | Enter first. Stop on unresolved alignment, leakage, incompatibility or infeasible budget; otherwise freeze P1 |
| P1: frozen packages | CNN and native ViT × 3 fresh readout seeds = 6 fits; 4,000 updates each | PushT validation q; earliest checkpoint on an exact tie | First formal comparison. If an arm passes required gates and has useful cost/retention, retain it for that consumer. If neither passes, preserve the reference and diagnose the failure |
| P2: memory diagnosis | Current-image-only, updated-memory-only, combined, and prior-memory-only probes × 3 seeds = 12 fits; 2,000 updates each | Validation velocity MSE for the existing Paddle state, reported in physical units | Recommended next cheap diagnostic using existing frozen E/U. Determine whether history information is accessible, lost in update, or mainly lost during rollout |
| P3: geometry objective/budget | Three paired deeper/on checkpoints, each continued +4,000 updates: unchanged objective versus tolerance-scaled pose objective = 6 fits | Existing validation q | Enter if geometry still fails and training/position scaling remain plausible limits. Keep original gates; stop architecture expansion if this already resolves the consumer's problem |
| P4: within-scale processing | Reference, two extra residual conv blocks/branch, two transformer blocks/branch × 3 seeds = 9 fits; 4,000 updates each | Validation q, with all other readouts reported | Enter only for a remaining spatial-representation question after cheaper diagnostics. Train all arms under the same recipe; do not compare their curves as though P1 trained its encoders |
| P5: decoder sharing/conditioning | Separate RGB/mask trunks versus shared trunk + typed heads × 3 seeds = 6 fits; optional shared + FiLM adds at most 3 fits; 4,000 updates each | Fixed equal-weight sum of per-domain RGB MSE; mask retention checked at that checkpoint | Enter when shared decoding or output conditioning has a consumer. FiLM runs only if unconditioned sharing loses or a prospectively named output needs context |
| P6: encoder conditioning | Late context-aware readout, earlier FiLM adapter, context-only control × 3 seeds = 9 fits; 4,000 updates each | Mean requested-object XY MAE across equally sampled PushT pusher/body queries | Enter for a named query consumer. Test context placement with both visual arms equally informed; this object-query result alone does not validate memory-conditioned perception |
| P7: compatible dynamics/control | Fresh compatible U/P, latent objective versus latent + verified physical multi-step objective × 3 seeds = 6 fits; proposed 4,000 updates, rollout horizon 5 | Validation five-step physical error normalized by declared tolerances | Enter only after that environment's existing perception/memory gates pass and the full control budget is frozen. Compare prediction, action ranking and closed-loop success; failure ends the claim of control readiness |

P2 is observational diagnosis of existing memory, not an alternative encoder competition. P3 and P4 are conditional causal comparisons within the custom family. P5/P6 concern extensibility and context, not prerequisites for every planner. P7 introduces a new compatible world-model package; it never attaches old predictor weights to an unfamiliar latent.

Keep the first-stage comparison at six fits. A randomly initialized frozen ViT may later test a pretraining-related question, but it cannot isolate every architecture/readout effect. Registers, sequential topology, additional scales, larger inputs and query decoders each require their own separately bounded protocol. No combinatorial sweep is proposed.

## 7. P1 protocol: six fits, one checkpoint rule, all outputs

Use the existing deeper/on seed-7107 encoder as the fixed CNN source, chosen by the already recorded lowest validation q among that family. Pin its checkpoint hash before execution. This selection uses previously inspected evidence and is exploratory. Use the already pinned DINOv2-S/14 source and weights. Three readout seeds (9107, 9108, 9109) measure **head-training variation conditional on these two fixed encoders**, not three independent encoder pretrains.

| Setting | Proposed value and rationale |
|---|---|
| Data | PushT CCHI configuration-group split: 20,493 / 2,651 / 2,506 frames, 164 / 20 / 22 groups; fixed COCO audit subset 4,096 / 512 / 512 images |
| Draws | Effective batch 64: 32 COCO + 32 PushT, sampled with replacement from train only; paired seed gives the same image indices/order across arms |
| Updates | 4,000 per fit; 24,000 total; 256,000 frame presentations per fit, 1,536,000 total; do not count separate output losses as extra observations |
| Trainable weights | Fresh decoder projections, RGB/mask trunks and pose head only. Encoder weights and buffers stay fixed in eval mode |
| Objective | RGB MSE averaged equally across the two domains; mask BCE over valid COCO pixels; original normalized PushT pose MSE. Sum these independent-head losses with weights 1/1/1; no shared trainable trunk in P1 |
| Optimizer | AdamW, learning rate 0.0003, weight decay 0.0001, clipping norm 1 separately per independent output; fixed learning rate |
| Precision/batching | FP32 readout training initially. Common microbatch 32 (16 images/domain) with two accumulation steps. Frozen feature cache may use FP16 only after measured round-trip error passes the existing 1e-6 relative-MSE criterion |
| Validation | Every 100 updates on fixed validation populations; save all output metrics and time. Select the smallest PushT q; tie → earliest update. Score every output at that selected checkpoint and also report final-checkpoint diagnostics |
| Pairing | Identical data draws and identical initial tensors wherever shapes agree. Separate RNG streams for unmatched input projections; record those differences rather than claiming impossible tensor identity across widths |
| Timing ceiling | 20 minutes per fit including training validation/checkpoints; completed equal-update comparison takes priority. If a fit hits its cap, mark truncation and report only genuinely matched step/time comparisons |

No per-arm hyperparameter rescue after test inspection. Tiny development runs check shape/gradient/numerical feasibility; freeze any resulting change for both arms before formal work. Cache keys include encoder and preprocessing hashes, dataset IDs and tensor layout. Cache construction must never mix train, validation and test labels or overwrite source checkpoints.

q remains `max(pusher_x_MAE/8, pusher_y_MAE/8, body_x_MAE/8, body_y_MAE/8, angle_MAE/10°)` with positions in simulator world units and wrapped angular error. **q ≤ 1** remains the existing PushT readiness gate. RGB metrics use the same [0,1] range and reduction across arms. Preserve the existing ignore/empty-mask conventions and report empty/full/training-mean mask baselines.

The P3 candidate loss expresses four position residuals in units of 8 world units and the unit orientation-vector chord error in units of `2 sin(10°/2)`, averaging the resulting five squared terms. Original pixel supervision and sampling remain matched. This changes the training objective, not the readiness definition. Verify unit conversion and gradients; resume both arms from the same full optimizer/RNG state, or explicitly classify both as matched optimizer restarts if that state is unavailable.

## 8. Curriculum and advancement rules

Start with label/geometry checks and frozen heads. Use mixed-domain reconstruction while learning each additional output. Once a package supports its required outputs, freeze E while training compatible state update and prediction modules. Only then test small encoder adapters or joint tuning if frozen representation proves inadequate, keeping generic/prior-task samples in the training mixture and checking retention after shared-weight changes.

| Requirement | Proposed decision rule | Interpretation |
|---|---|---|
| PushT perception | Existing validation q ≤ 1; report each coordinate and angle, every seed | A lower average cannot hide a failing required position component |
| Practical improvement | At least 10% lower q in all three paired readout seeds is a proposed screening signal | Engineering consistency criterion, not a significance test or a substitute for q ≤ 1 |
| RGB consumer | Each domain's MSE no more than 1.05× its corresponding reference at the selected checkpoint; inspect critical details | Proposed 5% retention tolerance to agree prospectively. Applies when images are a required output, not automatically to latent-space planning |
| Mask consumer | Beat declared constant baselines; no more than 0.02 absolute IoU loss versus the corresponding reference | Proposed retention margin; foreground union is a limited diagnostic, not a complete application gate |
| Generality | No universal average score. Report the Pareto tradeoff across required outputs, added-head cost and latency | Different consumers may retain different packages; a failure on a required property remains visible |
| World-model progression | Preserve the existing environment-specific memory, prediction and controller gates | New pose or RGB success cannot silently authorize a claim that those gates passed |

The 10%, 5% and 0.02 margins are proposal choices, not empirically estimated minimum detectable effects. Freeze applicable consumers and margins before formal execution. If neither package meets the essential gate, report “no qualified package at this budget.” Do not rank failures into a fictitious production-ready winner. A package that is adequate and cheaper can end architecture search for that consumer.

For memory, distinguish `m_(t-1)` from `m_t`. Updating memory with the legitimately observed current frame is not target leakage. Compare actual deployed availability, then add the prior-memory probe to identify what the latest update changes. No-image/context-only controls, history pairs with equal final images, occlusion cases and collision strata make shortcut explanations testable. Only create new history cohorts under a verified simulator/label protocol.

For P7, first audit physical readouts on real encoded trajectories; an auxiliary loss through a poor or exploitable readout is not reliable supervision. Train the two predictor arms on identical trajectories and matched action/horizon draws. The latent targets and target encoder remain frozen during this comparison. Verify that both arms use the same encoder hash, preprocessing version, feature schema and target exports; changing any of these invalidates the objective-only comparison. Report copy-last-feature, shuffled/zero-action prediction controls, horizon-wise drift, collisions and endpoints. Evaluate action ranking against simulator-verified alternatives, then a matched planner against existing random/no-model baselines. Any encoder update invalidates earlier downstream compatibility claims until rechecked.

End the perception architecture search for a named consumer once its required output gates and resource budget are met. If the memory diagnostic also finds the needed information accessible, leave the optional architecture/context branches unexecuted and proceed only to a separately frozen dynamics/control protocol. This ends the architecture investigation, not the claim of world-model validation; control still needs its own evidence.

## 9. Evaluation must expose errors and preserve the evidence boundary

Existing PushT and COCO held-outs have been inspected repeatedly. P1 is an **exploratory engineering comparison** on those populations. Do not rename another subset of them “untouched test.” Confirm a selected design on a prospectively frozen fresh cohort before broader generalization claims. New simulator configurations require rendering/reference-point audits and establish only that simulator population; new COCO images require duplicate/overlap checks, and pretraining overlap may remain unknown.

Report all seeds individually and aggregate frames at configuration-group/episode level where appropriate. Use paired group bootstrap intervals for PushT differences and per-image intervals for independent image audits, explaining their conditional scope. Three readout seeds do not characterize broad training variance. Preserve a fixed comparison count and one primary metric per stage; do not attach a post-hoc p-value to “all three improved.”

| View in the results report | What it should show | What it cannot establish alone |
|---|---|---|
| Learning and cost curves | Train/validation q and output losses versus updates and elapsed time; selected/final checkpoints, truncated runs | Convergence beyond the measured budget |
| Geometry panels | Predicted/true reference points and orientation; per-group, angle, position and boundary errors; worst cases | A complete object representation |
| Reconstruction and masks | Identical image panels across arms, signed/absolute RGB error, predicted/true masks, small-detail regions and both domains | Plausibility as physical correctness |
| Internal feature views | PCA fitted on train only, held-out projection with fixed scaling within each model; token norms, per-image spatial spectra, neighborhood structure | PCA axes/colors are not semantically aligned across independent models |
| Attention audit | Per-head entropy H/log(K) with valid key count, exact Q/K and masks, update magnitude; trained bypass/fusion comparisons where relevant | Entropy is not causal usefulness. Ablating attention only at test time can itself create distribution shift |
| Memory and prediction | History-paired probes, real versus imagined state readouts, horizon-wise errors and collision/occlusion cases | Readout improvement is not necessarily an improvement in the state itself |
| Control | Fixed episodes, action budget and planner horizon; paired outcomes, latency and failure videos | A PushT/Paddle result does not demonstrate general software use or driving |

Use common image cases without selecting only attractive reconstructions. Fit any PCA normalization or learned probe on training data only. Record exactly which state is shown: encoded observation, updated memory, predicted features or predicted memory. An observed-state plot must not be titled “imagination.”

## 10. Resource envelope, implementation order and stop conditions

Historical context: the completed deeper 4,000-update fits took about 309–336 seconds on the measured RTX 3050 8GB setup, including their training validation/checkpoints. That timing does not predict a native-ViT readout, new decoder, new resolution or full controller evaluation. New configurations need their own development profiles.

For **P0 + P1**, propose a maximum of 30 minutes of development compute, 45 minutes for feature extraction/cache validation, 120 minutes of formal fitting and 45 minutes of final evaluation/report generation: **four hours of local execution allowance**. This is a ceiling, not a runtime promise, and excludes implementation/review effort. Exceeding a category triggers a recorded feasibility/budget decision, not an unbounded continuation.

<!-- budget-chart -->

Formal fitting uses half of the proposed allowance. The other half explicitly covers preparation and checking the result, which short training-time estimates often omit. The figure shows planned caps, not measured durations.

| Later branch | Maximum formal readout/training fits | Proposed fit ceiling | Compute excluded from this number |
|---|---|---|---|
| P2 memory | 12 × 2,000 updates | 5 minutes each; 60 minutes total | New history generation, extraction, final evaluation/reporting |
| P3 objective/budget | 6 × 4,000 additional updates | 20 minutes each; 120 minutes total | Final evaluation/reporting |
| P4 processing | 9 × 4,000 updates | 20 minutes each; 180 minutes total | Development profiling and final evaluation/reporting |
| P5 decoder | 6 fits, at most 3 conditional fits | 20 minutes each; at most 180 minutes total | Development and final evaluation/reporting |
| P6 encoder context | 9 × 4,000 updates | 20 minutes each; 180 minutes total | Query preparation and final evaluation/reporting |
| P7 dynamics | 6 × 4,000 updates | 20 minutes each; 120 minutes total | Environment-specific U preparation, simulator branches and controller evaluation |

These branches are alternatives selected after results, not an overall runtime estimate. Their omitted work must receive an explicit profiled cap before activation. P7 additionally freezes batch size, physical tolerances, action-repeat semantics, exact dataset/checkpoints, controller cases and complete evaluation budget. It cannot launch from this sheet alone. The first comparison is much more specified because it is the next decision; later protocols remain conditional drafts.

Implementation order: add a separately versioned feature/readout contract; implement only the P1 native-feature reader and fresh heads; verify freezing, spatial ordering, label alignment, paired draws and selection semantics; profile; freeze the manifest; execute the six fits; evaluate and rebuild the canonical results dashboard from raw ledgers; publish a decision. Reuse the current harness and preserve legacy loaders. Do not build a universal plugin registry before a second real consumer needs it.

Stop a formal arm for nonfinite outputs, wrong labels, broken freezing/causality, corrupted cache or infeasible resource use; keep its logs and do not count it as a scientific loser. Correct protocol bugs transparently and rerun the affected comparison with a versioned protocol. Required software checks belong to the future implementation; this proposal itself only needs document, reference, manifest and rendered-report validation.

## 11. Extensions that need a named consumer

**Software use:** prepare a small instrumented local application with held-out layouts, control states and action sequences. Use genuine high-resolution observations for text and tiny controls; compare pixels with available structured/text observations as different input packages. Measure exact state reading, hidden-state persistence and action outcomes. This requires a separate data/protocol stage; a resized PushT frame cannot answer it.

**Variable objects and extent:** prepare matched instance masks and object identities, then compare independent dense outputs with a query decoder. Define visible versus amodal extent and orientation symmetries before choosing losses. COCO foreground union is an initial screen, not that task.

**Uncertainty:** where several futures remain possible given the history, specify distributions over the decision-relevant quantities and proper scoring/calibration tests. Stochastic image generation can help an image consumer, but attractive samples are not a substitute for calibrated action outcomes.

**More scales, register checkpoints, bigger transformers:** activate one only for a measured resolution, spatial-artifact, interaction or capacity limit. Compare the smallest intervention that tests the explanation. Capacity and training budget must be separated where possible; depth, parameter count and normalization changes still need to be reported together when not isolated.

Driving and open-ended “internal thinking” are motivating downstream examples, not capabilities established by this proposal. Driving would add sensor geometry, long-tail traffic interactions and domain-specific evaluation; internal search/planning requires temporal and computational mechanisms beyond an image pyramid.

## 12. Literature and Claude review: useful precedents, bounded conclusions

| Primary source | Design implication | Boundary |
|---|---|---|
| [DINO-WM](https://arxiv.org/html/2411.04983v2), §3 | Frozen spatial ViT features, separately learned dynamics and an optional convolutional visualization decoder support an asymmetric package | Does not establish our geometry, retention or controller gates |
| [DPT](https://arxiv.org/abs/2103.13413) and [SegFormer](https://arxiv.org/abs/2105.15203) | Transformer features can feed multiscale convolutional or lightweight dense decoders | Dense prediction precedents, not general world-model evidence |
| [FiLM](https://arxiv.org/abs/1709.07871) | Context-dependent feature scaling/offset is a concrete lightweight conditioning mechanism | It requires suitable context, targets and a fair baseline |
| [RT-1](https://arxiv.org/abs/2212.06817) | Instruction-conditioned visual processing is feasible in robot action learning | A conditioned policy is not itself a validated predictive world model |
| [Perceiver IO](https://arxiv.org/abs/2107.14795) | Output queries can support different output structures | New semantics still require data/objectives; not an automatic first decoder |
| [Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588) | Extra within-frame tokens can reduce high-norm background artifacts and improve dense feature maps | The paper does not supply persistent temporal memory or prove our coarse features need registers |

Claude reviewed a public conceptual brief, not repository code, local measurements or data. We accepted earlier memory diagnosis, consumer-specific RGB gates, one primary checkpoint rule, and conditional rather than automatic FiLM runs. We challenged automatic register-checkpoint substitution, treating current-frame memory updates as leakage, and making a random frozen ViT mandatory for a practical package decision. Claude accepted those corrections in the follow-up. We also added an explicit end to architecture search and an encoder/preprocessing identity check for the dynamics comparison. The final proposal preserves explicit timing, distinguishes practical selection from causal attribution, and keeps new pretrained checkpoints as declared alternatives. Peer agreement is design review, not experimental evidence.

## 13. Deliverable, unresolved choices and evidence notes

The immediate deliverable after P1 should answer: **which package, if any, meets the named outputs, at what cost, with which remaining failure—and therefore what single experiment comes next?** The preferred architecture remains provisional until that decision. Further questions are whether generic readout access survives adaptation, whether memory preserves the distinctions needed by control, whether predictions respect them across actions, and whether a second application changes the required resolution or interfaces.

This report is intentionally an answer-first design sheet followed by technical detail. The architecture figure, proposed-budget chart and lookup tables support the design decision. Historical comparisons are sparse and already visualized in the experiment dashboard, so their exact values appear in a lookup table here. No new model measurements were made; section 14 adds a read-only metadata and numerical-label audit. Values retain their original populations; there is no aggregate “encoder intelligence” score.

Sources for local evidence and current contracts: [study protocol](encoder-study-protocol-2026-09-08.md), [study results](encoder-study-results-2026-09-08.md), [decoder recovery](decoder-recovery-results-2026-09-08.md), [PCA/attention audit](encoder-visual-audit-2026-09-08.md), [requirements-first design](requirements-first-perception-design-2026-09-08.md), [decoder inputs](decoder-inputs-and-conditioning-2026-09-08.md), [encoder conditioning](encoder-conditioning-2026-09-08.md), and [collaboration workflow](claude-collaboration-workflow.md). Proposed settings are identified as such; they are not extracted historical measurements.

Reproducibility: the Markdown is the narrative source; the companion YAML is a non-runnable budget/decision manifest; `scripts/build_perception_proposal.py` packages these into the canonical portable-report input and an architecture figure. `scripts/deliver_perception_proposal.mjs` invokes the canonical builder and its unchanged verifier, with one isolated toolbar-sizing repair to the supplied runtime through the builder's supported runtime option. The original toolbar counted scrollbar width and produced real horizontal overflow; container-relative sizing fixes it without hiding content or changing verification. The receipt records both runtime hashes. The printable offline HTML passes desktop/narrow layout and source-interaction checks. Claude's public brief/replies and raw receipts are retained under `runs/perception_proposal_2026-09-08/collaboration/`.

## 14. Dataset readiness and the next architecture discriminator

The follow-up [dataset audit and Claude review](perception-data-readiness-2026-09-08.md) concludes that **no new download is needed for P0/P1 or the existing-memory diagnostic**. The current data include 82,783 COCO images, all 6,000 prepared Paddle episodes and all 206 prepared CCHI episodes. Numerical-label/time checks pass, and the recorded split groups do not overlap. This is a bounded readiness assessment; existing held-outs have already been inspected, and not all media were decoded again.

| Question | Existing sources | What still needs to be done |
|---|---|---|
| Initial static comparison | COCO RGB/union masks and labelled PushT | Pin/cache/validate the declared interfaces and freeze the protocol |
| Semantic transfer | COCO instance category annotations already present | Prepare labels for objects surviving the actual crop, then fit one separately budgeted fresh category-presence head per package/seed |
| Existing memory and prediction | Paddle actions, simulator velocities and collision records; PushT trajectories | Targeted equal-current-image, occlusion and action-alternative cohorts for stronger causal/persistence questions |
| More action-conditioned domains | Large PushT: 18,685 episodes but 185 exact initial-pose groups; TwoRoom: 10,000 episodes | Distinct adapters, near-configuration/trajectory and cross-source overlap audits; episode volume is not independent-start diversity |
| Natural video/audio | 9,848 Charades videos; 12,291 TAU audiovisual pairs, with all referenced files present | Quantitative passive objectives and decoding/timing audits; activity or scene labels are not our executable control commands |
| Higher resolution | Original COCO, RGB96 CCHI and RGB224 large archives | Reprocess the same source observations, preserving field of view and transformed labels |
| Software use | No prepared local source for the proposed consumer | Collect a small instrumented application with readable screenshots, actions, true state and resettable branches |

**P1-T, proposed semantic-transfer extension:** mean-pool native spatial features and fit the same small nonlinear category-presence head on each frozen encoder, using crop-aware COCO categories. Two packages × three seeds, up to 2,000 updates and ten minutes per fit: at most 60 additional fitting minutes, plus a separately profiled preparation/evaluation allowance. Freeze supported classes, crowd/visibility rules and the complete protocol first. Evaluate category-average AP, per-class support and train-prior baselines. This is global category access, not semantic localization or software understanding.

P1-T keeps the original P1 encoder/RGB/pose checkpoint fixed and selects only its new semantic head on validation AP; report its step separately. Do not retrospectively select another P1 checkpoint or quietly include its budget in P1's four-hour allowance. This extension is not yet launched. If geometry and semantics favor different packages, retain the tradeoff and choose by the consumer; test a hybrid only when both are required and no adequate package meets the cost constraints.

The most valuable later new source is a controlled software-interaction cohort. An instrumented local app gives a manageable first test; public interactive benchmarks such as [OSWorld](https://arxiv.org/abs/2404.07972) also exist, while [ScreenSpot-Pro](https://arxiv.org/abs/2504.07981) tests static high-resolution grounding. Fresh confirmation cases, rather than another partition of already inspected examples, are needed before broader claims. The detailed audit records what was verified, missing preparation and remaining uncertainty.

## 15. Where multiscale features and task queries meet

The [multiscale routing clarification](multiscale-routing-and-tasks-2026-09-08.md) makes feature access explicit. The deeper reference already has two processed grids, one simultaneous bidirectional cross-attention exchange, and a D/U/P interface that receives both levels. Its coarse branch starts from the shared stem, before fine residual processing; a sequential refined-fine-to-coarse route remains a separate experiment.

Preserve the declared processed map at each retained encoder stage. Give output readouts access to those maps directly or through a tested spatial fusion module. P1 starts with its already specified projections, aligned resampling and concatenation. An FPN-style top-down/lateral neck, HRNet-style repeated backbone exchange, more scales and more processing depth are distinct alternatives. Providing multiscale access does not require concatenating every preceding activation into every later layer.

| Boundary | Required clarity in the proposed interface | Conditional addition |
|---|---|---|
| Encoder export | Named levels, widths, grid/position conventions, valid regions, preprocessing, pre/post-fusion identity and time | More levels or different stage coupling |
| Fusion/readout | Declared level access, spatial information retained for dense outputs, independently measured output quality | Top-down/lateral fusion or query cross-attention |
| Context | Output selection or an actual object/task query; equally available in comparison arms | Earlier encoder conditioning where it can preserve needed distinctions |
| Future output | Every feature input causally produced from predicted state or declared retained past information | Richer appearance memory or feature generation, with distribution/compatibility checks |

For PushT, keep physical-state readouts usable across planning goals. For a requested object mask, give the readout an available object query and per-instance supervision; union foreground is insufficient. For albedo, retain spatial/material evidence and define reflectance/lighting supervision. The audited RGB/mask datasets do not establish albedo truth: this newly named consumer needs its own controlled rendering or intrinsic-image data protocol. Relative-reflectance judgments and dense absolute albedo are different targets.

[Mask2Former](https://arxiv.org/html/2112.01527v3) demonstrates one efficient choice: successive decoder layers access different resolutions in a cycle, with a multiscale pixel decoder also present. Its comparison does not show a consistent gain from naive all-scale concatenation at every layer. This is a useful precedent, not a prescribed replacement. Fine-only/coarse-only/both trained readouts can diagnose exported-map access, but our post-exchange maps already contain information from each other. None of these further comparisons is added to P0/P1 automatically.
