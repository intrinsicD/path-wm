# PATH-WM Design Decision Register (DDR) v0.1

Companion to PATH-WM v0.3 (2026-09-03). The defaults recorded here are the ones fixed in §5–§6 of the main document; this file keeps the options and the tests that would overturn each default. Every entry has the same shape: the question, the options, the default for v0.3, why, how we would find out it is wrong, and what depends on it.

Status codes: **NOW** — must be fixed before E1 is frozen. **E_n** — fixed at the named experiment. **DEFER** — implementation left open; the interface must not preclude it.

---

## 0. Summary

| # | Decision | v0.3 default | Status |
|---|---|---|---|
| 1 | Encoders: pretrained or from scratch | From scratch on E0 (LeWM recipe); frozen DINOv2-S added as an extra *foreign* encoder on external environments | NOW |
| 2 | Encoder zoo | CNN, ViT-S/8, conv–attention hybrid at matched ~5M params; SSM and equivariant later | NOW |
| 3 | Latent token space | 8×8 grid + 1 global token, d = 192, continuous, SIGReg-regularized, shared positional convention | NOW |
| 4 | Supervising the latent with labels | No. Labels are probe targets, viewer overlays, and goal masks only | NOW |
| 5 | Predictor | Markov in W, single-frame token Transformer, delta/residual output, action + Δt tokens, registers reset per call | NOW |
| 6 | Belief updater | Predict-then-correct: predictor is the prior, one gated cross-attention block is the correction | E4 |
| 7 | Actions, goals, cost | Action tokens via MLP; goal latent from goal observation with masked token cost; learned value later | NOW / E6 |
| 8 | Combining encoders / modalities | Union of adapted tokens into the updater, modality embedding + timestamp, modality dropout | DEFER (E10); interface NOW |
| 9 | Curriculum | Horizon curriculum and loss staging, fixed and pre-registered; data mixed from step 0 | NOW |
| 10 | Uncertainty | Deterministic → 5-ensemble → MoP-JEPA hard-assigned mixture, gated | E5 / U |
| 11 | Instrument panel | Probe suite, stopgrad debug decoders, rollout and planner diagnostics; built before E1 finishes | NOW |
| 12 | Failure taxonomy | Symptom → metric → cause → fix table | NOW |
| 13 | Contracts and part exchange | Versioned ABI spec + conformance tests; stable-worldmodel-compatible | NOW |
| 14 | Predesign vs evolve | Predesign contracts and instruments; evolve one implementation per experiment against a frozen reference | NOW |
| 15 | Scalability | Batch over (population × horizon); GPU-vectorized environment; two-scale replication of every core result | NOW |
| 16 | Language | Stopgrad text decoder from templated ground-truth captions; LLM prefix adapter later; text goals via a goal encoder | DEFER; interface NOW |
| 17 | Interaction | Real-time viewer with fork/rewind/edit (build early), Python API, text console later | NOW (viewer) |
| 18 | Data and environment engine | GPU-vectorized 2D physics with save/restore; exploration mixture; paired-intervention trees | NOW |
| 19 | Compute and reproducibility | ≥5 seeds, σ_pilot from E1, per-experiment GPU-hour budgets, hashed configs | NOW |
| 20 | Decision order | ABI spec → environment → instruments → E1 → viewer → E2 | — |
| 21 | Common multimodal base after E1-a audit | ABI v2 evidence streams → recurrent belief slots → action dynamics → frozen-model planning | NOW |

---

## 1. Encoders: pretrained or from scratch?

**Question.** Should $E_A$ and the encoder zoo be internet-pretrained frozen models, pretrained-then-fine-tuned, or trained inside E1?

**Options.**
- (a) Frozen pretrained (DINOv2, V-JEPA 2, LeVJEPA) with a learned predictor on top — the DINO-WM / Platonic-WM setup.
- (b) Pretrained, then fine-tuned with inverse dynamics. Frozen internet features are appearance-coupled until inverse-dynamics fine-tuning makes them action-relevant (2606.07687).
- (c) From scratch, end-to-end with the predictor — the LeWM recipe (~15M params, hours on one GPU).
- (d) Self-pretrained on E0 video with the LeVJEPA recipe (block-causal attention, SIGReg, token dropping), then a predictor.

**Default.** (c) for $E_A$ and the whole zoo on E0. E2 needs several *architectures* trained under identical conditions, and pretrained checkpoints exist almost only for ViTs. 64-px synthetic frames are far outside internet pretraining distributions. The coordinate-ownership question of H1 is cleanest when $E_A$ is not a foundation model carrying its own geometry. Add (a) as one more foreign encoder on the external environments (PushT, PointMaze, Wall): a frozen DINOv2-S stitched into a predictor trained with a from-scratch ViT is the most striking version of the gate and lines up with Platonic-WM's protocol for a direct comparison. (d) becomes $E_A$ for the first real-video stage after E7; the ABI does not change.

**How we find out it is wrong.** If grounding probes (object position, identity, container contents) from scratch-trained latents are worse than from a frozen DINOv2-S adapter on E0, the scratch encoders are undertrained. The fix is more data or (d), not switching to (a) for the zoo.

**Depends on.** §3 token format; §13 contracts (encoder output shape is free; the adapter normalizes it).

---

## 2. Encoder zoo and size matching

- CNN: small ResNet-style, stride-8 output, 8×8×d feature map → tokens.
- ViT-S/8: 8×8 patches on 64 px → 64 tokens.
- Hybrid: conv stem (stride 4) + 4 attention blocks.
- Later: SSM (Mamba-style raster scan over patches); equivariant network.

All at ≈5M parameters, same output layout, same training steps and data. $E_A$ = ViT (matches the predictor's token interface). Foreign order: CNN first (the hardest case; Platonic-WM excluded CNNs because their topology did not fit a patch-token predictor), hybrid second, frozen DINOv2-S on external environments third.

---

## 3. Latent token space — what $W$ is

**Question.** What are the units of $W$, how many, what dimension, continuous or discrete, how normalized, and where do positions live?

**Default.**
- $W = [\,64 \text{ grid tokens (8×8)} \;\|\; 1 \text{ global token}\,]$, $d = 192$.
- Continuous; SIGReg-regularized (isotropic Gaussian over the token batch); LayerNorm on every token.
- Fixed 2D positional embeddings on grid tokens (RoPE-2D or sinusoidal), **shared by all encoders' adapters**. Positions are part of the ABI, not learned per encoder.
- Registers $K_R \in \{0,4,8\}$ are appended by the predictor for the duration of a call and never stored in $W$.
- No discreteness in v0.3. A VQ head on top of $W$ exists only for E9-v2 tree search.

**Why.** Grid + global gives spatial grounding by construction (probes can be per-token), a place for identity to persist, a token mask for goal costs, and the same layout as DINO-WM, Platonic-WM and LeWM, so comparisons are direct. $d=192$ is LeWM scale. Continuous latents are what SMC, GRASP and the critic need; MoP-JEPA gives multimodality without discretization.

**Geometry requirement.** Planning costs are latent distances, so latent Euclidean distance must track task distance. Platonic-WM's Appendix F decomposes planning error into rollout error plus latent-geometry distortion, which is why a good one-step loss can coexist with bad planning. Track the correlation between $\|W_i - W_j\|$ and ground-truth state distance on E0 from day one. If it is poor, add Temporal Straightening's curvature regularizer as an ABI term — the one kind of geometry supervision that is allowed, because it constrains shape, not content.

**How we find out it is wrong.** Per-token spatial probes fail while a global probe succeeds (grid is not being used spatially); or distance correlation is low while one-step error is fine.

---

## 4. Should we supervise the latent with segmentation or other labels?

**Answer: no**, for the same reason the debug decoder is under stopgrad (Invariant 1). A segmentation target turns $W$ into a segmentation representation — legible, but not the representation the dynamics need. A label target also destroys the H1 question, because every encoder would then be trained toward the same human-chosen coordinates and stitching would be trivial.

Labels are used in exactly three ways: as **probe targets** (does $W$ contain positions, identities, hidden container contents?), as **viewer overlays**, and as **goal masks** (which tokens count in the cost).

What shapes the *content* of $W$ instead:
1. Architecture: grid, global, registers, the updater's gating (§6).
2. Action-anchored losses: inverse dynamics and counterfactual InfoNCE decide *what must be retained* — controllable, action-relevant state — without dictating *how* it is encoded. This is the Sensorimotor-World-Model argument (2606.20104: one inverse-dynamics regularizer both prevents collapse and aligns latents to action) and the Delta-JEPA argument (decode actions from latent differences so the transition itself is constrained).
3. SIGReg decides the *distribution*.

**Escape hatch.** If grounding probes stay poor after E1, add a self-supervised dense term (V-JEPA 2.1-style patch-level prediction), never a label term. Structure by *architecture* (ABI-1 entity slots) remains a deferred experiment.

---

## 5. Predictor architecture and exact operation

**Contract.** $P_\phi(W_t, a_{t:t+k}, \Delta t, R) \rightarrow \hat W_{t+\Delta t}$.

**The decision with the largest downstream effect: the predictor is Markov in $W$.** No context window; history lives in the updater. Reservoir snapshots, stitching, revalidation and the critic all rely on "the state is $W$". The context-window predictor (DINO-WM's $H$ frames) is E4's baseline only.

**Step by step.**
1. Inputs: 64 grid + 1 global tokens from $W_t$; $k$ action tokens (each $a_{t+i}$ → MLP → $d$-dim token with time embedding $i$); one $\Delta t$ token; $K_R$ fresh register tokens with learned initialization.
2. $L = 6$ pre-LN Transformer blocks, full attention among all tokens (a single frame per call, so no causal mask inside the call), 8 heads, MLP ratio 4.
3. Readout: the 65 state positions → linear → $\Delta W$; $\hat W_{t+\Delta t} = \mathrm{LN}(W_t + \Delta W)$. Delta form: identity by default, which helps permanence and one-step accuracy.
4. Registers discarded (Invariant 4).
5. Multi-step: feed $\hat W$ back. Chunked prediction: supply all $k$ action tokens with $\Delta t = k$, trained with variable-length supervision and a horizon curriculum (VLWM).
6. Stability: LayerNorm on the output; spectral-norm constraint on the readout (LoopWM) is enabled only if the compounding ratio (§11) exceeds threshold.
7. ~10M parameters.

**Alternatives.** Looped shared-weight (LoopWM), SSM, hybrid — E5, deferred; nothing in E1–E7 depends on them and the Transformer has the most direct comparisons. Diffusion/flow *predictors* are not used: Valdi reports no control gain per extra step; flow priors belong in the planner as proposals (LeFlow).

**How we find out it is wrong.** Compounding ratio > 1.5 before $H_{\rm plan}$ with curriculum and LN in place → move E5 forward. Delta form hurting stochastic variants → switch to full prediction for the mixture heads only.

---

## 6. Belief updater — exact operation

**Default: predict-then-correct.**

$$
\tilde W_t = P(W_{t-1}, a_{t-1}, \Delta t = 1) \qquad \text{(prior)}
$$

$$
W_t = \mathrm{LN}\Big(\tilde W_t + g_t \odot \mathrm{CA}\big(\tilde W_t \leftarrow \{A_m E_m(o_t^m)\}\big)\Big) \qquad \text{(posterior)}
$$

CA is one cross-attention block from prior tokens to adapted observation tokens (all modalities as one set, modality embedding per token); $g_t$ is a per-token sigmoid gate. No observation ⇒ $W_t = \tilde W_t$. Train with observation dropout (~30% of steps) so the prior path is exercised. This is RSSM's posterior/prior split with the predictor as the prior; $U_\psi$ is one block plus a gate, small enough to freeze for stitching. 

E4 compares this against (i) per-frame encoding, (ii) $K$-frame context window, and (iii) a block-causal encoder (LeVJEPA) that is its own updater.

**How we find out it is wrong.** Hidden-state probes (container contents) decay to chance within a few unobserved steps even though the prior path is trained → the gate is closing on the prior; inspect $g_t$ statistics.

---

## 7. Actions, goals and the cost function

**Actions.** Continuous 2D forces normalized to $[-1,1]$ → MLP → token; action-noise augmentation during training; VLWM's token-based action representation so chunk length is free. Discrete environments: embedding table. Later (E9-v2): VQ over action chunks for macro-actions.

**Goals.** (a) Goal observation → $W_G$ through the same encoder/adapter (DINO-WM). Cost $J = \sum_{i \in M} \|\hat W_H[i] - W_G[i]\|^2$ over a **token mask** $M$ (tokens where the goal differs from the start, or a task-specified mask). Masking is the cheapest mitigation of latent-MSE ≠ task-distance. (b) Learned goal-conditioned value / the E7 trajectory critic. (c) Text goals through a goal encoder — DEFER (§16).

**Constraints $C$.** For E0, a stopgrad probe from $W$ to collision / workspace violation, used as a penalty; verified against ground truth at evaluation.

**Rule.** Report every planning result under (a) and (b) separately. The cost function interacts with every other decision; it is a variable, not a constant.

---

## 8. Combining encoders and modalities

Interface fixed now, implementation deferred to E10. Every adapter emits tokens in the ABI layout with the shared positional convention, plus a modality embedding and a timestamp; the updater's cross-attention consumes the union; training uses modality dropout so any subset works; asynchronous rates are handled by the timestamp. No fusion at the predictor — the predictor sees only $W$. Test that the contract survives: with a single modality, the E10 code path must reproduce E1 numbers exactly.

---

## 9. Curriculum — what is taught, in what order

**What has evidence.** Horizon curriculum (VLWM), loss staging with warmups (LeWM, PLDM), passive-then-interactive (V-JEPA 2). **What does not.** Environment-complexity curricula for *training* — mixing all E0 variants from step 0 avoids forgetting and keeps one dataset per experiment.

**Default schedule (pre-registered).**
- Stage 0 (first 10% of steps): $L_{\rm reg}$ + one-step $L_{\rm action}$ + $L_{\rm inverse}$.
- Stage 1: add free-running rollout; $H_{\rm train}$ grows $1 \rightarrow H$ linearly over the next 40%.
- Stage 2: add counterfactual InfoNCE.
- Stage 3 (stochastic variant only): mixture heads with hard assignment.
- Adapter training (E2): the same schedule compressed 4×.

Complexity is staged in the *evaluation ladder* instead: deterministic → stochastic → occlusion → compositional OOD. Every model is evaluated on every rung.

---

## 10. Uncertainty (pointer)

Order: deterministic → 5-ensemble (epistemic; planner penalty $\beta\cdot$disagreement) → MoP-JEPA hard-assigned mixture (aleatoric; a searchable transition set the SMC planner branches over) → flow matching in feature space only if mixtures are insufficient. Gate as in v0.2. Valdi's finding stands as a rule: multimodality is for planning diversity, never for per-step accuracy.

---

## 11. Instrument panel — how we look inside

Built before E1 finishes; logged on every run; rendered in the viewer (§17).

**Probes** (linear and one-layer attentive, trained with stopgrad on $W$):
- from $W$: object positions, velocities, identities, container contents (hidden state), agent pose — $R^2$ / accuracy;
- per grid token: local occupancy (is the grid spatial?);
- from registers $R$: the same targets — the leakage ratio.

**Debug decoders** (stopgrad): image decoder for humans; text decoder for hidden-state questions (§16).

**Geometry.** m-kNN between $W$ and ground-truth state neighborhoods; distance correlation; token-norm histograms (outliers → register pressure); PCA of grid tokens; predictor attention maps (does a token attend locally, physically?).

**Dynamics.** One-step and $H$-step error; compounding ratio (open-loop error / teacher-forced error at the same horizon); kinematic-vs-dynamic error decomposition; per-object error; action-sensitivity ratio $s(w)$; counterfactual discrimination accuracy; calibration Spearman.

**Planner.** Homotopy-class coverage; reservoir age and drift histograms; critic-vs-verified cost scatter; predictor-call accounting; success-versus-budget curves.

---

## 12. Failure taxonomy — how we identify problems

| Symptom | Metric | Likely cause | First fix |
|---|---|---|---|
| Full collapse | $W$ variance → 0; probes at chance | regularizer weight, LR | SIGReg weight, warmup |
| Dimensional collapse | effective rank of $W$ low with SIGReg "fine" | projector / normalization | check projector, per-token vs global reg |
| Appearance coupling | appearance probes high, dynamics probes low, counterfactual acc ≈ 1/K | no action anchoring | raise $\lambda_i, \lambda_c$ |
| Action insensitivity | $s(w)$ small; predictor ≈ identity | identity shortcut | counterfactual term; action-dropout ablation |
| Compounding | CR > 1.5 before $H_{\rm plan}$ | no horizon curriculum / unnormalized output | curriculum, LN, spectral constraint, ensemble truncation |
| Register leakage | $R$-probe ≈ $W$-probe | $W$ too small | larger $W$; noise on $R$ |
| Geometry distortion | one-step loss good, planning bad, distance correlation low | latent metric ≠ task metric | masked cost, curvature reg, learned value |
| Mode averaging | deterministic prediction lands between modes | stochastic branching | mixture heads |
| Adapter escape (E2) | $s(w)$ drops in adapted region; inverse loss high | transition-only loss | $\lambda_i, \lambda_c$; check $P$'s manifold |
| Critic exploitation | $\hat J_C \ll$ verified $J$ on mutants | critic optimism | ensemble critic, verify more, penalize disagreement |
| Planner mode collapse | homotopy coverage = 1 | no diversity pressure | diversity term, slower λ anneal |
| Stale reservoir | high ages, large drift | δ_max too loose | tighten δ_max, revalidate deeper |
| Physics exploit | success with constraint violations | simulator artifact | fix engine; count violations as failures |

---

## 13. Module contracts and how parts are exchanged

**ABI spec v1** (a versioned YAML): token layout (64 grid + 1 global), $d$, normalization, positional convention, register count, action-token spec, $\Delta t$ convention, dtype. A breaking change is a major version and invalidates cross-version stitching results by design.

**Contracts.**

| Module | Signature | Frozen for stitching? |
|---|---|---|
| Encoder $E_m$ | $o \rightarrow z$ (any shape) | no |
| Adapter $A_m$ | $z \rightarrow W$ (ABI layout) | no (the thing being trained) |
| Updater $U_\psi$ | $(W, \{W^{(m)}_{\rm obs}\}, a) \rightarrow W$ | yes |
| Predictor $P_\phi$ | $(W, a_{t:t+k}, \Delta t, R) \rightarrow W$ | yes |
| Inverse $I_\omega$ | $(W, W') \rightarrow a$ | yes |
| Critic $C_\chi$ | $(W, a_{0:H-1}) \rightarrow \hat J$ | planner-side |
| Planner | $(W, G, P, C, \mathcal R) \rightarrow (a_{0:H-1}, \mathcal R')$ | — |
| Debug decoders | $W \rightarrow$ text / image | never trained into $W$ |

**Conformance tests** (any implementation must pass before it enters an experiment): shape and dtype; action-sensitivity $s(w) \ge s_{\min}$ on a fixed probe set; transition error $\le \epsilon_{\max}$ on the same set; for adapters, the §6.5 losses on held-out data; for planners, valid actions within budget.

**Signature resolutions (first slice, step 1, 2026-09-03; CLAUDE.md §2).** Recorded here because §16.1 is authoritative and these are corrections made before E1 is frozen. The code is `contracts.py`.

1. *Planner takes V.* §16.1's row $(W, G, P, C, \mathcal R)$ omits §7's optional value $V$. The Protocol is $(W, G, P, V, C, \mathcal R) \rightarrow (a_{0:H-1}, \mathcal R')$ with $V$ the critic, injected so that `evaluation/budget.py` can wrap it, and $C$ the hard constraints of §5.11.
2. *Registers are not a predictor argument.* §16.1 writes $P_\phi(W, a_{t:t+k}, \Delta t, R)$; the ABI says registers are never stored and reset per call, so the only valid $R$ input is "fresh". The Protocol is `predict(W, actions, delta_t)`; $K_R$ fresh registers are created inside every call (Invariant 4) and `n_registers` is a module attribute.
3. *Environment row added.* Not in §16.1: `reset(seed)`, `step(action)`, `save()`, `restore(snapshot)`, `render()`, `ground_truth()`, batched over $N$ worlds.
4. *Goal is a dataclass.* $G = (W_G, M)$ per §5.11 (a); a `None` mask means all tokens.
5. *Unit of a predictor call.* One unit per row of $W$ per call, whatever $\Delta t$: a batched call with $B$ rows costs $B$, a $\Delta t = k$ chunk call costs 1 per row (§7.6 crossover counts per trajectory per step; E9 compares flat versus chunked at matched calls). Critic calls are counted the same way, separately (Invariant 8). `planner.rollout_budget` is frozen in this unit.

**Step-2 additions (first slice, step 2, 2026-09-03; CLAUDE.md §2).** Decisions the structural conformance tests fix before any implementation exists. The tests are `tests/conformance/` and `tests/unit/test_environment.py`; the fixtures that obtain implementations are in `tests/conftest.py`.

6. *Builders.* One plain `build_<module>(cfg)` per family, `cfg` the whole parsed spec (each builder reads its own section plus `abi:`, so swapping a module is one YAML line), at the family's §16.2 home: `envs.build_env` (dispatches on `env.name`), `encoders.build_encoder` (`encoder.arch`), `encoders.adapters.build_adapter(cfg, z_shape)` (`adapter.kind`; z is "any shape" per §16.1, so the adapter is built for the encoder's output shape without the batch dim, as `build_updater(cfg, predictor)` is built for its predictor), `predictors.build_predictor` (`predictor.arch`), `world_state.inverse.build_inverse` (`inverse_dynamics.kind`). Builders build on CPU; the runner moves modules. The conformance tests use a module as built, in eval mode: the Protocols have no mode, and the structural layer tests the inference contract.
7. *Contract precisions* (docstrings in `contracts.py`, no signature change). `Predictor.predict` raises `ValueError` unless $1 \le \Delta t \le k \le$ max_chunk and never modifies its input $W$. `InverseDynamics.infer_action` returns float32 within the ABI action range — a bounded head, with the E1 $L_{\rm inverse}$ taken before the squash so that gradients do not die at the bounds. `Environment.ground_truth` returns a floating `full_state`, an integer `(N, res, res)` `segmentation`, and a `homotopy_signature` with leading dim $N$ and the engine's dtype (winding numbers accumulate as floats along a path). Every tensor the environment hands out belongs to the caller: no live buffers, or the dataset reads $o_t = o_{t+1}$.
8. *ABI clarifications* (fields added to `abi_v1.yaml`; not a breaking change, since no producer existed). `state.token_order: [global, grid]` — $W[:, 0]$ is the global token, $W[:, 1:]$ the 8 × 8 grid in row-major order, so that consumer-side positions mean the same thing for every adapter (§5.2 "identical for all adapters"). `state.normalization.per_token_affine: false` — the LayerNorm at the ABI boundary (adapter output, predictor readout) has no per-producer scale or shift: with an affine boundary a foreign adapter can escape through scale (§14 adapter escape) and the predictor consumes its own affine-scaled output on free-running rollouts, while the consumer's first linear layer absorbs any affine anyway. The structural tests check every token at $|\mu| < 0.02$, $|\sigma^2 - 1| < 0.05$ across input scales 0.1, 1, 10.
9. *Predictor readout is not zero-initialized.* §5.6's "identity by default" is the residual form $\mathrm{LN}(W + \Delta W)$, not a zero init of the readout: the structural test that actions and $\Delta t$ reach the output runs at random weights.
10. *Deferred E0-gate tests.* The RNG-stream half of "save/restore exact" cannot be exercised through the contract in the deterministic variant (RNG is consumed only in `reset`); its test arrives with the stochastic variant. The "≥ 2 homotopy classes on the fixed layout" test needs the layout and arrives with the E0 freeze task. Both stay named in the Now block's Gate line until written; the E0 row of `docs/preregistration.md` is not filled before they pass.
11. *Threshold layer.* The `threshold`-marked tests ($s(w)$, transition error) arrive in the same commit as `evaluation/`, in step 3, and call the predictor with $k = \Delta t$ as training does (E1 draws $\Delta t$ with a $k = \Delta t$ chunk); the $k > \Delta t$ contract case is structurally valid but untrained, so no number is measured on it.

**Step-3 additions (first slice, step 3, 2026-09-03; CLAUDE.md §2).** Readings the thin implementation fixes; each is a config field or a small file, never a branch spread across modules.

12. *Objective (§6.1–6.3, §6.9).* $L_{\rm action}$ is the $k = 1$ term of the $\gamma$-weighted rollout sum, scaled by `losses.action.weight`; the VLWM chunk term $\gamma^{\Delta t}\, d(P(W_0, a_{0:\Delta t}, \Delta t), \operatorname{sg}(W_{\Delta t}))$ with $\Delta t \sim U\{1..\min(\Delta t_{\max}, H)\}$ per batch joins from stage 1 with the same $\gamma^k$ weighting as the free-running term at that horizon; $d$ is the per-token MSE in fp32 on the LayerNormed tokens; every target is detached (§6.1), enforced at the line in `losses/rollout.py` and tested in `tests/unit/test_losses.py`. Weight-0 terms are never computed. The objective consumes observations and actions only (Invariant 11).
13. *SIGReg (§6.1).* LeWM's implementation and numbers (1024 random directions, 17 knots on $[0, 3]$, Epps–Pulley with a Gaussian window, statistic scaled by the sample count; $\lambda = 0.1$ as the dev value), statistics in fp32 (ABI `dtype_regularizer_stats`). `losses.reg.per_token: true` runs one test per token position over the frame batch, the analogue of LeWM's per-time-step test; `false` pools all tokens. This field is the §14 "per-token vs global reg" knob.
14. *Inverse head.* `mlp_pooled_pair`: mean-pooled token pairs $[w, w', w' - w]$ → MLP → `raw_action` (unbounded, what $L_{\rm inverse}$ sees) and `infer_action = clamp(raw_action, -1, 1)` (the contract). `raw_action` is the reference implementation's method, not a contract.
15. *Data.* `env.exploration` names the collector's policy (`uniform_random` in the dev config; the E0 data policy's mixture arrives with the data-policy slice, when the field's value becomes that mixture). Episodes are stored as `<dataset>/episodes.pt` with obs `(episodes, L+1, 3, 64, 64)` uint8 and actions `(episodes, L, 2)`; `run.py` collects when the dataset is absent. Training keys added to the E1 spec from the LeWM recipe: `train.weight_decay`, `train.grad_clip` (dev 1e-3, 1.0).
16. *Engine constants.* The E0 fixed design (layout rectangles, container, radii, dt, force gain, friction) lives in `envs/causal_world/physics.py`, pointed to by `E0_causal_world.yaml engine.physics_params`; it is frozen with E0 and is the one place numbers live in code, since they must not be editable per experiment.
17. *Threshold layer.* `tests/conformance/test_threshold.py` holds the `threshold`-marked tests for every module (s(w) and transition error now; the §6.5 adapter losses when E2's slice adds them). They take `--run-dir` (default `runs/dev/first_slice/0`), regenerate the probe set from the checkpoint's spec, write the number to `<run_dir>/threshold_record.json`, and skip with `threshold_unset` while the ABI threshold is null.

**Step-4 iteration 1 (first slice, 2026-09-04).** The inverse head was the sole changed implementation; the Step 3 pooled head and checkpoint remain the fallback and comparison point. A raw two-frame position probe recovered actions at MSE 0.235 versus the zero predictor's 0.336, but the saved pooled head measured 0.339 with near-zero correlation and mean pooling retained only 4% of the full latent-delta norm. `mlp_token_pair` therefore transforms aligned token pairs before mean/max pooling. After the matched 2,000-step run, held-out inverse MSE was 0.211 with per-axis correlations 0.639/0.636 and s(w) rose 26%, but transition error worsened and correct actions did not beat shuffled or zero actions. This is a mixed negative result: keep the implementation as an alternative, do not promote its checkpoint, and add action-correctness controls before the next intervention.

18. *Action sensitivity is necessary, not sufficient.* Opposite-action separation can increase even when the predictor's response is unrelated to the true transition. Development diagnostics pair s(w) with correct-, identity-, zero-action, and shuffled-action one-step errors; a candidate action anchor succeeds only when the correct action improves over those controls. All four errors use every one-step pair in the fixed probe trajectories. The deterministic shuffle rotates whole action trajectories by one probe example, preserving time index and the evaluation cohort while breaking action-transition correspondence. These are diagnostics, not post-hoc E1 thresholds.

19. *Paired-intervention data and counterfactual anchor.* `<env.dataset>/counterfactual.pt` holds initial observations `(G,C,H,W)`, K actions `(G,K,A)`, and successor observations `(G,K,C,H,W)`. The collector cycles initial-state warm-up depth across `env.episode_len` and calls `restore(snapshot)` before every branch; it never reads ground truth. `counterfactual_data` fixes training/probe group counts and disjoint seeds. Stage 2 begins after `stage0_fraction + horizon_growth_fraction` (50% under E1), samples a separately seeded stream so the ordinary training sequence stays matched, and applies §6.4's K × K negative-MSE logits with every successor target detached. `losses.counterfactual.batch_size` bounds the extra encoder/pairwise memory. The panel reports discrimination accuracy on fixed held-out paired groups; 1/K is chance.

**Step-4 iteration 2 (first slice, 2026-09-04).** A fresh same-GPU control and counterfactual run were identical through step 950. With K=4, weight 1 and the inherited inactive κ=0.1 placeholder from step 3, stage-2 training stayed at loss log(4) and minibatch accuracy 0.25; held-out accuracy was 0.250977 versus 0.248047 in the control. Correct-action error remained worse than zero action (0.000749412 versus 0.000748925), and one-/four-step errors worsened 1.15%/5.27%, so the checkpoint is not promoted. The paired RGB data are non-vacuous (96.22% of off-diagonal branch pairs differ), but κ=0.1 compresses the mean distance-row span to 0.00597 logits and gives a counterfactual-only gradient norm of 0.000273; κ=0.001 gives 1.048 on the same batch. Record this as a temperature-scale dead end, preserve the run, and test κ=0.001 as the sole next change before judging the predesigned anchor itself.

**Step-4 iteration 3 (first slice, 2026-09-04).** The κ=0.001 run was identical to both hardware-matched predecessors through logged step 950. Temperature calibration activated semantic branch ranking: final training accuracy was 0.75 and fixed-probe accuracy was 0.649414; correct-action error became 12.31% lower than shuffled-action error. It did not produce a usable reference. Correct-action error was 0.0191331, 2.36 times zero-action and 10.19 times identity error; one-/four-step error were 25.83/8.81 times the token-pair control. At the abrupt stage-2 onset, the raw pre-clip gradient norm was 178.74, versus a 2.54 mean for the matched control after step 1,000, and remained 30.94 on average. Weight-1 contrastive ranking therefore overwhelms the absolute-dynamics losses at this calibrated temperature. Preserve the checkpoint as a negative result and, before changing architecture or schedule, keep κ=0.001 while reducing only the counterfactual weight to 0.01; this scales the onset contribution to the same order as the ordinary gradient.

**Step-4 iteration 4 (first slice, 2026-09-04).** A user-requested informed search replaced the provisional single weight test. Seven log-spaced/refined weights at κ=0.001 and an equal-`weight/κ` temperature slice bracketed inactive, balanced, and over-ranked regimes. The best pure-InfoNCE point (κ=0.002, weight 0.04) reached 0.297852 fixed-probe discrimination, made correct actions beat zero/shuffled, and improved one-/four-step error 8.37%/8.82%, but remained 51.06% worse than identity. A gradient audit found that, on the matched control, encoder/adapter/predictor InfoNCE component norms were 52.71/5.55/2.45: the objective mostly reshaped perception rather than the predictor. Predictor-only routing at gradient-matched weight 1 still achieved ranking (0.46875) while exploding absolute error, proving the relative-only objective itself can overshoot. Add `positive_weight` for the diagonal paired MSE and `context_gradient` for explicit gradient routing; both default to the old behavior. A measured `{2,6,12,24,48}` positive-weight grid made weight 6 the semantic/accuracy Pareto checkpoint: discrimination 0.315430, correct 6.22%/14.04% better than zero/shuffled, and one-/four-step error 25.04%/21.28% below control. Correct remains 25.16% worse than identity, and the weight-48 boundary returns discrimination to chance, so stop objective search and do not promote.

**Step-4 iteration 5 (first slice, 2026-09-04).** A learning-onset audit separated the intentional stage-2 delay from representation and predictor bootstrap. At initialization W has unit RMS but only 0.00706 across-example variance, the default predictor's paired MSE is 0.31072 versus 0.001732 for identity, and all auxiliary gradient paths are nonzero: the issue is geometry/scale, not empty tensors or gradient starvation. Starting the full paired objective at step 0 with the default readout suppresses scene variance until about step 750 and stays at chance. Scale `predictor.readout_init_scale=0.03`, calibrated before training to 0.002131 initial paired MSE, makes direct training reach 0.52539 held-out discrimination. A 200-step joint dynamics warm-up followed by counterfactual training reaches the 3-SE onset threshold at step 1,000 versus step 2,000 in the diagnostic replay and ends at 0.53516. Encoder-only SIGReg+inverse pretraining is rejected: although scene variance reaches 0.9518 by step 200, paired identity MSE inflates to 0.1611, so it produces action-relevant but temporally discontinuous geometry and delays the predictor. Keep explicit stage-onset and diagnostic-checkpoint controls plus the optional residual scale; retain the short-joint schedule for E1-b comparison, but do not promote an E1-a checkpoint because correct action remains 2.81% worse than zero and 5.01% worse than identity in its strongest semantic condition. The exact hidden-velocity alias still requires E1-b history.

20. *Single-frame E1-a is observably non-Markov.* Opposite hidden velocities can have bit-identical RGB but different next RGB under the same action; a committed regression constructs this exact alias. Across 3,840 fresh random transitions, identity/action-only/two-frame/full-velocity physical next-position MSE was 0.000215142/0.000148172/0.000002944/0.000001315. Two frames remove 98.63% of identity error, so the next comparison is the predesigned E1-b updater, which carries history in W without changing the Markov predictor contract. A K-frame or block-causal encoder remains the E4 comparison; changing E0 to render velocity would remove the intended POMDP rather than solve it.

**Exchange procedure.** Implement → conformance → E2-style comparison against the frozen reference at equal budget → entry in the results ledger. **stable-worldmodel mapping:** `encode` = $A \circ E$ (+ $U_\psi$), `predict` = $P$, `rollout` = the loop, `criterion` = the cost, `get_cost` = the accounted call count.

---

## 14. Predesign or evolve?

Predesign the **contracts, the ABI spec, the instrument panel, the environment, the evaluation protocol and the thresholds**. Evolve **every implementation**, one module per experiment, against a frozen reference. Keep the simplest working implementation of each module forever as the baseline and the fallback. Change an interface only when two independent implementations both need the change. This is the same discipline as an ABI: the interface is stable so that implementations can move.

---

## 15. Scalability

- **Environment:** GPU-vectorized 2D physics (custom CUDA, Warp, or JAX), thousands of parallel worlds, deterministic save/restore for interventions and planner forks; ≥ $10^4$ transitions/s.
- **Model:** batch over (population × horizon); `torch.compile` or JAX; bf16; ≤ 20M parameters through E7 (LeWM shows 15M is enough at PushT scale).
- **Planner:** SMC is embarrassingly parallel — one predictor call per iteration over the whole population; critic batched; TSMCTS is the reference for GPU-parallel search.
- **Data:** sharded, memory-mapped; paired interventions stored as trees with parent ids.
- **Scientific scalability:** every core result at two model sizes (S, M) and two data sizes; report trend direction, not a point.
- **Path to real video:** swap $E_A$ for a LeVJEPA-pretrained block-causal encoder (a single consumer GPU suffices for a ViT-Tiny); token dropping; the ABI stays.

**Decision (first slice, step 1, 2026-09-03).** The E0 engine is batched PyTorch on one device, the same framework as the models: one framework is the simplest thing that can reach the $10^4$ transitions/s gate on one GPU, and Warp, JAX or custom CUDA remain a later, measured swap behind the `Environment` Protocol. Recorded in `experiments/E0_causal_world.yaml` (`engine.framework`).

---

## 16. Language — how it learns to talk

**Principle.** Language is an I/O modality; $W$ never receives language gradients (the H3 caution: a world state that is trained to be a caption becomes a caption).

- **L0 (with E0):** templated captions from ground-truth state ("red box inside the container; agent at (3,5) moving left; goal not visible") → train $D_{\rm text}(\operatorname{stopgrad}(W))$ as a small Transformer decoder. Evaluate on hidden-state questions ("what is in the container?") — this is the belief-state probe in natural language.
- **L1:** replace templates with a small LLM behind a prefix adapter ($W$ tokens → prefix embeddings), still stopgrad. Free-form questions; plan explanations by decoding the reservoir's predicted $\hat W_{1:H}$ ("route left around the wall, then push the box").
- **L2 (E10):** text as input — a goal encoder $E_{\rm text}$ + adapter to a goal latent or a goal mask; language-specified constraints.

What talking is not: a training signal for $W$.

---

## 17. Interaction — how we work with it

- **Python API** over the contracts (notebook-first); every experiment is a script that uses the same API.
- **Real-time viewer** — build it early, it is the microscope. Panels: environment; debug-image decode of $W$; probe overlays (predicted vs true positions, container contents); reservoir trajectories drawn on the world, colored by homotopy class, with age and cost; critic-vs-verified scatter; attention maps; the instrument panel (§11). Controls: set a goal by clicking; drag-edit a trajectory and re-verify; inject a perturbation; fork/rewind (the snapshot runtime); step the planner one SMC iteration at a time. The viewer is a client of the Python process over shared memory or a socket — a lightweight web client first; a Vulkan client in your own engine once the panels have stabilized.
- **Text console** (L1) for state queries and plan explanations.
- **Agent loop:** closed-loop MPC in the environment with full logging; every executed action has a verified rollout (Invariant 10).

---

## 18. Data and environment engine

E0 as specified in v0.2, plus a data policy: exploration mixture (uniform random, Brownian, scripted goal-directed, and — from E6 on — replayed planner actions), a coverage metric on ground-truth state, validation splits by object configuration (held-out combinations for compositional OOD), paired interventions stored as trees, seeded stochastic variants, an initial dataset of ~1M transitions. The engine must expose save/restore and a deterministic RNG so that interventions and planner forks are exact.

**Decisions that cannot be widened later (first slice, step 1, 2026-09-03).** *Batched-worlds layout:* every environment tensor has leading dimension $N$ (worlds), observations are $(N, 3, 64, 64)$ uint8 and actions $(N, 2)$ float32, all on one device. *Save/restore representation:* a snapshot is a clone of every state tensor plus the `torch.Generator` state; `restore(save())` followed by `step` reproduces the same observations bit-exactly, which is what paired interventions and planner forks require.

---

## 19. Compute, seeds and reproducibility

≥ 5 seeds per condition; $\sigma_{\rm pilot}$ from the 5-seed E1 baseline before any threshold is frozen; per-experiment GPU-hour budgets recorded in the spec; configs hashed into `docs/preregistration.md`; a results ledger that records every conformance test and comparison.

---

## 20. Decision order — what to fix this month

1. ABI spec v1 and the module contracts (§3, §13).
2. Environment engine and data policy (§18).
3. Instrument panel and failure taxonomy (§11, §12).
4. E1 reference: encoders, predictor, schedule (§1, §2, §5, §9).
5. Viewer v0 (§17).
6. Then E2.

Everything else is deliberately left open, with the interface written so that it can be closed later without touching the first four.

---

## 21. Common multimodal base after the E1-a audit

**Question.** Should the reference continue to make every encoder emit ABI-v1 visual-grid state and
jointly learn perception/dynamics from step zero, or should it separate sensory evidence, persistent
belief, action dynamics and planning?

**Options.** (a) keep ABI v1 and tune its losses; (b) add history while retaining one visual grid for
every modality; (c) introduce variable modality-native evidence and a fixed modality-neutral belief,
then train representation, belief, dynamics and planning in gated stages.

**Decision.** Adopt (c) as the v0.4 candidate; retain (a) as the measured E1-a control. ABI v2 has 64
latent belief slots plus a global token, variable timestamped evidence tokens, embodiment-specific
action adapters, a predict-then-correct updater and a Markov slot predictor. Video and audio are the
first implemented evidence producers. Future sensors reuse the evidence contracts; future uncertainty
and planning modules reuse the belief contract. See `docs/common-base-architecture.md`.

**Why.** The audit ruled out empty tensors and missing gradients. It instead found three structural
problems: an exact single-frame hidden-velocity alias; a random predictor 179 times above identity
transition scale; and stop-gradient targets without an EMA teacher. It also found that encoder-only
SIGReg+inverse warm-up raised scene variance while making adjacent frames discontinuous. Therefore a
curriculum is warranted, but its first stage must be temporally predictive EMA representation learning,
not generic encoder warm-up. Audio also makes the v1 assumption that all evidence has 8x8 visual
coordinates untenable.

**Promotion test.** ABI-v2 conformance and differentiable video+audio smoke tests first. Then each
curriculum stage advances only through the held-out gates in `docs/common-base-architecture.md` §4–5.
The world predictor must beat identity, zero and shuffled actions before rollout training; the frozen
world model must pass open-loop gates before any planner is trained.

**Depends on.** A real synchronized A/V dataset selection for R0/R1; E0 remains the action-labelled
causal testbed. ABI-v1 checkpoints are not stitch-compatible with ABI v2.
