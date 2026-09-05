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

---

## 22. R0/R1 data boundary and first real audiovisual corpus

**Question.** Which data dependency can exercise real synchronized audio/video without coupling the
representation learner to a dataset SDK, unstable hosted-video identifiers, or label supervision?

**Options.** (a) VGGSound/AudioSet identifiers that require re-fetching third-party videos; (b) Ego4D,
whose scale and access agreement are appropriate later but too heavy for the first gate; (c) the openly
archived TAU Urban Audio-Visual Scenes 2021 corpus; (d) a repository-specific media loader used directly
inside the training loop.

**Decision.** Use (c) for the first R0/R1 run and reject (d). Offline ingestion reads TAU's audio/video
pairs and official fold metadata, decodes each 10-second recording once into a normalized per-clip tensor
shard, and writes a versioned JSONL manifest. The training path depends only on the manifest/shard
contract. R0 samples video and audio independently; R1 samples synchronized current/future windows and
same-recording wrong-time negatives. Dataset labels are retained only as unused provenance and never
appear in a `RepresentationBatch`.

**Why.** TAU supplies directly archived media, stable checksums, synchronized 10-second audio/video,
and an official split whose recording-location identifiers allow leakage checks. Per-clip shards keep
random window reads bounded and make interrupted ingestion resumable without creating a monolithic
tensor store. The boundary remains reusable for larger or egocentric corpora.

**Promotion test.** Every manifest entry resolves to one valid shard; source audio/video paths pair by
metadata rather than filename guessing; train/eval clip ids and recording identifiers are disjoint; a
fixed seed reproduces sampled windows; R1 shifted views come from the same recording at the configured
nonzero time offset; and held-out panel metrics alone drive the curriculum gate.

**Depends on.** TAU Urban Audio-Visual Scenes 2021 development corpus, DOI
`10.5281/zenodo.4477542`. The 128 MB examples archive is development plumbing only; a promotion run uses
the complete development corpus and its official fold.

---

## 23. R0 dimensional-collapse guardrail

**Question.** The first real-data R0 panel showed healthy per-dimension standard deviation but very low
effective rank. Which intervention addresses redundant dimensions without undoing the temporal continuity
learned by the masked/future objectives?

**Options.** (a) weaken the effective-rank gate; (b) replace the temporal EMA objective with the earlier
SIGReg-only encoder warm-up; (c) add separate variance and off-diagonal covariance penalties to online
evidence; (d) whiten evidence or add a disposable high-dimensional regularizer projector.

**Decision.** Reject (a) and (b), and implement (c) as the smallest falsifiable intervention. For each
modality, valid evidence tokens are centered in fp32; the squared off-diagonal sample covariance is summed
and divided by evidence dimension. The existing standard-deviation hinge prevents the zero-covariance
solution obtained by shrinking all variation. Temporal prediction and EMA targets remain unchanged. This
is the variance/covariance separation introduced by VICReg (Bardes, Ponce & LeCun, arXiv:2105.04906),
applied directly where the effective-rank gate is measured. SIGReg and a disposable projector remain
recorded alternatives, not simultaneous changes.

**Why.** Per-dimension variance rules out constant features but not dimensions that move together. The
covariance term specifically penalizes that redundancy and does not force adjacent observations together
or apart. Replacing the temporal objective would repeat the E1-a failure in which variance rose while
adjacent-frame distance became 93 times worse. Weakening the gate would hide the measured failure.

**Development evidence.** On the exact initial seed/data batch, unweighted encoder-gradient norms were
0.578 for variance and 27.904 for covariance. The first declared covariance weight 0.01 therefore made
its weighted gradient 4.8 times the variance-floor gradient: after 20 steps, video/audio effective-rank
fractions moved from the no-covariance baseline 0.108/0.043 to 0.115/0.047, but feature standard deviations
shrank from 0.755/0.727 to 0.642/0.619. A follow-up coefficient of 0.002 was selected from that initialization
gradient audit, before reading its held-out panel. It retained more spread (0.732/0.702) but ranks moved only
to 0.110/0.045; audio masked/future advantages remained negative and the R0 gate still failed. The tiny
20-clip, 20-step fixture establishes plumbing and failure direction only; it does not select a winning
representation architecture.

**Promotion test.** Carry the gradient-balanced variant to the complete TAU development corpus with a
realistic batch and optimization budget, while retaining the official group-disjoint split and the frozen
0.25 rank, positive prediction-advantage, positive temporal-retrieval, and feature-variance gates. Compare
against the no-covariance control at matched seeds and budget. Retain covariance only if rank improves
without losing the temporal/variance guardrails; otherwise test one recorded alternative in a new isolated
iteration. Do not start R1 or B0 on a failed R0 representation.

**Depends on.** `configs/dev/common_base.yaml` is the no-covariance control;
`configs/dev/common_base_rank.yaml` preserves the overweight negative trial; and
`configs/dev/common_base_rank_balanced.yaml` is the next full-corpus candidate.


---

## 24. Full-corpus R0 execution and recovery

**Question.** Can the existing R0 objective learn non-redundant evidence at a realistic data/step
budget, and can that comparison survive an interrupted overnight acquisition or training job?

**Decision.** Keep the model, EMA decay, learning rate and all R0 thresholds fixed. Compare covariance
0.002 against 0 at batch 64, 10,000 steps, seeds 0 and 1, with 32 deterministic held-out batches.
The complete official TAU fold has 8,646 train and 3,645 evaluation clips. Matched long runs on the
20 example clips execute during acquisition to isolate optimization duration from corpus diversity;
they cannot promote R0 or select a new coefficient. Full-corpus rank must improve in both modalities
at both seeds, with all temporal/variance guardrails intact, to retain covariance. Every selected
full-corpus seed must pass R0 before R1. These are development runs, not a frozen E1 reference.

**Recovery.** A cached shard is bound to the actual source audio/video SHA-256 values and normalization
settings. Source or normalization changes force decoding; interrupted shards are rebuilt. Shards and
manifests publish through same-directory atomic replacement. Example discovery stays in `raw/examples`
even after development videos arrive. Decoder concurrency is a runtime CLI choice (`--workers`), not a
new dataset or loss. The full-corpus manifest and shard directory are separate from the example lineage.

Training writes a step-zero snapshot and an atomic snapshot every 500 steps. Snapshots include learner,
EMA teachers, optimizer, independent data/corruption RNGs, CPU/CUDA RNGs, original panel, config, seed,
and manifest fingerprint. Explicit `--resume` rejects mismatched config/data/device type and truncates
only log rows beyond the durable step before replay. Fresh invocation refuses to overwrite a run.
A CPU interruption regression reproduces uninterrupted weights and metrics exactly. Non-finite loss
or gradients fail before updating the model. CUDA resumption uses the same states; bitwise equivalence
across hardware/software versions is not asserted.

**Acquisition.** Three archive workers reuse partial downloads, verify the original published sizes and
MD5 values, and serialize writers with per-archive locks. Only the parent publishes whole-corpus
completion after all partitions succeed. The existing 20 GiB free-space floor remains active.
Source: https://zenodo.org/records/4477542. An overnight queue gives every subprocess the same absolute
deadline, serializes GPU runs, verifies the complete official split before full training, and preserves
logs and comparison JSON under `runs/overnight/`. A failed gate is an experimental result, never a reason
to weaken its threshold.

**Development validation.** The batch-64 250-step recovery smoke on the 20 examples took 43 seconds on
an RTX 3050. Audio masked/future advantages became +0.141/+0.143; video stayed positive. Video/audio
rank fractions 0.080/0.061 still failed 0.25. The dashboard refreshed with structural-only verification.
This validates execution and exposes the remaining rank failure; it is not the full-corpus decision.


---

## 25. R0 throughput without changing the experiment

**Question.** How can full-population window reads and temporal masking fit the overnight local budget
without changing the seeded training comparison?

**Decision.** Vectorize the existing corruption policy: one random tensor per modality, force the first
valid position when a row has no masked sample, and preserve the last valid position when every sample
is masked and at least two are valid. Batched indexing replaces per-row CUDA synchronization; outputs
and generator consumption stay exact. Map per-clip tensor files before selecting short windows so reads
do not copy unused frames. All data, model, objective, optimizer and panel settings remain unchanged.
Each new R0/R1 invocation records Git revision/dirty state, a digest of tracked Python source, normalized
config and manifest digests, device and Torch version in its run ledger. Recovery attempts append their
own identity rather than silently replacing the original invocation.

**Validation.** Forty padded CPU and forty padded CUDA comparisons match masks and random-stream states
exactly. A real batch-64 CUDA comparison matches every loss and final learner/EMA tensor across 50 AdamW
steps. Video masking drops from 4.33 to 0.218 ms, audio from 6.35 to 1.87 ms; steady step time drops from
164.7 to 155.9 ms (about 5.3%). A separate warm-cache sampler probe that forces full-population-like unique
clip loads drops from 90.0 to 72.2 ms/batch with identical sampled tensors. That probe uses virtual clip
identifiers solely to measure deserialization; it creates no new training data or experiment result.
All 168 fast tests pass, with two opt-in tests deselected. Proof and benchmark scripts are under
`runs/overnight/common_base_20260905/` (`masking_*_benchmark.json`, `loader_mmap_benchmark.json`).

**Rejected runtime changes.** Forcing one codec thread did not improve decoding. Eight clip workers
versus four reduced the 20-example probe only from 6.39 to 5.96 seconds, with identical decoded tensors;
the queue retains four workers. Linear extrapolation suggests roughly one hour for the full corpus,
but this is a throughput estimate rather than a completed ingestion time.

**Current experiment evidence.** The first balanced 10,000-step example seed passes both rank thresholds
(video 0.2614, audio 0.2700), but fails both future-advantage gates (-0.7377/-0.0480). Masked advantages
remain positive. The dashboard refreshed with structural-only verification. This is a duration diagnostic
on nine train and eleven eval clips; the full-corpus pair still decides the covariance intervention.
Frozen-checkpoint blank-input and teacher-copy controls are preserved as supplementary diagnostics,
including the possibility that position variation or student/teacher basis alignment inflates simple
metrics. They do not change the predeclared panel or establish a new architecture decision.


---

## 26. Same-basis future-copy diagnostic

**Question.** Does positive future prediction advantage reflect forecasting, or can it include alignment
between online and EMA latent coordinates?

**Decision.** Preserve every existing R0 metric and gate. Add supplementary
`{video,audio}_future_teacher_copy_advantage`: MSE(teacher-current, teacher-future) minus
MSE(learned prediction, teacher-future), using already-computed frozen evaluation views. The original
future-advantage metric uses online-current for its copy baseline and remains unchanged. Neither
training nor the sampler draws extra randomness for this diagnostic. A pure student/teacher scale
alignment with identical teacher-current/future targets must score zero on the new control.

**Evidence.** On a frozen step-2,500 example checkpoint, audio scores +0.01097 against the online copy
but -0.001985 against the teacher copy. At step 10,000, both modalities trail both controls, consistent
with the actual small-data future-prediction gate failure. A four-batch CPU replay of the final checkpoint
reproduces every original panel number and the entire gate result exactly after adding the diagnostic;
the additional teacher-copy advantages are -0.7252 video and -0.0558 audio for that supplementary cohort.
Those four batches are distinct from the official 32-batch CUDA panel and do not replace it. All 169 fast
tests pass (two opt-in tests deselected). Proof: `runs/overnight/common_base_20260905/teacher_copy_panel_compatibility.json`.

**Interpretation.** The original VICReg diversity statistics are computed across example embeddings
([§4.1](https://arxiv.org/pdf/2105.04906)); flattening our tokens also counts position differences. This
motivates the supplementary blank-input/within-position probes, without establishing a replacement
objective. Full-corpus measurements must distinguish representation diversity, temporal generalization,
and latent-coordinate alignment before a representation is treated as useful for the next stage.


---

## 27. Overlap acquisition with bounded cache ingestion

**Question.** Can unused CPU capacity remove the post-download decoding delay while the full-corpus
R0 comparison still sees exactly the official split and source bytes?

**Decision.** Add an operational cache-only mode that decodes at most a bounded number of available,
uncached source pairs and never writes a manifest. Ingestion processes serialize on the shard root's
file lock. The final normal ingestion still requires every source, rehashes all source bytes, rebuilds
any changed or truncated cache, and atomically publishes the complete manifest. The prefill loop stops
at acquisition completion or the shared overnight deadline. A partially extracted source can at most
produce a reusable-or-rejected cache candidate; it cannot enter training without full revalidation.
No scientific setting, source selection or training trajectory changes.

**Validation.** The missing-source test permits bounded prefill but rejects final ingestion until the
source arrives; changing a prefilled source forces a rebuild. On the real 20 examples, prefill of five
clips followed by full ingestion reuses those five shards and reproduces the original manifest byte for
byte. All 170 fast tests pass (two opt-in tests deselected). Proof is in
`runs/overnight/common_base_20260905/prefill_compatibility.json`.


---

## 28. Make representation outcomes visible in the experiment dashboard

**Problem.** Once a representation run became the latest completed seed, headline cards still selected
metrics from the union of historical action-model runs and displayed only missing values. R0 metrics
were available only in the exact result table, and `max_steps` was absent from the inventory.

**Decision.** Headline cards explicitly describe the latest completed run and select its own metric
family. R0 exposes the recorded gate result, both rank fractions and both future advantages; small
future values use a labeled ×10³ display scale. Native comparison charts retain unscaled raw rank and
future scores by run/modality. The inventory records corpus, batch size, covariance weight and the
correct training-step field. Prior action-model charts, raw ledgers, gate outcomes and thresholds stay
intact. The selected-run control is described as an enhanced-reader control; the semantic fallback
labels all included runs.

**Validation.** A mixed-ledger test verifies that a latest R0 run has no missing action-model headline,
that every chart value matches its source, and that tiny negative future values keep their sign after
display scaling. All 171 fast tests pass (two opt-in tests deselected). Canonical artifact packaging
succeeds with structural-only verification. An isolated installed-Chrome rendering exposed the original
missing-card problem, but Chrome fails the canonical extractor's requested-environment check, so full
browser QA is still unavailable; no full visual-verification claim is made.


---

## 29. One-batch prefetch with consumed-only random-state checkpoints

**Question.** Can the common evidence frontend overlap full-population CPU window reads with CUDA
training without changing H1 / E1_common_base's seeded comparison or recovery semantics?

**Decision.** `train.prefetch_batches` is 0 (serial) or 1 (one queued CPU batch). The worker samples with
a private generator. Only taking that batch advances the caller's data stream, so optimizer snapshots
exclude queued but unused draws. Explicit generator closure joins pending work on exit. The full-corpus
pair enables 1 after validation; the tiny base/rank templates share the explicit disabled default. No
source, model, optimizer, seed, panel or threshold changes. Recovery still requires identical configs.

**Validation.** All 175 fast tests pass, including a test that waits for the second draw to finish while
the checkpoint stream still points immediately after the first, exact serial/prefetched CPU training,
and interrupted recovery with and without prefetch. Four 100-step CUDA trajectories in ABBA order match
every loss, learner/EMA tensor, optimizer state and random stream exactly. An interrupted actual CUDA
runner also matches its uninterrupted model, optimizer, random states and final panel exactly.
The loading-heavy warm-cache probe averages 256.7 ms/step serial and 183.2 ms/step prefetched, a 28.6%
reduction. Virtual clip identifiers force unique loads of the same example source tensors solely for
this timing probe; they are not additional research data. Actual full-corpus throughput is still pending.

The real 250-step batch-64 prefetch smoke also matches the earlier saved recovery smoke's complete
model/EMA, optimizer, random streams and every original initial/final panel metric exactly. Its unchanged
rank failure is preserved, and the dashboard refresh reports structural-only verification. Proof files:
`runs/overnight/common_base_20260905/prefetch_gpu_benchmark.json` and
`runs/overnight/common_base_20260905/prefetch_smoke_compatibility.json`.

**Completed duration control.** All four 10,000-step example runs are now complete. Covariance improves
rank in both modalities at both seeds: video 0.2614/0.2599 versus 0.1189/0.1155, audio 0.2700/0.2771 versus
0.0613/0.0587. Every run still fails both future-prediction gates. The example bundle has only nine train
and eleven eval clips, with three eval scene categories absent from training. These are duration and
generalization diagnostics; the predeclared full-corpus pair still decides coefficient retention.


---

## 30. Separate position diversity from variation across windows in the R0 panel

**Question.** Can a high token-rank score be explained by fixed position structure rather than
variation in sensory inputs? For the current layouts, 120 audio positions can supply a rank fraction
up to 119/192 even when every window is identical; the corresponding 32 video positions supply at
most 31/192. This is a diagnostic limitation, not a reason to change the fixed rank threshold.

**Decision.** Add `{video,audio}_within_position_variation_fraction` as supplemental panel context.
For each batch, divide squared variation around each position's mean across windows by squared
variation around the global valid-token mean. Exclude padding, including NaN padding, from both;
constant evidence returns zero. Average the per-batch fractions using the existing held-out cohort.
Position-only evidence returns zero, but a larger fraction does not establish semantic content.
The metric uses already computed online evidence, with no additional forwards, random draws,
training changes, gate changes, or rewriting of historical ledgers.

**Validation.** Tests reproduce position-only evidence that passes the rank threshold yet has zero
within-position variation, content-only evidence with fraction one, uneven padding with an exact
8/35 fraction, and constant evidence. All 177 fast tests pass (two opt-in tests deselected).
A CUDA replay of the frozen example balanced seed 0 on the official 32×64 evaluation cohort matches
every prior metric, original ledger field, gate outcome, and random stream exactly. The new fractions
are video 0.78657 and audio 0.32339; both future-prediction gates still fail. Proof:
`runs/overnight/common_base_20260905/position_panel_compatibility.json`.


---

## 31. Give synchronized R1 modalities one physical time origin

**Problem.** The sampler independently subtracted each modality's final sample time. With video at
8 fps and audio at 16 kHz, samples captured at the same physical instant differed by 0.1249375 seconds
in their reported timestamps. That contradicts the shared belief-update reference in ABI-v2 and the
synchronized R1 data contract. Independent R0 streams do not claim this cross-modal correspondence.

**Decision.** R1 timestamps use the exclusive window end as a shared reference, so every observed
sample precedes the update time and aligned video/audio instants have the same timestamp. Each
current/future/shifted view uses its own window end: timestamp features alone cannot disclose which
view is the shifted negative. R0 retains its existing last-sample reference and all its source draws.
This is a sampler correction before any real R1 run, not an R0 model intervention or an R1 promotion.
No ABI shape, gate, source population, normalization or training-setting changes.

**Validation.** The new conformance test failed on the old physical-clock mismatch and now checks
aligned instants, causal timestamps, and identical relative grids across the shifted controls. All
178 fast tests pass (two opt-in tests deselected). On real data, 16 batches of 64 per stage across both
splits and both seeds reproduce R0 values/timestamps/masks/random streams exactly; R1 values/masks/
random streams also match exactly, while its clock discrepancy becomes zero. The previously validated
training module remains byte-identical. Proof:
`runs/overnight/common_base_20260905/shared_clock_compatibility.json`.


---

## 32. Bind R1 entry to a completed R0 checkpoint and retain collapse guards

**Problem.** Selecting `representation_av` previously built a fresh model, with no R0 handoff or
prerequisite check. Its configured exit gate checked only A/V scores even though the architecture
requires that neither modality collapse. These gaps must close before a real R1 run.

**Decision.** A fresh R1 seed requires `train.r0_initialization[seed]` containing a checkpoint path
and full SHA-256. Read and deserialize those exact bytes; verify completed R0 stage/budget and panel
cohort, matching seed and manifest, compatible model/data configuration, and unchanged prerequisite
thresholds. Recompute the R0 gate from the metrics bound inside the checkpoint. Reject a failed source
before training. Transfer the entire learner/EMA state, then start a fresh optimizer and R1 random
streams. Store source identity, metrics and gate provenance in target snapshots and final ledgers.
Resume uses its own initialized snapshot and does not reopen the original source file.

R1 inherits the video/audio feature-standard-deviation and rank conditions from the configured R0
gate, in addition to its A/V conditions. Conflicting R1 overrides are rejected. No R0 condition or
threshold changes. The optional initialization field defaults to null in the small R0 templates and
is rejected if supplied to an R0 run. This work was prepared in an isolated worktree while the paired
full R0 comparison used its committed runtime; it does not authorize a real R1 run before that decision.

**Validation.** All 192 fast tests pass (two opt-in tests deselected). Invalid checksums, incomplete or
failed sources, wrong stage/seed/manifest, incompatible models, and changed R0 thresholds are rejected.
The initial R1 snapshot exactly matches all source weights including EMA, with an empty optimizer;
interrupted R1 with prefetch reproduces uninterrupted weights, optimizer, random streams and metrics
even after the source fixture is removed. These transfer tests stub panel measurements solely to test
mechanics; they are not experimental evidence of a passing representation.

A real 250-step CUDA R0 replay matches the original learner/EMA, optimizer, CPU/CUDA/data/corruption RNG,
all prior initial/final metrics and every final training value exactly. The real 10,000-step example
balanced checkpoint is rejected for its two failed future gates, and no R1 checkpoint is created.
Proof: `runs/overnight/common_base_20260905/handoff_r0_compatibility.json`.


---

## 33. Measure aligned audiovisual changes within a recording

**Question.** How much of R1's projected correspondence concerns change over time, alongside the
existing retrieval and wrong-time accuracy gates?

**Decision.** Add `audiovisual_temporal_change_alignment` as supplemental R1 context: half the dot
product of current-minus-shifted video/audio vectors after unit normalization, averaged over held-out
windows. It is zero if either modality's embedding stays constant over time. Positive and negative
values distinguish aligned and opposed projected changes. Use the already computed embeddings, with
no additional forwards, random draws, objectives, thresholds, or gate changes.

**Validation.** Constant embeddings produce zero, matching changes produce +1 in the constructed
unit-vector case, and reversed changes produce -1. All 193 fast tests pass (two opt-in tests deselected).
A read-only CPU evaluation of the completed full R0 balanced seed 0 in R1 format (32×64 windows) matches
every prior metric, gate outcome and random stream before/after the addition. No R1 training or source
selection occurs. Proof: `runs/overnight/common_base_20260905/av_change_panel_compatibility.json`.

**New transfer finding.** That preview also exposes a large video feature shift when the corrected
R1 window-end timestamps reach time embeddings trained under R0's last-sample reference: video rank
falls to 0.1825 and future advantage becomes -0.5017. This is a checkpoint-transfer problem to resolve
before R1 training, not a failure of the running R0 comparison or of this supplemental metric.


---

## 34. Preserve learned time functions when transferring R0 into R1

**Problem.** DDR §31 fixes physical timestamps, but loading R0's time coefficients unchanged shifts
its learned Fourier features. On a read-only R1-format preview of full balanced seed 0, video rank
falls from roughly 0.34 to 0.1825 and future advantage becomes -0.5017 before any R1 optimization.

**Decision.** At fresh, validated R0-to-R1 entry only, translate each first-coordinate Fourier linear
projection so f_new(t - delta) = f_old(t), where delta is one sample interval for that modality.
Rotate sine/cosine coefficient pairs and add the linear-coordinate contribution to the bias. Match
the runtime's fp32 frequencies, calculate the rotation in fp64, then store the original parameter
dtype. Apply this to both encoder and evidence-adapter time embeddings, online and EMA. Keep all
other parameters exact. Timestamps retain the shared physical window-end reference. The transfer
uses no random draws, inherits no optimizer state, and is never reapplied on target resume.
Record the source/target references, offsets and affected module paths in handoff provenance.
This supersedes DDR §32's literal time-weight copy with function-preserving coordinate conversion;
the source checkpoint and running R0 implementation remain unchanged.

**Validation.** All 195 fast tests pass (two opt-in tests deselected). Tests check translated Fourier
functions with other axes and RNG exact, all non-time source parameters exact, online/EMA evidence
and prediction functions before ABI rounding, an empty target optimizer, and exact interrupted R1
recovery without reopening the source. The ABI still rounds evidence to bf16; crossing a rounding
bin prevents bit-exact evidence after a mathematically equivalent coefficient transformation.
A real 32×64 held-out R1-format replay compares the original model on legacy timestamps with the
converted model on the same physical shared-clock windows. Every panel difference is below 2.2e-7;
video rank is 0.34235 and future advantage +0.04813. Random streams and non-time parameters are exact,
and the same audiovisual gate conditions fail. This is a read-only transfer proof, not R1 training
or covariance selection. Proof: `runs/overnight/common_base_20260905/time_transport_compatibility.json`.


---

## 35. Display R1's audiovisual gate measurements beside its collapse guards

**Decision.** The latest-run dashboard cards include video-to-audio retrieval margin,
audio-to-video retrieval margin and synchrony accuracy above chance whenever those measurements
exist. R0's available cards remain unchanged. This is a view of existing reconciled result fields;
no measurement, source ledger, objective or gate is altered.

**Validation.** A mixed-history R0/R1 fixture requires every available audiovisual card to contain
its source value and forbids invented action metrics. All 196 fast tests pass (two opt-in tests
deselected). Canonical rendering of the current real run history succeeds with structural-only
verification; browser verification remains unavailable. The first real R1 ledger is still pending.


---

## 36. Retain covariance after the complete matched R0 comparison and enter R1

**Decision.** Apply DDR §24's predeclared rule to the complete TAU population: retain covariance
weight 0.002. At 10,000 updates, batch 64, seeds 0/1, it improves both modalities' effective rank
at each seed and preserves every other R0 condition. Both selected sources pass all ten R0 checks;
both zero-covariance controls fail the two rank checks. No threshold was changed.

| Seed | Covariance | Video rank | Audio rank | R0 gate |
|---|---:|---:|---:|---|
| 0 | 0 | 0.104915 | 0.062662 | fail: both ranks |
| 0 | 0.002 | 0.341511 | 0.300382 | pass |
| 1 | 0 | 0.113408 | 0.059612 | fail: both ranks |
| 1 | 0.002 | 0.333895 | 0.271944 | pass |

All four runs use the same clean runtime revision `0bb78d5`, identical model/data/budget settings
except covariance, and exactly matching initial panels within each seed. Checkpoint metrics,
summary/metric/threshold ledgers, copied specs and recomputed gate outcomes reconcile. Proof:
`runs/overnight/common_base_20260905/full_comparison_validation.json`.

**Independent checks.** Both selected models also beat the teacher-current copy on the separate
all-eval-clip future audit: the recording-group bootstrap intervals are above zero for video and
audio at both seeds. Blank-input and fixed-position controls show content dependence alongside
substantial position structure, especially in audio. Frozen scene readouts include untrained
baselines; higher rank does not establish a scene-recognition advantage over the control.
Native track starts agree on the fixed 80-clip QA sample, but 8 Hz frame selection can lag native
exposure by up to 32.9 ms. These are supplementary controls, not modified gates or formal E1 results.

**R1 continuation.** `configs/dev/common_base_full_av.yaml` binds the two passing full source
checkpoints by SHA-256. Run 10,000 further updates per seed with the existing model, preprocessing,
loss weights, learning rate, EMA and thresholds; use the fresh optimizer/RNG and analytic clock
transport from DDR §34. The actual CUDA R1 interruption proof matches uninterrupted weights/EMA,
optimizer, all random streams, initial/final panels and training values exactly after recovery
from step 2 of 8. That tiny mechanics probe is not a curriculum result. Proof:
`runs/overnight/common_base_20260905/r1_cuda_recovery_compatibility.json`.
The final handoff implementation also exactly replays the original 250-step CUDA R0 trajectory.
All 196 fast tests pass (two opt-in tests deselected). Keep the absolute 09:25 Berlin stop time,
report incomplete work explicitly, and do not launch B0 from the overnight queue.

---

## 37. Preserve the R1 baseline and resolve timing before belief bootstrap

**Product and question.** The ABI-v2 evidence frontend prepares the modality-neutral recurrent
belief and the replaceable H1 interface through E1_common_base development. This comparison asks
whether the R1 continuation learns temporal audiovisual correspondence while preserving R0 evidence.
It does not test interface replaceability or freeze a formal reference.

**Completed result.** Both seeds complete 10,000 additional updates under clean runtime `e49c451`;
1/2 pass the unchanged seven audiovisual/collapse conditions. Source checkpoints are SHA-bound
and retain their own learned time functions through the analytic R0-to-R1 conversion. Final
checkpoints, raw panels, copied specs, summary/metric/threshold ledgers and recomputed gates reconcile.

| Seed | Video→audio margin | Audio→video margin | Official synchrony above chance (pp) | Video rank | Audio rank | R1 gate |
|---|---:|---:|---:|---:|---:|---|
| 0 | 0.144661 | 0.146318 | +1.758 | 0.335264 | 0.318711 | pass |
| 1 | 0.153440 | 0.157171 | -0.977 | 0.333221 | 0.298217 | fail: synchrony_accuracy_above_chance=-0.00976562 does not satisfy greater 0 |

**Independent timing.** The predeclared audit uses one fixed current/+2-second pair for every
one of the 3,645 held-out clips from 126 disjoint recording groups. Balanced two-time accuracy compares
both aligned pairs against their crossed assignment, with half-credit for ties. Ten thousand paired
recording-group resamples quantify uncertainty conditional on each fitted model.

| Seed | R0 above chance (pp) | R1 above chance (pp) | Paired change (pp) | Paired 95% group interval (pp) |
|---|---:|---:|---:|---|
| 0 | +0.453 | +1.413 | +0.960 | [-1.399, +3.341] |
| 1 | +0.261 | +0.892 | +0.631 | [-1.587, +2.848] |

Both paired intervals include zero; seed 1 also fails the original synchrony condition (-0.977 pp).
The evidence does not establish a reliable timing improvement in either seed. Independent future-head
advantages over copying the same-basis teacher remain positive in video and audio at both R1 seeds; this concern
is temporal audiovisual correspondence, not an observed loss of all future prediction.

**Exploratory diagnosis, added after seed 0.** On the same frozen windows, average retrieval margins
are compared against all eligible negatives from other clips, the same scene in another recording,
the same recording in another clip, and the same clip two seconds later. No label enters training.
Intervals resample query recording groups with negative candidate pools fixed. The margins shrink
as controls share context in both seeds. This is consistent with substantial scene/recording signal,
not a causal decomposition or proof of a unique limitation. These controls are explicitly post-outcome exploration.

**Metric counterexample.** Let current video/audio both be an orthogonal clip axis e; let shifted
video be 0.5e + sqrt(3)/2 f and shifted audio 0.5e - sqrt(3)/2 f. The three audiovisual statistics
pass (retrieval margin 1, synchrony +0.5), while balanced assignment is always wrong and change
alignment is -0.25. This constructed example isolates the audiovisual metric semantics; it is not
a trained checkpoint or a test of its separate rank/std guards. The current anchor need not enforce
agreement of the shifted aligned pair.

**Decision.** Preserve the measured checkpoints and unchanged gate results; defer B0 while the
independent timing question is unresolved. The next thin experiment uses a controlled audiovisual
source with a known common physical signal, then compares the present ranking with supervision
of both aligned times at matched data, initialization and budget. Keep source model/data binding
exact and scope any loss choice explicitly to R1. Validate shifted and constant-signal controls
before returning to the real-data comparison. The concrete plan is
`docs/common-base-next-experiment.md`; it is not implemented or launched in this overnight run.

**Validation and artifacts.** All 196 fast tests pass (two opt-in tests deselected); the measured
R0 CUDA replay and actual R1 interruption/recovery are exact. Each completed seed refreshes the
offline dashboard. The source-backed report uses the canonical portable renderer and receives
structural verification; browser layout/chart rendering are not verified. The E0 freeze gaps,
ABI-v1 measured control, Phase and Target remain unchanged. Main receipts and report:

- `runs/overnight/common_base_20260905/r1_comparison_validation.json`
- `runs/overnight/common_base_20260905/av_temporal_paired_seed{0,1}.json`
- `runs/overnight/common_base_20260905/av_context_exploratory_paired_seed{0,1}.json`
- `runs/overnight/common_base_20260905/audit_current_anchor_counterexample.json`
- `runs/overnight/common_base_20260905/report.html` and `report_evidence.json`

---

## 38. Validate known physical audiovisual correspondence before changing its objective

**Product and question.** E1_common_base's ABI-v2 frontend needs reliable temporal evidence before
belief bootstrap and H1 interface testing. The TAU R1 pair fails to establish that correspondence.
The next source control asks whether a shared physical signal is observable and learnable under
the existing frontend before testing current-anchor versus balanced-time supervision.

**Source decision.** Four visible emitters occupy separate image quadrants. Their eight continuous
x/y oscillator coordinates modulate eight audio carrier amplitudes. Both sensors observe the same
8 Hz state; amplitudes are held between those physical updates. Each carrier completes integer
cycles per frame. Independent group seeds fix phases, frequencies and background nuisance; group
IDs, split, absolute clip time and truth never enter the training batch. A separate audit sidecar
preserves physical coordinates. This is an intentionally controlled source, not natural audio/video.

**Interface and checks.** A data utility writes the existing normalized shard/manifest contract;
`RepresentationData` and model/ABI signatures remain unchanged. Repeated generation must reproduce
source bytes without touching global RNG, reject a changed definition at a published path, and keep
groups disjoint. Pixel centroids and waveform Fourier amplitudes must recover the common state.
The fixed raw audit requires at least 95% two-time matching, at most 5% after swapping audio times,
mean coordinate error below 0.03, and exactly chance on constant signals. No model success is inferred.

**Planned budget.** The new development config declares two 5,000-update R0 sources and the existing
unrelaxed R0 gates. Only passing sources can initialize a later matched 2,000-update R1 comparison
at both seeds. That next iteration changes only the R1 time-ranking objective; all model, source,
other loss, optimizer, temperature and audit settings remain matched. Known-source success is a
prerequisite for widening the objective comparison to TAU. This is not a formal freeze task.

**Source implementation outcome.** Both dynamic and constant versions contain 192 clips, with
64 train and 32 held-out recording groups (two clips each). The all-64-eval raw dynamic audit
scores 1.0 balanced assignment and 0.0 swapped assignment, coordinate MAE 0.00207896; the
constant control scores 0.5 in both assignments with tie fraction 1.0. Dynamic manifest SHA-256
`932f6eb649a3a6f243bf22b2aa6a5827defa3a1810b67dde7ace571d2fc357c5`; renderer SHA-256
`3bd43e064be5cd6fcb932b726b937ff9cd782fa6d76f32be361f3ba976a6900b`. Raw receipts live beside
each manifest as `manifest.raw_audit.json`; generated truth stays outside observations.

All 200 fast tests pass (two opt-in tests deselected). A separate 100-update CUDA mechanics
run writes `runs/dev/common_base_controlled_smoke/0/` and refreshes the dashboard with
structural verification only. Its rank fractions are video 0.032723 / audio 0.056412 and its
video temporal margin is negative; these three model conditions fail. This short run verifies
the training path and does not replace the predeclared 5,000-update source budget.
