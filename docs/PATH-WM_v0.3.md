# PATH-WM v0.3

## Modular JEPA World Models with Transition-Compatible Latents and Path-Space Planning

> **Architecture update (2026-09-04).** The ABI-v1/E1-a design below is retained as a measured control.
> Its single-frame visual-grid state and joint moving-target training were not promoted. The replacement
> reference candidate is [the common base architecture](common-base-architecture.md) with
> [ABI v2](abi/abi_v2.yaml): EMA-trained video/audio evidence, modality-neutral recurrent belief slots,
> separate action-conditioned dynamics, and gate-based planning curriculum.

**Status:** pre-registration draft, 2026-09-03. This is the consolidated research document: it integrates the v0.1 proposal, the v0.2 amendments, the 2026 literature sweep (`docs/literature-2026.md`), and the design decision register (`docs/design-decisions.md`). Sections marked *frozen* in §9 must not change after the corresponding experiment's first outcome-producing run; later changes are appended as numbered amendments, never edited in place.

---

## 0. Changes from v0.2

| Area | Change |
|---|---|
| Prior art | "Platonic Representation Hypothesis on World Models" (2608.23720) identified as the closest prior art to the encoder-stitching gate; the novelty claim is sharpened against it (§4 H1, §12). Sensorimotor World Models (2606.20104) and Delta-JEPA (2606.31232) adopted as the precedent for inverse-dynamics anchoring. |
| Design decisions | All open architecture questions are now fixed for v0.3 (§5): from-scratch encoder zoo plus one frozen pretrained foreign encoder; ABI spec v1 (64 grid + 1 global tokens, d = 192, shared positional convention); Markov predictor in W with delta output and action/Δt tokens; predict-then-correct updater; no label supervision of W; masked-token goal cost. |
| Training | SIGReg + inverse dynamics as the anti-collapse pair (LeWM / LeJEPA recipe); pre-registered horizon-and-loss curriculum (VLWM); optional curvature term for latent geometry (Temporal Straightening). |
| Uncertainty | MoP-JEPA's hard-assigned predictor mixture replaces the ad-hoc winner-take-all head; its conditional-mean-collapse argument is the justification. Valdi's multimodality-vs-control trade-off recorded as a rule. |
| Planner | Tempered SMC with rejuvenation (2604.21456) adopted as the reference realization of the annealed-SMC planner; GRASP (2602.00475) added as baseline and as the path-repair operator; TSMCTS as the parallel-search reference; transition-path-sampling shooting moves as the mutation family. |
| Hierarchy | HWM as v1 baseline with "Mind the Gap" (2607.12547) as the failure-mode reference; VLWM's variable-length supervision as the training recipe for the Δt-conditioned predictor. |
| Belief state | LeVJEPA-style block-causal encoder added as a fourth E4 condition (the encoder as its own updater). MemoBench-style disappear–reappear protocol adopted for E4 evaluation. |
| Evaluation | Compounding ratio (SkyJEPA), kinematic-vs-dynamic decomposition, Platonic-WM's normalized retention score, distance–state correlation, m-kNN, and POKEWORLD-style paired-intervention identifiability added to §10. |
| New sections | §14 Instrumentation and failure taxonomy; §15 Language and interaction; §20 Roadmap. |
| Infrastructure | LeWM code as the E1 starting point; stable-worldmodel as harness; repository layout in §16. |

---

## 1. Project goal

Build and experimentally validate a modular learned world-model architecture in which

$$
\boxed{\text{Perception}}
\rightarrow
\boxed{\text{Canonical World State}}
\rightarrow
\boxed{\text{Predictive Dynamics}}
\rightarrow
\boxed{\text{Path-Space Planning}}
$$

are separable components.

**Central hypothesis.** A sufficiently well-defined, transition-compatible latent world-state interface can decouple perception from dynamics and planning. Different encoder architectures and modalities should be replaceable through small adapters, while a shared action-conditioned predictor and planner continue to operate on the same learned world.

The interface is not a coordinate system owned by a reference encoder. It is defined operationally by the frozen components that consume it — the predictor $P$, the inverse-dynamics head $I$, and the belief updater $U_\psi$. A representation is on the interface if and only if those components behave correctly on it. Static similarity to any particular encoder is neither necessary nor sufficient.

**Second hypothesis.** Planning should treat trajectories as persistent structured objects in path space, rather than discarding them at every step and refitting a single distribution around freshly sampled action sequences.

Two connected research themes:

1. What is the correct learned representation and interface of a world?
2. How should a system search over possible futures represented by that world model?

---

## 2. Scope

The system is not intended to be a chatbot, a video generator, an LLM, a general-purpose robot foundation model, or a photorealistic simulator. It exists to answer architecture questions under controlled conditions.

Generation is separated from reasoning. Decoders are diagnostics only:

$$
D_{\text{text}}(\operatorname{stopgrad}(W_t)),\qquad D_{\text{img}}(\operatorname{stopgrad}(W_t)).
$$

No gradient from any decoder enters the world model. Language is an I/O modality (§15), never a training target for $W$.

### 2.1 Thesis scope

| Tier | Experiments |
|---|---|
| Core | E0 (environment), E1 (reference model), E2 (encoder stitching — the gate), E4 (belief state, part of the interface question), E6 (planner baselines), E7 (path-space planner ladder) |
| Side | E3 (registers; one-week budget on the E1 model) |
| Optional | E9-v1 (hierarchy replication, run with a stitched encoder) |
| Deferred | E5 (predictor architecture sweep), E8 (bidirectional planning), E9-v2 (learned macro-actions), E10 (multimodal modularity), ABI-1 (structured state) |

Deferred experiments remain specified so they can be pre-registered later without redesign. Nothing in the core depends on a deferred result.

---

## 3. Scientific question

**Primary.** Can perception, predictive dynamics, and planning be made independently replaceable by communicating through a transition-compatible learned world-state interface?

**Planning.** Can path-space inference — persistent diverse trajectories, structured mutations, revalidated reuse, learned guiding — outperform strong shooting-based, gradient-based and amortized planners (iCEM, SV-MPC, MPPI with a policy prior, GRASP, LeFlow-style flow priors) at an equal world-model rollout budget?

---

## 4. Core hypotheses

### H1 — Transition-compatible latent interface

Different encoders describe the same world in different native coordinates, $z_t^{(m)} = E_m(o_t^{(m)})$, and small adapters $A_m : z_t^{(m)} \rightarrow W_t$ map them into a shared world state. The requirement is not coordinate-wise equality but preserved dynamical structure. The test is transition compatibility,

$$
P\big(A_mE_m(o_t), a_t\big) \approx A_mE_m(o_{t+1}),
$$

and the strongest form is a foreign encoder driving an unchanged $P$.

**Degeneracy.** With $P$ frozen, $L_{\rm ABI} = d\big(P(AE(o_t), a_t), AE(o_{t+1})\big)$ alone is minimized by any adapter whose outputs lie where $P$ is nearly action-insensitive — a fixed point, a collapsed subspace, or off-manifold territory where the frozen $P$ acts as identity. Transition consistency is necessary but not sufficient. The interface is anchored by a second frozen consumer whose loss cannot be satisfied by action-insensitive latents, the inverse-dynamics head $I(W_t, W_{t+1}) \rightarrow a_t$, and by a counterfactual InfoNCE term (§6.4). $I$ forces adapted latents to vary with the action; $L_{\rm ABI}$ forces the frozen $P$'s action response to reproduce that variation; since $P$ cannot change, the only way to satisfy both is to place latents where $P$'s dynamics are the true dynamics. The 2026 precedent is Sensorimotor World Models (2606.20104), where a single inverse-dynamics regularizer both prevents collapse and aligns latents to action, and Delta-JEPA (2606.31232), which decodes actions from latent *differences* so the transition itself is constrained. Diagnostic: the action-sensitivity ratio $s(w) = \|P(w,a_1) - P(w,a_2)\| / \|w\|$, reported on adapted and native latents.

**Ownership of the interface.** If $P$ is trained with a single encoder, the interface is that encoder's coordinate system. Two alternatives are tested:

- **H1a — Multi-encoder interface formation.** Train $P$ and $I$ jointly with two encoders (separate adapters) plus the cross-stitched term $P(A_AE_A(o_t), a_t) \approx A_BE_B(o_{t+1})$ and its mirror; then stitch a third encoder. Hypothesis: the third encoder's adapter-size curve dominates the single-encoder curve — interface *formation*, not stitching.
- **H1b — ABI-0-rel.** The canonical state is a relative representation over a fixed anchor bank, $W_t[i] = \cos(z_t, z^{(i)})$, so the coordinate system is defined by anchors rather than by any encoder. If encoder geometries agree up to angle-preserving maps, stitching needs a near-trivial adapter by construction.

**Adapter budget.** Adapters are swept over {linear, 1-layer MLP, 2-layer MLP, 2-layer token transformer}, capped at 10% of the foreign encoder's parameters, and reported as planning success versus adapter parameters with "retrain $P$ on $E_B$ at matched compute" as ceiling and a random-initialized adapter as floor.

**Prior art and the precise claim.** Platonic-WM (2608.23720) shows that DINO-WM predictors trained on heterogeneous frozen encoders (DINOv2, SigLIP, MAE) converge toward similar internal geometry and that *predictor halves* of two trained world models can be spliced with a lightweight MLP while retaining planning success (retention score $\ge 0.7$ on PointMaze; strong depth sensitivity on Wall). It stitches predictor-to-predictor, trains the adapter against static targets from both trained endpoints, and excludes convolutional encoders as topologically incompatible. PATH-WM's claim is the complement: a foreign *encoder* — including a CNN and a frozen internet-pretrained ViT — in front of a predictor that never saw it, with the adapter anchored by frozen consumers rather than by static targets, evaluated as a curve over adapter size. As of this writing no published result shows a predictor planning with an encoder architecture it never trained with.

### H2 — World state and computational scratch space should be separate

A token representing a location, object, or observation should not double as arbitrary working memory (registers, Darcet et al. 2024). We distinguish $W_t$ from $R_{\rm enc}, R_{\rm pred}$: registers have no physical semantics, receive no prediction target, and $R_{\rm pred}$ is reset at every predictor call (Invariant 4). A register that persisted across rollout steps would be a hidden state by another name. The leakage probe (§14) measures whether registers are functioning as state. $R_{\rm plan}$ exists only once a learned proposal $q_\theta$ exists. Prior art for reset-per-call scratch tokens is pause tokens (Goyal et al. 2024); memory tokens that persist across recursion steps (2604.21999) are the *contrast* condition, not the proposal.

### H3 — Passive JEPA pretraining learns a world prior, not a causal world model

Passive learning constrains $p(W_{t+1}\mid W_{\le t})$; planning requires $p(W_{t+1}\mid W_t, \operatorname{do}(a_t))$. The project separates a passive stage from an interactive stage, as V-JEPA 2 does at scale. In E0 passive video is as cheap as interaction data, so the data-efficiency result transfers weakly; H3 is reported as a curve in E1 (action-conditioned data required for fixed planning success versus passive pretraining amount), a secondary result. Latent-action world models (2601.05230, 2510.26433, 2509.18428) are the reference for what passive video can supply — including a claim that latent-action pretraining can outperform ground-truth-action pretraining on LIBERO — and are the E9-v2 recipe for macro-actions.

### H4 — Predictor architecture need not match encoder architecture (deferred)

Transformer, looped shared-weight (LoopWM, 2606.18208, with a spectral-norm constraint for contractive rollouts), SSM (Mamba-3-class), and hybrid predictors are compared in E5 at matched parameters and rollout compute. No 2026 head-to-head on latent rollout stability exists; E5 is a real gap but nothing in the core depends on it.

### H5 — A world model should maintain a belief state

$W_t = U_\psi(W_{t-1}, E(o_t), a_{t-1})$ approximates a POMDP belief. The updater/predictor split is RSSM's posterior/prior split; the open question is whether it matters in a non-reconstructive JEPA model where nothing forces the recurrent state to retain unobserved content. The functional taxonomy of 2605.01694 (storage substrate × update rule × access pattern) frames E4; the four conditions are per-frame encoding, $K$-frame context window, the predict-then-correct updater (§5.5), and a block-causal encoder that is its own updater (LeVJEPA, 2608.27395, which trains block-causal attention at no accuracy cost). Since foreign encoders must feed $U_\psi$, the updater is part of the interface and E4 belongs to the Paper A program.

**Uncertainty $U$.** A deterministic JEPA predictor under stochastic branching collapses to the conditional mean — an invalid state between modes. MoP-JEPA (2607.05238) proves this and recovers one successor mode per hard-assigned predictor, turning the output into a *searchable transition set*, which is exactly the object the SMC planner branches over. Order of adoption: deterministic (E1/E2) → 5-ensemble for epistemic uncertainty (planner penalty $\beta\cdot$disagreement; calibration by Spearman correlation with true $H$-step error) → MoP-JEPA mixture for the stochastic E0 variant, behind a gate (multimodality enabled only if the deterministic model demonstrably lands between modes) → flow matching in feature space (2606.29059) only if mixtures are insufficient. Rule from Valdi (2607.00917): multimodality is built for planning diversity, never for per-step accuracy.

### H6 — Planning is path-space inference

A trajectory $\tau = (W_0, a_0, \ldots, a_{H-1}, W_H)$ is the object being sampled or optimized, with target $p^*(\tau) \propto p_0(\tau)\exp[-J(\tau)/\lambda]$ (Kappen 2005; Toussaint 2009; Levine 2018). The planner is specified as annealed sequential Monte Carlo over a trajectory population (§7.2); $\lambda$ is an annealing schedule; in the single-iteration Gaussian-proposal limit the planner reduces to MPPI. The closest existing realization is Tempered SMC with HMC rejuvenation and differentiable rollouts (2604.21456); GRASP (2602.00475) treats states as optimization variables with soft dynamics constraints and is both a baseline and the path-repair operator of §7.7; TSMCTS (2511.14220, ICML 2026) is the reference for GPU-parallel search and names the failure mode — path degeneracy — that the reservoir targets.

Three disanalogies with the rendering and chemistry sources are recorded so the analogies are not over-applied:

1. **Cost structure is inverted relative to ReSTIR.** Evaluating a rollout is the expensive operation. Reuse pays only if evaluation is made cheap (two-tier evaluator, §7.5) or stale costs are accepted under an explicit revalidation rule (§7.7).
2. **No cheap connection operator.** Latent dynamics has no operator joining $W_j^A$ to a suffix that started at $W_j^B$. State-level splice is deferred to the hierarchical planner, where the low-level planner is the connection operator. v0.3 uses action-sequence crossover.
3. **MLT samples a density; planning wants optimization plus mode coverage.** The SMC framing makes explicit what the resampling rule estimates at each temperature.

No 2026 work was found that imports ReSTIR/MIS spatiotemporal reuse into replanning; this remains an unclaimed contribution.

### H7 — Long-horizon planning requires temporal hierarchy

HWM (2604.03208) realizes inference-time hierarchical MPC on latent world models with large success gains and up to 4× lower planning cost; "Mind the Gap" (2607.12547) documents when hierarchy fails to help a frozen flat planner. E9-v1 replicates HWM's structure with fixed-length action chunks and a $\Delta t$-conditioned predictor trained with VLWM's variable-length supervision (2606.21775); learned macro-actions (VQ over chunks, latent-action models) are E9-v2.

---

## 5. Architecture — the v0.3 reference design

Every component below is the *reference implementation*: the simplest version that works, kept permanently as baseline and fallback. Alternatives enter only through the experiments of §9, one module at a time, against this reference.

### 5.1 Observation layer

At time $t$ the system receives an arbitrary subset $O_t = \{o_t^{\rm RGB}, o_t^{\rm alt}, o_t^{\rm occ}, o_t^{\rm proprio}, \dots\}$, each with its own encoder $E_m(o_t^m) = z_t^m$. Encoder output shape is unconstrained; the adapter normalizes it. Through E7 only $o^{\rm RGB}$ is used; the other modalities exist in E0 for E10.

### 5.2 ABI specification v1

The canonical world state is a fixed token layout shared by every encoder's adapter:

| Field | Value |
|---|---|
| Grid tokens | 64 (8×8 over 64 px), $d = 192$ |
| Global token | 1, $d = 192$ |
| Positional convention | fixed 2D embedding (RoPE-2D or sinusoidal), **identical for all adapters**; positions belong to the ABI, not to the encoder |
| Normalization | LayerNorm per token; SIGReg on the token batch (isotropic Gaussian) |
| Registers | $K_R \in \{0,4,8\}$, appended by the predictor per call, never stored in $W$ |
| Discreteness | none; a VQ head over $W$ exists only for E9-v2 |
| Action token | $a \in [-1,1]^2$ → MLP → $d$; time embedding $i$ within a chunk; discrete actions via embedding table |
| $\Delta t$ token | learned embedding of the chunk length |
| dtype | bf16 activations, fp32 SIGReg statistics |

A breaking change to this table is a new major ABI version and invalidates cross-version stitching results by design. Two further variants share the spec: **ABI-0-rel** ($W_t[i] = \cos(z_t, z^{(i)})$ over a fixed anchor bank of $K \in \{256, 1024\}$ observations, anchors per token position or shared) and **ABI-1** (structured: spatial, entity slots, global, memory — deferred). Identifiable Token Correspondence (2605.16457), a decoding-time copy-or-generate assignment over tokens, is the cheap ABI-1 precursor to test before entity slots.

**Geometry requirement.** Planning costs are latent distances, and Platonic-WM's Appendix F decomposes planning error into rollout error plus latent-geometry distortion. The correlation between $\|W_i - W_j\|$ and ground-truth state distance is tracked from E1 onward; if it is poor, Temporal Straightening's curvature regularizer (2603.12231) is added as an ABI term — the one geometry supervision allowed, because it constrains shape, not content.

**No label supervision of $W$.** Segmentation or state labels never enter $W$'s objective: a label target makes $W$ a label representation and trivializes H1 by pulling every encoder toward the same human coordinates. Labels are probe targets, viewer overlays, and goal masks. Content is shaped by architecture (grid, global, registers, gating), by action-anchored losses (§6.3–6.4), and by SIGReg. If grounding probes stay poor after E1, a *self-supervised* dense term (V-JEPA 2.1-style patch prediction) is added, never a label term.

### 5.3 Encoders

**From scratch on E0** (LeWM recipe, ~5M parameters each, identical data and steps): CNN (small ResNet-style, stride-8), ViT-S/8, conv–attention hybrid; later SSM and equivariant variants. $E_A$ = ViT. Foreign order for E2: CNN first (the case Platonic-WM excluded), hybrid second, and a **frozen internet-pretrained DINOv2-S** third on the external environments — the most striking form of the gate and a direct comparison with Platonic-WM's setup. Frozen pretrained features are appearance-coupled until inverse-dynamics fine-tuning (2606.07687); that is a condition, not an obstacle. For the first real-video stage after E7, $E_A$ is replaced by a LeVJEPA-pretrained block-causal encoder (95% token dropping; a ViT-Tiny trains in 12 h on a consumer GPU); the ABI does not change.

### 5.4 Adapters

$A_m : z^{(m)} \rightarrow W$ in the ABI layout. Sweep: linear; 1-layer MLP; 2-layer MLP; 2-layer token transformer; cap 10% of $E_m$'s parameters. Trained with §6.5 only; the encoder is either frozen or fine-tuned (two conditions).

### 5.5 World-state updater — predict-then-correct

$$
\tilde W_t = P(W_{t-1}, a_{t-1}, \Delta t = 1)\quad\text{(prior)},\qquad
W_t = \mathrm{LN}\Big(\tilde W_t + g_t \odot \mathrm{CA}\big(\tilde W_t \leftarrow \{A_mE_m(o_t^m)\}\big)\Big)\quad\text{(posterior)}.
$$

CA is one cross-attention block from prior tokens to adapted observation tokens (all modalities as one set, modality embedding and timestamp per token); $g_t$ is a per-token sigmoid gate; no observation ⇒ $W_t = \tilde W_t$. Trained with observation dropout (~30%) so the prior path is exercised. $U_\psi$ is one block plus a gate — small enough to freeze for stitching. Multi-encoder fusion (E10) is the same block over the union of adapted tokens with modality dropout; with one modality the E10 code path must reproduce E1 numbers exactly.

### 5.6 Predictor — Markov in $W$

$$
P_\phi(W_t, a_{t:t+k}, \Delta t, R) \rightarrow \hat W_{t+\Delta t}
$$

The predictor has **no context window**: history lives in the updater, and "the state is $W$" is what reservoir snapshots, revalidation, the critic and stitching rely on. The $K$-frame context predictor (DINO-WM) is E4's baseline only.

1. Inputs: 64 grid + 1 global tokens; $k$ action tokens; one $\Delta t$ token; $K_R$ fresh registers with learned initialization.
2. $L = 6$ pre-LN Transformer blocks, full attention among all tokens (one frame per call, no causal mask inside a call), 8 heads, MLP ratio 4.
3. Readout: the 65 state positions → linear → $\Delta W$; $\hat W_{t+\Delta t} = \mathrm{LN}(W_t + \Delta W)$. Delta form is identity by default, which helps permanence and one-step accuracy.
4. Registers discarded.
5. Multi-step: feed $\hat W$ back. Chunked prediction: all $k$ action tokens with $\Delta t = k$, trained with variable-length supervision (VLWM).
6. Stability: LayerNorm on the output; a spectral-norm constraint on the readout (LoopWM) is enabled only if the compounding ratio (§10) exceeds threshold.
7. ~10M parameters. Diffusion/flow *predictors* are not used (Valdi: no control gain per extra denoising step); flow priors belong in the planner as proposals.

### 5.7 Inverse-dynamics head

$I_\omega(W_t, W_{t+1}) \rightarrow \hat a_t$: an MLP over pooled token pairs (or a two-token cross-attention block). Trained jointly in E1; frozen with $P$ for stitching. Roles: anti-collapse during E1, interface anchor during E2, inverse proposal source for the planner.

### 5.8 Uncertainty

Deterministic in E1/E2. Epistemic: 5-ensemble, disagreement as $u$. Aleatoric (stochastic E0 only, gated): MoP-JEPA hard-assigned mixture of $K$ predictor heads with a hypothesis-weight classifier; evaluated by oracle-best-of-$K$ versus single-head error; the planner branches at high-entropy steps with a capped branching factor.

### 5.9 Trajectory critic (planner-side)

$C_\chi(W_0, a_{0:H-1}) \rightarrow \hat J$: an ensemble of three small sequence models trained online from verified rollouts. It never touches raw observations, never enters the world model's objective, and never decides an executed action without a verifying rollout (Invariant 10). Value-guided JEPA planning (2601.00844) is the precedent.

### 5.10 Debug decoders

$D_{\rm img}$ and $D_{\rm text}$ on $\operatorname{stopgrad}(W_t)$, trained separately (§15). Their quality is never a world-model objective.

### 5.11 Actions, goals and cost

Actions as in §5.2, with action-noise augmentation during training. Goals: (a) goal observation → $W_G$ through the same encoder/adapter; cost $J = \sum_{i \in M}\|\hat W_H[i] - W_G[i]\|^2$ over a token mask $M$ (tokens where goal and start differ, or a task mask) — the cheapest mitigation of latent-MSE ≠ task-distance; (b) the learned critic / goal-conditioned value; (c) text goals via a goal encoder (deferred, §15). Hard constraints $C$: a stopgrad probe from $W$ to collision/workspace violation, used as a penalty and verified against ground truth. Every planning result is reported under (a) and (b) separately: the cost function is a variable.

---

## 6. Training objectives and curriculum

$$
L = \lambda_{\rm reg} L_{\rm reg} + \lambda_a L_{\rm action} + \lambda_r L_{\rm rollout} + \lambda_i L_{\rm inverse} + \lambda_c L_{\rm cf} + \lambda_x L_{\rm cross} + \lambda_{\rm ABI} L_{\rm ABI} + \lambda_m L_{\rm masked} + \lambda_t L_{\rm temporal} \;(+\lambda_g L_{\rm geom}).
$$

### 6.1 Anti-collapse (mandatory)

$L_{\rm reg}$ = SIGReg (LeJEPA, 2511.08544; a single hyperparameter, provably collapse-free) applied to $W$ tokens, paired with $L_{\rm inverse}$. This is the LeWM / PLDM recipe; no EMA target encoder. An EMA target is a recorded fallback if regularized training is unstable. Sub-JEPA's subspace variant (2605.09241) is the first alternative if SIGReg's isotropy conflicts with the grid structure. All prediction targets carry $\operatorname{stopgrad}$.

### 6.2 Rollout loss (free-running)

$$
\hat W_{t+1} = P(W_t, a_t),\qquad \hat W_{t+k} = P(\hat W_{t+k-1}, a_{t+k-1}),\qquad
L_{\rm rollout} = \sum_{k=1}^{H_{\rm train}} \gamma^{k} d\big(\hat W_{t+k}, \operatorname{sg}(W_{t+k})\big),\ H_{\rm train} \ge H_{\rm plan}/2 .
$$

$L_{\rm action}$ is the $k=1$ term; $d$ is per-token MSE on normalized tokens; $\gamma = 0.9$ placeholder. Variable-length targets at $\Delta t \in \{1,\dots,k\}$ are drawn per batch (VLWM).

### 6.3 Inverse dynamics

$L_{\rm inverse} = \ell(I_\omega(W_t, W_{t+1}), a_t)$, MSE or cross-entropy. Its known side effect — over-emphasis of controllable features — is counterbalanced by $L_{\rm cf}$ and $L_{\rm rollout}$. Joint inverse-model + world-model training can collapse if the latent action channel is unconstrained (2510.26433); here actions are ground truth in E0, so this applies only to E9-v2.

### 6.4 Counterfactual discrimination

For $K$ paired interventions from an identical $X_0$:

$$
L_{\rm cf} = -\log \frac{\exp\big(-d(P(W_0, a_i), W_1^{(i)})/\kappa\big)}{\sum_{j=1}^{K} \exp\big(-d(P(W_0, a_i), W_1^{(j)})/\kappa\big)},
$$

with the metric **counterfactual discrimination accuracy** (fraction of $i$ with $\arg\min_j = i$) and an $H$-step variant. Collapse makes all $K$ distances equal, so this term cannot be satisfied by degenerate latents.

### 6.5 Adapter objective (E2)

With $P$, $I_\omega$ (and $U_\psi$) frozen:

$$
L_{A_B} = L_{\rm ABI}^{(H)} + \lambda_i L_{\rm inverse} + \lambda_c L_{\rm cf} + \lambda_{\rm reg} L_{\rm reg},
$$

where $L_{\rm ABI}^{(H)}$ is the free-running rollout loss with the frozen $P$ and adapter-produced targets. Static-matching conditions (1) and (2) of E2 add $d(A_BE_B(o), \operatorname{sg}(A_AE_A(o)))$ on paired observations; they are comparison conditions, since (1) requires paired observations and ties the interface to $E_A$'s coordinates and (2) constrains geometry rather than dynamics.

### 6.6 Multi-encoder interface formation (H1a)

$L_{\rm cross} = d(P(A_AE_A(o_t), a_t), \operatorname{sg}(A_BE_B(o_{t+1}))) + \text{mirror}$, added when $P$ is trained with two encoders.

### 6.7 Optional geometry term

$L_{\rm geom}$: Temporal Straightening's curvature regularizer on latent trajectories, enabled only if the distance–state correlation (§10) is below threshold after E1.

### 6.8 Passive losses

$L_{\rm masked}$ and $L_{\rm temporal}$ as in v0.1, used for the H3 curve only.

### 6.9 Curriculum (pre-registered)

What has evidence: horizon curricula (VLWM), loss staging with warmups (LeWM, PLDM), passive-then-interactive (V-JEPA 2). What does not: environment-complexity curricula for *training*; all E0 variants are mixed from step 0 and complexity is staged in the *evaluation ladder* (deterministic → stochastic → occlusion → compositional OOD).

- Stage 0 (first 10% of steps): $L_{\rm reg}$ + one-step $L_{\rm action}$ + $L_{\rm inverse}$.
- Stage 1: add free-running rollout; $H_{\rm train}$ grows $1 \rightarrow H$ linearly over the next 40%.
- Stage 2: add $L_{\rm cf}$.
- Stage 3 (stochastic variant only): mixture heads with hard assignment.
- Adapter training (E2): the same schedule compressed 4×.

---

## 7. Path-space planner

The planner receives $(W_t, G, P, V, C, \mathcal R)$ — belief, goal, predictor, optional value, hard constraints, persistent reservoir — and never raw observations.

### 7.1 Trajectory object

$$
\tau = \big(W_0,\ a_{0:H-1},\ \hat W_{1:H},\ J,\ \hat J_C,\ u,\ \text{source},\ \text{lineage},\ \text{validity},\ \text{age}\big)
$$

$J$ is the verified rollout cost (may be absent), $\hat J_C$ the critic estimate, $u$ ensemble disagreement along the path, *age* the number of replanning steps since the last verified rollout.

### 7.2 Annealed SMC loop

Per replanning step, for $n = 1,\dots,N$:

1. **Propose** from the mixture $q = \sum_j \alpha_j q_j$ (§7.3), including shifted reservoir members.
2. **Score.** Tier 1: every candidate gets $\hat J_C$. Tier 2: the top-$M$ by $\hat J_C$ (plus any candidate about to be executed) receive a verified rollout and $J$.
3. **Weight** $w_i \propto \exp(-J_i/\lambda_n)$, using $\hat J_C$ where $J$ is absent, with penalties for age, infeasibility and $u$.
4. **Resample** with diversity preservation: SVGD-style repulsion in action-sequence space or a $k$-DPP over a kernel on action sequences (latent-trajectory kernel as second option).
5. **Rejuvenate / mutate** survivors (§7.6).
6. **Anneal** $\lambda_{n+1} < \lambda_n$.

Degenerate settings recover the baselines: $N=1$, Gaussian $q$, no diversity, no critic is MPPI; $N>1$ with elite refit and no reservoir is CEM/iCEM. Tempered SMC (2604.21456) — annealed SMC with HMC rejuvenation and differentiable-rollout gradients, treating rollout randomness as auxiliary variables — is the reference realization; where $P$ is differentiable, its gradient-based rejuvenation is one mutation operator among ours. TSMCTS (2511.14220) is the reference for parallel search and for path-degeneracy mitigation.

### 7.3 Proposal mixture

Random exploration; iCEM-style colored noise around elites; MPPI-style perturbations of the incumbent; shifted reservoir members; learned policy prior; inverse-dynamics proposals from $I_\omega$ toward a subgoal; GRASP-style virtual-state optimization from a reservoir member; later a flow trajectory prior (LeFlow structure). The analogy is multiple importance sampling: no single proposal finds every class of useful path. Mixture weights $\alpha_j$ are fixed in E7; adaptive weighting by success attribution is a later ablation.

### 7.4 Persistent trajectory reservoir

$\mathcal R$ persists across replanning steps and must preserve distinct solution modes rather than the top-$K$ near-duplicates. Mode assignment uses action-sequence distance (L2 or DTW) first and latent-trajectory distance second; ground-truth homotopy signatures in E0 are for evaluation only. Guardrail: shifted incumbents and elite buffers exist in iCEM, diverse particle sets in SV-MPC; persistence alone is not the contribution (§12).

### 7.5 Two-tier evaluation

The critic scores all reservoir members, proposals and mutants; only the top-$M$ per iteration are verified by rollout; verified rollouts refresh the critic's buffer; critic-ensemble disagreement flags exploitation and forces verification. This restores the ReSTIR regime — cheap target evaluation, expensive verification. Predictor calls (tier 2) and critic calls (tier 1) are reported separately; equal-budget comparisons are on predictor calls; a planner that wins only by hiding compute in the critic is not a win.

### 7.6 Mutations

Local action mutation; segment resampling; prefix-preserving regeneration; **action-sequence crossover** (apply $\tau_B$'s suffix actions from $\tau_A$'s own state and re-roll, counting the $H-j$ calls); large-mode jump to a different reservoir cluster; **shooting moves** from transition-path sampling (perturb the state at an interior time, re-roll forward, keep the prefix — the TPS analogue of the prefix-preserving move, cf. 2504.18506 / 2506.01904); subgoal and macro-action mutations in the hierarchical planner only. State-level splice is deferred to §8.

### 7.7 Temporal reuse by partial revalidation, and path repair

After executing $a_0$ and observing $W_{t+1}^{\rm obs}$: shift each stored $\tau$; re-roll $k \in \{2,4\}$ steps from the new belief; drift $\delta = \tfrac{1}{k}\sum_{s\le k} d(\hat W_s^{\rm new}, \hat W_{s+1}^{\rm stored})$; keep the tail with $J \leftarrow J + \beta\delta$ and increment *age* if $\delta < \delta_{\max}$, else re-roll fully or discard. **Path repair:** instead of re-rolling, run a few GRASP iterations on the stored trajectory — virtual states with a soft dynamics penalty — from the new belief; this is the optimization form of the learned repair mapping of v0.1 and costs gradient evaluations rather than sequential rollouts. The claim under test: fewer predictor calls at equal success than full re-evaluation. AdaJEPA-style test-time adaptation of $P$ itself is complementary and not in v0.3.

### 7.8 Path guiding (later)

Verified successful paths train $q_\theta(\tau \mid W, G)$, one proposal source among many. Amortized trajectory priors (Diffuser; LeFlow, which generates a latent path between current and goal embeddings, decodes actions by inverse dynamics, and verifies by rollout) are proposal families inside this framework and baselines outside it.

---

## 8. Hierarchical planner

**v1 (E9-v1).** HWM-style inference-time hierarchy: action chunks $k \in \{4,8,16\}$; the $\Delta t$-conditioned predictor as the coarse model; the §7.2 loop over chunk sequences at the high level; the same loop at $\Delta t=1$ within the next chunk. "Mind the Gap" (2607.12547) supplies the conditions under which hierarchy does not help; those are E9's negative controls.

**v2 (E9-v2, deferred).** VQ-VAE over action chunks (codebook 64–256) or a latent-action model (2601.05230 recipe: inverse dynamics trained jointly with the world model, latent actions regularized by noise, sparsification or quantization) → discrete macro-actions → tree search at the top level.

**State splice lives here.** High-level subgoals are latent states; splicing two high-level paths at a shared subgoal is valid because the low-level planner performs the connection. E8's reachability model would later supply goal-side subpaths.

---

## 9. Experimental program

### E0 — Controlled causal world *(frozen once built)*

- 2D top-down rigid-body physics; $64\times64$ RGB; one agent, 2–4 movable objects, static walls arranged so that at least two homotopy classes of paths exist between spawn and goal regions; one container in which an object becomes invisible; continuous 2D force actions; friction and collisions.
- Variants: deterministic; stochastic (random perturbation forces, random object drift).
- Modalities generated: RGB; alternate render style (silhouette/edge); coarse occupancy grid; low-dimensional proprioception. Depth is deferred to a 3D variant for E10.
- Ground truth retained: full state, segmentation, homotopy signature (winding numbers around obstacle centroids).
- Paired interventions by state save/restore: $K \in \{4, 8\}$ actions from each $X_0$, multi-step paired sequences, and hidden-parameter variants (mass, friction behind visually identical objects) after the POKEWORLD protocol (2607.27017) for identifiability tests.
- Permanence protocol after MemoBench (2606.27537): object disappears (container / occluder) and reappears; an object-reappearance score is computed from probes on $W$.
- Data policy: exploration mixture (uniform random, Brownian, scripted goal-directed, and from E6 on replayed planner actions); coverage metric on ground-truth state; validation splits by object configuration; paired interventions stored as trees; ~1M transitions initially.
- External environments through stable-worldmodel (2605.21800): PushT, PointMaze, Wall, plus Rope and Granular for the Platonic-WM comparison.
- Engine: GPU-vectorized, deterministic save/restore and RNG, $\ge 10^4$ transitions/s.

### E1 — Reference JEPA world model *(frozen before E2)*

Start from the LeWM codebase. $E_A$ = ViT-S/8; $A_A$ linear; ABI v1; predictor §5.6; $I_\omega$ §5.7; recipe §6.1–6.4 with the §6.9 curriculum; CEM at a fixed rollout budget. Two configurations: **E1-a** per-frame ($W_t = A_AE_A(o_t)$), **E1-b** with the predict-then-correct updater. H3 curve as secondary result. Five seeds; $\sigma_{\rm pilot}$ is fixed here.

### E2 — Encoder stitching, the gate *(frozen)*

- Freeze $P$, $I_\omega$ (and $U_\psi$ for E1-b). Train adapters for the CNN, the hybrid, and the frozen DINOv2-S (external environments).
- Conditions: (1) static latent matching; (2) relative/anchor-geometry matching; (3) transition-only — expected to degenerate, kept as negative control; (4) transition + inverse + counterfactual (proposed); (5) condition 4 plus static matching.
- Adapter sweep, $E_B$ frozen versus fine-tuned, ceiling (retrain $P$ with $E_B$ at matched compute) and floor (random adapter).
- Variants: H1a (two-encoder-trained $P$, stitch a third); H1b (ABI-0-rel).
- Platonic-WM replication: their PointMaze/Wall protocol (CEM, $H=5$) and normalized retention score $\mathrm{RS} = \mathrm{SSR}/(S_A S_B)$ reported alongside ours, so that encoder-to-predictor stitching can be compared with their predictor-to-predictor numbers.
- Decisive metric: planning success with the identical planner at equal rollout budget, foreign versus native, as a curve over adapter size. Supporting: $\epsilon_{\rm transition}$, $s(w)$, counterfactual discrimination accuracy, m-kNN to native latents.

### E3 — Registers *(side, one week)*

$K_R \in \{0,4,8,16\}$ on the E1 model; matched-capacity control; leakage probe; token-norm outliers; grounding probes; planning success.

### E4 — Persistent belief state *(core, Paper A)*

Occlusion tasks (object placed in container → invisible → a later action depends on it) and the MemoBench-style reappearance protocol; memory horizon $\{1,4,8,16,32\}$. Four conditions: per-frame; $K$-frame context; predict-then-correct updater; block-causal encoder (LeVJEPA-style, the encoder as its own updater). Then repeat E2 on the recurrent and causal configurations with foreign adapters feeding the frozen updater.

### E5 — Predictor architecture *(deferred)*

Transformer; looped shared-weight with spectral constraint (LoopWM); SSM; hybrid; deterministic versus mixture. Matched parameters, training and rollout compute. This is an open gap in the 2026 literature and may be worth pulling forward if E1's compounding ratio is poor.

### E6 — Planner baseline suite *(frozen)*

One frozen world model. Baselines: CEM; warm-start CEM; MPPI; iCEM; SV-MPC; MPPI + learned policy prior (TD-MPC2 style); GRASP; Tempered SMC in its published form; flow trajectory prior + fixed-budget verification (LeFlow structure); GC-IDM-style amortized inverse-dynamics planning (2605.08732); MCTS/TSMCTS where the action space permits. Three predictor-call budgets spanning one order of magnitude; success-versus-budget curves, never single points.

### E7 — Path-space planner ablation ladder *(frozen)*

From the MPPI limit, add one feature at a time: (a) annealed iterations; (b) reservoir; (c) diversity-preserving resampling; (d) mutations including crossover and shooting moves; (e) proposal mixture; (f) two-tier critic; (g) partial-revalidation reuse and GRASP repair; (h) path guiding. Then leave-one-out. Report predictor calls, critic calls, wall-clock, success, homotopy-class coverage, recovery after disturbance.

### E8 — Bidirectional planning *(deferred)*

Reachability/preimage model; goal-side subpath growth; connection at the subgoal level of §8.

### E9 — Hierarchical temporal planning

**v1:** replicate HWM on E0 and the external environments; flat versus chunked at matched predictor calls; $\Delta t$-conditioned versus per-scale predictors; the "Mind the Gap" negative controls; and the cross-cutting question — does the hierarchy survive a stitched foreign encoder? **v2 (deferred):** macro-actions and top-level tree search.

### E10 — Multimodal modularity *(deferred)*

RGB → +alt render → +occupancy → +proprioception with modality dropout and asynchronous observations; the strong result is a new modality encoder improving belief quality without retraining $P$ or the planner.

---

## 10. Evaluation

### Representation
- Transition error $\epsilon_{\rm transition}$, one-step and $H$-step.
- Action-sensitivity ratio $s(w)$, adapted versus native.
- Counterfactual discrimination accuracy, native and through adapters.
- Register leakage ratio (probe accuracy from $R_{\rm pred}$ / probe accuracy from $W$).
- Distance–state correlation (latent distance versus ground-truth state distance) and m-kNN between $W$ and ground-truth neighborhoods; m-kNN between native and adapted latents (Platonic-WM protocol).
- Grounding probes (positions, velocities, identities, hidden container contents), per-token spatial probes, hidden-state memory probes, cross-encoder stitching success and normalized retention score.

Static CKA and cosine similarity are supporting evidence only (Invariant 7).

### World model
One-step error; multi-step degradation; **compounding ratio** CR$_k$ = open-loop error / teacher-forced error at horizon $k$ (SkyJEPA); kinematic-versus-dynamic error decomposition; per-object error; action sensitivity; counterfactual discrimination; calibration (Spearman correlation of ensemble disagreement with true $H$-step error); oracle-best-of-$K$ versus single-head error for mixtures; object-reappearance score under the permanence protocol; hidden-parameter identifiability under paired interventions; out-of-distribution composition.

### Planner
Task success; cumulative cost; predictor calls; critic calls; wall-clock; complete versus early-terminated trajectories; homotopy-class coverage; recovery after disturbance; constraint violations; long-horizon success; model error encountered along executed paths. All comparisons at equal predictor-call budgets with curves over at least three budgets, reported separately for observation-goal cost and learned-value cost.

---

## 11. Falsification criteria

Numeric thresholds are placeholders fixed from $\sigma_{\rm pilot}$ (five-seed E1) before the corresponding experiment is frozen.

| Hypothesis | Succeeds if | Fails if |
|---|---|---|
| World Latent ABI (H1) | a foreign encoder retains $\ge 90\%$ of native planning success at equal rollout budget with an adapter $\le 10\%$ of encoder parameters | $< 70\%$ even at $25\%$ of encoder parameters |
| Interface formation (H1a) | the third-encoder adapter curve under two-encoder training dominates the single-encoder curve at every budget by $> 2\sigma_{\rm pilot}$ | no budget shows a difference beyond $1\sigma_{\rm pilot}$ |
| ABI-0-rel (H1b) | a linear adapter reaches the H1 threshold | it needs the largest adapter to match ABI-0 |
| Registers (H2) | registers beat matched-capacity controls by $> 2\sigma_{\rm pilot}$ with leakage ratio $< 0.5$ | controls match within $1\sigma_{\rm pilot}$, or leakage $> 0.8$ |
| Belief state (H5) | the updater or the causal encoder beats the $K$-frame window at some memory horizon by $> 2\sigma_{\rm pilot}$ | the window matches at all horizons |
| Mixture predictor (U) | oracle-best-of-$K$ error beats single-head by $> 2\sigma_{\rm pilot}$ *and* planning success improves | modes are covered but planning does not improve (Valdi outcome) |
| Path-space planner (H6) | beats the best of {iCEM, SV-MPC, MPPI+prior, GRASP, Tempered SMC, flow prior} by $\ge 2\sigma_{\rm pilot}$ at equal predictor calls on $\ge 3$ of 4 environments | does not beat iCEM at any budget |
| Two-tier evaluation | the critic cuts predictor calls at equal success by $\ge 25\%$ | no reduction, or wins only if critic cost is counted as free |
| Temporal reuse / repair | partial revalidation or GRASP repair cuts predictor calls at equal success | no reduction, or success drops |
| Hierarchy (H7) | advantage survives matching total predictor calls | advantage disappears after matching |
| Passive JEPA (H3) | pretraining reduces required action-conditioned data by $\ge 2\times$ | $< 2\times$ (weakened) |

Negative outcomes are reported as results.

---

## 12. Novelty boundaries

**Existing ingredients** (not claimed): JEPA family (I-JEPA, V-JEPA, V-JEPA 2/2.1, LeJEPA/SIGReg, LeVJEPA, LeWM, Sub-JEPA, Delta-JEPA); decoder-free latent world models with planning (TD-MPC2, DINO-WM, PLDM); frozen-encoder world models with adapters (Reconstruction or Semantics?); inverse-dynamics anchoring (Sensorimotor World Models; What Makes Latents Action-Relevant); RSSM belief states; belief-state taxonomy (Latent State Design); memory mechanisms (Tensor Memory, Identifiable Token Correspondence); stochastic JEPA variants (MoP-JEPA, VJEPA, Var-JEPA, UWM-JEPA, flow matching in feature space, Valdi); stability mechanisms (LoopWM, LaWM, VLWM); planners (CEM, MPPI, iCEM, SV-MPC, GRASP, Tempered SMC, TSMCTS, PMCTS, Monte Carlo Tree Diffusion, Planning as Descent); amortized priors (Diffuser, LeFlow, GC-IDM); hierarchical latent MPC (HWM); test-time adaptation (AdaJEPA); latent-action world models (LAWM, Olaf-World, Co-Evolving LAM, LAC-WM, DreamDojo); cross-embodiment alignment (Demo-JEPA, DyPES-VLA, UniT); representation convergence and stitching (Platonic-WM; relative representations; model stitching); registers and pause tokens; transition-path sampling; MLT, MIS, path guiding, ReSTIR; benchmarks (stable-worldmodel, OGBench, MemoBench, POKEWORLD, WorldTest).

**Contribution A — Transition-compatible interface.** Functional, planning-level encoder interchangeability anchored by frozen consumers $(P, I, U_\psi)$, across encoder *architectures* including CNN and frozen pretrained ViT; multi-encoder interface formation; relative-anchor ABI. Complement of Platonic-WM's predictor-to-predictor stitching. Unclaimed as of 2026-09.

**Contribution B (reduced) — Register leakage methodology and belief state in JEPA.** Likely a section of Paper A; a null result is acceptable.

**Contribution C (narrowed) — Persistent path-space search with two-tier evaluation.** Diversity-preserving, mutation-based, revalidated trajectory populations with a learned critic and GRASP-style repair, shown to beat amortized priors, gradient planners and warm-start/buffer MPC at equal predictor calls; the first import of ReSTIR/MIS-style spatiotemporal reuse into replanning. Persistence alone is explicitly not the claim.

**Contribution D (deferred) — Unified multiscale latent planning.**

---

## 13. Publication path

- **Paper A** — E0, E1, E2, E4 (E3 as a section): transition-compatible world representations, interface formation, belief-state stitching, with the Platonic-WM head-to-head.
- **Paper C** — E6, E7, optionally E9-v1: path-space planning with learned world models against the 2026 baseline suite.
- Order A, then C; C depends on A only for the frozen world model. Paper B folded into A; Paper D deferred with E5, E8, E9-v2, E10.

---

## 14. Instrumentation and failure taxonomy

The instrument panel is built before E1 finishes, logged on every run, and rendered in the viewer (§15). It is the debugger.

**Probes** (linear and one-layer attentive, stopgrad): from $W$ and from $R$ to object positions, velocities, identities, hidden container contents, agent pose; per grid token to local occupancy.
**Debug decoders** (stopgrad): image for humans; text for hidden-state questions.
**Geometry:** m-kNN, distance–state correlation, token-norm histograms, PCA of grid tokens, predictor attention maps.
**Dynamics:** the §10 world-model metrics, per step.
**Planner:** homotopy coverage, reservoir age and drift histograms, critic-versus-verified scatter, call accounting, success-versus-budget.

| Symptom | Metric | Likely cause | First fix |
|---|---|---|---|
| Full collapse | $W$ variance → 0; probes at chance | regularizer weight, LR | SIGReg weight, warmup |
| Dimensional collapse | effective rank low | projector / normalization | per-token vs global reg; Sub-JEPA |
| Appearance coupling | appearance probes high, dynamics probes low, counterfactual acc ≈ 1/K | no action anchoring | raise $\lambda_i, \lambda_c$ |
| Action insensitivity | $s(w)$ small; predictor ≈ identity | identity shortcut in delta form | $L_{\rm cf}$; action-dropout ablation |
| Compounding | CR > 1.5 before $H_{\rm plan}$ | no horizon curriculum / unnormalized output | curriculum, LN, spectral constraint, ensemble truncation |
| Register leakage | $R$-probe ≈ $W$-probe | $W$ too small | larger $W$; noise on $R$ |
| Geometry distortion | one-step loss good, planning bad, distance correlation low | latent metric ≠ task metric | masked cost, $L_{\rm geom}$, learned value |
| Mode averaging | deterministic prediction between modes | stochastic branching | mixture heads |
| Adapter escape (E2) | $s(w)$ drops on adapted latents; inverse loss high | transition-only loss | $\lambda_i, \lambda_c$; inspect $P$'s manifold |
| Critic exploitation | $\hat J_C \ll$ verified $J$ on mutants | critic optimism | ensemble critic, verify more, penalize disagreement |
| Planner mode collapse | homotopy coverage = 1 | no diversity pressure | diversity term; slower anneal |
| Stale reservoir | high ages, large drift | $\delta_{\max}$ too loose | tighten $\delta_{\max}$; repair instead of keep |
| Physics exploit | success with constraint violations | simulator artifact | fix engine; count violations as failures |

---

## 15. Language and interaction

**Principle.** Language is an I/O modality; $W$ never receives language gradients.

- **L0 (with E0):** templated captions from ground-truth state ("red box inside the container; agent at (3,5) moving left") train $D_{\rm text}(\operatorname{stopgrad}(W))$, a small Transformer decoder. Hidden-state questions ("what is in the container?") are the belief-state probe in natural language.
- **L1:** a small LLM behind a prefix adapter ($W$ tokens → prefix embeddings), still stopgrad; free-form questions; plan explanations by decoding the reservoir's predicted $\hat W_{1:H}$.
- **L2 (E10):** text as input — a goal encoder plus adapter to a goal latent or goal mask; language-specified constraints.

**Interaction.** (1) A Python API over the module contracts (§16), notebook-first; every experiment is a script on the same API. (2) A real-time viewer, built before E2: environment; debug-image decode of $W$; probe overlays; reservoir trajectories drawn on the world, colored by homotopy class with age and cost; critic-versus-verified scatter; attention maps; the instrument panel. Controls: click to set a goal; drag-edit a trajectory and re-verify; inject a perturbation; fork/rewind on the snapshot runtime; step the planner one SMC iteration at a time. A lightweight web client first; a Vulkan client once the panels stabilize. (3) The text console (L1). (4) The agent loop: closed-loop MPC with full logging; every executed action has a verified rollout.

---

## 16. Module contracts and repository layout

### 16.1 Contracts

| Module | Signature | Frozen for stitching |
|---|---|---|
| Encoder $E_m$ | $o \rightarrow z$ (any shape) | no |
| Adapter $A_m$ | $z \rightarrow W$ (ABI v1 layout) | no (the object under training) |
| Updater $U_\psi$ | $(W, \{W^{(m)}_{\rm obs}\}, a) \rightarrow W$ | yes |
| Predictor $P_\phi$ | $(W, a_{t:t+k}, \Delta t, R) \rightarrow W$ | yes |
| Inverse $I_\omega$ | $(W, W') \rightarrow a$ | yes |
| Critic $C_\chi$ | $(W, a_{0:H-1}) \rightarrow \hat J$ | planner-side |
| Planner | $(W, G, P, C, \mathcal R) \rightarrow (a_{0:H-1}, \mathcal R')$ | — |
| Debug decoders | $W \rightarrow$ text / image | never trained into $W$ |

**Conformance tests** any implementation must pass before entering an experiment: shape and dtype against the ABI spec; $s(w) \ge s_{\min}$ on a fixed probe set; transition error $\le \epsilon_{\max}$ on the same set; for adapters the §6.5 losses on held-out data; for planners valid actions within budget. **Exchange procedure:** implement → conformance → E2-style comparison against the frozen reference at equal budget → results ledger. **stable-worldmodel mapping:** `encode` = $A \circ E$ (+ $U_\psi$), `predict` = $P$, `rollout` = the loop, `criterion` = the cost, `get_cost` = the accounted call count.

**Predesign versus evolve.** Contracts, the ABI spec, the instrument panel, the environment, the evaluation protocol and the thresholds are predesigned. Every implementation evolves one module per experiment against the frozen reference. Interfaces change only when two independent implementations both need the change.

### 16.2 Repository layout

```
path-wm/
├── README.md
├── docs/
│   ├── PATH-WM_v0.3.md          # this document
│   ├── design-decisions.md      # decision register (options, defaults, tests)
│   ├── literature-2026.md       # annotated bibliography by problem area
│   ├── preregistration.md       # frozen spec hashes and thresholds
│   └── abi/abi_v1.yaml          # the ABI specification
├── envs/
│   ├── causal_world/            # E0: physics, renderers, interventions, permanence protocol, homotopy signatures
│   └── external/                # stable-worldmodel harness: PushT, PointMaze, Wall, Rope, Granular
├── encoders/                    # cnn/, vit/, hybrid/, pretrained/ (frozen DINOv2-S, LeVJEPA)
│   └── adapters/                # linear, mlp, token_transformer; budget accounting
├── world_state/
│   ├── abi.py                   # ABI-0, ABI-0-rel; ABI-1 deferred
│   ├── updater.py               # predict-then-correct
│   ├── registers.py             # reset-per-call enforced here
│   └── inverse.py               # I_omega
├── predictors/
│   ├── transformer.py           # §5.6 reference, Δt-conditioned
│   ├── ensemble.py  mixture.py  # epistemic ensemble; MoP-JEPA hard-assigned mixture
│   └── looped.py  ssm.py        # E5, deferred
├── planners/
│   ├── smc.py                   # §7.2 loop; MPPI/CEM/iCEM as degenerate configs
│   ├── trajectories.py  reservoir.py  proposals.py  mutations.py
│   ├── critic.py  revalidation.py  repair.py     # two-tier evaluation; §7.7; GRASP-style repair
│   ├── baselines/               # icem, svmpc, policy_prior, grasp, tempered_smc, flow_prior, gcidm, tsmcts
│   └── hierarchical.py
├── losses/                      # sigreg.py, rollout.py, inverse.py, counterfactual.py, cross.py, geom.py
├── evaluation/                  # probes.py, homotopy.py, calibration.py, compounding.py, budget.py, retention.py
├── decoders/                    # text_debug.py, image_debug.py
├── viewer/                      # web client; panels mirror evaluation/
├── experiments/                 # one YAML per experiment; frozen specs listed by hash in docs/preregistration.md
└── tests/conformance/           # contract tests per module
```

Experiment specifications are immutable once frozen; the freeze is the commit of the spec's hash into `docs/preregistration.md`.

---

## 17. Project invariants

1. Decoders are not part of cognition.
2. The planner never consumes raw sensory input.
3. *(Hypothesis, not invariant.)* The predictor should not depend on a particular encoder implementation; H1 tests this.
4. Scratch registers are not canonical world state; $R_{\rm pred}$ resets at every predictor call.
5. Planning-relevant uncertainty must not be silently collapsed.
6. Action-conditioned evidence is required for causal or control claims.
7. Representation similarity never establishes interoperability; interoperability is established by frozen consumers behaving correctly.
8. Planner comparisons report predictor calls and critic calls separately and use budget curves.
9. Previous trajectories are information; they are revalidated or repaired, never blindly reused or discarded.
10. Every executed action sequence has a verified rollout; the critic alone never decides an action.
11. Labels never enter $W$'s objective.
12. No architectural component is protected from falsification.

---

## 18. Immediate first research target

$$
\boxed{\textbf{Can a predictor trained with one encoder plan using another?}}
$$

1. Build E0 (deterministic variant, RGB only, paired interventions, permanence protocol).
2. Build the instrument panel and the conformance tests.
3. Train $E_A + A_A + P + I_\omega$ (E1-a) from the LeWM codebase with §6.1–6.4 and the §6.9 curriculum. Freeze $P$ and $I_\omega$. Fix $\sigma_{\rm pilot}$.
4. Train $A_B$ for the CNN and the hybrid under E2 conditions (1)–(5) across the adapter sweep, with ceiling and floor; then the frozen DINOv2-S on PushT/PointMaze/Wall with the Platonic-WM protocol alongside.
5. Run the identical CEM planner at equal rollout budget: native versus foreign.
6. Run H1a and H1b.
7. Repeat on E1-b and on the block-causal configuration.

Decide by §11. If condition (3) fails and (4) succeeds, the anchoring argument of H1 is confirmed. If (4) fails at every adapter size, the interface needs more structure (ABI-0-rel, then ABI-1) before any investment in path-space planning.

---

## 19. Long-term target

$$
\boxed{
\text{multimodal observations}
\rightarrow
\text{persistent probabilistic world state}
\rightarrow
\text{action-conditioned latent dynamics}
\rightarrow
\text{hierarchical path-space inference}
}
$$

with independent perception, world-state maintenance, predictive dynamics, planning, memory, and diagnostic decoding. The objective is not another world model but a determination of whether a modular internal model of reality with a stable transition-level interface can be the substrate on which perception, prediction and planning evolve independently.

---

## 20. Roadmap

| Phase | Deliverables | Gate |
|---|---|---|
| 1 | ABI v1 spec; contracts and conformance tests; E0 engine with save/restore, interventions, permanence protocol; data policy; instrument panel; viewer v0 | E0 frozen; $\ge 10^4$ transitions/s |
| 2 | E1-a and E1-b reference models (five seeds); $\sigma_{\rm pilot}$; thresholds committed to `preregistration.md` | E1 frozen |
| 3 | E2 gate: CNN, hybrid, frozen DINOv2-S; conditions (1)–(5); H1a; H1b; Platonic-WM head-to-head | §11 decision |
| 4 | E4 (four belief-state conditions) and E2 on recurrent/causal configurations; E3 side experiment | Paper A draft |
| 5 | E6 baseline suite on the frozen world model; E7 ladder; viewer v1 with planner panels | Paper C draft |
| 6 | E9-v1 with a stitched encoder; decide on E5 / E8 / E9-v2 / E10 | Paper D scoping |

---

## References

*JEPA, collapse, world-model training*
- Assran et al. 2023, I-JEPA. Bardes et al. 2024, V-JEPA. Assran et al. 2025, V-JEPA 2 (2506.09985). Mur-Labadia et al. 2026, V-JEPA 2.1 (2603.14482).
- Balestriero & LeCun 2025, LeJEPA / SIGReg (2511.08544). Bardes, Ponce & LeCun 2022, VICReg.
- Kuhn, Maes, Serra, Le Lidec, LeCun, Balestriero, Buettner 2026, LeVJEPA (2608.27395).
- Maes, Le Lidec, Scieur, LeCun & Balestriero 2026, LeWorldModel (2603.19312).
- Sub-JEPA 2026 (2605.09241). Delta-JEPA 2026 (2606.31232). Ivashkov et al. 2026, Sensorimotor World Models (2606.20104). What Makes Video World Model Latents Action-Relevant 2026 (2606.07687). Klindt, LeCun & Balestriero 2026, When does LeJEPA learn a world model? (2605.26379). Terver et al. 2026, Energy-Based JEPA library (2602.03604).
- Huang 2026, VJEPA variational (2601.14354). Var-JEPA 2026 (2603.20111). MoP-JEPA 2026 (2607.05238). UWM-JEPA 2026 (2605.25313). Flow Matching in Feature Space 2026 (2606.29059). Lindenberg & Chitta 2026, Valdi (2607.00917).

*Latent world models and planning*
- Hafner et al. 2019–2023, PlaNet / Dreamer v1–v3 (RSSM). Hansen, Su & Wang 2024, TD-MPC2. Schrittwieser et al. 2020, MuZero.
- Zhou, Pan, LeCun & Pinto 2024, DINO-WM (2411.04983). Sobal et al. 2025, PLDM.
- Li, Ma, Xiong, Chen & Zhang 2026, Platonic Representation Hypothesis on World Models (2608.23720).
- Reconstruction or Semantics? 2026 (2605.06388). Temporal Straightening for Latent Planning 2026 (2603.12231). Learning Invariant Visual Representations for Planning 2026 (2602.18639). NE-Dreamer 2026 (2603.02765).
- HWM 2026, Hierarchical Planning with Latent World Models (2604.03208). Mind the Gap 2026 (2607.12547). VLWM 2026 (2606.21775). LaWM 2026 (2605.08279). Looped World Models 2026 (2606.18208). SkyJEPA 2026 (2606.23444). Imagined Rollouts are Kinematic, Not Dynamic 2026. FF-JEPA 2026 (2606.09311).
- Huang et al. 2026, LeFlow (2608.24855). Nguyen, Xu & Huang 2026, Latent Geometry Beyond Search / GC-IDM (2605.08732). Wang, Bounou, LeCun & Ren 2026, AdaJEPA (2606.32026). Value-guided JEPA planning 2026 (2601.00844). Planning as Descent 2025/26 (2512.17846).
- Psenka, Rabbat, Krishnapriyan, LeCun & Bar 2026, GRASP (2602.00475). Tempered SMC for trajectory and policy optimization 2026 (2604.21456). Oren, de Vries, van der Vaart, Spaan & Böhmer 2026, TSMCTS (2511.14220). PMCTS 2026 (2605.08982). Yoon et al. 2025/26, Monte Carlo Tree Diffusion (2502.07202). Janner et al. 2022, Diffuser.
- Williams et al. 2017, MPPI. Pinneri et al. 2020, iCEM. Lambert et al. 2020, SV-MPC. Kappen 2005; Toussaint 2009; Levine 2018. Del Moral, Doucet & Jasra 2006, SMC samplers. Liu & Wang 2016, SVGD. Kulesza & Taskar 2012, DPPs.

*Belief state, memory, structure*
- Latent State Design for World Models under Sufficiency Constraints 2026 (2605.01694). Identifiable Token Correspondence 2026 (2605.16457). Tensor Memory 2026 (2605.27686). Persistent Computational State 2026 (2607.21686). Permanence Fields 2026 (2606.28455). Singh et al. 2021, Structured World Belief.
- Darcet et al. 2024, registers. Goyal et al. 2024, pause tokens. Universal Transformers Need Memory 2026 (2604.21999).

*Latent actions, cross-embodiment*
- Learning Latent Action World Models In The Wild 2026 (2601.05230). Co-Evolving Latent Action World Models 2025 (2510.26433). LAWM 2025 (2509.18428). Olaf-World 2026 (2602.10104). LAC-WM, ICML 2026. DeFI, ICLR 2026 (2604.16391). Latent Particle World Models 2026 (2603.04553). Demo-JEPA 2026 (2605.20811). DyPES-VLA 2026 (2608.06374). Gao et al. 2025, AdaWorld (2503.18938).

*Representation alignment*
- Lenc & Vedaldi 2015; Bansal, Nakkiran & Barak 2021 — model stitching. Moschella et al. 2023 — relative representations. Huh et al. 2024 — Platonic Representation Hypothesis. Rupprecht et al. 2017 — multiple-hypothesis prediction.

*Benchmarks and harnesses*
- stable-worldmodel 2026 (2605.21800). Park et al. 2025, OGBench. MemoBench 2026 (2606.27537). POKEWORLD 2026 (2607.27017). WorldTest / AutumnBench 2025 (2510.19788). WorldBench 2026 (2601.21282). LIBERO-Mem 2025 (2511.11478). A Definition and Roadmap for World Models 2026 (2607.06401).

*Path-space sampling in rendering and chemistry*
- Veach & Guibas 1995 (MIS), 1997 (MLT). Lafortune & Willems 1993; Veach & Guibas 1994 (BDPT). Müller et al. 2017, practical path guiding. Bitterli et al. 2020, ReSTIR. Sawhney et al. 2022, MCMC-mutated ReSTIR. Zeng et al. 2025, ReSTIR path guiding.
- Bolhuis, Chandler, Dellago & Geissler 2002, transition path sampling. Raja et al. 2025, TPS with generative models by action minimization (2504.18506). Machine-learned sampling of conditioned path measures 2025 (2506.01904).
- Bhattacharya, Likhachev & Kumar 2012, homotopy constraints in path planning.
