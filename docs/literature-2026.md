# PATH-WM — Annotated 2026 literature by problem area

Companion to `PATH-WM_v0.3.md`. Each entry: arXiv ID or venue, what it solves, mechanism, and bearing on PATH-WM.
Flags: **[read]** — primary abstract or text was inspected directly during the sweep; **[sweep]** — surfaced by the automated literature sweep and its cited sources; verify against the primary PDF before citing in a paper.

## A. Encoder interchangeability, stitching, representation convergence (→ H1, E2)

- **Platonic Representation Hypothesis on World Models** — 2608.23720, Li, Ma, Xiong, Chen, Zhang (HKUST-GZ), Aug 2026. [read] DINO-WM predictors on DINOv2/SigLIP/MAE/ResNet converge in m-kNN geometry; predictor *halves* of two trained models are spliced with an MLP (k=3), CEM H=5; retention score RS = SSR/(S_A·S_B); ResNet excluded as topologically incompatible; Appendix F: planning error = rollout error + latent-geometry distortion. **Closest prior art.** Bearing: E2 baseline protocol and metric; H1 novelty statement; geometry criterion in §5.2.
- **Reconstruction or Semantics? What Makes a Latent Space Useful for Robotic World Models** — 2605.06388. [read] Frozen encoder + optional frozen adapter + DiT transition; compares semantic vs reconstruction latents at compute parity. Bearing: choice of E_A latent type; adapter-as-frozen-projection condition.
- **What Makes Video World Model Latents Action-Relevant** — 2606.07687. [sweep] Frozen V-JEPA/DINO/SigLIP features are appearance-coupled until inverse-dynamics fine-tuning. Bearing: frozen-pretrained foreign encoder condition; §6.3.
- **Learning Invariant Visual Representations for Planning with JEPA World Models** — 2602.18639. [read] Bisimulation encoder over DINOv2/SimDINOv2/iBOT features; MPC with CEM. Bearing: task-irrelevance suppression as an adapter variant.
- **Temporal Straightening for Latent Planning** — 2603.12231. [read] Curvature regularization for planning-friendly latent geometry on top of DINO-WM. Bearing: optional $L_{\rm geom}$ (§6.7).
- **Relative representations** (Moschella et al. 2023), **model stitching** (Lenc & Vedaldi 2015; Bansal et al. 2021), **Platonic Representation Hypothesis** (Huh et al. 2024). Bearing: H1b, E2 conditions (1)–(2).
- Cross-embodiment latent alignment: **LAC-WM** (ICML 2026), **Demo-JEPA** (2605.20811), **DyPES-VLA** (2608.06374), **UniT** (2604.19734). [sweep] Align *action* spaces under a shared predictor — the dual of PATH-WM; cite as adjacent.

## B. Collapse prevention and training recipes (→ §6)

- **LeJEPA / SIGReg** — 2511.08544, Balestriero & LeCun. [read via LeVJEPA] Sketched isotropic-Gaussian regularization; provable collapse exclusion; one hyperparameter. Bearing: $L_{\rm reg}$.
- **LeWorldModel** — 2603.19312, Maes, Le Lidec, Scieur, LeCun, Balestriero. [read] End-to-end JEPA world model from pixels with two loss terms; ~15M params; single GPU; plans up to 48× faster than foundation-model world models. Bearing: E1 codebase.
- **LeVJEPA** — 2608.27395, Kuhn et al. [read] Collapse-free video encoder; 95% token dropping; block-causal attention at no accuracy cost; consumer-GPU pretraining. Bearing: E4 causal-encoder condition; real-video $E_A$.
- **Sub-JEPA** — 2605.09241. [sweep] Subspace Gaussian regularization on LeWM. Bearing: first alternative to SIGReg.
- **Sensorimotor World Models** — 2606.20104, Ivashkov et al. [sweep] One inverse-dynamics regularizer prevents collapse and aligns latents to action. Bearing: H1 anchor precedent; §6.3.
- **Delta-JEPA** — 2606.31232. [sweep] Decode actions from latent differences so the transition is constrained. Bearing: §5.6 delta form; §6.3.
- **When does LeJEPA learn a world model?** — 2605.26379, Klindt, LeCun, Balestriero. [sweep] Theory of when the objective yields a usable model. **Energy-Based JEPA library** — 2602.03604. [sweep]
- **VJEPA (variational)** — 2601.14354; **Var-JEPA** — 2603.20111. [read] Probabilistic JEPA formulations. Bearing: later $U$ options.

## C. Long-horizon rollout stability (→ §5.6, §6.2, §10)

- **VLWM: Variable-Length Latent World Models** — 2606.21775. [read] Variable-offset prediction conditioned on action segments; horizon curriculum; token-based actions; adaptive chunk sizes in planning. Bearing: §6.2, §6.9, E9-v1.
- **Looped World Models** — 2606.18208. [read] Parameter-shared looped transformer with spectral-norm constraint (contractive). Bearing: §5.6 stability option; E5.
- **LaWM: Least Action World Models** — 2605.08279. [read] Least-action constraint on latent transitions. Bearing: E5 physics-structured option.
- **SkyJEPA** — 2606.23444. [read] Compounding ratio and error-rate metrics; latent model CR≈1.4 at k=60 vs 2.4 baseline. Bearing: §10 metric.
- **Imagined Rollouts are Kinematic, Not Dynamic** — July 2026. [read] Kinematic-vs-dynamic error decomposition. Bearing: §10.
- **FF-JEPA** — 2606.09311. [read] Latent planners on LeWM for goal-free long-horizon PushT. Bearing: E6 baseline.
- **Mamba-3** — 2603.15569 (ICLR 2026). [sweep] SSM improvements; not benchmarked for latent rollout stability. Bearing: E5 candidate; the matched-compute bake-off is an open gap.

## D. Belief state, memory, permanence (→ H5, E4)

- **Latent State Design for World Models under Sufficiency Constraints** — 2605.01694. [read] Functional taxonomy; memory design space (substrate × update × access); evaluation setups with occlusion, revisitation, delayed cues. Bearing: E4 framework.
- **NE-Dreamer** — 2603.02765. [read] Next-embedding prediction with a temporal transformer; gains on memory tasks. Bearing: E4 context-window baseline.
- **Identifiable Token Correspondence** — 2605.16457. [read] Copy-or-generate token assignment at decoding time for persistence. Bearing: ABI-1 precursor.
- **Tensor Memory** — 2605.27686. [read] Fixed-size voxel-grid recurrent state. **Persistent Computational State** — 2607.21686. [read] Snapshot/fork/backtrack runtime. Bearing: reservoir infrastructure.
- **Permanence Fields** — 2606.28455. [read] Object-permanence diagnostics in passive object-state models. **MemoBench** — 2606.27537 (ECCV 2026). [sweep] Disappear–reappear benchmark; no model exceeds 0.6 object-reappearance score. Bearing: E4 protocol.
- **LIBERO-Mem** — 2511.11478; **MIKASA-Robo**. [sweep] Non-Markovian robot memory benchmarks.
- **Universal Transformers Need Memory** — 2604.21999. [read] Memory tokens persisting across recursion steps. Bearing: H2 contrast condition.

## E. Uncertainty and multimodal futures (→ U)

- **MoP-JEPA** — 2607.05238. [read] Proof of conditional-mean collapse under stochastic branching; hard-assigned predictor mixtures yield a searchable transition set. Bearing: §5.8.
- **Flow Matching in Feature Space** — 2606.29059. [sweep] Flow matching in JEPA feature space restores multimodality. Bearing: §5.8 fallback.
- **Valdi** — 2607.00917. [read] Single-step latent diffusion matches deterministic MLP in TD-MPC; more steps add variety, not control. Bearing: §11 mixture criterion.
- **UWM-JEPA** — 2605.25313. [read] Density-matrix latent with unitary predictor. Bearing: exotic $U$ option.

## F. Planning with learned world models (→ §7, E6, E7)

- **GRASP** — 2602.00475, Psenka, Rabbat, Krishnapriyan, LeCun, Bar (ICML 2026). [read] Virtual states with soft dynamics constraints, Langevin stochasticity, gradient reshaping; beats CEM and GD on long horizons. Bearing: E6 baseline; §7.7 repair operator.
- **Tempered SMC for trajectory and policy optimization** — 2604.21456. [sweep] Annealed SMC with HMC rejuvenation and differentiable rollouts. Bearing: §7.2 reference realization.
- **TSMCTS** — 2511.14220, Oren et al. (ICML 2026). [read] SMC as parallel MCTS alternative; addresses variance and path degeneracy. **TRT-SMC** — 2504.06048 (ICML 2025). [read] **PMCTS** — 2605.08982. [read] Bearing: §7.2 parallel search references.
- **LeFlow** — 2608.24855. [read] Rectified-flow latent trajectory prior + inverse-dynamics decoder + frozen-model verification. Bearing: proposal family; E6 baseline.
- **Latent Geometry Beyond Search / GC-IDM** — 2605.08732. [read] Amortized goal-conditioned inverse dynamics; sweeps CEM/MPPI/iCEM/gradient. Bearing: E6 baseline.
- **HWM** — 2604.03208 (NYU, on PLDM). [read] Inference-time hierarchical latent MPC; up to 3–4× less planning compute. **Mind the Gap** — 2607.12547. [sweep] When hierarchy fails. Bearing: E9.
- **AdaJEPA** — 2606.32026. [read] Test-time adaptation inside MPC. Bearing: §7.7 complement.
- **Value-guided JEPA planning** — 2601.00844. [sweep] Learned value for candidate scoring. Bearing: §5.9 critic precedent.
- **Monte Carlo Tree Diffusion** — 2502.07202. [sweep] **Planning as Descent** — 2512.17846. [sweep] **HDFlow** — 2605.04525. [sweep] **WorldPlanner** — 2511.03077. [read] Bearing: E6 extended baselines.
- Rendering and chemistry sources: **ReSTIR** (Bitterli 2020), **MCMC-mutated ReSTIR** (Sawhney et al.), **ReSTIR path guiding** (Zeng et al. 2025), **TPS with generative models** (2504.18506), **conditioned path measures** (2506.01904). [sweep for the last three] No 2026 planning paper imports ReSTIR/MIS reuse into replanning — open.

## G. Latent actions and passive-to-interactive (→ H3, E9-v2)

- **Learning Latent Action World Models In The Wild** — 2601.05230 (Meta). [read] Inverse dynamics trained jointly with the world model; latent actions regularized by noise, sparsification, quantization. Bearing: E9-v2 recipe.
- **Co-Evolving Latent Action World Models** — 2510.26433. [read] Joint IDM + world model training collapses without care. Bearing: §6.3 caveat.
- **LAWM** — 2509.18428; **Olaf-World** — 2602.10104 (ICML 2026); **LAC-WM** (ICML 2026); **DreamDojo** — 2602.06949; **ViPRA** — 2511.07732; **AdaWorld** — 2503.18938. [sweep] Passive video → latent actions; LAWM reports beating ground-truth-action pretraining on LIBERO.
- **DeFI** — 2604.16391 (ICLR 2026). [read] Decoupled forward and inverse dynamics pretraining. Bearing: $I_\omega$ as a first-class component.
- **Latent Particle World Models** — 2603.04553. [read] Object-centric stochastic dynamics with latent actions. Bearing: ABI-1.

## H. Benchmarks, harnesses, metrics (→ E0, E6, §10)

- **stable-worldmodel** — 2605.21800. [read] Common MPCPolicy wrapper (encode/predict/rollout/criterion/get_cost); DINO-WM baseline. Bearing: harness.
- **OGBench** (Park et al. 2025); PushT / PointMaze / Wall / Rope / Granular (DINO-WM protocol). Bearing: external environments.
- **POKEWORLD** — 2607.27017. [sweep] Paired-intervention identifiability of hidden physical parameters. Bearing: E0 intervention protocol.
- **MemoBench** — 2606.27537; **WorldTest / AutumnBench** — 2510.19788; **WorldBench** — 2601.21282; **Omni-WorldBench** — 2603.22212. [sweep]
- **A Definition and Roadmap for World Models** — 2607.06401. [read] Survey; compounding-error framing. Bearing: §12 positioning.

## Open gaps (no 2026 work found)
1. Foreign-encoder → frozen-dynamics-predictor stitching with transition/inverse-dynamics anchoring and counterfactual contrast (the PATH-WM gate).
2. ReSTIR/MIS spatiotemporal reuse imported into replanning.
3. Predictor-architecture bake-off for latent rollout stability at matched compute.
4. Whether register/scratch tokens leak world state in dynamics predictors.
