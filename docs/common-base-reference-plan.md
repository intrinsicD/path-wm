# Reproduce a working world-model reference before extending the common base

Product: a learned dynamics reference for the modular PATH-WM stack.
Research: establish the E1 prerequisite for encoder replacement (H1); H1 and path-space planning
(H6) remain untested. This is a development reference plan, not a formal E1 freeze.

Status, 5 September 2026: literature and implementation comparison complete; no reference training
or planning evaluation has run. The user directs us to establish expected baseline learning before
modifying the method. LeWM is the proposed first reproduction, starting with video and recorded
actions. Audio-first versus video-first was asked as an optional preference and is unconfirmed.

## Finding

The executable ABI-v2 common base is a custom research candidate. It is not a reproduction of a
published world model. R0/R1 train its encoders, evidence adapters and disposable pretext heads.
They neither call nor optimize the persistent-state updater or action-conditioned world predictor.
The runner explicitly supports only these representation stages. Passing their gates cannot
establish action dynamics, open-loop rollouts, planning, or encoder replacement.

This is verified in `training/representation.py:195`, `training/common_base.py:221`, and
`training/curriculum.py:30`. Instantiating the measured R0 configuration gives 262,272 video encoder
parameters and 120,096 audio encoder parameters. The 941,952-parameter world predictor, action
adapter and updater have zero trainable parameters in that stage. Total trainable representation
parameters are 2,427,168, including the adapters and pretext heads.

## What the literature actually does

| Reference | Relevant recipe | Difference from our current training |
| --- | --- | --- |
| [LeWM paper](https://arxiv.org/html/2603.19312v3), [author implementation](https://github.com/lucas-maes/le-wm/tree/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac) | Jointly learn visual embeddings and action-conditioned causal dynamics with prediction MSE and SIGReg. Use the learned dynamics for goal-conditioned planning. | We use EMA teachers, several objectives, short passive windows, and a separate untrained dynamics core. Our earlier ABI-v1 adaptations also changed the original target gradients, state layout and losses. |
| [V-JEPA 2](https://arxiv.org/html/2506.09985v1) | Masked video-token prediction against EMA targets; later freeze the encoder and train an autoregressive predictor on robot observations, actions and end-effector states. Pretraining uses 22M samples and 300M–1B-scale encoders. | R0/R1 resemble sensory pretraining in broad purpose, but our data, capacity, masking and heads differ. The action-conditioned second stage has no trained counterpart in our common base. |
| [DINO-WM](https://arxiv.org/html/2411.04983v1) | Freeze pretrained DINOv2 patch features; train a causal transition model over observation/action history and plan against image goals. | We train our frontend from scratch and have not trained the common transition model. Its frozen-feature recipe offers a useful later control for separating perception from dynamics. |
| [DreamerV3](https://arxiv.org/html/2301.04104v2) | Jointly train recurrent deterministic/stochastic state, observation reconstruction, action-conditioned transitions, reward and continuation prediction; train behavior using imagined trajectories. | Our deterministic slot updater is an unvalidated design, and currently frozen. Dreamer's full recipe also requires training signals and policy machinery outside our reward-free reference goal. |
| [CAV-MAE](https://github.com/yuangongnd/cav-mae), [CAV-MAE Sync](https://arxiv.org/html/2505.01237v1) | Audiovisual masked reconstruction and contrastive learning. Sync adds temporal audio instances and separates global objectives; evaluates retrieval, classification and localization. | These are appropriate frontend comparators. Their audiovisual scores do not substitute for an action-conditioned world-model benchmark. Our all-token mean readout removes temporal identity before the alignment loss. |
| [LeVJEPA](https://arxiv.org/abs/2608.27395v1) | Recent video representation pretraining using view invariance and SIGReg, with token dropping and a causal variant. | A possible future encoder candidate, not evidence that our common dynamics or audio branch works. |

There is no single shared EMA/pretrain-first recipe across these methods. In particular, LeWM
learns targets jointly without stopping their gradients. Dreamer trains perception and dynamics
together. The previous common-base document overstated this intersection; its staged EMA design
is our choice, not a requirement established by all these papers.

## Material implementation differences

1. **Data and actions.** Our measured input is four 64x64 video frames in a half-second span,
   plus audio, followed by another passive span. The official [TAU description](https://zenodo.org/records/4477542)
   provides urban scene recordings; our manifest has no controller-action trajectories. This can
   support representation learning, but cannot directly reproduce action-conditioned control.
   Missing controller actions cannot be replaced by numeric zeros. Learning latent actions would
   be another method change requiring its own evidence.
2. **Representation and target gradients.** The [released LeWM configuration](https://huggingface.co/quentinll/lewm-pusht/blob/22b330c28c27ead4bfd1888615af1340e3fe9052/config.json)
   uses a ViT-Tiny, learned CLS readout, BatchNorm projectors and a six-layer predictor. Its
   [training function](https://github.com/lucas-maes/le-wm/blob/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac/train.py)
   differentiates through both prediction and target embeddings. We use one-layer sensory
   encoders, non-affine per-token LayerNorm, EMA targets, variance/covariance penalties and
   audiovisual losses. Transplanting only SIGReg or only EMA would not reproduce either recipe.
3. **Masking and readout.** Our corruption replaces entire video frames with gray and independently
   zeros audio samples, then scores all valid output tokens. V-JEPA 2 drops spatiotemporal tokens
   and uses positional mask queries. CAV-MAE masks input patches. Also, mean pooling is used for
   our audiovisual loss and retrieval; the unimodal future head still sees the full token layout.
   The frozen readout experiments identify a limitation of that pooling on the controlled source,
   not proof that every global representation fails: LeWM itself uses a learned global CLS token.
4. **Temporal state.** A world model must train the state/history consumed during imagination.
   Our current future head predicts one next span and is discarded. The planned 64-slot belief
   has neither a learning result nor an action-conditioned rollout result.
5. **Validation.** Rank and synchrony are diagnostics for our frontend. The original failed outcomes
   remain valid records. They are not published prerequisites for LeWM, and they should not block
   an independent faithful reference experiment. Our synthetic coordinate R2 and TAU metrics
   cannot be numerically compared with another paper's control success rate.

These differences explain why we lack a calibrated baseline comparison. They do not identify a
unique causal explanation for any individual failed run.

## First reference: LeWM on PushT

Use the author's native computation graph:

```text
observation frames -> shared visual encoder/projector -> latent history
recorded action blocks -------------------------------> causal predictor -> future latents
future observations -> same trainable encoder/projector -> prediction targets
                         prediction MSE + SIGReg
frozen learned model + image goal ---------------------> original MPC/CEM evaluation
```

This is an external development control until a behavior-preserving PATH-WM wrapper is verified.
Do not force the native latent history into 64 belief slots or add inverse, covariance, EMA,
audiovisual alignment, or our planner during reproduction. The historical ABI-v1 and ABI-v2
checkpoints remain controls. No contract is silently changed and no formal milestone is advanced.

PushT is the proposed scored reference because its public evaluation defaults agree more closely
with the paper than TwoRoom's do. TwoRoom remains a useful optional cheap mechanics check.
The paper's Appendix G reports 96.0 +/- 2.83 percent PushT success across three training seeds on
50 evaluation trajectories, with goals 25 steps ahead and a 50-step execution budget. This is a
published comparison target under that protocol, not an expected TAU score or a guaranteed local
result. The uncertainty notation is retained as reported, not relabeled as a confidence interval.

### Resolve these reproduction details first

The source audit is pinned in [the receipt](evidence/common-base-reference-audit-2026-09-05.json).

| Item | Observed discrepancy or missing prerequisite | Required resolution |
| --- | --- | --- |
| Training duration | Paper describes 10 epochs; checked-in `config/train/lewm.yaml` says 100. | Explicitly use the paper's 10-epoch budget for the paper-recipe run; label any default-100-epoch run separately. Pin resolved scheduler settings too. |
| TwoRoom history/evaluation | Paper states history 1, goal offset 100, budget 150 and 10 CEM iterations; published config/checkpoint use history 3, offset 25, budget 50 and common CEM defaults of 30. | Do not equate them. Prefer PushT for the first scored replication; retain both TwoRoom protocol descriptions if tested. |
| Runtime dependencies | `stable-worldmodel`, `stable-pretraining`, Hydra, Lightning and HDF5 support are absent locally. Upstream dependencies are not locked by this repository. | Pin a compatible environment and environment implementation; audit current versus historical data APIs before execution. Keep external logging disabled. |
| Entry-point configuration | Upstream training reads `cfg.wandb.enabled`, absent from the checked-in base config. Evaluation leaves `max_episode_steps` mandatory. | Supply explicit local-only logger settings and episode budget. Record wrapper-only overrides. |
| Dataset and evaluation identities | Model metadata/configs are inspected; PushT action data and checkpoint weights have not been downloaded in this task. | Verify revisions/checksums, action normalization/blocking, frame transforms, simulator resets, and selected start/goal pairs. Do not infer training history solely from the checkpoint's architecture JSON. |
| Validation separation | Upstream code splits dataset indices after computing normalization statistics on the dataset. Exact episode isolation depends on the pinned data implementation. | Audit and reproduce the author protocol for comparability; also reserve a separate untouched trajectory cohort for our generalization checks. Clearly distinguish the two. |
| Hardware | Local GPU is an RTX 3050 with 8 GiB. Reference training-memory feasibility is unmeasured. | Profile forward/backward at the original shape before a full run. Gradient accumulation changes BatchNorm/SIGReg statistics unless deliberately preserved; do not describe a smaller-batch variant as identical. |

### Completion criteria and sequence

1. **Establish the evaluator.** Evaluate the released checkpoint and simple/random controls with
   the same pinned planner, observation transforms, actions and start/goal identities. Save
   episode-level outcomes. Failure here sends us to the loader/evaluator, before retraining.
2. **Establish implementation fidelity.** Write essential tests before a wrapper: identical native
   and wrapped latents, losses, target-branch gradients and action-conditioned predictions on
   a fixed batch; correct action/frame alignment; no episode-boundary crossing; future frames
   unavailable during open-loop rollout. The first integration panel is finite real-data loss
   and a reproducible checkpoint-evaluation receipt, not a passing world-model claim.
3. **Train the declared recipe.** Retain original architecture, preprocessing, action blocks,
   normalization, objectives and batch semantics. Check a small real training batch can learn,
   then run the full declared budget with three seeds. A mechanics overfit check is not held-out
   success. Record every material hardware adaptation as a separate baseline variant.
4. **Require useful learning.** Compare action-conditioned one-step and open-loop errors at
   horizons 1, 3, 5 and 10 against identity, shuffled/zero actions and a simple fitted predictor.
   Use one shared feature basis when comparing errors across models; native latent MSE scales
   from separately trained encoders are not directly comparable. Require positive paired
   improvements on an untouched trajectory cohort, and report failure at each horizon.
5. **Define parity before inspecting new training results.** Use the released checkpoint as the
   local calibration, retaining the paper number as a separate target. Proposed acceptance:
   the lower 95% paired interval for new-model minus reference control success exceeds -5
   percentage points, together with the predictive-control checks above. This tolerance is our
   proposed engineering criterion, not a paper threshold. Fix evaluation size, seed aggregation
   and interval construction before training; 50 episodes can leave parity inconclusive. Failure
   to reject a difference is not evidence of parity. Confirm exact-paper replication only if the
   original protocol and data identities can actually be matched.
6. **Freeze, then extend one component.** First verify the PATH-WM wrapper preserves the native
   reference. Then separately test a different state interface/history mechanism, audio evidence,
   missing observations, and encoder replacement. Each gets the same task, budget and controls.
   A successful visual baseline does not validate multimodal belief; that capability needs its
   own training objective and paired evaluation after the reference is established.

TAU remains available for the later audiovisual representation/transfer comparison, including an
appropriate pretrained audiovisual reference. It is not discarded, nor is its timing uncertainty
reinterpreted as proof that the recordings lack temporal information.

## Work completed in this assessment

Checked primary papers and author code, downloaded a hash-bound LeWM source snapshot and two
checkpoint architecture/metadata records, and verified local trainable groups. Corrected the
common-base plan's universal-recipe claims and made reproduction the next priority. No package
installation, model training, checkpoint-weight download, benchmark evaluation or successful
baseline claim is part of this assessment.
