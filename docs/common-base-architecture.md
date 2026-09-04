# PATH-WM common base architecture (v0.4 candidate)

Status: implementation candidate, 2026-09-04. ABI v1 and E1-a remain an immutable measured control;
this document defines the replacement reference that must pass the gates below before it is promoted.

## 1. Decision

There is no literal architecture shared by LeWM, V-JEPA 2, VLWM, DINO-WM, Delta-JEPA and Dreamer.
Their useful intersection is narrower:

1. learn temporally useful sensory features before relying on an action-conditioned dynamics model;
2. predict in latent space, with a stable target rather than a target that moves freely with the student;
3. give the dynamics model history, either explicitly or in a sufficient recurrent state;
4. keep the planner outside the representation objective and train/use it only after dynamics works;
5. treat action correctness, open-loop stability and partial observability as separate failure modes.

PATH-WM adopts that intersection and adds the modularity needed by H1. The default is an EMA-target,
multimodal JEPA frontend feeding a deterministic recurrent belief and a separate action-conditioned
predictor. Planning, stochastic heads, language and additional sensors attach later through interfaces;
they are not empty trainable modules in the initial graph.

## 2. Runtime graph

```text
 video clip ── video encoder ── video evidence adapter ─┐
                                                       │
 audio span ── audio encoder ── audio evidence adapter ─┼─> predict–correct updater ─> W_t
                                                       │              ▲
 future sensor ─ encoder ─ evidence adapter ────────────┘              │ prior
                                                                      │
 raw actions ─ embodiment action adapter ─> action tokens ─> world predictor

 W_t ──────────────────────────────────────────────────────> planner / probes / debug decoders
```

The three representations have deliberately different ownership:

- **Native tokens** retain video space-time or audio time-frequency geometry. Their count and width
  are encoder-specific.
- **Evidence tokens** have a shared width, modality id, validity mask and timestamp, but retain native
  token identity. Evidence is variable length and is never rolled forward by the planner.
- **Belief state `W`** is 64 modality-neutral latent slots plus one global token. It is fixed-shape,
  persistent, Markov by contract, and is the only state consumed by dynamics and planning.

This avoids the ABI-v1 error of requiring an audio patch to occupy a visual 8x8 coordinate. Spatial and
frequency positions remain available inside evidence; cross-attention decides what enters persistent
state. A future modality needs only an `EvidenceEncoder` and `EvidenceAdapter`. The predictor and
planner do not change.

## 3. Core modules

### 3.1 Video encoder

A tubelet ViT (or a compatible block-causal video transformer) consumes `(B,T,C,H,W)`. It encodes
space-time positions before emitting native tokens. Training uses masked/future latent prediction with
an EMA target. The retained inference artifact is the online encoder; the pretext prediction head is
discarded.

### 3.2 Audio encoder

A spectrogram-patch transformer consumes waveform spans `(B,C,S)`. It computes a differentiable STFT,
encodes time-frequency positions, and emits timestamped native tokens. It has its own normalization and
EMA target; it is not forced to imitate video patch coordinates.

### 3.3 Evidence adapters

Each adapter projects native width to 192, adds a modality embedding, and applies non-affine per-token
LayerNorm. It preserves token timestamps and padding masks. An adapter may later become a resampler,
but no adapter may claim its native token positions are canonical world-state positions.

### 3.4 Belief updater

At the first observation, learned state queries cross-attend to all available evidence. Later:

`prior_t = P(W_(t-1), action_tokens, delta_t)`

`W_t = LN(prior_t + observed_gate * correction(prior_t, evidence_t))`

The correction is one small cross-attention block. With no valid evidence it returns the prior exactly.
Modality dropout and whole-observation dropout exercise video-only, audio-only and prior-only paths.
This is where temporal history and occluded information live; the predictor remains Markov in `W`.

### 3.5 World predictor

A slot transformer consumes fixed belief slots, timestamped action tokens, a continuous `delta_t`
token and fresh scratch registers. It predicts a small normalized residual. Its readout is calibrated
near the empirical transition scale (the v1 audit measured 0.03 as a useful initial scale), rather than
starting 179 times above identity transition error. Unknown passive actions use a learned token, never
the numerical zero-action token.

### 3.6 EMA targets and pretext heads

Every representation-stage encoder has a no-gradient exponential-moving-average target. Stop-gradient
is applied only to that teacher. This replaces the incoherent v1 hybrid of stop-gradient targets without
an EMA teacher. Pretext predictors belong to representation training and are not the world predictor.

### 3.7 Later modules

The existing `InverseDynamics`, `Critic`, `Planner`, `Constraints` and `DebugDecoder` contracts remain.
Uncertainty heads implement the same predictor state ABI; new sensors implement the evidence contracts;
language is an input/output adapter; hierarchical planning changes action proposals and `delta_t`, not
the belief layout. Interfaces are present now, but implementation files are added only when an experiment
uses them.

## 4. Training and data curriculum

Stages advance by held-out gates, not just elapsed steps. A maximum step budget remains a safety cap.

| Stage | Trainable modules | Data | Objective | Exit gate |
|---|---|---|---|---|
| R0 unimodal representation | video/audio encoders, adapters, disposable pretext heads; EMA teachers track | diverse unlabelled clips/spans, sampled separately | masked and future latent prediction; variance/rank guardrail | non-collapse, temporal retrieval beats static/identity controls in each modality |
| R1 audiovisual representation | same | synchronized A/V plus deliberately shifted negatives; modality dropout | R0 losses plus cross-modal future/synchrony prediction | A/V synchrony and cross-modal retrieval beat chance without either modality collapsing |
| B0 belief bootstrap | adapters + updater; encoders frozen | short synchronized sequences, missing/asynchronous modalities, observation dropout | filtered belief predicts EMA evidence; consistency across missing-modality views | velocity/history and occlusion probes beat per-frame baseline; prior-only path finite |
| D0 one-step dynamics | action adapter + updater + world predictor; encoders frozen | exploratory action-labelled trajectories with broad action/state coverage | absolute one-step latent prediction + inverse/delta anchor | correct action beats identity, zero and shuffled controls; counterfactual ranking above chance |
| D1 rollout dynamics | D0 modules; optionally top encoder blocks at 0.1x LR after stability | longer trajectories, action chunks, paired interventions | free-running horizon curriculum, variable `delta_t`, calibrated residual | bounded compounding ratio and correct-action advantage through planning horizon |
| U uncertainty, conditional | ensemble/mixture heads only | genuinely stochastic branches | calibration or best-of-K objective | enabled only after deterministic mean-between-modes failure is measured |
| P0 planning | world model frozen; planner/critic only | task goals and closed-loop rollouts | CEM/MPC baseline first, then learned proposals/critic | success-versus-predictor-call curve; every executed action verified by rollout |

Important corrections to “train encoders first”:

- R0/R1 must teach temporal continuity. The measured v1 encoder-only `SIGReg + inverse` warm-up created
  high variance but made adjacent frames 93 times farther apart; it is not a valid representation phase.
- B0 precedes action dynamics because a single frame is provably non-Markov in E0. Two consecutive
  frames removed 98.63% of the identity physical error in the audit.
- Counterfactual InfoNCE is a late causal discriminator, not a bootstrap loss. It activates only after
  absolute dynamics is competitive with identity/zero/shuffle controls.
- Planning never backpropagates into the base world model in the reference curriculum.

## 5. Mandatory diagnostics

The following are promotion gates, not optional dashboard decoration:

- representation: per-modality feature standard deviation, effective rank, temporal retrieval, masked
  prediction versus identity/static baselines, A/V synchrony with time-shift controls;
- belief: hidden-velocity probe, observation-dropout reconstruction in latent space, occlusion memory,
  updater gate statistics, video-only/audio-only/both/prior-only parity;
- dynamics: absolute one-step error, identity/zero/shuffled/wrong-time controls, action sensitivity,
  paired counterfactual accuracy, teacher-forced versus open-loop error and compounding ratio;
- planning: oracle-dynamics ceiling, random-action floor, success versus predictor calls, constraint
  violations, and model error on states actually visited by the planner.

## 6. Pitfalls ruled out by construction

| Pitfall | Architectural or training guardrail |
|---|---|
| moving/collapsing targets | EMA teacher plus variance/effective-rank gates |
| identity shortcut | future masking, calibrated residual, absolute error and identity control |
| hidden velocity / occlusion | persistent predict–correct belief, not a single-frame state |
| ignored or mislabelled actions | unknown-action token, inverse/delta anchor, zero/shuffle/counterfactual gates |
| modality dominance | separate encoders/normalization, modality dropout, per-modality metrics |
| fake cross-modal alignment | time-shift negatives and modality-specific collapse checks |
| open-loop drift | free-running horizon curriculum after one-step correctness |
| mean-between-modes state | gated ensemble/mixture extension after a measured stochastic failure |
| planner exploitation | frozen model, oracle ceiling, verified receding-horizon execution |
| sensor topology leaking into ABI | variable native evidence -> modality-neutral belief slots |

## 7. Migration and first experiment

`E1_reference`/ABI v1 remains the negative/control lineage. The v0.4 candidate is promoted in this order:

1. structural conformance for ABI v2, video, audio, action adapter, predictor and updater;
2. differentiable two-modality smoke training and EMA-target checks;
3. R0/R1 on real synchronized A/V or a documented public dataset;
4. B0/D0 on E0 video first, with a deterministic audio observation added only if it carries a declared
   physical signal rather than a label leak;
5. equal-budget E1-b comparison against the best v1 checkpoint;
6. planning only after D1 passes its promotion gate.

The immediate implementation slice covers items 1–2. It does not claim a scientific result; the first
result is the held-out R0/R1 representation panel on real A/V data.
