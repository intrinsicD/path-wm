"""Module contracts for PATH-WM: the §16.1 signatures as typing.Protocols.

What: one Protocol per §16.1 row, plus the environment. Nothing here has an
implementation; every module and every test imports from this file.
How: a Protocol states the call and the tensor shapes; the structural
conformance tests (tests/conformance/) check the shapes against
docs/abi/abi_v1.yaml, and runtime_checkable lets a test assert
`isinstance(impl, Predictor)` on any implementation.
Why: H1 (§4) claims that a frozen consumer behaves correctly on a foreign
producer. That is testable only if producers and consumers share a written
interface. Interfaces are predesigned, implementations evolve (DDR §14).
Until E1 is frozen a signature may be corrected with a DDR §13 entry
(CLAUDE.md §3); after that only when two implementations both need it.

Shape conventions (first-slice step-1 decisions, DDR §18):
  obs      (N, 3, 64, 64) uint8          N parallel worlds, RGB at 64 px (E0 spec)
  action   (N, 2) float32 in [-1, 1]     one 2D force per world (abi_v1.yaml actions)
  W        (B, 65, 192) bf16             64 grid + 1 global token (ABI v1, world_state/abi.py)
  actions  (B, k, 2), 1 <= delta_t <= k <= 16   an action chunk (abi_v1.yaml delta_t)
The model-side batch B is whatever the caller stacks (worlds x time).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable

import torch

Tensor = torch.Tensor
Snapshot = Any    # opaque: what Environment.save() returns and restore() consumes
Reservoir = Any   # the persistent trajectory reservoir (§7.1, §7.4); its type is fixed in the planner slice


# ABI v2 data contracts.  ABI v1 remains below as the measured E1-a control; v2 separates
# modality-native evidence from the persistent, planner-facing world state.  These are containers,
# not model implementations.  Producers own preprocessing and positional encoding, while the
# evidence boundary owns only width, dtype, validity, modality identity, and time.
@dataclass(frozen=True)
class TemporalObservation:
    """One padded temporal batch for a single modality.

    ``values`` keeps the modality-native layout: video is (B,T,C,H,W), audio is (B,C,S), and a
    future sensor may use another layout. ``timestamps`` and ``valid_mask`` share that modality's
    temporal axis: (B,T) for video and (B,S) for waveform audio. Times are seconds relative to the
    belief-update time, so asynchronous sensors do not have to pretend they were sampled together.
    """

    values: Tensor
    timestamps: Tensor
    valid_mask: Tensor


@dataclass(frozen=True)
class EvidenceTokens:
    """Variable-length observation evidence before or after the ABI-v2 projection.

    Shape is tokens (B,N,D), timestamps/valid_mask (B,N). ``D`` is encoder-native before an
    EvidenceAdapter and ``abi.evidence_dim`` afterwards. Native spatial/frequency coordinates have
    already been encoded into the tokens; they never become canonical world-state coordinates.
    """

    tokens: Tensor
    timestamps: Tensor
    valid_mask: Tensor
    modality: str


@dataclass(frozen=True)
class ActionSequence:
    """Raw, padded action sequence owned by an embodiment-specific ActionAdapter.

    ``values`` is (B,K,A); timestamps, valid_mask, and observed_mask are (B,K). An unobserved action
    is not represented as zero: observed_mask=False selects the learned passive/unknown-action token.
    """

    values: Tensor
    timestamps: Tensor
    valid_mask: Tensor
    observed_mask: Tensor


@dataclass(frozen=True)
class ActionTokens:
    """ABI-v2 action condition: tokens (B,K,D_a) plus time, padding, and known/unknown masks."""

    tokens: Tensor
    timestamps: Tensor
    valid_mask: Tensor
    observed_mask: Tensor


@runtime_checkable
class EvidenceEncoder(Protocol):
    """A modality frontend: native samples -> variable-length, modality-native tokens (ABI v2)."""

    modality: str

    def encode_observation(self, observation: TemporalObservation) -> EvidenceTokens:
        ...


@runtime_checkable
class EvidenceAdapter(Protocol):
    """Project native evidence tokens to the shared evidence width without changing token identity."""

    modality: str

    def adapt_evidence(self, evidence: EvidenceTokens) -> EvidenceTokens:
        ...


@runtime_checkable
class ActionAdapter(Protocol):
    """Map an environment/embodiment action space to ABI-v2 action tokens."""

    def adapt_actions(self, actions: ActionSequence) -> ActionTokens:
        ...


@runtime_checkable
class WorldPredictorV2(Protocol):
    """Markov prior over canonical belief slots, conditioned on timestamped action tokens."""

    n_registers: int

    def predict_state(self, W: Tensor, actions: ActionTokens, delta_t: Tensor) -> Tensor:
        ...


@runtime_checkable
class BeliefUpdaterV2(Protocol):
    """Initialize and predict-then-correct a persistent belief from any available evidence subset."""

    def initialize(self, evidence: Mapping[str, EvidenceTokens]) -> Tensor:
        ...

    def update(
        self,
        W_prev: Tensor,
        evidence: Mapping[str, EvidenceTokens],
        actions: ActionTokens,
        delta_t: Tensor,
    ) -> Tensor:
        ...


@runtime_checkable
class WorldModelCore(Protocol):
    """Planner-facing multimodal core. Raw observations stop at this boundary."""

    def initialize(self, observations: Mapping[str, TemporalObservation]) -> Tensor:
        ...

    def observe(
        self,
        W_prev: Tensor,
        observations: Mapping[str, TemporalObservation],
        actions: ActionSequence,
        delta_t: Tensor,
    ) -> Tensor:
        ...

    def imagine(self, W: Tensor, actions: ActionSequence, delta_t: Tensor) -> Tensor:
        ...


@runtime_checkable
class Environment(Protocol):
    """E0 engine (§9 E0, DDR §18): N parallel worlds with deterministic save/restore.

    Not a §16.1 row; added by the first slice (DDR §13). Ground truth is for
    probes, viewer overlays and goal masks only (Invariant 11).
    Every tensor the environment hands out (obs, ground truth) belongs to the
    caller: a later step never changes it (DDR §18; a live render buffer would
    give a dataset with o_t == o_{t+1}).
    """

    n_worlds: int

    def reset(self, seed: int) -> Tensor:
        """Seed every world and its RNG stream; returns obs (N, 3, 64, 64) uint8."""
        ...

    def step(self, action: Tensor) -> Tensor:
        """One physics step with action (N, 2) in [-1, 1]; returns the next obs."""
        ...

    def save(self) -> Snapshot:
        """Full physical state plus the RNG stream.

        restore(save()) must be bit-exact: paired interventions and planner
        forks depend on it (E0 gate, docs/preregistration.md).
        """
        ...

    def restore(self, snapshot: Snapshot) -> None:
        ...

    def render(self) -> Tensor:
        """Current obs (N, 3, 64, 64) uint8 without stepping."""
        ...

    def ground_truth(self) -> Mapping[str, Tensor]:
        """full_state, segmentation, homotopy_signature (E0 spec), each with leading dim N.

        full_state is floating; segmentation is (N, res, res) with an integer dtype (pixel ids, the
        §14 per-token occupancy probe target); homotopy_signature's dtype is the engine's (winding
        numbers accumulate as floats along a path). DDR §13 step-2 additions.
        """
        ...


@runtime_checkable
class Encoder(Protocol):
    """E_m: o -> z, any shape (§16.1, §5.3). Not part of the frozen reference; E2 sweeps it frozen vs fine-tuned."""

    def encode(self, obs: Tensor) -> Tensor:
        """obs exactly as the environment renders it: (B, 3, 64, 64) uint8 for E0, the harness's
        frames for envs/external. Scaling and normalization are the encoder's own preprocessing,
        so build_encoder(cfg) swaps encoders without touching the data pipeline."""
        ...


@runtime_checkable
class Adapter(Protocol):
    """A_m: z -> W in the ABI v1 layout (§5.4). The object under training in E2."""

    def adapt(self, z: Tensor) -> Tensor:
        """Returns W (B, 65, 192) in the ABI dtype, every token LayerNormed without affine parameters
        (abi_v1.yaml state.normalization; DDR §13 step-2 additions)."""
        ...


@runtime_checkable
class Updater(Protocol):
    """U_psi: (W, {W_obs^(m)}, a) -> W, predict-then-correct (§5.5). Frozen for stitching.

    E1-a has none (W_t = A E(o_t)); the Protocol exists so that E1-b and E10
    swap in by one config line.
    """

    def update(self, W_prev: Tensor, W_obs: Mapping[str, Tensor], action: Tensor) -> Tensor:
        """W_prev (B, 65, 192); action (B, 2) = a_{t-1}; W_obs values (B, 65, 192), keyed by modality.

        The prior P(W_prev, action[:, None, :], delta_t=1) is computed inside: an updater is
        built with the predictor it corrects (build_updater(cfg, predictor)), the same frozen P
        that E2 stitches against. An empty mapping means W_t = the prior (§5.5).
        """
        ...


@runtime_checkable
class Predictor(Protocol):
    """P_phi: (W_t, a_{t:t+k}, delta_t) -> W_hat_{t+delta_t} (§5.6). Markov in W; frozen for stitching.

    Registers are not an argument: K_R fresh registers are created inside every
    call and discarded (Invariant 4; abi_v1.yaml registers.reset). Two calls
    with the same inputs give the same output regardless of history.
    Every call is counted by evaluation/budget.py (Invariant 8): one unit per
    row of W per call, whatever delta_t. A batched call with B rows costs B; a
    delta_t=k chunk call costs 1 per row (§7.6 crossover, E9 flat vs chunked).
    Critic.score is counted the same way, separately (DDR §13).
    """

    n_registers: int

    def predict(self, W: Tensor, actions: Tensor, delta_t: int) -> Tensor:
        """actions (B, k, 2) with 1 <= delta_t <= k <= max_chunk, else ValueError; returns W_hat (B, 65, 192),
        every token LayerNormed without affine parameters (§5.6 readout LN(W + dW)); W itself is never modified."""
        ...


@runtime_checkable
class InverseDynamics(Protocol):
    """I_omega: (W_t, W_{t+1}) -> a_hat_t (§5.7). Anti-collapse in E1, interface anchor in E2."""

    def infer_action(self, W: Tensor, W_next: Tensor) -> Tensor:
        """Returns (B, 2) float32 within the ABI action range: a bounded head, since the output feeds
        Environment.step and planner proposals (§7.3); the E1 loss is taken before the squash (DDR §13)."""
        ...


@runtime_checkable
class Critic(Protocol):
    """C_chi: (W_0, a_{0:H-1}) -> J_hat (§5.9). Planner-side; never in the world model's objective.

    Every call is counted separately from predictor calls (Invariant 8).
    """

    def score(self, W0: Tensor, actions: Tensor) -> Tensor:
        """W0 (B, 65, 192), actions (B, H, 2) -> J_hat (B,)."""
        ...


@dataclass(frozen=True)
class Goal:
    """G (§5.11 a): the goal observation through the same encoder and adapter, plus a token mask.

    Cost J = sum over masked tokens i of ||W_hat_H[i] - W_G[i]||^2. mask None means all tokens.
    """

    W_G: Tensor                 # (1, 65, 192) or (N, 65, 192): always batched, so ABI.check_state applies
    mask: Tensor | None = None  # (1, 65) or (N, 65) bool, broadcast over W_hat (N, 65, 192); None = all tokens


@runtime_checkable
class Constraints(Protocol):
    """C (§5.11): hard constraints as a stopgrad probe from W to collision / workspace violation."""

    def violation(self, W: Tensor) -> Tensor:
        """W (B, 65, 192) -> penalty (B,) >= 0."""
        ...


@runtime_checkable
class Planner(Protocol):
    """(W, G, P, V, C, R) -> (a_{0:H-1}, R') as in §7; §16.1's row omits V (DDR §13 resolution).

    Never consumes raw observations (Invariant 2). The executed sequence has a
    verified rollout; the critic alone never decides an action (Invariant 10).
    """

    def plan(
        self,
        W: Tensor,
        goal: Goal,
        predictor: Predictor,
        critic: Critic | None,
        constraints: Constraints | None,
        reservoir: Reservoir,
    ) -> tuple[Tensor, Reservoir]:
        """W (N, 65, 192) -> actions (N, H, 2) and the updated reservoir."""
        ...


@runtime_checkable
class DebugDecoder(Protocol):
    """D_img / D_text: stopgrad(W) -> image or text (§5.10, §15). Never trained into W (Invariant 1)."""

    def decode(self, W: Tensor) -> Any:
        ...
