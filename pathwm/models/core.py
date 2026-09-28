"""Shared core: one weight-tied block stack for observe, predict, think, induce and apply.

Design: docs/core-design.md (E1-E12). Plan and contracts: docs/shared-core-plan.md (S1).

State (`CoreState`, batch B, slots S):
  h        [B,S,D]    continuous slot state
  logits   [B,S,G,C]  categorical code per slot (None with continuous_only)
  presence [B,S]      presence logit per slot
  scene    [B,4,D]    scene tokens
  work     [B,16,D]   workspace W; only `think` writes it
Every operation runs the same blocks (`rounds` inner rounds, each re-reading the same
context). A mode embedding and small heads tell the operations apart. Slots carry no
position embedding, so every operation is equivariant to slot order. There is no step
counter input. Write rights:
  predict  -> prior state (imagination: only on a hypothetical copy)
  observe  -> posterior; the only operation that corrects a live state; bumps `revision`
  think    -> W and proposals (finish, retrieval query, action query)
  induce   -> concept code Z [B,K,D], recomputable from its evidence
  apply    -> outputs only
"""

from dataclasses import dataclass, field

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class CoreConfig:
    width: int = 128
    heads: int = 4
    blocks: int = 2
    rounds: int = 2
    slots: int = 8
    scene: int = 4
    work: int = 16
    groups: int = 4
    classes: int = 16
    concept_tokens: int = 4
    evidence_width: int = 128
    action_width: int = 128
    unimix: float = 0.01
    tied: bool = True
    continuous_only: bool = False


@dataclass
class CoreState:
    h: torch.Tensor
    logits: torch.Tensor | None
    presence: torch.Tensor
    scene: torch.Tensor
    work: torch.Tensor
    revision: int = 0
    work_revision: int = -1
    hypothetical: bool = False
    assignment: torch.Tensor | None = field(default=None, repr=False)  # derived, not saved

    def branch(self):
        """A hypothetical copy; the tensors are shared but never modified in place."""
        return CoreState(self.h, self.logits, self.presence, self.scene, self.work,
                         self.revision, self.work_revision, True)

    def detach(self):
        d = lambda t: None if t is None else t.detach()
        return CoreState(d(self.h), d(self.logits), d(self.presence), d(self.scene), d(self.work),
                         self.revision, self.work_revision, self.hypothetical)

    def to_dict(self):
        c = lambda t: None if t is None else t.detach().cpu().clone()
        return dict(h=c(self.h), logits=c(self.logits), presence=c(self.presence),
                    scene=c(self.scene), work=c(self.work), revision=self.revision,
                    work_revision=self.work_revision, hypothetical=self.hypothetical)

    @classmethod
    def from_dict(cls, d, device=None):
        m = lambda t: None if t is None else t.to(device) if device else t.clone()
        return cls(m(d["h"]), m(d["logits"]), m(d["presence"]), m(d["scene"]), m(d["work"]),
                   d["revision"], d["work_revision"], d["hypothetical"])


@dataclass
class ThinkResult:
    state: CoreState
    finish: torch.Tensor  # [B] logit
    retrieve: torch.Tensor  # [B,D] retrieval query
    action: torch.Tensor  # [B,D] action query (scored against candidate action tokens)


@dataclass
class ApplyResult:
    tokens: torch.Tensor  # [B,Q,D]
    outcome: torch.Tensor  # [B,Q] logit


class Block(nn.Module):
    """Cross-attention to a fixed context, self-attention, feed-forward (pre-norm)."""

    def __init__(self, width, heads):
        super().__init__()
        self.x_norm, self.c_norm, self.self_norm, self.ff_norm = (nn.LayerNorm(width) for _ in range(4))
        self.cross = nn.MultiheadAttention(width, heads, batch_first=True)
        self.self_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

    def forward(self, x, context, valid=None):
        c = self.c_norm(context)
        ignore = None if valid is None else ~valid
        x = x + self.cross(self.x_norm(x), c, c, key_padding_mask=ignore, need_weights=False)[0]
        h = self.self_norm(x)
        x = x + self.self_attention(h, h, h, need_weights=False)[0]
        return x + self.ff(self.ff_norm(x))


def categorical_kl(post_logits, prior_logits, unimix=0.01):
    """KL(post || prior) per row, summed over slots and groups: [B,S,G,C] -> [B]."""
    def logp(logits):
        p = (1 - unimix) * logits.softmax(-1) + unimix / logits.shape[-1]
        return p.log()
    lq, lp = logp(post_logits), logp(prior_logits)
    return (lq.exp() * (lq - lp)).sum(-1).sum((-1, -2))


TYPES = ("slot", "scene", "work", "concept", "evidence", "action", "time", "memory",
         "branch", "goal", "null", "seed", "query")


class SharedCore(nn.Module):
    OPERATIONS = ("observe", "predict", "think", "induce", "apply")

    def __init__(self, config=CoreConfig()):
        super().__init__()
        c = self.config = config
        d = c.width
        self.blocks = nn.ModuleList(Block(d, c.heads) for _ in range(c.blocks))
        # Untied control (E2): one stack per operation; `observe` keeps `self.blocks`.
        self.untied = nn.ModuleDict() if c.tied else nn.ModuleDict(
            {f"{op}_blocks": nn.ModuleList(Block(d, c.heads) for _ in range(c.blocks)) for op in self.OPERATIONS[1:]})
        self.modes = nn.Embedding(len(self.OPERATIONS), d)
        self.types = nn.Embedding(len(TYPES), d)
        self.slot_mu = nn.Parameter(torch.zeros(d))
        self.slot_log_sigma = nn.Parameter(torch.zeros(d))
        self.scene_init = nn.Parameter(torch.randn(c.scene, d) * 0.02)
        self.work_init = nn.Parameter(torch.randn(c.work, d) * 0.02)
        self.seeds = nn.Parameter(torch.randn(c.concept_tokens, d) * 0.02)
        code = c.groups * c.classes
        self.code_embed = None if c.continuous_only else nn.Linear(code, d, bias=False)
        self.presence_embed = nn.Linear(1, d)
        self.evidence_in = nn.Linear(c.evidence_width, d)
        self.action_in = nn.Linear(c.action_width, d)
        self.action_flag = nn.Embedding(2, d)  # missing vs present action
        self.time_in = nn.Linear(2, d)
        # A demonstrated transition (pre entity token, action, post entity token) for induce.
        self.event_in = nn.Sequential(nn.Linear(2 * c.evidence_width + c.action_width, 2 * d), nn.GELU(),
                                      nn.Linear(2 * d, d))
        self.norm = nn.LayerNorm(d)
        # predict heads
        self.prior_head = None if c.continuous_only else nn.Linear(d, code)
        self.h_step = nn.Linear(d, d)
        self.scene_step = nn.Linear(d, d)
        self.presence_head = nn.Linear(d, 1)
        for layer in (self.h_step, self.scene_step):  # predict starts as a copy of h/scene
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        # observe heads
        self.post_head = None if c.continuous_only else nn.Linear(d, code)
        self.h_correct = nn.Linear(d, d) if c.continuous_only else None
        self.post_presence = nn.Linear(d, 1)  # prior and posterior share no head
        self.scene_correct = nn.Linear(d, d)
        nn.init.zeros_(self.scene_correct.weight)
        nn.init.zeros_(self.scene_correct.bias)
        self.assign_q, self.assign_k = nn.Linear(d, d), nn.Linear(d, d)
        # think heads
        self.finish_head = nn.Linear(d, 1)
        self.retrieve_head = nn.Linear(d, d)
        self.action_head = nn.Linear(d, d)
        # induce / apply
        self.concept_norm = nn.LayerNorm(d)
        self.outcome_head = nn.Linear(d, 1)

    # ------------------------------------------------------------------ plumbing

    def stack(self, op):
        return self.blocks if self.config.tied or op == "observe" else self.untied[f"{op}_blocks"]

    def run(self, op, x, context, valid=None, rounds=None):
        x = x + self.modes.weight[self.OPERATIONS.index(op)]
        for _ in range(self.config.rounds if rounds is None else rounds):
            for block in self.stack(op):
                x = block(x, context, valid)
        return x

    def type(self, name):
        return self.types.weight[TYPES.index(name)]

    def code(self, logits):
        """Straight-through sampled one-hot while training, arg-max one-hot otherwise."""
        c = logits.shape[-1]
        probs = (1 - self.config.unimix) * logits.softmax(-1) + self.config.unimix / c
        if self.training:
            index = torch.multinomial(probs.reshape(-1, c), 1).reshape(probs.shape[:-1])
            hard = F.one_hot(index, c).to(probs.dtype)
            return hard + probs - probs.detach()
        return F.one_hot(logits.argmax(-1), c).to(logits.dtype)

    def tokens(self, state):
        """Slot tokens [B,S,D]: h + code embedding + presence."""
        x = state.h + self.presence_embed(state.presence.sigmoid()[..., None])
        if state.logits is not None:
            x = x + self.code_embed(self.code(state.logits).flatten(-2))
        return x

    def state_tokens(self, state, kind="slot"):
        return torch.cat((self.tokens(state) + self.type(kind), state.scene + self.type("scene")), 1)

    def _ones(self, x):
        return torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)

    def _null(self, b, device):
        return (self.type("null")).expand(b, 1, -1), torch.ones(b, 1, dtype=torch.bool, device=device)

    def _join(self, parts):
        """[(tokens [B,N,D], valid [B,N] or None)] -> concatenated context and validity."""
        parts = [(t, self._ones(t) if v is None else v) for t, v in parts if t is not None]
        return torch.cat([t for t, _ in parts], 1), torch.cat([v for _, v in parts], 1)

    # ------------------------------------------------------------------ state

    def initial_state(self, batch, generator=None, device=None):
        c, device = self.config, device or self.slot_mu.device
        noise = torch.randn(batch, c.slots, c.width, generator=generator).to(device)
        h = self.slot_mu + F.softplus(self.slot_log_sigma) * noise
        logits = None if c.continuous_only else torch.zeros(batch, c.slots, c.groups, c.classes, device=device)
        return CoreState(h, logits, torch.zeros(batch, c.slots, device=device),
                         self.scene_init.expand(batch, -1, -1), self.work_init.expand(batch, -1, -1))

    def bind(self, evidence, present):
        """Initial state bound to perceived entities: slot i <- evidence token i [B,K,E], K <= S.

        Remaining slots are absent. The posterior then comes from `observe`."""
        c = self.config
        b, k, _ = evidence.shape
        if k > c.slots:
            raise ValueError("More entities than slots")
        h = torch.zeros(b, c.slots, c.width, device=evidence.device)
        h[:, :k] = self.evidence_in(evidence)
        presence = torch.full((b, c.slots), -8.0, device=evidence.device)
        presence[:, :k] = torch.where(present, 8.0, -8.0)
        logits = None if c.continuous_only else torch.zeros(b, c.slots, c.groups, c.classes, device=evidence.device)
        return CoreState(h, logits, presence, self.scene_init.expand(b, -1, -1), self.work_init.expand(b, -1, -1))

    def event(self, pre, action, post):
        """Token [n,D] of one demonstrated transition for `induce`."""
        return self.event_in(torch.cat((pre, action, post), -1))

    def work_valid(self, state):
        return state.work_revision == state.revision

    # ------------------------------------------------------------------ operations

    def predict(self, state, action=None, action_present=None, dt=None, z=None, z_valid=None, rounds=None):
        """Prior after an optional action and elapsed time dt [B]. Rows without an action and
        with dt == 0 are returned exactly; an action at dt == 0 is applied once."""
        b, device = state.h.shape[0], state.h.device
        dt = torch.zeros(b, device=device) if dt is None else dt.to(device, torch.float32)
        if action_present is None:
            action_present = torch.full((b,), action is not None, dtype=torch.bool, device=device)
        active = action_present | (dt != 0)
        if not bool(active.any()):
            return state
        present = action_present.long()
        act = self.type("action") + self.action_flag(present)
        if action is not None:
            act = act + self.action_in(action) * action_present[:, None]
        time = self.time_in(torch.stack((dt.clamp_min(0).log1p(), (dt > 0).float()), -1)) + self.type("time")
        queries = self.state_tokens(state)
        context, valid = self._join([(queries, None), (act[:, None], None), (time[:, None], None),
                                     (None if z is None else z + self.type("concept"), z_valid)])
        out = self.norm(self.run("predict", queries, context, valid, rounds))
        s = self.config.slots
        slots, scene = out[:, :s], out[:, s:]
        h = state.h + self.h_step(slots)
        logits = None if state.logits is None else self.prior_head(slots).view_as(state.logits)
        presence = self.presence_head(slots).squeeze(-1)
        scene = state.scene + self.scene_step(scene)
        keep = lambda new, old: torch.where(active.view(-1, *[1] * (new.dim() - 1)), new, old)
        return CoreState(keep(h, state.h), None if logits is None else keep(logits, state.logits),
                         keep(presence, state.presence), keep(scene, state.scene), state.work,
                         state.revision, state.work_revision, state.hypothetical)

    def imagine(self, state, action=None, action_present=None, dt=None, z=None, z_valid=None, rounds=None):
        return self.predict(state.branch(), action, action_present, dt, z, z_valid, rounds)

    def observe(self, prior, evidence, valid, memory=None, memory_valid=None, rounds=None):
        """Posterior from evidence tokens [B,N,E] with validity [B,N]. h stays the prior's
        (the code carries the correction) except with continuous_only."""
        if prior.hypothetical:
            raise ValueError("Cannot observe into a hypothetical branch")
        b = len(evidence)
        ev = self.evidence_in(evidence) + self.type("evidence")
        queries = self.state_tokens(prior)
        context, cvalid = self._join([self._null(b, ev.device), (ev, valid),
                                      (None if memory is None else memory + self.type("memory"), memory_valid)])
        out = self.norm(self.run("observe", queries, context, cvalid, rounds))
        s = self.config.slots
        slots, scene = out[:, :s], out[:, s:]
        h = prior.h if self.h_correct is None else prior.h + self.h_correct(slots)
        logits = None if prior.logits is None else self.post_head(slots).view_as(prior.logits)
        score = torch.einsum("bsd,bnd->bsn", self.assign_q(slots), self.assign_k(ev)) / self.config.width ** 0.5
        assignment = score.masked_fill(~valid[:, None], float("-inf"))
        return CoreState(h, logits, self.post_presence(slots).squeeze(-1), prior.scene + self.scene_correct(scene),
                         prior.work, prior.revision + 1, prior.work_revision, False, assignment)

    def think(self, state, goal=None, memory=None, memory_valid=None, branches=(), z=None, z_valid=None,
              rounds=None):
        """Updates only the workspace; branch results enter as marked context."""
        b = state.h.shape[0]
        parts = [(self.state_tokens(state), None),
                 (None if z is None else z + self.type("concept"), z_valid),
                 (None if memory is None else memory + self.type("memory"), memory_valid),
                 (None if goal is None else goal + self.type("goal"), None)]
        parts += [(self.state_tokens(br) + self.type("branch"), None) for br in branches]
        context, valid = self._join(parts)
        work = self.run("think", state.work + self.type("work"), context, valid, rounds)
        pooled = self.norm(work).mean(1)
        new = CoreState(state.h, state.logits, state.presence, state.scene, work,
                        state.revision, state.revision, state.hypothetical)
        return ThinkResult(new, self.finish_head(pooled).squeeze(-1), self.retrieve_head(pooled),
                           self.action_head(pooled))

    def induce(self, tokens, valid, rounds=None):
        """Evidence tokens [B,N,D] with validity [B,N] -> concept code Z [B,K,D]."""
        b = len(tokens)
        tokens = tokens.masked_fill(~valid[..., None], 0) + self.type("evidence")
        context, cvalid = self._join([self._null(b, tokens.device), (tokens, valid)])
        seeds = (self.seeds + self.type("seed")).expand(b, -1, -1)
        return self.concept_norm(self.run("induce", seeds, context, cvalid, rounds))

    def apply(self, queries, z, z_valid=None, rounds=None):
        """Query tokens [B,Q,D] read Z [B,K,D]; returns outputs, never state."""
        b = len(queries)
        context, valid = self._join([self._null(b, queries.device), (z + self.type("concept"), z_valid)])
        out = self.norm(self.run("apply", queries + self.type("query"), context, valid, rounds))
        return ApplyResult(out, self.outcome_head(out).squeeze(-1))

    # ------------------------------------------------------------------ losses

    def kl(self, post, prior):
        """KL(post || prior) per row [B]; squared h distance with continuous_only."""
        if post.logits is None:
            return (post.h - prior.h).pow(2).mean(-1).sum(-1)
        return categorical_kl(post.logits, prior.logits, self.config.unimix)
