"""Shared latent core for R1: one tied block induces codes (G) and applies them (T).

G: support evidence tokens -> concept code Z [K,D].
T: [target machine, role a, role b] tokens read Z -> unweighted outcome logit and a
   residual next-machine token (hypothetical, never evidence).
`search` is the bounded planner: it enumerates executable action sequences and
consumes T's learned forecasts through the shared TaskContract utility.
No hidden rule, kind, split or oracle value enters any function here.
"""

from dataclasses import dataclass, field

import torch
from torch import nn
from torch.nn import functional as F

EVIDENCE, NULL, SEED, MACHINE, ROLE_A, ROLE_B = range(6)


class Block(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.x_norm, self.c_norm, self.self_norm, self.ff_norm = (
            nn.LayerNorm(width) for _ in range(4)
        )
        self.cross = nn.MultiheadAttention(width, heads, batch_first=True)
        self.self_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width)
        )

    def forward(self, x, context, context_valid=None):
        c = self.c_norm(context)
        ignore = None if context_valid is None else ~context_valid
        x = x + self.cross(self.x_norm(x), c, c, key_padding_mask=ignore, need_weights=False)[0]
        h = self.self_norm(x)
        x = x + self.self_attention(h, h, h, need_weights=False)[0]
        return x + self.ff(self.ff_norm(x))


def key_head(width, key_width):
    """Appearance-key readout; S2's key head and the optional S1 identity objective share it."""
    return nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, key_width))


class LatentCore(nn.Module):
    def __init__(self, width=64, heads=4, loops=2, code_tokens=4, key_width=32):
        super().__init__()
        self.width, self.loops, self.code_tokens = width, loops, code_tokens
        self.block = Block(width, heads)  # the tied operator shared by G and T
        self.evidence_mlp = nn.Sequential(
            nn.Linear(4 * width, 2 * width), nn.GELU(), nn.Linear(2 * width, width)
        )
        self.null = nn.Parameter(torch.randn(width) * 0.02)
        self.seeds = nn.Parameter(torch.randn(code_tokens, width) * 0.02)
        self.types = nn.Embedding(6, width)
        self.code_norm = nn.LayerNorm(width)
        self.head_norm = nn.LayerNorm(width)
        self.outcome = nn.Linear(width, 1)
        self.next = nn.Linear(width, width)
        nn.init.zeros_(self.next.weight)  # m_hat starts as a copy of m_pre
        nn.init.zeros_(self.next.bias)
        self.key_head = key_head(width, key_width)

    def evidence(self, m_pre, a, b, m_post):
        return self.evidence_mlp(torch.cat((m_pre, a, b, m_post), -1))

    def induce(self, tokens, valid, loops=None):
        """tokens [B,N,D] with boolean valid [B,N] -> Z [B,K,D]; invalid rows ignored."""
        b = len(tokens)
        tokens = tokens.masked_fill(~valid[..., None], 0) + self.types.weight[EVIDENCE]
        null = (self.null + self.types.weight[NULL]).expand(b, 1, -1)
        context = torch.cat((null, tokens), 1)
        context_valid = torch.cat(
            (torch.ones(b, 1, dtype=torch.bool, device=valid.device), valid), 1
        )
        x = (self.seeds + self.types.weight[SEED]).expand(b, -1, -1)
        for _ in range(self.loops if loops is None else loops):
            x = self.block(x, context, context_valid)
        return self.code_norm(x)

    def apply(self, m, a, b, z, loops=None):
        """Per-query [B,D] tokens and Z [B,K,D] -> (outcome logit [B], m_hat [B,D])."""
        x = torch.stack(
            (
                m + self.types.weight[MACHINE],
                a + self.types.weight[ROLE_A],
                b + self.types.weight[ROLE_B],
            ),
            1,
        )
        for _ in range(self.loops if loops is None else loops):
            x = self.block(x, z)
        h = self.head_norm(x[:, 0])
        return self.outcome(h).squeeze(-1), m + self.next(h)

    def key(self, m):
        return F.normalize(self.key_head(m), dim=-1)


# ------------------------------------------------------------------ training objective


def cross_entropy(logits, target):
    """Mean cross-entropy via one-hot (deterministic backward on CUDA)."""
    onehot = F.one_hot(target, logits.shape[-1]).to(logits.dtype)
    return -(onehot * logits.log_softmax(-1)).sum(-1).mean()


@dataclass
class TransitionTokens:
    m_pre: torch.Tensor
    a: torch.Tensor
    b: torch.Tensor
    m_post: torch.Tensor  # detached perceived post-frame slot of the target machine
    outcome: torch.Tensor  # float target lamp after the press (loss/evaluation only)
    episode: torch.Tensor
    step: torch.Tensor = field(default=None)

    def __len__(self):
        return len(self.episode)

    def select(self, mask):
        return TransitionTokens(
            *(None if v is None else v[mask] for v in vars(self).values())
        )


@dataclass
class EpisodeTokens:
    support: TransitionTokens
    query: TransitionTokens
    chain: TransitionTokens
    keys: torch.Tensor  # [E,S>=2,D] target machine slots from different scenes
    episodes: int
    kinds: torch.Tensor | None = None  # train-only labels to mask same-kind negatives


def pad_by_episode(values, episode, episodes):
    """Flat [n,D] rows -> [E,Nmax,D] plus validity; zero rows allowed."""
    counts = torch.bincount(episode, minlength=episodes)
    width = int(counts.max()) if len(episode) else 0
    padded = values.new_zeros(episodes, width, values.shape[-1])
    valid = torch.zeros(episodes, width, dtype=torch.bool, device=values.device)
    if len(episode):
        order = torch.argsort(episode, stable=True)
        starts = torch.cumsum(counts, 0) - counts
        rank = torch.arange(len(episode), device=values.device) - starts[episode[order]]
        padded[episode[order], rank] = values[order]
        valid[episode[order], rank] = True
    return padded, valid


def codes_for(core, support, episodes, loops=None):
    e = core.evidence(support.m_pre, support.a, support.b, support.m_post)
    padded, valid = pad_by_episode(e, support.episode, episodes)
    return core.induce(padded, valid, loops=loops)


def episode_loss(core, tokens, variance, *, temperature=0.1, loops=None, key_weight=0.2):
    """Unweighted outcome BCE + next-token + 2-step rollout + key InfoNCE.

    `key_weight=0` removes the appearance-key objective (used by the symbolic
    diagnostic, whose machine tokens carry no appearance to identify a kind).
    """
    z = codes_for(core, tokens.support, tokens.episodes, loops)
    q = tokens.query
    logit, predicted = core.apply(q.m_pre, q.a, q.b, z[q.episode], loops)
    outcome = F.binary_cross_entropy_with_logits(logit, q.outcome)
    first = tokens.chain.select(tokens.chain.step == 0)
    second = tokens.chain.select(tokens.chain.step == 1)
    if not torch.equal(first.episode, second.episode):
        raise ValueError("Chain steps must be paired per episode in the same order")
    c_logit1, rolled = core.apply(first.m_pre, first.a, first.b, z[first.episode], loops)
    c_logit2, rolled2 = core.apply(rolled, second.a, second.b, z[second.episode], loops)
    step1 = torch.cat((predicted - q.m_post, rolled - first.m_post))
    next1 = (step1.square() / variance).mean()
    next2 = ((rolled2 - second.m_post).square() / variance).mean()
    rollout = F.binary_cross_entropy_with_logits(c_logit2, second.outcome)
    total = outcome + next1 + next2 + 0.5 * rollout
    if key_weight:
        keys = core.key(tokens.keys[:, :2])
        similarity = keys[:, 0] @ keys[:, 1].T / temperature
        if tokens.kinds is not None:
            same = tokens.kinds[:, None] == tokens.kinds[None]
            same.fill_diagonal_(False)
            similarity = similarity.masked_fill(same, -1e4)
        target = torch.arange(len(keys), device=keys.device)
        key_loss = cross_entropy(similarity, target) if len(keys) > 1 else keys.sum() * 0
        total = total + key_weight * key_loss
    with torch.no_grad():
        metrics = dict(
            loss=float(total),
            outcome_bce=float(outcome),
            next1=float(next1),
            next2=float(next2),
            rollout_bce=float(rollout),

            query_accuracy=float(((logit > 0).float() == q.outcome).float().mean()),
            rollout_accuracy=float(((c_logit2 > 0).float() == second.outcome).float().mean()),
        )
        if key_weight:
            metrics.update(
                key_nce=float(key_loss),
                key_accuracy=float((similarity.argmax(-1) == target).float().mean()),
            )
    return total, metrics


# ------------------------------------------------------------------ bounded planning


@dataclass
class Decision:
    option: str  # stop, abstain or press
    action: tuple | None  # (machine, object a, object b) first press of best sequence
    value: float
    values: dict
    sequences: int
    sequence: tuple = ()


def _goal_probability(logit, goal_bit):
    p = torch.sigmoid(logit)
    return p if goal_bit else 1 - p


@torch.no_grad()
def search(
    core,
    machines,
    objects,
    codes,
    lamp_logits,
    goal,
    contract,
    *,
    budget_left,
    presses_done,
    loops=None,
    chunk=1024,
):
    """Exhaustive latent search over <= budget_left presses (factorized dynamics).

    machines [2,D] current machine tokens, objects [n,D], codes [2,K,D] retrieved
    per machine, lamp_logits [2] perceived lamp logits, goal (g_left, g_right).
    Each press predicts only its target machine; the other machine is unchanged.
    """
    goal = tuple(int(g) for g in goal)
    p_now = [_goal_probability(lamp_logits[m], goal[m]) for m in (0, 1)]
    values = {
        "stop": contract.expected_stop(float(p_now[0] * p_now[1]), presses_done),
        "abstain": contract.utility("abstain", presses_done),
    }
    n = len(objects)
    actions = [(m, i, j) for m in (0, 1) for i in range(n) for j in range(n) if i != j]
    best = (max(values.values()), "stop" if values["stop"] >= values["abstain"] else "abstain", ())
    count = 0
    if budget_left < 1 or not actions:
        return Decision(best[1], None, best[0], values, 0)
    device = machines.device
    idx = torch.tensor(actions, device=device)

    def step(tokens, probs, act):
        # tokens [S,2,D], probs [S,2], act [S,3] -> updated copies
        m = act[:, 0]
        rows = torch.arange(len(act), device=device)
        logit, following = core.apply(
            tokens[rows, m], objects[act[:, 1]], objects[act[:, 2]], codes[m], loops
        )
        goal_bits = torch.tensor(goal, device=device)[m]
        p = torch.where(goal_bits.bool(), torch.sigmoid(logit), 1 - torch.sigmoid(logit))
        tokens = tokens.clone()
        probs = probs.clone()
        tokens[rows, m] = following
        probs[rows, m] = p
        return tokens, probs

    start_tokens = machines[None].expand(len(actions), -1, -1)
    start_probs = torch.stack(p_now)[None].expand(len(actions), -1)
    tokens1, probs1 = step(start_tokens, start_probs, idx)
    value1 = [
        contract.expected_stop(float(p), presses_done + 1) for p in probs1.prod(-1)
    ]
    count += len(actions)
    candidates = [(v, (a,)) for v, a in zip(value1, actions)]
    if budget_left >= 2:
        pairs = torch.cartesian_prod(
            torch.arange(len(actions), device=device), torch.arange(len(actions), device=device)
        )
        for start in range(0, len(pairs), chunk):
            part = pairs[start : start + chunk]
            t2, p2 = step(tokens1[part[:, 0]], probs1[part[:, 0]], idx[part[:, 1]])
            for (first, second), p in zip(part.tolist(), p2.prod(-1).tolist()):
                candidates.append(
                    (
                        contract.expected_stop(p, presses_done + 2),
                        (actions[first], actions[second]),
                    )
                )
        count += len(pairs)
    sequence_value, sequence = max(candidates, key=lambda c: c[0])
    values["best_sequence"] = sequence_value
    if sequence_value > best[0] + 1e-9:
        return Decision("press", sequence[0], sequence_value, values, count, sequence)
    return Decision(best[1], None, best[0], values, count)
