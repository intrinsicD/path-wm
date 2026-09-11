"""Bounded key-box integration. Planner dynamics are supplied, not learned."""

import copy
import math
from functools import lru_cache

import torch
from torch import nn
from pathwm.io import digest, state_hash
from pathwm.models.modalities import Observation, Attend, bytes_batch
from pathwm.models.entity_state import EntityStateMemory
from pathwm.models.belief_state import BeliefState


def neutral_event(agent, state, time):
    values, valid = bytes_batch(["event"] * len(state.tokens))
    observation = Observation(
        values,
        torch.full(values.shape, float(time)),
        valid,
    )
    return agent.observe(
        state, {"text": observation}, time=float(time), replay=agent.training
    )


class KeyBoxReader(nn.Module):
    """Entity values reach output only through the actual agent working tokens."""

    def __init__(self, agent, matcher, cell):
        super().__init__()
        self.agent, self.matcher, self.cell = agent, matcher, cell
        self.agent.requires_grad_(False)
        self.matcher.requires_grad_(False)
        self.cell.requires_grad_(False)
        self.agent.thinker.requires_grad_(True)
        self.projection = nn.Linear(cell.width, agent.width)
        self.query = nn.Parameter(torch.randn(1, agent.width) * 0.02)
        self.readout = Attend(agent.width)
        self.head = nn.Linear(agent.width, 2)

    def train(self, mode=True):
        super().train(mode)
        self.agent.eval()
        self.agent.thinker.train(mode)
        self.matcher.eval()
        self.cell.eval()
        return self

    def forward(self, state, entity_latent):
        working = self.agent.think(
            state, steps=2, goal=self.projection(entity_latent)[:, None]
        )
        tokens = working.tokens[:, self.agent.layout["working"]]
        read = self.readout(self.query.expand(len(tokens), -1, -1), tokens)[:, 0]
        return self.head(read), working


def plan_key(belief, opened, horizon=4):
    """Expectimax over three hypotheses: key in box0, box1, or absent."""
    if (
        len(belief) != 3
        or any(not math.isfinite(p) or p < 0 for p in belief)
        or abs(sum(belief) - 1) > 1e-6
    ):
        raise ValueError("Expected normalized key belief")
    if len(opened) != 2 or type(horizon) is not int or not 0 <= horizon <= 4:
        raise ValueError("Invalid planning state or horizon")

    def condition(q, box, present):
        mass = q[box] if present else 1 - q[box]
        return tuple(
            (p if (i == box) == present else 0.0) / mass for i, p in enumerate(q)
        )

    @lru_cache(None)
    def search(q, doors, depth):
        best, best_action = 0.0, ("stop", -1)
        if depth == 0:
            return best_action, best
        for box in range(2):
            p = q[box]
            choices = []
            if 1e-8 < p < 1 - 1e-8:
                value = -0.25
                for present, mass in ((True, p), (False, 1 - p)):
                    value += (
                        mass * search(condition(q, box, present), doors, depth - 1)[1]
                    )
                choices.append((("inspect", box), value))
            if not doors[box] and p > 1e-8:
                changed = tuple(True if i == box else d for i, d in enumerate(doors))
                choices.append((("open", box), -1 + search(q, changed, depth - 1)[1]))
            if doors[box] and p > 1e-8:
                value = -1 + 10 * p
                if p < 1 - 1e-8:
                    value += (1 - p) * search(
                        condition(q, box, False), doors, depth - 1
                    )[1]
                choices.append((("retrieve", box), value))
            for action, value in choices:
                if value > best + 1e-9:
                    best_action, best = action, value
        return best_action, best

    action, value = search(tuple(belief), tuple(opened), horizon)
    return action, dict(value=value, nodes=search.cache_info().currsize)


class KeyBoxSession:
    """Explicit entity state owner, with atomic delivered-event/agent commit."""

    def __init__(self, model, descriptors, session_id="key-box"):
        self.model = model
        self.descriptors = copy.deepcopy(descriptors)
        self.memory = EntityStateMemory(model.matcher, model.cell, capacity=2)
        self.state = model.agent.initial_state(1, session_id=session_id)
        self.known = [False, False]
        self.events = {}

    @torch.no_grad()
    def observe(self, event_id, box=None, bit=None, invalidate=False):
        if box is not None and (box not in (0, 1) or bit not in (0, 1)):
            raise ValueError("Invalid box observation")
        payload = digest(dict(box=box, bit=bit, invalidate=invalidate))
        if event_id in self.events:
            if self.events[event_id] != payload:
                raise ValueError("Conflicting event retry")
            return
        if len(self.events) >= 64:
            raise ValueError("Event budget exhausted")
        staged = EntityStateMemory.restore(
            self.model.matcher, self.model.cell, self.memory.snapshot()
        )
        known = self.known.copy()
        time = len(self.events) + 1
        if box is not None:
            receipt = staged.observe(
                event_id, self.descriptors[box], time, torch.eye(4)[bit]
            )
            if receipt["entity_id"] is None:
                raise LookupError("Unresolved box")
            known[box] = True
        if invalidate:
            known = [False, False]
        state = neutral_event(self.model.agent, self.state, time)
        self.memory, self.known, self.state = staged, known, state
        self.events[event_id] = payload

    @torch.no_grad()
    def probabilities(self):
        probabilities = []
        for box in range(2):
            if not self.known[box]:
                probabilities.append(0.5)
                continue
            identity = self.memory.memory.lookup(self.descriptors[box])["entity_id"]
            if identity is None:
                raise LookupError("Unresolved read")
            latent = torch.tensor(self.memory.latents[identity])[None]
            logits, self.state = self.model(self.state, latent)
            probabilities.append(float(logits.softmax(-1)[0, 1].clamp(1e-6, 1 - 1e-6)))
        a, b = probabilities
        raw = (a * (1 - b), b * (1 - a), (1 - a) * (1 - b))
        total = sum(raw)
        return tuple(p / total for p in raw), probabilities

    def snapshot(self):
        return dict(
            model=state_hash(self.model),
            descriptors=copy.deepcopy(self.descriptors),
            memory=self.memory.snapshot(),
            state=self.state.to_dict(),
            known=self.known.copy(),
            events=self.events.copy(),
        )

    @classmethod
    def restore(cls, model, record):
        if record["model"] != state_hash(model):
            raise ValueError("Incompatible key-box model")
        result = cls(model, record["descriptors"])
        result.memory = EntityStateMemory.restore(
            model.matcher, model.cell, record["memory"]
        )
        result.state = BeliefState.from_dict(record["state"])
        result.known, result.events = record["known"].copy(), record["events"].copy()
        return result
