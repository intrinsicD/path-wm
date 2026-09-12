"""Read a selected visual history through actual state and detached episodic memory."""

from dataclasses import replace

import torch
from torch import nn

from .modalities import Attend, Observation, position


class FactHead(nn.Module):
    def __init__(self, width, depth=1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, width) * 0.02)
        self.readers = nn.ModuleList([Attend(width) for _ in range(depth)])
        self.output = nn.Linear(width, 8)

    def forward(self, tokens):
        query = self.query[None].expand(len(tokens), -1, -1)
        for reader in self.readers:
            query = reader(query, tokens)
        return self.output(query[:, 0])


class MemoryOutput(nn.Module):
    def __init__(self, agent, teacher):
        super().__init__()
        self.agent, self.teacher = agent, teacher
        self.facts = FactHead(agent.width)
        self.direct = FactHead(agent.width, depth=2)

    def working(self, state):
        return torch.cat(
            [state.tokens[:, self.agent.layout[k]] for k in ("working", "reasoning")], 1
        )

    def observe_history(self, images):
        if images.ndim != 5 or images.shape[1:] != (3, 3, 64, 64):
            raise ValueError("Need three RGB64 history frames")
        state = self.agent.initial_state(len(images))
        stored = None
        for t in range(3):
            obs = Observation(
                images[:, t : t + 1], images.new_full((len(images), 1), float(t))
            )
            state = self.agent.observe(state, {"image": obs}, time=t)
            if t < 2:
                stored = state
                state = self.agent.remember(state, source=f"observed-frame-{t}")
        return dict(stored=stored, final=state)

    def query(self, state, hidden, mode="ordinary"):
        if mode not in (
            "ordinary",
            "ordinary_no_bank",
            "reset",
            "reset_erased",
            "reset_swapped",
        ):
            raise ValueError("Unknown memory-output query condition")
        bank = state.memory
        if mode.startswith("reset"):
            fresh = self.agent.initial_state(len(hidden), time=2)
            obs = Observation(hidden[:, None], hidden.new_full((len(hidden), 1), 2.0))
            state = self.agent.observe(fresh, {"image": obs}, time=2)
        if mode in ("ordinary_no_bank", "reset_erased"):
            bank = None
        if mode == "reset_swapped":
            if bank is None or len(hidden) % 2:
                raise ValueError("Swapping needs a bank and complete adjacent pairs")
            permutation = torch.arange(len(hidden), device=hidden.device) ^ 1
            bank = replace(
                bank,
                keys=bank.keys[permutation],
                values=bank.values[permutation],
                times=bank.times[permutation],
            )
        return self.agent.think(replace(state, memory=bank), steps=2)

    def output(self, state):
        tokens = self.working(state)
        decoder = self.agent.decoders["image"]
        features = decoder.features(tokens)
        return dict(
            facts=self.facts(tokens), image=decoder.head(features), features=features
        )

    def direct_tokens(self, images):
        with torch.no_grad():
            b = len(images)
            times = images.new_tensor([0.0, 1.0]).expand(b, -1)
            encoded = self.agent.encoders["image"](
                Observation(images[:, :2], times)
            ).as_tokens()
            return (encoded.values + position(encoded.times, self.agent.width)).detach()

    def forward(self, images, mode="ordinary"):
        if mode == "erased_history":
            images = images[:, -1:].expand(-1, 3, -1, -1, -1)
            mode = "ordinary"
        history = self.observe_history(images)
        return self.output(self.query(history["final"], images[:, -1], mode))
