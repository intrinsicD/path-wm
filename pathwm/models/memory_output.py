"""Read a selected visual history through actual state and detached episodic memory."""

from dataclasses import replace

import torch
from torch import nn

from .modalities import Attend, Observation, position


class FrozenFeatureNormalization(nn.Module):
    """Fixed training-observation channel statistics; retain complete provenance."""

    def __init__(self, base, levels, width):
        super().__init__()
        if any(p.requires_grad for p in base.parameters()):
            raise ValueError("Input normalization expects a frozen encoder")
        self.base, self.levels, self.code_width = base, levels, base.code_width
        for i in range(levels):
            self.register_buffer(f"mean_{i}", torch.zeros(1, 1, width))
            self.register_buffer(f"std_{i}", torch.ones(1, 1, width))

    @torch.no_grad()
    def calibrate(self, observations):
        totals, squares, counts = (
            [None] * self.levels,
            [None] * self.levels,
            [0] * self.levels,
        )
        for obs in observations:
            if obs.valid is not None and not obs.valid.any():
                continue
            for i, scale in enumerate(self.base(obs).scales):
                values = scale.values[scale.valid].double()
                if len(values) == 0:
                    continue
                if not torch.isfinite(values).all():
                    raise ValueError("Calibration requires finite valid features")
                total, square = values.sum(0), values.square().sum(0)
                totals[i] = total if totals[i] is None else totals[i] + total
                squares[i] = square if squares[i] is None else squares[i] + square
                counts[i] += len(values)
        if not all(counts):
            raise ValueError(
                "Calibration needs valid training observations at every level"
            )
        for i, count in enumerate(counts):
            mean = totals[i] / count
            std = (
                (squares[i] / count - mean.square()).clamp_min(0).sqrt().clamp_min(0.01)
            )
            getattr(self, f"mean_{i}").copy_(mean[None, None])
            getattr(self, f"std_{i}").copy_(std[None, None])

    def forward(self, observation, **kwargs):
        pyramid = self.base(observation, **kwargs)
        scales = tuple(
            replace(
                scale,
                values=(
                    (scale.values - getattr(self, f"mean_{i}"))
                    / getattr(self, f"std_{i}")
                ).masked_fill(~scale.valid[..., None], 0),
            )
            for i, scale in enumerate(pyramid.scales)
        )
        trace = kwargs.get("trace")
        if trace is not None:
            for i, (raw, scaled) in enumerate(zip(pyramid.scales, scales)):
                trace[f"normalization.{i}.input"] = raw.values.detach().cpu().clone()
                trace[f"normalization.{i}.mean"] = (
                    getattr(self, f"mean_{i}").detach().cpu().clone()
                )
                trace[f"normalization.{i}.std"] = (
                    getattr(self, f"std_{i}").detach().cpu().clone()
                )
                trace[f"scale.{i}.values"] = scaled.values.detach().cpu().clone()
        return replace(pyramid, scales=scales)


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
        if mode in ("cue_erased", "last_seen_erased"):
            images = images.clone()
            images[:, 0 if mode == "cue_erased" else 1] = images[:, -1]
            mode = "reset"
        if mode == "erased_history":
            images = images[:, -1:].expand(-1, 3, -1, -1, -1)
            mode = "ordinary"
        history = self.observe_history(images)
        return self.output(self.query(history["final"], images[:, -1], mode))
