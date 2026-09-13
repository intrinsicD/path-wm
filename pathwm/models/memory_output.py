"""Read a selected visual history through actual state and detached episodic memory."""

from dataclasses import replace

import torch
from torch import nn

from .modalities import Attend, Observation, position
from .agent import (
    MultimodalAgent,
    ObservationUpdate,
    LatentDynamics,
    Thinker,
    ActionHead,
    ErrorMonitor,
)
from .agent_state import EpisodicMemory
from .decoders import DenseHead, PatchDetailHead, StateFeatureDecoder
from .encoders import PyramidEncoder, PatchDetailEncoder
from .perception import Perception


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


class TokenProbe(nn.Module):
    def __init__(self, training):
        super().__init__()
        if training.ndim != 3 or not torch.isfinite(training).all():
            raise ValueError("Probe calibration requires finite training tokens")
        values = training.detach().double()
        self.register_buffer(
            "mean", values.mean((0, 1), keepdim=True).to(training.dtype)
        )
        self.register_buffer(
            "std",
            values.std((0, 1), correction=0, keepdim=True)
            .clamp_min(1e-4)
            .to(training.dtype),
        )
        self.head = FactHead(training.shape[-1], depth=2)

    def forward(self, tokens):
        return self.head((tokens - self.mean) / self.std)


def load_workspace_reference(path, width, device="cpu"):
    """Load only the stored-working diagnostic; exported agent owns its own copy."""
    record = torch.load(path, map_location="cpu", weights_only=True)
    if record["width"] != width or "stored_working" not in record["stages"]:
        raise ValueError("Reference width/stage must match the working state")
    reference = TokenProbe(torch.zeros(1, 1, width))
    prefix = "stored_working."
    reference.load_state_dict(
        {
            k[len(prefix) :]: v
            for k, v in record["model"].items()
            if k.startswith(prefix)
        },
        strict=True,
    )
    if (
        not all(torch.isfinite(v).all() for v in reference.state_dict().values())
        or not (reference.std > 0).all()
    ):
        raise ValueError("Reference requires finite weights and positive scales")
    return reference.to(device).requires_grad_(False).eval()


class CalibratedMemory(EpisodicMemory):
    """Read-only affine calibration; raw storage and retrieval selection are intact."""

    def __init__(self, width, capacity=4, retrieve_count=2):
        super().__init__(capacity, retrieve_count)
        self.register_buffer("mean", torch.zeros(1, 1, width))
        self.register_buffer("std", torch.ones(1, 1, width))

    @torch.no_grad()
    def calibrate(self, training_values):
        total = square = None
        count = 0
        for value in training_values:
            if value.ndim != 4 or value.shape[-1] != self.mean.shape[-1]:
                raise ValueError("Calibration needs [B,snapshots,tokens,width]")
            flat = value.detach().flatten(0, 2).double()
            if not len(flat) or not torch.isfinite(flat).all():
                raise ValueError("Calibration needs nonempty finite training values")
            total = flat.sum(0) if total is None else total + flat.sum(0)
            square = (
                flat.square().sum(0)
                if square is None
                else square + flat.square().sum(0)
            )
            count += len(flat)
        if not count:
            raise ValueError("Calibration needs training values")
        mean = total / count
        std = (square / count - mean.square()).clamp_min(0).sqrt().clamp_min(1e-4)
        self.mean.copy_(mean[None, None])
        self.std.copy_(std[None, None])

    def read(self, state, trace=None, name="memory"):
        raw = super().read(state, trace=trace, name=name)
        result = (raw - self.mean) / self.std
        if trace is not None:
            trace[name + "_raw_values"] = raw.detach().cpu().clone()
            trace[name + "_calibrated_values"] = result.detach().cpu().clone()
        return result


def configure_recall_repair(model, *, relative_time=False):
    """Only the native reader/workspace, factual head and image producer can learn."""
    memory = model.agent.memory
    if not isinstance(memory, CalibratedMemory):
        model.agent.memory = CalibratedMemory(
            model.agent.width, memory.capacity, memory.retrieve_count
        ).to(model.agent.initial)
    model.agent.memory.relative_time = relative_time
    model.requires_grad_(False)
    model.agent.thinker.requires_grad_(True)
    model.facts.requires_grad_(True)
    model.agent.decoders["image"].requires_grad_(True)
    model.agent.decoders["image"].head.requires_grad_(False)
    return model


class TokenNormalization(nn.Module):
    """Identity or fixed per-channel statistics from this route's training tokens."""

    def __init__(self, width):
        super().__init__()
        self.register_buffer("mean", torch.zeros(1, 1, width))
        self.register_buffer("std", torch.ones(1, 1, width))

    @torch.no_grad()
    def calibrate(self, tokens):
        if (
            tokens.ndim != 3
            or tokens.shape[-1] != self.mean.shape[-1]
            or not tokens.numel()
            or not torch.isfinite(tokens).all()
        ):
            raise ValueError("Normalization requires finite nonempty training tokens")
        x = tokens.detach().double()
        self.mean.copy_(x.mean((0, 1), keepdim=True))
        self.std.copy_(x.std((0, 1), correction=0, keepdim=True).clamp_min(1e-4))

    def forward(self, tokens):
        return (tokens - self.mean) / self.std


def configure_output_readout(model, stage):
    """Freeze both routes; learn only native facts and image feature production."""
    if stage not in ("native", "stored"):
        raise ValueError("Readout stage must be native or stored")
    configure_recall_repair(model)
    model.agent.thinker.requires_grad_(False)
    model.readout_stage = stage
    if not isinstance(model.output_normalization, TokenNormalization):
        model.output_normalization = TokenNormalization(model.agent.width).to(
            model.agent.initial
        )
    return model


def frozen_tensors(model):
    """All frozen named parameters AND buffers, including shared codec aliases."""
    trainable = {
        name
        for name, p in model.named_parameters(remove_duplicate=False)
        if p.requires_grad
    }
    return {
        name: value
        for name, value in model.state_dict().items()
        if name not in trainable
    }


class MemoryOutput(nn.Module):
    def __init__(self, agent, teacher):
        super().__init__()
        self.agent, self.teacher = agent, teacher
        self.facts = FactHead(agent.width)
        self.direct = FactHead(agent.width, depth=2)
        self.workspace_reference = None
        self.readout_stage = "native"
        self.output_normalization = nn.Identity()

    def working(self, state):
        return torch.cat(
            [state.tokens[:, self.agent.layout[k]] for k in ("working", "reasoning")], 1
        )

    def observe_history(self, images):
        if images.ndim != 5 or images.shape[1:] != (3, 3, 64, 64):
            raise ValueError("Need three RGB64 history frames")
        state = self.agent.initial_state(len(images))
        stored = initial = None
        for t in range(3):
            obs = Observation(
                images[:, t : t + 1], images.new_full((len(images), 1), float(t))
            )
            state = self.agent.observe(state, {"image": obs}, time=t)
            if t == 0:
                initial = state
            if t < 2:
                stored = state
                state = self.agent.remember(state, source=f"observed-frame-{t}")
        return dict(initial=initial, stored=stored, final=state)

    def query(self, state, hidden, mode="ordinary"):
        if mode not in (
            "ordinary",
            "ordinary_no_bank",
            "reset",
            "reset_erased",
            "reset_swapped",
            "reset_time_erased",
            "reset_time_swapped",
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
        if mode in ("reset_time_erased", "reset_time_swapped") and bank is not None:
            # Diagnostic metadata interventions; never mutate observed history.
            times = (
                state.time[:, None].expand_as(bank.times)
                if mode == "reset_time_erased"
                else bank.times.flip(1)
            )
            bank = replace(bank, times=times)
        state = replace(state, memory=bank)
        if self.readout_stage == "stored":
            # Reuse causal/shape guards; selection below uses complete-bank times.
            self.agent.memory.read(state)
            if bank is None:
                values = torch.zeros_like(state.tokens)
            else:
                latest = bank.times == bank.times.amax(1, keepdim=True)
                weights = latest.to(bank.values.dtype)
                weights = weights / weights.sum(1, keepdim=True)
                values = (bank.values * weights[:, :, None, None]).sum(1)
            tokens = state.tokens.clone()
            for group in ("working", "reasoning"):
                section = self.agent.layout[group]
                tokens[:, section] = values[:, section]
            return replace(state, tokens=tokens)
        return self.agent.think(state, steps=2)

    def output(self, state):
        return self.output_tokens(self.working(state))

    def output_tokens(self, raw_tokens):
        tokens = self.output_normalization(raw_tokens)
        decoder = self.agent.decoders["image"]
        features = decoder.features(tokens)
        output = dict(
            facts=self.facts(tokens), image=decoder.head(features), features=features
        )
        if self.workspace_reference is not None:
            # Diagnostic/training readout only; never an input to native outputs.
            output["reference_facts"] = self.workspace_reference(raw_tokens)
        return output

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


def make_codec(width=32, levels=3, depth=2, fusion_depth=2, weights=None):
    encoder = PyramidEncoder(
        width=width, levels=levels, depth=depth, fusion_depth=fusion_depth
    )
    options = dict(levels=tuple(encoder.feature_spec), retain_statistics=True)
    base = Perception(
        encoder,
        dict(
            rgb=DenseHead(
                encoder.feature_spec, channels=3, activation="sigmoid", **options
            ),
            mask=DenseHead(encoder.feature_spec, **options),
        ),
    )
    if weights is not None:
        base.load_state_dict(
            torch.load(weights, map_location="cpu", weights_only=True)["model"],
            strict=True,
        )
    return Perception(
        PatchDetailEncoder(base.encoder), dict(rgb=PatchDetailHead(base.heads["rgb"]))
    ).requires_grad_(False)


def build_model(codec, width=32, normalize_input=False):
    encoder = codec.encoder.base.encoder
    if normalize_input:
        encoder = FrozenFeatureNormalization(
            encoder, len(codec.encoder.base.feature_spec), width
        )
    agent = MultimodalAgent(
        width=width,
        encoders={"image": encoder},
        decoders={
            "image": StateFeatureDecoder(
                width, codec.encoder.feature_spec, codec.heads["rgb"]
            )
        },
        updater=ObservationUpdate(width),
        dynamics=LatentDynamics(width),
        thinker=Thinker(width),
        memory=EpisodicMemory(capacity=4, retrieve_count=2),
        action_head=ActionHead(width),
        monitor=ErrorMonitor(width),
    )
    return MemoryOutput(agent, codec.encoder)


def load_model(path, device="cpu"):
    record = torch.load(path, map_location="cpu", weights_only=True)
    settings = record["settings"]
    codec = make_codec(
        **{k: settings[k] for k in ("width", "levels", "depth", "fusion_depth")}
    )
    model = build_model(
        codec, settings["width"], settings.get("normalize_input", False)
    )
    if settings.get("recall_repair"):
        configure_recall_repair(
            model, relative_time=settings["recall_repair"] == "temporal"
        )
    if settings.get("workspace_reference_sha256"):
        model.workspace_reference = TokenProbe(torch.zeros(1, 1, settings["width"]))
        model.workspace_reference.requires_grad_(False)
    if settings.get("readout_stage"):
        configure_output_readout(model, settings["readout_stage"])
    model.load_state_dict(record["model"], strict=True)
    return model.to(device).eval()
