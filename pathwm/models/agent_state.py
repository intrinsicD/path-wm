"""Caller-owned state, provenance and detached bounded episodic memory."""

from dataclasses import dataclass, replace

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class MemoryBank:
    keys: torch.Tensor  # [B,E,D]
    values: torch.Tensor  # [B,E,N,D]
    times: torch.Tensor  # [B,E]
    sources: tuple[str, ...]  # one provenance label per batched write

    def to_dict(self):
        return {
            k: v.detach().clone() if isinstance(v, torch.Tensor) else v
            for k, v in vars(self).items()
        }


@dataclass(frozen=True)
class LatentState:
    tokens: torch.Tensor  # [B,N,D]; role slices belong to the model
    log_scale: torch.Tensor  # conditional diagonal Gaussian scale
    time: torch.Tensor  # [B] environmental time in seconds
    observed_time: torch.Tensor  # [B] time of last assimilation
    imagined: bool = False
    thinking_steps: int = 0
    memory: MemoryBank | None = None
    observation_count: int = 0

    def to_dict(self):
        return {
            "schema": "pathwm-latent-v1",
            **{
                k: v.detach().clone() if isinstance(v, torch.Tensor) else v
                for k, v in vars(self).items()
                if k != "memory"
            },
            "memory": None if self.memory is None else self.memory.to_dict(),
        }

    def to(self, device, dtype=None):
        """Move caller-owned state independently of the model's parameters."""

        def move(value):
            return value.to(device=device, dtype=dtype or value.dtype)

        bank = self.memory
        if bank is not None:
            bank = MemoryBank(
                move(bank.keys),
                move(bank.values),
                bank.times.to(device=device),
                bank.sources,
            )
        return replace(
            self,
            tokens=move(self.tokens),
            log_scale=move(self.log_scale),
            time=self.time.to(device=device),
            observed_time=self.observed_time.to(device=device),
            memory=bank,
        )

    @classmethod
    def from_dict(cls, record):
        record = dict(record)
        if record.pop("schema") != "pathwm-latent-v1":
            raise ValueError("Unknown latent state schema")
        bank = record.pop("memory")
        return cls(**record, memory=None if bank is None else MemoryBank(**bank))


class EpisodicMemory(nn.Module):
    """Bounded FIFO snapshots with cosine retrieval, entirely owned by the state.

    Retrieval is nondifferentiable selection. Selected values feed differentiable
    attention. Batch members are separate episode streams, never mixed together.
    """

    def __init__(self, capacity=16, retrieve_count=2):
        super().__init__()
        if min(capacity, retrieve_count) < 1:
            raise ValueError("Memory capacity and retrieval count must be positive")
        self.capacity, self.retrieve_count = capacity, retrieve_count

    def extra_repr(self):
        return f"capacity={self.capacity}, retrieve_count={self.retrieve_count}"

    def write(self, state, source):
        if state.imagined:
            raise ValueError("Cannot store imagined states as observed memories")
        if state.observation_count < 1:
            raise ValueError("Cannot remember an unobserved initial state")
        if not isinstance(source, str) or not source.strip():
            raise ValueError("Memory needs an explicit observation source")
        values = state.tokens.detach().clone()[:, None]
        keys = F.normalize(values.mean(2), dim=-1)
        times = state.observed_time.detach().clone()[:, None]
        sources = (source,)
        if state.memory is not None:
            bank = state.memory
            if (times[:, 0] < bank.times[:, -1]).any():
                raise ValueError("Memory cannot be written backward in time")
            keys, values, times = (
                torch.cat((old, new), 1)
                for old, new in (
                    (bank.keys, keys),
                    (bank.values, values),
                    (bank.times, times),
                )
            )
            sources = bank.sources + sources
        bank = MemoryBank(
            keys[:, -self.capacity :],
            values[:, -self.capacity :],
            times[:, -self.capacity :],
            sources[-self.capacity :],
        )
        return replace(state, memory=bank)

    def read(self, state, trace=None, name="memory"):
        bank = state.memory
        if bank is None:
            return state.tokens[:, :0]
        if (
            bank.values.shape[0] != len(state.tokens)
            or bank.values.shape[-2:] != state.tokens.shape[-2:]
        ):
            raise ValueError("Memory shape differs from latent state")
        if (bank.times > state.observed_time[:, None]).any():
            raise ValueError("Memory contains future observations")
        score = torch.einsum(
            "bd,bed->be", F.normalize(state.tokens.mean(1), dim=-1), bank.keys
        )
        indices = score.topk(min(self.retrieve_count, score.shape[1]), dim=1).indices
        values = bank.values[
            torch.arange(len(score), device=score.device)[:, None], indices
        ]
        if trace is not None:
            trace[name + "_indices"] = indices.detach().cpu().clone()
            trace[name + "_scores"] = score.detach().cpu().clone()
            trace[name + "_sources"] = list(bank.sources)
        return values.flatten(1, 2)
