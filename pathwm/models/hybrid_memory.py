"""Bounded two-view session memory; independent reads and differentiable compression."""

from dataclasses import replace
import math

import torch
from torch import nn

from .belief_state import MemoryRecord, SessionMemory, tree_map
from .modalities import Attend, position


def detached(value):
    return tree_map(value, lambda x: x.detach().clone())


def belief_tokens(record, projection):
    if record.logits is None:
        return record.belief
    probabilities = projection(record.logits.softmax(-1))
    groups = torch.arange(
        probabilities.shape[1], device=probabilities.device, dtype=probabilities.dtype
    )
    probabilities = probabilities + position(groups, probabilities.shape[-1])[None]
    return torch.cat((record.belief, probabilities), 1)


class MemoryRead(nn.Module):
    def __init__(self, width, latent_codes=8):
        super().__init__()
        self.distribution = nn.Linear(latent_codes, width)
        self.attention = nn.ModuleList([Attend(width) for _ in range(3)])
        self.gate = nn.Linear(width, 4)
        self.view = nn.Parameter(torch.randn(2, width) * 0.02)
        self.clocks = nn.ModuleList(
            [nn.Linear(width * 3, width), nn.Linear(width * 3, width)]
        )

    def forward(self, query, groups, now, trace=None, name="memory"):
        reads, available = [], []
        for i, records in enumerate(groups):
            values, masks = [], []
            for record in records:
                if ((record.end_time > now) & record.valid).any() or (
                    record.state_time is not None
                    and ((record.state_time > now) & record.valid).any()
                ):
                    raise ValueError(
                        "Memory contains future evidence or inferred state"
                    )
                end_age = position(now - record.end_time, query.shape[-1]).to(
                    query.dtype
                )
                start_age = position(now - record.start_time, query.shape[-1]).to(
                    query.dtype
                )
                # Both views remain labelled and separately encoded; attention may use either.
                for v, tokens in enumerate(
                    (record.evidence, belief_tokens(record, self.distribution))
                ):
                    reference = (
                        record.end_time
                        if v == 0 or record.state_time is None
                        else record.state_time
                    )
                    state_age = position(now - reference, query.shape[-1]).to(
                        query.dtype
                    )
                    clock = self.clocks[v](
                        torch.cat((start_age, end_age, state_age), -1)
                    )
                    values.append(tokens + self.view[v] + clock[:, None])
                    masks.append(record.valid[:, None].expand(-1, tokens.shape[1]))
            valid = (
                torch.cat(masks, 1)
                if masks
                else torch.zeros(len(query), 0, device=query.device, dtype=torch.bool)
            )
            present = valid.any(1)
            # One null key for empty members avoids all-masked attention/NaNs.
            context = torch.cat(
                [*values, query.new_zeros(len(query), 1, query.shape[-1])], 1
            )
            valid = torch.cat((valid, ~present[:, None]), 1)
            output = self.attention[i](
                query, context, valid=valid, trace=trace, name=f"{name}.{i}"
            )
            reads.append(output - query)
            available.append(present)
        mask = torch.stack([*available, torch.ones_like(available[0])], -1)
        weights = self.gate(query).masked_fill(~mask[:, None], -torch.inf).softmax(-1)
        result = sum(read * weights[:, :, i : i + 1] for i, read in enumerate(reads))
        if trace is not None:
            trace[name + ".gates"] = weights.detach().cpu()
        return result


class HybridMemory(nn.Module):
    def __init__(
        self,
        width,
        *,
        recent=32,
        block=8,
        blocks=16,
        compressed_tokens=8,
        protected=8,
        consolidated_tokens=8,
        latent_codes=8,
    ):
        super().__init__()
        if (
            min(
                recent, block, blocks, compressed_tokens, protected, consolidated_tokens
            )
            < 1
        ):
            raise ValueError("Memory capacities must be positive")
        self.width = width
        self.distribution = nn.Linear(latent_codes, width)
        self.capacity, self.block_size, self.block_capacity = recent, block, blocks
        self.protected_capacity = protected
        self.compressed_tokens, self.consolidated_tokens = (
            compressed_tokens,
            consolidated_tokens,
        )
        self.queries = nn.Parameter(torch.randn(2, compressed_tokens, width) * 0.02)
        self.initial_consolidated = nn.Parameter(
            torch.randn(2, consolidated_tokens, width) * 0.02
        )
        self.compressors = nn.ModuleList([Attend(width), Attend(width)])
        self.clocks = nn.ModuleList(
            [nn.Linear(width * 2, width), nn.Linear(width * 2, width)]
        )
        self.consolidators = nn.ModuleList([Attend(width), Attend(width)])
        self.consolidation_gates = nn.ModuleList(
            [nn.Linear(width * 2, width) for _ in range(2)]
        )
        self.readers = nn.ModuleDict(
            {
                name: MemoryRead(width, latent_codes)
                for name in ("perception", "prediction", "thinking")
            }
        )
        self.mark_score = nn.Sequential(
            nn.Linear(width * 2, width), nn.GELU(), nn.Linear(width, 1)
        )

    def extra_repr(self):
        return f"recent={self.capacity}, block={self.block_size}, blocks={self.block_capacity}, protected={self.protected_capacity}, views=2"

    def groups(self, bank):
        if bank is None:
            return ((), (), ())
        return (
            bank.recent + bank.staging,
            bank.compressed
            + (() if bank.consolidated is None else (bank.consolidated,)),
            bank.protected,
        )

    def read(
        self, state, trace=None, name="memory", *, query=None, consumer="thinking"
    ):
        query = state.tokens if query is None else query
        if state.memory is not None and state.memory.session_id != state.session_id:
            raise ValueError("Memory belongs to another session")
        if state.memory is None:
            return torch.zeros_like(query)
        groups = self.groups(state.memory)
        if any(record.ordinal > state.ordinal for group in groups for record in group):
            raise ValueError("Memory contains a later event ordinal")
        return self.readers[consumer](query, groups, state.time, trace, name)

    def summarize(self, records, old=None):
        if not records:
            raise ValueError("Cannot compress empty history")
        first = records[0]
        outputs = []
        for i, view in enumerate(("evidence", "belief")):
            values, masks = [], []
            for record in records:
                # Explicit chronology survives permutation-invariant compression.
                start = position(record.start_time, self.width).to(record.belief.dtype)
                end_time = (
                    record.end_time
                    if view == "evidence" or record.state_time is None
                    else record.state_time
                )
                end = position(end_time, self.width).to(record.belief.dtype)
                clock = self.clocks[i](torch.cat((start, end), -1))
                tokens = (
                    record.evidence
                    if view == "evidence"
                    else belief_tokens(record, self.distribution)
                )
                values.append(tokens + clock[:, None])
                masks.append(record.valid[:, None].expand(-1, tokens.shape[1]))
            values = torch.cat(values, 1)
            valid = torch.cat(masks, 1)
            present = valid.any(1)
            values = torch.cat(
                (values, values.new_zeros(len(values), 1, values.shape[-1])), 1
            )
            valid = torch.cat((valid, ~present[:, None]), 1)
            if old is None:
                query = self.queries[i][None].expand(len(values), -1, -1)
                value = self.compressors[i](query, values, valid=valid)
            else:
                query = getattr(old, view)
                candidate = self.consolidators[i](query, values, valid=valid)
                gate = self.consolidation_gates[i](
                    torch.cat((query, candidate), -1)
                ).sigmoid()
                value = query + gate * (candidate - query)
            outputs.append(
                torch.where(
                    present[:, None, None],
                    value,
                    torch.zeros_like(value) if old is None else query,
                )
            )
        sources = tuple(dict.fromkeys(s for r in records for s in r.sources))
        support = ((old,) if old is not None else ()) + tuple(records)
        starts = torch.stack(
            [r.start_time.masked_fill(~r.valid, torch.inf) for r in support]
        )
        ends = torch.stack(
            [r.end_time.masked_fill(~r.valid, -torch.inf) for r in support]
        )
        valid = torch.stack([r.valid for r in support]).any(0)
        start = torch.where(valid, starts.min(0).values, first.start_time)
        end = torch.where(valid, ends.max(0).values, first.end_time)
        # Bounded source support explicitly records loss of exact old IDs.
        if old is not None:
            sources = tuple(dict.fromkeys((*old.sources, *sources)))
        return MemoryRecord(
            outputs[1],
            outputs[0],
            valid,
            start,
            end,
            sources[-32:],
            records[-1].event_id,
            records[-1].ordinal,
            omitted_sources=sum(r.omitted_sources for r in support)
            + max(0, len(sources) - 32),
            state_time=torch.where(
                valid,
                torch.stack(
                    [
                        (
                            r.end_time if r.state_time is None else r.state_time
                        ).masked_fill(~r.valid, -torch.inf)
                        for r in support
                    ]
                )
                .max(0)
                .values,
                first.end_time,
            ),
        )

    def consolidate(self, block, old):
        if old is None:
            b = len(block.belief)
            empty = torch.zeros_like(block.valid)
            old = replace(
                block,
                belief=self.initial_consolidated[1][None].expand(b, -1, -1),
                evidence=self.initial_consolidated[0][None].expand(b, -1, -1),
                valid=empty,
                sources=(),
                omitted_sources=0,
            )
        return self.summarize((block,), old=old)

    def write(self, state, source=None, *, replay=False):
        if state.imagined or state.phase != "posterior":
            raise ValueError(
                "Only sealed observed live events can enter evidence memory"
            )
        if not state.evidence_valid.any():
            return state
        bank = state.memory or SessionMemory(state.session_id)
        if bank.session_id != state.session_id:
            raise ValueError("Memory belongs to another session")
        if bank.recent and bank.recent[-1].ordinal >= state.ordinal:
            if bank.recent[-1].event_id == state.event_id:
                return state
            raise ValueError("Memory events must be ordered")
        record = detached(
            MemoryRecord(
                state.tokens[:, : state.h.shape[1]],
                state.evidence,
                state.evidence_valid,
                state.evidence_start,
                state.evidence_end,
                state.sources,
                state.event_id,
                state.ordinal,
                state.h,
                state.logits,
                state.z,
                state_time=state.time,
                last_observed_time=state.observed_time,
                source_valid=state.source_valid,
                source_start=state.source_start,
                source_end=state.source_end,
            )
        )
        recent, staging = bank.recent + (record,), bank.staging
        blocks, consolidated = bank.compressed, bank.consolidated
        if len(recent) > self.capacity:
            staging, recent = staging + (recent[0],), recent[1:]
        if len(staging) == self.block_size:
            compressed = self.summarize(detached(staging))
            blocks, staging = blocks + (compressed,), ()
            if len(blocks) > self.block_capacity:
                # Replay gradients enter this consolidation update, not unbounded old graphs.
                consolidated = self.consolidate(
                    detached(blocks[0]), detached(consolidated)
                )
                blocks = blocks[1:]
        updated = SessionMemory(
            bank.session_id, recent, staging, blocks, bank.protected, consolidated
        )
        if not replay:
            updated = detached(updated)
        return replace(state, memory=updated)

    def propose_mark_score(self, record, task):
        return self.mark_score(
            torch.cat((record.belief.mean(1), task.mean(1)), -1)
        ).squeeze(-1)

    def mark(self, state, *, author, detail, score=None):
        if author not in ("user", "agent") or not detail or len(detail) > 256:
            raise ValueError(
                "Mark requires user/agent author and bounded selected-detail description"
            )
        if state.imagined or state.memory is None or not state.memory.recent:
            raise ValueError("Mark requires a live observed event")
        record = state.memory.recent[-1]
        score = (
            float(self.propose_mark_score(record, state.tokens).detach().mean())
            if score is None
            else float(score)
        )
        if not math.isfinite(score):
            raise ValueError("Mark score must be finite")
        if author == "agent" and score <= 0:
            return state
        record = replace(record, author=author, detail=detail, score=score)
        protected = list(state.memory.protected)
        for i, old in enumerate(protected):
            if old.event_id == record.event_id:
                if old.author == "user" and author == "agent":
                    return state
                protected[i] = record
                return replace(
                    state, memory=replace(state.memory, protected=tuple(protected))
                )
        if len(protected) == self.protected_capacity:
            candidates = [
                (r.score, i) for i, r in enumerate(protected) if r.author == "agent"
            ]
            if not candidates:
                if author == "user":
                    raise ValueError(
                        "Protected capacity is full of user marks; additional admission rejected"
                    )
                return state
            minimum, index = min(candidates)
            if author == "agent" and score <= minimum:
                return state
            protected.pop(index)
        protected.append(record)
        return replace(state, memory=replace(state.memory, protected=tuple(protected)))

    @staticmethod
    def storage_bytes(bank):
        """Allocated tensor payload, conservatively counts shared protected references twice.

        Strings/containers and autograd activations are additional; source/detail lengths
        and all record counts are bounded independently.
        """
        count = 0

        def visit(x):
            nonlocal count
            count += x.numel() * x.element_size()
            return x

        tree_map(bank, visit)
        return count

    def capacity_bytes(
        self,
        context_tokens,
        groups,
        codes,
        evidence_tokens,
        *,
        batch_size=1,
        element_size=4,
    ):
        """Conservative maximum tensor payload, including full source-support tables.

        Python/string overhead, live state, model weights and transient activations
        are additional. Compressed source ID tables are bounded at 32 strings.
        """
        full = (
            (2 * context_tokens + evidence_tokens) * self.width + groups * codes
        ) * element_size
        full += groups * 8 + 4 * 8 + 1 + 32 * (1 + 2 * 8)
        compressed = 2 * self.compressed_tokens * self.width * element_size + 3 * 8 + 1
        consolidated = (
            2 * self.consolidated_tokens * self.width * element_size + 3 * 8 + 1
        )
        return batch_size * (
            (self.capacity + self.block_size - 1 + self.protected_capacity) * full
            + self.block_capacity * compressed
            + consolidated
        )
