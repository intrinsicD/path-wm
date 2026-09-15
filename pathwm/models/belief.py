"""Categorical predict/correct agent, using the existing modality and task interfaces."""

from dataclasses import replace
import uuid

import torch
from torch import nn
from torch.nn import functional as F

from .agent import MultimodalAgent, batch_time, record_state
from .belief_state import BeliefState, Packet, PendingEvent
from .modalities import Attend, Observation, observation_values, position
from .multiscale import FeaturePyramid


def distribution(logits, unimix=0.01):
    return ((1 - unimix) * logits.softmax(-1) + unimix / logits.shape[-1]).log()


def draw(logits, noise=None, *, sample=True):
    probabilities = logits.softmax(-1)
    if sample:
        noise = torch.rand_like(probabilities[..., :1]) if noise is None else noise
        code = (
            (noise > probabilities.cumsum(-1)).sum(-1).clamp_max(logits.shape[-1] - 1)
        )
    else:
        code = logits.argmax(-1)
    hard = F.one_hot(code, logits.shape[-1]).to(logits.dtype)
    return code, hard + (probabilities - probabilities.detach())


class BeliefDynamics(nn.Module):
    def __init__(self, width, groups, codes, action_width):
        super().__init__()
        self.action_width = action_width
        self.code = nn.Linear(groups * codes, width, bias=False)
        self.action = nn.Linear(action_width + 1, width)
        self.transition = Attend(width)
        self.head = nn.Linear(width, groups * codes)
        self.groups, self.codes = groups, codes

    def prior(self, h):
        return distribution(
            self.head(h.mean(1)).reshape(len(h), self.groups, self.codes)
        )

    def forward(self, state, action, dt, memory, trace=None):
        b, width = len(state.h), state.h.shape[-1]
        value = state.h.new_zeros(b, self.action_width) if action is None else action
        if (
            value.shape != (b, self.action_width)
            or value.device != state.h.device
            or not torch.isfinite(value).all()
        ):
            raise ValueError(
                "Action must be finite [B,action_width] on the state device"
            )
        present = value.new_full((b, 1), float(action is not None))
        condition = self.action(torch.cat((value, present), -1)) + position(
            dt, width
        ).to(state.h.dtype)
        query = state.h + self.code(state.stochastic.flatten(1))[:, None]
        # No multiplication by dt: instantaneous recorded actions remain meaningful.
        h = self.transition(
            query,
            torch.cat((query, memory, condition[:, None]), 1),
            trace=trace,
            name="imagine.attention.0",
        )
        return h, self.prior(h)


class BeliefCorrection(nn.Module):
    def __init__(self, width, groups, codes, action_width):
        super().__init__()
        self.action_width = action_width
        self.observation = Attend(width)
        self.refine = Attend(width)
        self.head = nn.Linear(width, groups * codes)
        self.groups, self.codes = groups, codes

    def forward(self, prior, values, valid, memory, trace=None):
        query = self.observation(
            prior.h, values, valid=valid, trace=trace, name="observe.attention"
        )
        query = query + memory.read(
            prior,
            query=query,
            consumer="perception",
            trace=trace,
            name="observe.memory",
        )
        query = self.refine(
            query, values, valid=valid, trace=trace, name="observe.refine"
        )
        return distribution(
            self.head(query.mean(1)).reshape(len(query), self.groups, self.codes)
        )


class BeliefAgent(MultimodalAgent):
    def __init__(
        self,
        *,
        width=32,
        context_tokens=16,
        latent_groups=8,
        latent_codes=8,
        evidence_tokens=8,
        time_unit="steps",
        **components,
    ):
        if min(context_tokens, latent_groups, evidence_tokens) < 1 or latent_codes < 2:
            raise ValueError(
                "Belief dimensions must be positive with at least two codes"
            )
        if time_unit not in ("steps", "seconds"):
            raise ValueError("Declare steps or seconds as the duration unit")
        super().__init__(
            width=width,
            groups=dict(world=context_tokens, working=4, reasoning=4),
            **components,
        )
        self.latent_groups, self.latent_codes = latent_groups, latent_codes
        memory_projection = getattr(self.memory, "distribution", None)
        if (
            isinstance(memory_projection, nn.Linear)
            and memory_projection.in_features != latent_codes
        ):
            raise ValueError("Memory latent_codes must match the agent's latent_codes")
        self.evidence_tokens, self.time_unit = evidence_tokens, time_unit
        self.evidence_queries = nn.Parameter(torch.randn(evidence_tokens, width) * 0.02)
        self.evidence_encoder = Attend(width)
        self.readout = nn.Linear(latent_groups * latent_codes, width, bias=False)

    def initial_state(self, batch_size, time=0.0, *, session_id=None):
        if session_id is not None and (
            not isinstance(session_id, str) or not session_id or len(session_id) > 256
        ):
            raise ValueError("Session identity must be a bounded nonempty string")
        base = super().initial_state(batch_size, time)
        h = base.tokens[:, self.layout["world"]]
        logits = self.dynamics.prior(h)
        z, stochastic = draw(logits)
        state = BeliefState(
            **vars(base),
            h=h,
            logits=logits,
            z=z,
            stochastic=stochastic,
            prior_logits=logits,
            evidence=h.new_zeros(batch_size, self.evidence_tokens, self.width),
            evidence_valid=torch.zeros(batch_size, dtype=torch.bool, device=h.device),
            evidence_start=base.time.clone(),
            evidence_end=base.time.clone(),
            source_valid=torch.zeros(batch_size, 0, dtype=torch.bool, device=h.device),
            source_start=base.time[:, None][:, :0],
            source_end=base.time[:, None][:, :0],
            observation_counts=torch.zeros(
                batch_size, dtype=torch.long, device=h.device
            ),
            session_id=session_id or uuid.uuid4().hex,
            time_unit=self.time_unit,
        )
        return self._readout(state)

    def validate_state(self, state):
        if not isinstance(state, BeliefState):
            raise ValueError(
                "Categorical agent requires a BeliefState; Gaussian checkpoints need their original model"
            )
        super().validate_state(state)
        b = len(state.tokens)
        if (
            state.h.shape != (b, self.groups["world"], self.width)
            or state.logits.shape != (b, self.latent_groups, self.latent_codes)
            or state.stochastic.shape != state.logits.shape
            or state.z.shape != (b, self.latent_groups)
            or state.z.dtype != torch.long
            or ((state.z < 0) | (state.z >= self.latent_codes)).any()
            or state.time_unit != self.time_unit
            or state.time.dtype != torch.float64
            or state.observed_time.dtype != torch.float64
            or not all(
                torch.isfinite(x).all()
                for x in (state.h, state.logits, state.stochastic)
            )
        ):
            raise ValueError(
                "Invalid categorical belief dimensions, codes, clock or values"
            )
        if state.memory is not None and state.memory.session_id != state.session_id:
            raise ValueError("Memory belongs to another session")

    def _readout(self, state):
        world = state.h + self.readout(state.stochastic.flatten(1))[:, None]
        tokens = torch.cat((world, state.tokens[:, self.groups["world"] :]), 1)
        return replace(state, tokens=tokens)

    def _advance(self, state, action, dt, *, sample=True, trace=None):
        self.validate_state(state)
        interval = batch_time(dt, state.tokens)
        if (interval < 0).any():
            raise ValueError("Transition duration cannot be negative")
        query = state.h + self.readout(state.stochastic.flatten(1))[:, None]
        context = self.memory.read(
            state,
            query=query,
            consumer="prediction",
            trace=trace,
            name="predict.memory",
        )
        h, logits = self.dynamics(state, action, interval, context, trace)
        z, stochastic = draw(logits, sample=sample)
        result = replace(
            state,
            h=h,
            logits=logits,
            prior_logits=logits,
            z=z,
            stochastic=stochastic,
            time=state.time + interval,
            thinking_steps=0,
            phase="prior",
            evidence=torch.zeros_like(state.evidence),
            evidence_valid=torch.zeros_like(state.evidence_valid),
            sources=(),
            evidence_start=state.observed_time,
            evidence_end=state.observed_time,
            source_valid=state.source_valid[:, :0],
            source_start=state.source_start[:, :0],
            source_end=state.source_end[:, :0],
        )
        return self._readout(result)

    def begin_event(
        self, state, *, event_id, ordinal, time, action=None, replay=False, trace=None
    ):
        if state.imagined:
            raise ValueError("Cannot promote an imagined branch into a live event")
        if not isinstance(event_id, str) or not event_id or len(event_id) > 256:
            raise ValueError("Event identity must be a nonempty bounded string")
        if (
            not isinstance(ordinal, int)
            or ordinal <= state.ordinal
            or event_id == state.event_id
        ):
            raise ValueError(
                "Event ordinal must increase and event identity must be new"
            )
        now = batch_time(time, state.tokens)
        prior = self._advance(state, action, now - state.time, trace=trace)
        prior = replace(prior, event_id=event_id, ordinal=ordinal)
        return PendingEvent(
            prior, prior, torch.rand_like(prior.logits[..., :1]), replay=replay
        )

    def _features(self, pending, packets, trace=None):
        prior = pending.prior
        for packet in packets:
            observation = packet.observation
            if packet.modality not in self.encoders:
                raise ValueError("Packet modality has no encoder")
            if observation.provenance is not None:
                raise ValueError(
                    "Generated content cannot enter source evidence; use reflect"
                )
            _, times, valid = observation_values(observation)
            if (
                len(times) != len(prior.tokens)
                or ((times > prior.time[:, None]) & valid).any()
            ):
                raise ValueError("Packet batch mismatch or future observation")
        values, validity, labels, scales = [], [], [], []
        for packet in packets:
            encoder = self.encoders[packet.modality]
            local_trace = {} if trace is not None else None
            # Source-only path: no current belief, workspace, task or controller conditioning.
            if hasattr(encoder, "code_width"):
                features = encoder(
                    packet.observation, condition=None, trace=local_trace
                )
            else:
                features = encoder(packet.observation)
            if trace is not None and local_trace:
                trace.update(
                    {f"encode.{packet.modality}.{k}": v for k, v in local_trace.items()}
                )
            if isinstance(features, FeaturePyramid):
                item = features.as_tokens()
                scales.extend(
                    f"{packet.modality}.scale.{i}"
                    for i, scale in enumerate(features.scales)
                    for _ in range(scale.values.shape[1])
                )
            else:
                item = features
                scales.extend([packet.modality] * item.values.shape[1])
            tokens = item.values + position(
                item.times - prior.time[:, None], self.width
            ).to(item.values.dtype)
            values.append(tokens.masked_fill(~item.valid[:, :, None], 0))
            validity.append(item.valid)
            labels.extend([packet.modality] * tokens.shape[1])
        values, valid = torch.cat(values, 1), torch.cat(validity, 1)
        present = valid.any(1)
        values = torch.cat((values, values.new_zeros(len(values), 1, self.width)), 1)
        valid = torch.cat((valid, ~present[:, None]), 1)
        if trace is not None:
            trace["observe.input_modalities"] = labels + ["empty_member_null"]
            trace["observe.input_scales"] = scales + ["empty_member_null"]
            trace["observe.valid"] = valid.detach().cpu()
        return values, valid, present

    def add_packet(self, pending, packet, *, trace=None):
        if (
            not isinstance(packet, Packet)
            or not packet.source
            or len(packet.source) > 256
        ):
            raise ValueError("Packet needs a bounded source identity")
        # Canonical sanitized content makes masked NaNs immaterial to deduplication too.
        values, times, valid = observation_values(packet.observation)
        packet = replace(
            packet,
            observation=Observation(
                values.clone(),
                times.clone(),
                valid.clone(),
                packet.observation.provenance,
            ),
        )
        for old in pending.packets:
            if old.source == packet.source:
                equal = (
                    old.modality == packet.modality
                    and old.observation.provenance == packet.observation.provenance
                )
                equal = equal and all(
                    torch.equal(
                        getattr(old.observation, k), getattr(packet.observation, k)
                    )
                    for k in ("values", "times", "valid")
                )
                if equal:
                    return pending
                raise ValueError("Conflicting duplicate source packet")
        if len(pending.packets) >= 32:
            raise ValueError(
                "Event packet capacity exceeded; seal or use another event"
            )
        packets = tuple(
            sorted((*pending.packets, packet), key=lambda p: (p.modality, p.source))
        )
        return self.correct_packets(replace(pending, packets=packets), trace=trace)

    def correct_packets(self, pending, *, trace=None):
        canonical = {}
        if len(pending.packets) > 32:
            raise ValueError("Event packet capacity exceeded")
        for packet in pending.packets:
            if (
                not packet.source
                or len(packet.source) > 256
                or packet.source in canonical
            ):
                raise ValueError("Correction requires unique bounded source identities")
            values, times, valid = observation_values(packet.observation)
            canonical[packet.source] = replace(
                packet,
                observation=Observation(
                    values, times, valid, packet.observation.provenance
                ),
            )
        pending = replace(
            pending,
            packets=tuple(
                sorted(canonical.values(), key=lambda p: (p.modality, p.source))
            ),
        )
        if not pending.packets:
            return replace(pending, state=pending.prior)
        values, valid, present = self._features(pending, pending.packets, trace)
        prior = pending.prior
        if not present.any():
            return replace(pending, state=prior)
        logits = self.updater(prior, values, valid, self.memory, trace)
        logits = torch.where(present[:, None, None], logits, prior.logits)
        z, stochastic = draw(logits, pending.noise)
        z = torch.where(present[:, None], z, prior.z)
        stochastic = torch.where(present[:, None, None], stochastic, prior.stochastic)
        evidence = self.evidence_encoder(
            self.evidence_queries[None].expand(len(values), -1, -1), values, valid=valid
        )
        evidence = evidence * present[:, None, None]
        supported = [p for p in pending.packets if p.observation.valid.any()]
        source_valid = torch.stack([p.observation.valid.any(1) for p in supported], 1)
        starts = torch.stack(
            [
                p.observation.times.to(torch.float64)
                .masked_fill(~p.observation.valid, torch.inf)
                .min(1)
                .values
                for p in supported
            ],
            1,
        )
        ends = torch.stack(
            [
                p.observation.times.to(torch.float64)
                .masked_fill(~p.observation.valid, -torch.inf)
                .max(1)
                .values
                for p in supported
            ],
            1,
        )
        evidence_start = torch.where(present, starts.min(1).values, prior.observed_time)
        evidence_end = torch.where(present, ends.max(1).values, prior.observed_time)
        starts = torch.where(source_valid, starts, prior.observed_time[:, None])
        ends = torch.where(source_valid, ends, prior.observed_time[:, None])
        state = replace(
            prior,
            logits=logits,
            z=z,
            stochastic=stochastic,
            evidence=evidence,
            evidence_valid=present,
            observed_time=torch.maximum(prior.observed_time, evidence_end),
            phase="posterior",
            evidence_start=evidence_start,
            evidence_end=evidence_end,
            source_valid=source_valid,
            source_start=starts,
            source_end=ends,
            sources=tuple(p.source for p in supported),
        )
        result = replace(pending, state=self._readout(state))
        record_state(trace, "observe", result.state)
        return result

    def commit_event(self, pending):
        """Pure seal: repeated calls return the same values; no hidden mutable write log."""
        if not isinstance(pending, PendingEvent):
            raise ValueError("Only an open PendingEvent can be sealed")
        state = pending.state
        if not state.evidence_valid.any():
            return state
        state = replace(
            state,
            observation_count=state.observation_count + 1,
            observation_counts=state.observation_counts + state.evidence_valid.long(),
        )
        return self.memory.write(state, replay=pending.replay)

    def observe(
        self,
        state,
        observations,
        *,
        time,
        previous_action=None,
        feature_code=None,
        trace=None,
        replay=False,
    ):
        if feature_code is not None:
            raise ValueError(
                "Source evidence is unconditioned; use feature controls on task/reflection reads"
            )
        ordinal = state.ordinal + 1
        pending = self.begin_event(
            state,
            event_id=f"event-{ordinal}",
            ordinal=ordinal,
            time=time,
            action=previous_action,
            replay=replay,
            trace=trace,
        )
        # One dictionary call is a complete event; callers with partial arrivals use Packet explicitly.
        packets = tuple(
            Packet(f"{ordinal}/{name}", name, observation)
            for name, observation in sorted(observations.items())
        )
        for packet in packets:
            pending = self.add_packet(pending, packet, trace=trace)
        return self.commit_event(pending)

    def remember(self, state, *, source):
        # observe() already commits once. The old recipe's extra call is idempotent.
        self.validate_state(state)
        if state.imagined:
            raise ValueError("Cannot remember an imagined branch as evidence")
        return state

    def mark(self, state, **kwargs):
        return self.memory.mark(state, **kwargs)

    def imagine(self, state, action=None, *, dt=1.0, sample=True, trace=None):
        result = replace(
            self._advance(state, action, dt, sample=sample, trace=trace), imagined=True
        )
        record_state(trace, "imagine", result)
        return result

    def sample_beliefs(self, state, samples=4):
        if not isinstance(samples, int) or not 1 <= samples <= 4:
            raise ValueError(
                "Decision sampling supports one to four temporary branches"
            )
        branches = []
        for _ in range(samples):
            z, stochastic = draw(state.logits)
            branches.append(self._readout(replace(state, z=z, stochastic=stochastic)))
        return branches

    def think(self, state, *, steps=1, goal=None, feedback=None, trace=None):
        if feedback is None:
            p = state.logits.softmax(-1)
            entropy = -(p * p.clamp_min(1e-12).log()).sum(-1).mean(-1)
            feedback = torch.stack(
                (
                    self.monitor(state.tokens),
                    entropy,
                    state.tokens.new_full(state.time.shape, state.thinking_steps),
                ),
                -1,
            )
        return super().think(
            state, steps=steps, goal=goal, feedback=feedback, trace=trace
        )

    def intervene(self, state, role, value):
        if role == "world":
            raise ValueError(
                "A belief intervention must specify h or categorical codes explicitly"
            )
        return super().intervene(state, role, value)
