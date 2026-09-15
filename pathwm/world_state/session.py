"""Inference owner joining the store and existing BeliefAgent at an explicit boundary.

Training uses the modules' functional forwards and Run; commits intentionally do
not retain BPTT graphs. A session freezes model identity, not global parameters.
External actions are described here, never executed. Supplied candidates remain
an explicit observation adapter responsibility.
"""

from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
import copy
import torch

from pathwm.io import atomic_torch, digest, state_hash
from pathwm.models.belief_state import BeliefState
from .inspection import snapshot_diff
from .modules import BindingDecision
from .records import identifier
from .retrieval import ExactRetriever, Query, RetrievalBudget
from .store import WorldStore, primitive


def candidate_record(c):
    for value in (c.id, c.source, c.modality, c.space, c.model_version):
        identifier(value)
    for value in (c.key, c.value):
        if (
            value.ndim != 1
            or not value.is_floating_point()
            or not torch.isfinite(value).all()
        ):
            raise ValueError("Candidates need finite floating feature vectors")
    if c.key.norm() == 0:
        raise ValueError("Recognition key cannot be zero")
    return dict(
        id=c.id,
        source=c.source,
        modality=c.modality,
        key=c.key.detach().cpu().tolist(),
        value=c.value.detach().cpu().tolist(),
        space=c.space,
        model_version=c.model_version,
        content_ref=c.content_ref,
        exclusive_group=c.exclusive_group,
    )


def packet_record(packet):
    observation = packet.observation
    if observation.provenance is not None:
        raise ValueError("Generated/recalled content is not source observation")
    from pathwm.models.modalities import observation_values

    values, times, valid = observation_values(observation)
    return dict(
        source=packet.source,
        modality=packet.modality,
        shape=list(values.shape),
        dtype=str(values.dtype),
        values=values.detach().cpu().tolist(),
        times=times.detach().cpu().tolist(),
        valid=valid.detach().cpu().tolist(),
    )


@dataclass(frozen=True)
class _Bundle:
    store: WorldStore
    state: object
    decisions: dict
    rng: torch.Tensor
    cuda_rng: tuple


class WorldSession:
    def __init__(
        self,
        *,
        agent,
        binder,
        updater,
        context_encoder,
        retriever=None,
        store=None,
        state=None,
        state_version="state-v1",
        binding_budget=None,
    ):
        self.agent, self.binder, self.updater, self.context_encoder = (
            agent,
            binder,
            updater,
            context_encoder,
        )
        self.retriever = retriever if retriever is not None else ExactRetriever()
        self.state_version = state_version
        self.binding_budget = binding_budget or RetrievalBudget(
            entities=32, components=64, values=8192
        )
        self._signature = self._models_signature()
        # Runtime stochastic state is session-owned; diagnostics/other sessions do not advance it.
        self._bundle = _Bundle(
            store.clone() if store is not None else WorldStore(),
            state
            if state is not None
            else agent.initial_state(1, session_id="world-state"),
            {},
            torch.get_rng_state().clone(),
            tuple(torch.cuda.get_rng_state_all())
            if torch.cuda.is_initialized()
            else (),
        )
        if len(self._bundle.state.tokens) != 1 or self._bundle.state.imagined:
            raise ValueError("WorldSession needs one non-imagined agent state")
        self._check_models()

    def _models_signature(self):
        modules = dict(
            agent=self.agent,
            scorer=self.binder.scorer,
            updater=self.updater,
            context=self.context_encoder,
        )
        return digest(
            dict(
                modules={
                    name: dict(type=repr(module), weights=state_hash(module))
                    for name, module in modules.items()
                },
                context_contract=dict(
                    representations=self.context_encoder.representations,
                    max_tokens=self.context_encoder.max_tokens,
                ),
                retriever=type(self.retriever).__module__
                + "."
                + type(self.retriever).__qualname__,
                state_version=self.state_version,
            )
        )

    def _check_models(self):
        if self._models_signature() != self._signature:
            raise ValueError(
                "Runtime model changed; retrain outside the session and explicitly migrate state"
            )
        if any(
            m.training
            for module in (
                self.agent,
                self.binder.scorer,
                self.updater,
                self.context_encoder,
            )
            for m in module.modules()
        ):
            raise ValueError(
                "Runtime modules must be in eval mode; use functional forwards for training"
            )

    @property
    def store(self):
        return self._bundle.store

    @property
    def state(self):
        return BeliefState.from_dict(self._bundle.state.to_dict())

    @contextmanager
    def _rng(self):
        devices = list(range(len(self._bundle.cuda_rng)))
        with torch.random.fork_rng(devices=devices):
            torch.set_rng_state(self._bundle.rng)
            if devices:
                torch.cuda.set_rng_state_all(list(self._bundle.cuda_rng))
            yield

    @torch.no_grad()
    def observe(
        self,
        event_id,
        *,
        occurred_at,
        available_at,
        candidates=(),
        packets=(),
        action=None,
        trace=None,
        save_to=None,
    ):
        self._check_models()
        if len({c.id for c in candidates}) != len(candidates):
            raise ValueError("Candidate IDs must be unique within an event")
        # Fingerprint input content before inference so completed retries bypass a now-changed graph.
        payload = dict(
            candidates=[candidate_record(c) for c in candidates],
            packets=[packet_record(p) for p in packets],
            action=None if action is None else action.detach().cpu().tolist(),
            occurred_at=occurred_at,
            available_at=available_at,
        )
        payload = primitive(payload)
        existing = self.store.receipt(event_id, payload)
        if existing is not None:
            return copy.deepcopy(self._bundle.decisions[event_id])
        original = self._bundle
        base_revision = original.store.revision
        draft = original.store.clone()
        tx = draft.begin(
            event_id,
            occurred_at=occurred_at,
            available_at=available_at,
            payload=payload,
        )
        decisions, claimed = [], set()
        with self._rng():
            for c in candidates:
                evidence = tx.add_evidence(
                    c.source,
                    c.modality,
                    content_ref=c.content_ref,
                    data=dict(
                        candidate=c.id,
                        space=c.space,
                        model_version=c.model_version,
                        key=c.key.detach().cpu().tolist(),
                        value=c.value.detach().cpu().tolist(),
                    ),
                )
                view = tx.preview()
                context = self.retriever(
                    view,
                    Query(key=c.key, space=c.space, model_version=c.model_version),
                    self.binding_budget,
                    trace=trace,
                )
                decision = self.binder(c, context, trace=trace)
                owner = decision.entity_id
                collision = (
                    (c.exclusive_group, view.canonical(owner))
                    if owner is not None
                    else None
                )
                if c.exclusive_group is not None and collision in claimed:
                    decision = BindingDecision(
                        "unresolved",
                        None,
                        decision.scores,
                        "exclusive candidates collide",
                    )
                if decision.status == "new":
                    owner = tx.create_entity()
                elif decision.status != "matched":
                    owner = None
                if owner is not None:
                    previous = (
                        view.latest(owner, "state")
                        if decision.status == "matched"
                        else None
                    )
                    if previous is not None and (
                        previous.space != "belief"
                        or previous.model_version != self.state_version
                    ):
                        raise ValueError("Incompatible stored state representation")
                    requires_replay = previous is None and any(
                        not c.active for c in view.components(owner, "state")
                    )
                    if requires_replay or (
                        previous is not None and occurred_at < previous.valid_from
                    ):
                        # Keep the evidence; do not run a forward-only cell backwards in time.
                        decision = BindingDecision(
                            "unresolved",
                            None,
                            decision.scores,
                            "invalidated or late state evidence requires replay/reconciliation",
                        )
                        owner = None
                    else:
                        old = (
                            previous.tensor(device=c.value.device, dtype=c.value.dtype)[
                                None
                            ]
                            if previous is not None
                            else c.value.new_zeros(1, self.updater.state_width)
                        )
                        dt = (
                            occurred_at - previous.valid_from
                            if previous is not None
                            else 0.0
                        )
                        updated = self.updater(old, c.value[None], dt)
                        if (
                            updated.shape != (1, self.updater.state_width)
                            or not torch.isfinite(updated).all()
                        ):
                            raise ValueError("Updater returned an invalid state")
                        if trace is not None:
                            trace["update." + c.id + ".before"] = old
                            trace["update." + c.id + ".after"] = updated
                        recognition = tx.put_component(
                            owner,
                            "recognition",
                            c.key.detach(),
                            space=c.space,
                            model_version=c.model_version,
                            evidence=(evidence,),
                        )
                        tx.put_component(
                            owner,
                            "state",
                            updated[0].detach(),
                            space="belief",
                            model_version=self.state_version,
                            evidence=tuple(
                                dict.fromkeys(
                                    (evidence,)
                                    + (
                                        previous.evidence
                                        if previous is not None
                                        else ()
                                    )
                                )
                            ),
                            role="inferred",
                            parents=(recognition,)
                            + ((previous.id,) if previous is not None else ()),
                        )
                        if c.exclusive_group is not None:
                            claimed.add(
                                (
                                    c.exclusive_group,
                                    view.canonical(owner)
                                    if decision.status == "matched"
                                    else owner,
                                )
                            )
                decisions.append(
                    dict(
                        candidate=c.id,
                        status=decision.status,
                        entity_id=owner,
                        reason=decision.reason,
                        scores=decision.scores,
                        evidence=evidence,
                    )
                )
            receipt = draft.commit(tx)
            pending = self.agent.begin_event(
                original.state,
                event_id=event_id,
                ordinal=original.state.ordinal + 1,
                time=available_at,
                action=action,
                trace=trace,
            )
            for packet in packets:
                pending = self.agent.add_packet(pending, packet, trace=trace)
            next_state = self.agent.commit_event(pending)
            result = dict(receipt=receipt, bindings=decisions)
            stored_decisions = {**original.decisions, event_id: result}
            bundle = _Bundle(
                draft,
                next_state,
                stored_decisions,
                torch.get_rng_state().clone(),
                tuple(torch.cuda.get_rng_state_all()) if original.cuda_rng else (),
            )
            if self._bundle is not original or original.store.revision != base_revision:
                raise ValueError("stale session update")
            if save_to is not None:
                atomic_torch(save_to, self._snapshot(bundle))
            self._bundle = bundle
        if trace is not None:
            trace.record("commit", snapshot_diff(original.store, draft))
        return copy.deepcopy(result)

    @torch.no_grad()
    def think(self, query, *, budget=None, steps=1, trace=None, context=None):
        self._check_models()
        retrieved = (
            context
            if context is not None
            else self.retriever(self.store, query, budget, trace=trace)
        )
        if retrieved.revision > self.store.revision:
            raise ValueError("Future retrieval revision")
        encoded = self.context_encoder(
            retrieved, now=float(self._bundle.state.time[0]), trace=trace
        )
        state = self.agent.think(
            self._bundle.state,
            steps=steps,
            goal=encoded.values if encoded.values.shape[1] else None,
            trace=trace,
        )
        # Workspace update never writes observations or changes the persistent graph.
        self._bundle = replace(self._bundle, state=state)
        return self.state, retrieved, encoded

    def _snapshot(self, bundle):
        return dict(
            schema="pathwm-world-session-v1",
            model=self._signature,
            state_version=self.state_version,
            binding_policy=dict(
                match_threshold=self.binder.match_threshold,
                new_threshold=self.binder.new_threshold,
                margin=self.binder.margin,
            ),
            binding_budget=asdict(self.binding_budget),
            world=bundle.store.snapshot(),
            state=bundle.state.to_dict(),
            decisions=copy.deepcopy(bundle.decisions),
            rng=bundle.rng.clone(),
            cuda_rng=[x.clone() for x in bundle.cuda_rng],
        )

    def snapshot(self):
        self._check_models()
        return self._snapshot(self._bundle)

    def save(self, path):
        atomic_torch(path, self.snapshot())

    @classmethod
    def restore(cls, snapshot, **modules):
        snapshot = copy.deepcopy(snapshot)
        if snapshot["schema"] != "pathwm-world-session-v1":
            raise ValueError("Incompatible session schema")
        result = cls(
            **modules,
            store=WorldStore.restore(snapshot["world"]),
            state=BeliefState.from_dict(snapshot["state"]).to(
                next(modules["agent"].parameters()).device
            ),
            state_version=snapshot["state_version"],
            binding_budget=RetrievalBudget(**snapshot["binding_budget"]),
        )
        if (
            snapshot["model"] != result._signature
            or snapshot["binding_policy"]
            != result._snapshot(result._bundle)["binding_policy"]
        ):
            raise ValueError("Incompatible session model or binding policy")
        for event_id, value in snapshot["decisions"].items():
            if result.store.receipt(event_id) != value["receipt"]:
                raise ValueError("Session decisions mismatch committed events")
        result._bundle = replace(
            result._bundle,
            decisions=snapshot["decisions"],
            rng=snapshot["rng"],
            cuda_rng=tuple(snapshot["cuda_rng"]),
        )
        return result


@torch.no_grad()
def rebuild_entity_state(
    transaction, store, entity_id, updater, *, model_version="state-v1"
):
    """Explicit replay after reattribution, using retained supplied-candidate evidence.

    This adapter knows the foundation candidate payload. Replace it for another
    evidence/updater format. It never fabricates new source observations.
    """
    records = sorted(
        (c for c in store.components(entity_id, "recognition") if c.active),
        key=lambda c: (c.valid_from, c.available_at, c.id),
    )
    if not records:
        raise ValueError("No retained observations for state replay")
    if transaction.event.occurred_at != records[-1].valid_from:
        raise ValueError(
            "Replay state time must equal the last retained observation time"
        )
    proofs = {e.id: e for e in store.evidence()}
    param = next(updater.parameters(), torch.empty(0))
    state = torch.zeros(1, updater.state_width, device=param.device, dtype=param.dtype)
    last = records[0].valid_from
    sources = []
    for c in records:
        source = [proofs[e] for e in c.evidence if "value" in proofs[e].data]
        if len(source) != 1 or len(source[0].data["value"]) != updater.input_width:
            raise ValueError(
                "Retained evidence is incompatible with the replay adapter"
            )
        observation = torch.tensor(
            source[0].data["value"], device=state.device, dtype=state.dtype
        )[None]
        state = updater(state, observation, c.valid_from - last)
        last = c.valid_from
        sources.extend(c.evidence)
    return transaction.put_component(
        entity_id,
        "state",
        state[0].detach(),
        space="belief",
        model_version=model_version,
        evidence=tuple(dict.fromkeys(sources)),
        role="inferred",
        parents=tuple(c.id for c in records),
        data={"replayed_from": [c.id for c in records]},
    )
