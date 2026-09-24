"""Small stateless client for evidence-backed entity and conversation state.

The session owns the store, clock and restart state. This client supplies protocol
checks and selected latent reads, not learned language or calibrated uncertainty.
"""

from dataclasses import asdict, dataclass

import torch

from .records import identifier
from .session import SourceItem
from .store import primitive


@dataclass(frozen=True)
class StateRead:
    entity_id: str
    components: tuple
    omitted: tuple[str, ...]
    names: tuple[str, ...]
    representations: tuple
    heads: tuple
    sources: tuple
    omission_reasons: tuple[tuple[str, str], ...] = ()
    revision: int = -1


class EpisodeClient:
    def __init__(self, session, *, representations):
        self.session = session
        self.representations = dict(representations)
        self._versions()

    def _versions(self):
        for name, contract in self.representations.items():
            identifier(name)
            if not isinstance(contract, (tuple, list)) or len(contract) != 2:
                raise ValueError("Representations need (space, model_version)")
            for item in contract:
                identifier(item)
        return tuple(sorted((k, tuple(v)) for k, v in self.representations.items()))

    @property
    def store(self):
        return self.session.store

    def _entity(self, entity_id):
        self.session.check()
        identifier(entity_id)
        if entity_id not in {e.id for e in self.store.entities()}:
            raise ValueError("Unknown entity")
        return self.store.entity(entity_id)

    def _internal(self, event_id, payload):
        self.session.check()
        payload = primitive(payload)
        if self.store.receipt(event_id, payload) is not None:
            return None
        return self.store.begin(event_id, occurred_at=self.session.time,
                                available_at=self.session.time, kind="internal", payload=payload)

    def create(self, event_id, entity_id, *, kind="conversation"):
        identifier(entity_id)
        identifier(kind)
        tx = self._internal(event_id, dict(operation="episode-create", entity=entity_id, kind=kind))
        if tx is not None:
            tx.create_entity(kind=kind, entity_id=entity_id)
            self.session.commit(tx, advance_belief=False)
        return entity_id

    def observe(self, event_id, entity_id, *, source, modality, occurred_at,
                available_at, data=None, content_ref="", content_hash="",
                supersedes=None, provenance=None):
        """Ingest external evidence; caller must carry generated provenance through."""
        self._entity(entity_id)
        if provenance is not None:
            raise ValueError("Generated/recalled material cannot be source evidence")
        identifier(source)
        identifier(modality)
        if not isinstance(content_ref, str) or not isinstance(content_hash, str):
            raise ValueError("Content references and hashes must be strings")
        if data is not None and not isinstance(data, dict):
            raise ValueError("Evidence metadata must be a mapping")
        payload = dict(entity_id=entity_id, detail=primitive(data or {}))
        if supersedes is not None:
            old = next((e for e in self.store.evidence() if e.id == supersedes), None)
            if old is None or old.data.get("entity_id") != entity_id:
                raise ValueError("Replacement must belong to this entity")
        self.session.observe(event_id, occurred_at=occurred_at, available_at=available_at,
                             evidence=(SourceItem(source, modality, content_ref, content_hash,
                                                  payload, supersedes),))
        return next(e.id for e in self.store.evidence() if e.event_id == event_id)

    def _utterances(self, episode_id):
        return [e for e in self.store.evidence()
                if e.data.get("entity_id") == episode_id
                and e.data.get("detail", {}).get("protocol") == "utterance-v1"]

    def utterance(self, event_id, episode_id, *, turn_id, phase, source, time,
                  text="", audio_ref="", provenance=None):
        """Record a source turn's start/chunk/end/cancel; text is an assertion record."""
        if self._entity(episode_id).kind != "conversation":
            raise ValueError("Utterances require a conversation episode")
        identifier(turn_id)
        if phase not in {"start", "chunk", "end", "cancel"}:
            raise ValueError("Unknown utterance phase")
        if not isinstance(text, str) or not isinstance(audio_ref, str):
            raise ValueError("Utterance text and audio reference must be strings")
        # Completed retries go through the session's exact payload check, without
        # interpreting their old phase against a newer protocol state.
        if self.store.receipt(event_id) is None:
            turns = self._utterances(episode_id)
            last = turns[-1] if turns else None
            detail = last.data["detail"] if last else {}
            open_turn = detail.get("phase") in {"start", "chunk"}
            if phase == "start":
                if open_turn or any(e.data["detail"]["turn_id"] == turn_id for e in turns):
                    raise ValueError("Finish/cancel the active turn; turn IDs cannot be reused")
            elif (not open_turn or detail["turn_id"] != turn_id or last.source != source):
                raise ValueError("Chunk/end/cancel must match the open source turn")
        return self.observe(event_id, episode_id, source=source,
                            modality="audio" if audio_ref else "text",
                            occurred_at=time, available_at=time, content_ref=audio_ref,
                            provenance=provenance,
                            data=dict(protocol="utterance-v1", turn_id=turn_id, phase=phase,
                                      text=text, audio_ref=audio_ref))

    def publish(self, event_id, entity_id, name, tensor, *, evidence=(), parents=(),
                data=None, confidence=None):
        """Publish a detached interpretation, never an observation or truth label."""
        self._entity(entity_id)
        self._versions()
        if name not in self.representations:
            raise ValueError("Undeclared state representation")
        if data is not None and not isinstance(data, dict):
            raise ValueError("State metadata must be a mapping")
        value = torch.as_tensor(tensor)
        if (value.requires_grad or value.grad_fn is not None or
                not value.is_floating_point() or not torch.isfinite(value).all()):
            raise ValueError("State must be explicitly detached finite floating values")
        space, version = self.representations[name]
        payload = dict(operation="episode-publish", entity=entity_id, name=name,
                       values=value.cpu().tolist(), shape=list(value.shape), space=space,
                       model_version=version, evidence=list(evidence), parents=list(parents),
                       data=data or {}, confidence=confidence)
        tx = self._internal(event_id, payload)
        if tx is None:
            return next(c.id for c in self.store.components(entity_id, name)
                        if c.id.startswith(event_id + "/component/"))
        result = tx.put_component(entity_id, name, value, space=space,
                                  model_version=version, evidence=evidence, parents=parents,
                                  role="inferred", data=data, confidence=confidence)
        self.session.commit(tx, advance_belief=False)
        return result

    def _pins(self, entity_id, names):
        heads = []
        for name in names:
            head = self.store.latest(entity_id, name)
            heads.append((name, None if head is None else (head.id, head.revision)))
        retracted = self.store.retractions()
        sources = tuple((e.id, e.id in retracted) for e in self.store.evidence()
                        if e.data.get("entity_id") == entity_id)
        return tuple(heads), sources

    def publish_unknown(self, event_id, entity_id, name, *, reason):
        """Explicit caller revision to unknown; never a fabricated observation.

        Consumers must interpret this marker deliberately instead of projecting
        its empty value as a normal code. Earlier evidence/history is retained.
        """
        identifier(reason)
        return self.publish(event_id, entity_id, name, torch.empty(0),
                            data={"availability": "unknown", "reason": reason})

    def load(self, entity_id, names, *, max_values=1024, max_components=16):
        """Selected current states, with explicit omissions; never old-state fallback."""
        self._entity(entity_id)
        names = tuple(names)
        if len(set(names)) != len(names) or any(n not in self.representations for n in names):
            raise ValueError("Select distinct declared component names")
        if any(type(x) is not int or x < 0 for x in (max_values, max_components)):
            raise ValueError("Read budgets must be nonnegative integers")
        versions = self._versions()
        components, omitted, reasons, used = [], [], [], 0
        for name in names:
            c = self.store.latest(entity_id, name)
            reason = None
            if c is None:
                reason = "invalidated" if self.store.components(entity_id, name) else "never_observed"
            elif (c.space, c.model_version) != tuple(self.representations[name]):
                reason = "representation"
            elif len(components) >= max_components or used + len(c.values) > max_values:
                reason = "capacity"
            if reason is not None:
                omitted.append(name)
                reasons.append((name, reason))
            else:
                components.append(c)
                used += len(c.values)
        heads, sources = self._pins(entity_id, names)
        return StateRead(entity_id, tuple(components), tuple(omitted), names, versions,
                         heads, sources, tuple(reasons), self.store.revision)

    def validate(self, read):
        self._entity(read.entity_id)
        if (read.representations != self._versions() or
                (read.heads, read.sources) != self._pins(read.entity_id, read.names)):
            raise ValueError("Stale state read; retrieve current evidence and state again")
        if any(c != self.store.component(c.id) for c in read.components):
            raise ValueError("Read component content was changed after retrieval")
        return True

    def _completed(self, episode_id, turn_id):
        aborted = self._aborted()
        return any(e.id not in aborted and e.kind == "internal" and
                   e.payload.get("operation") == "episode-emit" and
                   e.payload.get("entity") == episode_id and
                   e.payload.get("turn_id") == turn_id and e.payload.get("complete")
                   for e in self.store.events())

    def _aborted(self):
        return {output for e in self.store.events()
                if e.kind == "internal" and e.payload.get("operation") == "episode-abort"
                for output in e.payload["outputs"]}

    def _turn(self, read):
        source_ids = {source for source, _ in read.sources}
        turns = [e for e in self._utterances(read.entity_id) if e.id in source_ids]
        if not turns:
            raise ValueError("No input turn in the output read")
        return turns[-1].data["detail"]["turn_id"]

    def record_abort(self, event_id, episode_id, read, *, reason="input"):
        """Withdraw earlier outputs for this turn without answering it.

        A stale read is allowed here precisely because invalidation/interruption
        may trigger the abort. Retrying requires new reads taken after the abort;
        the caller explicitly repairs state first. No automatic response runs.
        """
        if reason not in {"input", "invalidation"} or read.entity_id != episode_id:
            raise ValueError("Abort needs a matching episode and declared reason")
        if self._entity(episode_id).kind != "conversation":
            raise ValueError("Abort requires a conversation episode")
        turn_id = self._turn(read)
        # Acknowledged retries retain the originally named outputs even if a
        # later rederived response has since been committed for the same turn.
        old = next((e for e in self.store.events() if e.id == event_id), None)
        outputs = old.payload.get("outputs", []) if old else [
            e.id for e in self.store.events()
            if e.payload.get("operation") == "episode-emit"
            and e.payload.get("entity") == episode_id and e.payload.get("turn_id") == turn_id]
        tx = self._internal(event_id, dict(operation="episode-abort", entity=episode_id,
                                          turn_id=turn_id, reason=reason, read=asdict(read), outputs=outputs))
        if tx is not None:
            self.session.commit(tx, advance_belief=False)
        return event_id

    def emission_status(self, event_id):
        """Whether retained derived output is still supported; history is never erased."""
        self.session.check()
        event = next((e for e in self.store.events() if e.id == event_id), None)
        if event is None or event.payload.get("operation") != "episode-emit":
            raise ValueError("Unknown emission event")
        payload = event.payload
        status = "aborted" if event_id in self._aborted() else "current"
        if status == "current":
            for saved in (payload["read"], *payload["dependencies"]):
                heads, sources = self._pins(saved["entity_id"], saved["names"])
                if (saved["representations"] != primitive(self._versions()) or
                        saved["heads"] != primitive(heads) or saved["sources"] != primitive(sources)):
                    status = "stale"
                    break
        return dict(status=status, complete=payload["complete"])

    def response(self, episode_id, read, *, needs_clarification=False):
        """Protocol readiness only; this does not select words or infer user intent."""
        if read.entity_id != episode_id or self._entity(episode_id).kind != "conversation":
            raise ValueError("Response read must belong to this conversation")
        self.validate(read)
        turns = self._utterances(episode_id)
        if not turns or turns[-1].data["detail"]["phase"] != "end":
            return "wait"
        # Withdrawal of a phase makes that turn unsuitable for response until
        # an explicit new turn arrives; don't reinterpret older speech as fresh.
        turn_id = turns[-1].data["detail"]["turn_id"]
        withdrawn = self.store.retractions()
        if any(e.id in withdrawn and e.data["detail"]["turn_id"] == turn_id for e in turns):
            return "wait"
        if self._completed(episode_id, turn_id):
            return "wait"
        return "clarify" if needs_clarification else "respond"

    def emit(self, event_id, episode_id, read, *, text="", audio_ref="",
             dependencies=(), complete=True):
        """Commit a derived output/chunk after revalidation; no external I/O.

        Each audio chunk is its own event. Completion consumes this input turn's
        response opportunity. Repeated identical events remain idempotent, even
        after new input, but a new event may never emit against an obsolete read.
        """
        if not isinstance(text, str) or not isinstance(audio_ref, str) or type(complete) is not bool:
            raise ValueError("Output needs text/audio references and boolean completion")
        if read.entity_id != episode_id:
            raise ValueError("Output read belongs to another episode")
        dependencies = tuple(dependencies)
        # Derive turn identity from the pinned sources, not today's active turn:
        # an old acknowledged event must still admit an identical retry.
        turn_id = self._turn(read)
        payload = dict(operation="episode-emit", entity=episode_id, turn_id=turn_id,
                       read=asdict(read), dependencies=[asdict(r) for r in dependencies],
                       text=text, audio_ref=audio_ref, complete=complete)
        tx = self._internal(event_id, payload)
        if tx is None:
            return event_id
        # Pin the transaction first: an intervening writer after the checks also
        # invalidates its base revision at the session's atomic commit boundary.
        if self.response(episode_id, read) == "wait":
            raise ValueError("Input turn is incomplete, interrupted, withdrawn or answered")
        abort_revision = max((self.store.receipt(e.id)["revision"] for e in self.store.events()
                              if e.payload.get("operation") == "episode-abort"
                              and e.payload.get("entity") == episode_id
                              and e.payload.get("turn_id") == turn_id), default=-1)
        for dependency in (read, *dependencies):
            self.validate(dependency)
            if dependency.revision < abort_revision:
                raise ValueError("Retry after abort requires fresh state/dependency reads")
            if any(reason != "never_observed" for _, reason in dependency.omission_reasons):
                raise ValueError("Unavailable state requires repair/retrieval before output")
        self.session.commit(tx, advance_belief=False)
        return event_id
