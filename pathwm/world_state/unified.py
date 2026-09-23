"""One composed runtime over one store: session identity plus concept memory.

R2 software slice (docs/integrated-architecture-plan.md §18). `WorldSession` owns
the store, the clock and temporal instance identity; `ConceptMemory` (session
client) owns concept membership and codes. Each frame is encoded once and shared by
the slot consumer and the belief. The shared latent core is the only forecaster a
decision reads. Goals are explicit `GoalSpec` records; free text asks. Untrained
or supplied weights/keys make this a software contract, not a capability.

Atomicity: `WorldSession.observe` publishes source evidence, identity decisions and
the belief update atomically. Concept records (appearance, attribution, transitions,
binding, key, code) are derived afterwards in separate internal transactions, each
a pure function of retained source events and the identity now in force; an
interrupted derive loses nothing and `recover()` completes it.
"""

from dataclasses import asdict, dataclass, replace
import hashlib
import math

import numpy as np
import torch
from torch import nn

from pathwm.io import digest, state_hash
from pathwm.models.belief_state import Packet
from pathwm.models.modalities import Observation
from pathwm.models.slots import at_time, pointer
from .concepts import SOURCE, BoundedCache, ConceptAgent, SceneView, from_uint8, slot_candidates, to_uint8
from .records import finite_time, identifier
from .session import SourceItem

OPS = ("press",)
PREDICATES = ("lamp_state",)
RECEIPTS = ("ok", "miss", "same_object", "budget_exceeded")
STEP = 1.0  # session-clock units between consecutive observation events


def _integer(value):
    return type(value) is int


@dataclass(frozen=True)
class GoalSpec:
    """Explicit goal record; exact predicate and costs stay outside latent inference.

    budget: most presses the caller allows (the domain contract caps it further).
    deadline: latest session-clock time (same units as `WorldSession.time`) at which
    an action's effect may be observed; each press is observed `STEP` later.
    """

    task_id: str
    predicate: str
    targets: tuple  # ((instance entity id, value), ...)
    budget: int = 2
    deadline: float | None = None

    def __post_init__(self):
        identifier(self.task_id)
        identifier(self.predicate)
        if not isinstance(self.targets, tuple) or not self.targets or any(
            not isinstance(t, tuple) or len(t) != 2 or not isinstance(t[0], str) or not _integer(t[1])
            for t in self.targets
        ):
            raise ValueError("Goal targets must be a nonempty tuple of (entity id, integer value)")
        if not _integer(self.budget) or self.budget < 0:
            raise ValueError("Goal budget must be a nonnegative integer")
        if self.deadline is not None:
            finite_time(self.deadline)

    def digest(self):
        return digest(asdict(self))


@dataclass(frozen=True)
class TypedAction:
    """Executable action: op plus entity/object references, validated before encoding."""

    op: str
    machine: str
    a: int
    b: int
    task_id: str = ""

    def __post_init__(self):
        if not isinstance(self.op, str) or not isinstance(self.machine, str) or not isinstance(self.task_id, str):
            raise ValueError("Action op, machine and task must be strings")
        if not (_integer(self.a) and _integer(self.b)):
            raise ValueError("Action object references must be integers")


@dataclass(frozen=True)
class VerificationRecord:
    """Caller-owned verifier result, published as source evidence."""

    task_id: str
    goal_hash: str
    status: str  # success, failure or unknown
    verifier_id: str
    observed_at: float

    def __post_init__(self):
        if self.status not in ("success", "failure", "unknown"):
            raise ValueError("Verification status is success, failure or unknown")
        for value in (self.task_id, self.goal_hash, self.verifier_id):
            identifier(value)
        finite_time(self.observed_at)


@dataclass(frozen=True)
class Dispatch:
    kind: str  # plan, ask, unsupported or expired
    reason: str
    goal: tuple | None = None  # (left, right) lamp targets for the R1 planner


class ActionEncoder(nn.Module):
    """Typed action -> belief-filter action vector (learned; untrained here).

    Role-typed argument tokens; for `press` exactly the core's [m, a, b] roles.
    Unknown operations are rejected before encoding.
    """

    def __init__(self, width, action_width):
        super().__init__()
        self.op = nn.Embedding(len(OPS), width)
        self.role = nn.Embedding(3, width)
        self.out = nn.Linear(width, action_width)

    def tokens(self, op, args):
        if op not in OPS:
            raise ValueError(f"Unknown action operation {op!r}")
        if args.ndim != 3 or args.shape[1:] != self.role.weight.shape:
            raise ValueError("Press arguments must be [B,3,width] role tokens")
        head = self.op.weight[OPS.index(op)].expand(len(args), 1, -1)
        return torch.cat((head, args + self.role.weight[None]), 1)

    def forward(self, op, args):
        return self.out(self.tokens(op, args).mean(1))


@dataclass
class LiveView(SceneView):
    event_id: str
    decisions: dict  # slot index -> session binding decision


def _r1_only(name, instead):
    def method(self, *args, **kwargs):
        raise NotImplementedError(
            f"{name} writes the R1 standalone record layout; the unified runtime uses {instead}"
        )

    method.__name__ = name
    return method


def model_versions(perception, core, candidate_encoder, action_encoder):
    modules = dict(perception=perception, core=core, candidates=candidate_encoder, actions=action_encoder)
    return {name: state_hash(module)[:16] for name, module in modules.items()}


class UnifiedAgent(ConceptAgent):
    """Concept logic of `ConceptAgent` over session-owned identity and one store."""

    def __init__(self, perception, core, session, memory, candidate_encoder, action_encoder,
                 contract, settings=None, *, device="cpu"):
        if memory.session is not session:
            raise ValueError("ConceptMemory must be a client of this session")
        if session.agent.encoders["image"] is not perception.encoder:
            raise ValueError("Belief and slot consumer must share one image encoder")
        versions = model_versions(perception, core, candidate_encoder, action_encoder)
        if memory.versions != versions:
            raise ValueError(f"Memory versions {memory.versions} differ from models {versions}")
        super().__init__(perception, core, memory, contract, settings, device=device)
        self.session, self.candidate_encoder = session, candidate_encoder
        self.action_encoder = action_encoder
        self._pyramids = BoundedCache(self.settings.percept_cache)
        self.view = None
        self.check()

    # Inherited R1 writers assume the standalone layout (new instance per scene,
    # transitions parented on appearance). They fail before any blob/store/model effect;
    # `repair`, `receive_retract` and `receive_supersede` are re-implemented below.
    observe_scene = _r1_only("observe_scene", "observe()")
    observe_session = _r1_only("observe_session", "observe(transitions=...)")
    receive_claim = _r1_only("receive_claim", "no testimony/claim path yet (declared absent in R2)")
    _act = _r1_only("_act", "execute()")
    _attach = _r1_only("_attach", "attribution through observe()/execute()")
    _instances = _r1_only("_instances", "WorldSession identity decisions")
    _transitions_component = _r1_only("_transitions_component", "_rebind()")

    def check(self):
        """Integrated model identity: session modules plus perception (incl. decoder and
        heads), core, candidate and action encoders. Changed weights are rejected, never
        silently combined with caches or codes derived from the old ones."""
        self.session.check()
        modules = (self.perception, self.core, self.candidate_encoder, self.action_encoder)
        if model_versions(*modules) != self.memory.versions:
            raise ValueError("Runtime model weights changed; retrain outside the agent and migrate")
        if any(m.training for module in modules for m in module.modules()):
            raise ValueError("Runtime modules must be in eval mode")

    # perception: one encoder call per distinct frame --------------------------
    @torch.no_grad()
    def encode(self, rgb):
        array = to_uint8(rgb)
        sha = hashlib.sha256(array.tobytes()).hexdigest()
        if sha not in self._pyramids:
            self._pyramids[sha] = self.perception.pyramid(from_uint8(array, self.device)[None])
        return sha, self._pyramids[sha]

    @torch.no_grad()
    def percept(self, rgb):
        sha, pyramid = self.encode(rgb)
        if sha not in self._percepts:
            self._percepts[sha] = self.perception.from_pyramid(pyramid).detach()
        return sha, self._percepts[sha]

    def candidates(self, percept, rgb=None):
        """Default candidate source: learned slot keys (`slot-k`)."""
        return slot_candidates(percept, self.candidate_encoder, source=SOURCE,
                               model_version=self.memory.versions["candidates"])

    # observation ----------------------------------------------------------------
    @torch.no_grad()
    def observe(self, rgb, *, transitions=(), attribute_with=None, action=None,
                verification=None, goal=None, candidates=None, supersedes=None):
        """Publish one observation event, then derive concept records from source.

        transitions: (pre, record, receipt status, post) source items of this event;
        each records the observation it was chosen in (`perceived_in`: `attribute_with`
        or this event) so attribution can always be re-derived from source.
        `verification(t)` returns this event's VerificationRecord (for `goal`, if
        given); an invalid record is not published and raises after the rest is.
        `candidates(percept, frame)` replaces the slot-key source (SOFTWARE fixtures).
        `supersedes` = {transition index: old evidence id}.
        """
        self.check()
        t = self.session.time + STEP
        count = sum(e.kind == "observation" for e in self.session.store.events())
        event_id = f"obs-{count + 1:06d}"
        sha, pyramid = self.encode(rgb)
        _, percept = self.percept(rgb)
        ref, blob = self.memory.put_blob(to_uint8(rgb)[None])
        supersedes = supersedes or {}
        perceived_in = event_id if attribute_with is None else attribute_with.event_id
        items = [SourceItem(SOURCE, "image", ref, blob)]
        for k, (pre, record, status, post) in enumerate(transitions):
            r, h = self.memory.put_blob(np.stack((to_uint8(pre), to_uint8(post))))
            data = dict(action=record.as_dict(), receipt=status, perceived_in=perceived_in)
            items.append(SourceItem(SOURCE, "transition", r, h, data, supersedes.get(k)))
        problem = None
        if verification is not None:
            try:
                record = verification(t)
            except Exception as error:  # a broken verifier must not erase this event
                problem = error
            else:
                reason = self._verification_problem(record, t, goal)
                if reason is None:
                    items.append(SourceItem(record.verifier_id, "verification", data=asdict(record)))
                else:
                    problem = ValueError(f"Verification record not published: {reason}")
        frame = from_uint8(to_uint8(rgb), self.device)
        packet = Packet(SOURCE, "image", Observation(frame[None, None], torch.full((1, 1), t)))
        result = self.session.observe(
            event_id, occurred_at=t, available_at=t,
            candidates=(candidates or self.candidates)(percept, frame), packets=(packet,),
            action=None if action is None else self._action_vector(action),
            evidence=tuple(items), packet_features={SOURCE: at_time(pyramid, [t])},
        )
        # Published atomically above; everything below is re-derivable (recover()).
        view = self._event_view(event_id)
        if supersedes:
            self._repair()  # withdrawn items invalidated dependents; this event's view follows
        self._memberships(view)
        self._attribute(result["evidence"][1 : 1 + len(transitions)])
        self._memberships(view)
        self.view = view
        if problem is not None:
            raise problem
        return view, result

    def _verification_problem(self, record, t, goal):
        if not isinstance(record, VerificationRecord):
            return "verifier must return a VerificationRecord"
        if record.observed_at != t:
            return "verification time differs from its observation event"
        if record.verifier_id == SOURCE:
            return "a verifier cannot speak as the camera source"
        if goal is not None and (record.task_id, record.goal_hash) != (goal.task_id, goal.digest()):
            return "verification names another task or goal"
        return None

    def _event_view(self, event_id):
        """Live view of an observation event from its retained camera frame and the
        session's published decisions; None when that event retained no frame."""
        frame = next((e for e in self.memory.view()["evidence"].values()
                      if e.event_id == event_id and e.source == SOURCE and e.modality == "image"
                      and e.content_ref), None)
        if frame is None:
            return None
        rgb = from_uint8(self.memory.get_blob(frame.content_ref, frame.content_hash)[0], self.device)
        sha, percept = self.percept(rgb)
        machines, objects = self._layout(percept)
        decisions = {int(d["candidate"].split("-")[1]): d
                     for d in self.session.decision(event_id)["bindings"] if d["candidate"].startswith("slot-")}
        return LiveView(sha, percept, machines, objects, True, event_id, decisions)

    def _machine_instance(self, view, slot):
        d = view.decisions.get(slot)
        if d is None or d["recognition"] is None:
            return None, None
        r = self.memory.view()["components"].get(d["recognition"])
        return (r.entity_id, r.id) if r is not None and r.active else (None, None)

    def _memberships(self, view):
        """Machine instance (session identity), appearance and current concept membership."""
        created = []
        for m in view.machines:
            instance, recognition = self._machine_instance(view, m["slot"])
            m.update(instance=instance, recognition=recognition, concept=None, binding=None)
            if instance is not None and self._appearance_from(recognition) is None:
                created.append(recognition)
        if created:
            tx = self.memory.begin("internal")
            for recognition in created:
                self._restore_appearance(tx, self.memory.view()["components"][recognition])
            self.memory.commit(tx)
        for m in view.machines:
            if m["instance"] is None:
                continue
            bindings = [c for c in self.memory.view()["components"].values()
                        if c.entity_id == m["instance"] and c.name == "binding"]
            binding = max(bindings, key=lambda c: (c.valid_from, c.revision, c.id), default=None)
            if binding is None:  # first sighting: appearance proposal only
                appearance = self._appearance_from(m["recognition"])
                d = self._bind(m["instance"], appearance.id, appearance.tensor(device=self.device), None, [])
            elif not binding.active:  # invalidated membership: re-derive, never re-propose blank
                d = self._rebind(m["instance"])
            else:
                d = dict(concept=binding.data["concept"], binding=binding.id)
            m.update(concept=d.get("concept"), binding=d.get("binding"))

    def _appearance_from(self, recognition):
        found = [c for c in self.memory.view()["components"].values()
                 if c.name == "appearance" and c.active and c.parents == (recognition,)]
        return found[-1] if found else None

    def _restore_appearance(self, tx, recognition):
        """Appearance key derived from the retained frame of the recognition's event."""
        view = self.memory.view()
        candidate = view["evidence"][recognition.evidence[0]]
        frame = next(e for e in view["evidence"].values()
                     if e.event_id == candidate.event_id and e.source == SOURCE
                     and e.modality == "image" and e.content_ref)
        rgb = from_uint8(self.memory.get_blob(frame.content_ref, frame.content_hash)[0], self.device)
        _, percept = self.percept(rgb)
        slot = int(candidate.data["candidate"].split("-")[1])
        tx.put_component(recognition.entity_id, "appearance", self._key(percept.slots[0, slot]).detach().cpu(),
                         space="machine-key", model_version=self.core_version, role="inferred",
                         evidence=(frame.id,), parents=(recognition.id,))

    # attribution: a pure function of one transition's own source item -----------------
    def _target(self, evidence_id):
        """(instance, recognition) a transition teaches about, or None.

        Reads only the item's own receipt, action and `perceived_in` observation, and
        the identity decision now in force for the slot at the action's machine pixel.
        Failed receipts, withdrawn items and unresolved identity stay raw evidence.
        """
        view = self.memory.view()
        e = view["evidence"][evidence_id]
        if (e.source != SOURCE or e.modality != "transition" or e.id in view["retracted"]
                or e.data.get("receipt") != "ok"):
            return None
        live = self._event_view(e.data.get("perceived_in", e.event_id))
        if live is None:
            return None
        xy = torch.tensor([list(e.data["action"]["machine_xy"])], device=self.device, dtype=torch.float32)
        instance, recognition = self._machine_instance(live, int(pointer(live.percept.alpha, xy)))
        return None if instance is None else (instance, recognition)

    def _attribute(self, evidence_ids):
        rows = [(e,) + target for e in evidence_ids if (target := self._target(e)) is not None]
        if not rows:
            return {}
        tx = self.memory.begin("internal")
        restored = set()
        for e, instance, recognition in rows:
            if self._appearance_from(recognition) is None and recognition not in restored:
                self._restore_appearance(tx, self.memory.view()["components"][recognition])
                restored.add(recognition)
            tx.put_component(instance, "attribution", torch.ones(1), space="attribution",
                             model_version=self.core_version, role="inferred", evidence=(e,),
                             parents=(recognition,), data=dict(transition=e))
        self.memory.commit(tx)
        return {instance: self._rebind(instance) for instance in dict.fromkeys(r[1] for r in rows)}

    def _rebind(self, instance):
        """Transitions from the instance's active attributions; re-verify membership."""
        view = self.memory.view()
        attributions = sorted(
            (c for c in view["components"].values()
             if c.entity_id == instance and c.name == "attribution" and c.active),
            key=lambda c: (view["evidence"][c.data["transition"]].available_at, c.data["transition"]),
        )
        appearances = [c for c in view["components"].values()
                       if c.entity_id == instance and c.name == "appearance" and c.active]
        if not appearances:
            return dict(instance=instance, status="no_active_appearance")
        appearance = max(appearances, key=lambda c: (c.valid_from, c.revision, c.id))
        evidence = list(dict.fromkeys(c.data["transition"] for c in attributions))
        transitions_id = None
        current = max((c for c in view["components"].values() if c.entity_id == instance and c.name == "transitions"),
                      key=lambda c: (c.valid_from, c.revision, c.id), default=None)
        if evidence and current is not None and current.active and list(current.evidence) == evidence:
            transitions_id = current.id  # already derived (idempotent recovery)
        elif evidence:
            tx = self.memory.begin("internal")
            transitions_id = tx.put_component(
                instance, "transitions", torch.tensor([float(len(evidence))]), space="count",
                model_version=self.core_version, role="inferred", evidence=tuple(evidence),
                parents=tuple(c.id for c in attributions),
            )
            self.memory.commit(tx)
        # An invalidated membership is a hypothesis to re-check, not forgotten: its
        # concept's key was invalidated with it and would not be proposed by appearance.
        bindings = [c for c in view["components"].values() if c.entity_id == instance and c.name == "binding"]
        previous = max(bindings, key=lambda c: (c.valid_from, c.revision, c.id), default=None)
        prior = None
        if previous is not None and not previous.active and previous.data["concept"] in view["entities"]:
            prior = previous.data["concept"]
        if prior is not None and previous.data["supports"] and 0 < len(evidence) < self.settings.min_check:
            # Too little evidence for a behavioural check: keep the prior membership
            # (unverified) instead of re-proposing by appearance among other concepts.
            decision = dict(instance=instance, repaired_from=previous.id)
            return self._commit_binding(instance, prior, appearance.id, transitions_id, evidence,
                                        False, True, None, decision, "retained")
        return self._bind(instance, appearance.id, appearance.tensor(device=self.device), transitions_id,
                          evidence, prior=prior)

    # corrections and recovery ----------------------------------------------------------
    @torch.no_grad()
    def correct(self, transaction):
        """Publish a caller's identity/source correction, then repair dependents."""
        self.check()
        receipt = self.session.commit(transaction)
        return receipt, self.repair_identity()

    @torch.no_grad()
    def repair_identity(self):
        """Re-derive every transition whose latest attribution was invalidated.

        Each is re-derived from its OWN source item (receipt, action, observation) and
        the identity now in force: a reassigned recognition moves it to the new entity;
        a recognition invalidated by a withdrawn identity link leaves it raw. Withdrawn
        or superseded items are never re-attached; a replacement is attributed from
        its own item when it is published (or by `recover`). Touched instances are
        re-derived and re-verified; untouched records stay.
        """
        self.check()
        repaired = self._repair()
        self._refresh()
        return repaired

    def _refresh(self, view=None):
        """Bring a live view's identities and memberships to the heads now in force.

        Only derived records move: the frame, percept, event and clock stay those of
        the last camera observation. `_memberships` re-derives an invalidated binding
        once and reads consistent ones, so repeated refreshes write nothing.
        """
        view = self.view if view is None else view
        if view is not None:
            self._memberships(view)
        return view

    def _repair(self):
        view = self.memory.view()
        latest = {}
        for c in sorted((c for c in view["components"].values() if c.name == "attribution"),
                        key=lambda c: (c.revision, c.id)):
            latest[c.data["transition"]] = c
        stale = [e for e, c in latest.items() if not c.active]
        repaired = self._attribute(stale)
        view = self.memory.view()
        touched = [e.id for e in view["entities"].values()
                   if e.kind == "instance" and e.id not in repaired and not self._consistent(e.id, view)]
        return repaired | {instance: self._rebind(instance) for instance in sorted(touched)}

    def _consistent(self, instance, view):
        """Derived membership matches the instance's active attributions: the latest
        transitions cover exactly them and the latest binding is active and built on
        those transitions. Anything else is re-derived (e.g. after an interruption)."""
        def latest(name):
            found = [c for c in view["components"].values() if c.entity_id == instance and c.name == name]
            return max(found, key=lambda c: (c.valid_from, c.revision, c.id), default=None)

        attributed = {c.data["transition"] for c in view["components"].values()
                      if c.entity_id == instance and c.name == "attribution" and c.active}
        binding, transitions = latest("binding"), latest("transitions")
        if not attributed:
            return binding is None or binding.active
        return (transitions is not None and transitions.active and set(transitions.evidence) == attributed
                and binding is not None and binding.active and transitions.id in binding.parents)

    @torch.no_grad()
    def recover(self):
        """Complete derivations an interrupted process may have lost: transitions that
        never received an attribution decision are derived from their own source."""
        self.check()
        view = self.memory.view()
        seen = {c.data["transition"] for c in view["components"].values() if c.name == "attribution"}
        pending = sorted((e for e in view["evidence"].values()
                          if e.modality == "transition" and e.id not in seen),
                         key=lambda e: (e.available_at, e.id))
        repaired = self._attribute([e.id for e in pending]) | self._repair()
        self._refresh()
        return repaired

    def repair(self):
        """Legacy name, R2 meaning (R1 `repair` assumes the standalone record layout)."""
        return self.repair_identity()

    def _withdrawable(self, evidence_id):
        view = self.memory.view()
        old = view["evidence"].get(evidence_id)
        if old is None:
            raise ValueError(f"Unknown evidence {evidence_id!r}")
        if evidence_id in view["retracted"]:
            raise ValueError("Evidence already withdrawn")
        if old.source != SOURCE or old.modality != "transition":
            raise ValueError("Only camera transition items are corrected through this legacy API")
        return old

    @torch.no_grad()
    def receive_retract(self, evidence_id):
        """Same-source withdrawal of one transition, published as a session correction;
        dependents are invalidated by the store and re-derived from source."""
        self.check()
        old = self._withdrawable(evidence_id)
        t = self.session.time + STEP
        tx = self.session.store.begin(
            f"correction-{self.session.store.revision + 1:06d}", occurred_at=old.occurred_at,
            available_at=t, kind="correction", payload=dict(source=old.source),
        )
        tx.retract_evidence(evidence_id)
        return self.correct(tx)

    @torch.no_grad()
    def receive_supersede(self, old_id, transition):
        """Replace one transition item by a corrected one in a frameless observation event.

        The replacement keeps the original `perceived_in` observation, so it is
        attributed through the identity decisions the action was chosen on, but from
        its OWN receipt and action. No camera frame is published: the live view and
        resume point stay the last real camera observation.
        """
        self.check()
        old = self._withdrawable(old_id)
        pre, record, status, post = transition
        if status not in RECEIPTS or not hasattr(record, "as_dict"):
            raise ValueError("A replacement needs an ActionRecord and a known receipt status")
        action = record.as_dict()
        ref, blob = self.memory.put_blob(np.stack((to_uint8(pre), to_uint8(post))))
        data = dict(action=action, receipt=status, perceived_in=old.data.get("perceived_in", old.event_id))
        count = sum(e.kind == "observation" for e in self.session.store.events())
        t = self.session.time + STEP
        result = self.session.observe(
            f"obs-{count + 1:06d}", occurred_at=old.occurred_at, available_at=t,
            evidence=(SourceItem(SOURCE, "transition", ref, blob, data, old_id),),
        )
        new = result["evidence"][0]
        self._repair()
        self._attribute([new])
        self._refresh()
        return new

    # goals, planning and execution -------------------------------------------------
    def remaining(self, goal, presses_done):
        """Presses still allowed: caller budget, contract cap and deadline (session clock)."""
        left = min(goal.budget, self.contract.budget) - presses_done
        if goal.deadline is not None:
            left = min(left, math.floor((goal.deadline - self.session.time) / STEP + 1e-9))
        return max(left, 0)

    def dispatch(self, request=None, goal=None):
        """Structured goals plan; free text or unresolvable references ask."""
        self.check()
        self._refresh()  # resolve references against the identities now in force
        if goal is None:
            return Dispatch("ask", "no structured GoalSpec; free text is not interpreted")
        if request is not None and request.task_id != goal.task_id:
            return Dispatch("unsupported", "request and goal name different tasks")
        if goal.predicate not in PREDICATES:
            return Dispatch("unsupported", f"unsupported predicate {goal.predicate!r}")
        if goal.deadline is not None and self.session.time > goal.deadline:
            return Dispatch("expired", "goal deadline has passed on the session clock")
        entities = self.memory.view()["entities"]
        if any(ref not in entities or entities[ref].kind != "instance" or value not in (0, 1)
               for ref, value in goal.targets):
            return Dispatch("unsupported", "unknown entity reference or target value")
        visible = [m["instance"] for m in self.view.machines] if self.view is not None else []
        if len(visible) != 2 or None in visible:
            return Dispatch("ask", "both visible machines must be identified instances")
        canonical = self.session.store.canonical
        bits = [None, None]
        for ref, value in goal.targets:
            hits = [i for i, e in enumerate(visible) if canonical(e) == canonical(ref)]
            if len(hits) != 1:
                return Dispatch("ask", "target reference is not visible" if not hits else "ambiguous target reference")
            if bits[hits[0]] not in (None, value):
                return Dispatch("ask", "conflicting targets for one machine")
            bits[hits[0]] = value
        if None in bits:
            return Dispatch("ask", "the R1 planner needs a target for each visible machine")
        return Dispatch("plan", "structured goal", tuple(bits))

    def _live(self, view):
        """Exact live dependencies of a decision: latest source observation, the view's
        observation, its machine identity decisions and object references."""
        observations = [e.id for e in self.session.store.events() if e.kind == "observation"]
        return (
            ("latest_observation", observations[-1] if observations else None),
            ("view", view.event_id),
            ("machines", tuple((m["slot"], m.get("instance"), m.get("recognition")) for m in view.machines)),
            ("objects", tuple(o["slot"] for o in view.objects)),
        )

    def is_ready(self, read):
        """Stored heads unchanged AND the live observation/identity/object references
        the action was chosen on are still the current ones."""
        if not read.live or read.live != self._live(self.view) or not self.memory.is_current(read):
            return False
        components = self.memory.view()["components"]
        return all(r is None or (components[r].active and components[r].entity_id == instance)
                   for _, instance, r in dict(read.live)["machines"])

    @torch.no_grad()
    def plan(self, view, goal, *, presses_done, control=None, budget=None):
        self.check()
        self._refresh(view)  # a fresh plan reads only current identities and heads
        decision, read = super().plan(view, goal, presses_done=presses_done, control=control, budget=budget)
        return decision, replace(read, live=self._live(view))

    @torch.no_grad()
    def predict(self, view, record, *, code=None, loops=None):
        self.check()
        self._refresh(view)
        return super().predict(view, record, code=code, loops=loops)

    def action_for(self, decision, task_id=""):
        m, i, j = decision.action
        return TypedAction("press", self.view.machines[m]["instance"], i, j, task_id)

    def _check_action(self, action):
        if not isinstance(action, TypedAction):
            raise ValueError("Actions must be TypedAction records")
        if action.op not in OPS:
            raise ValueError(f"Unknown action operation {action.op!r}")
        machines = [m["instance"] for m in self.view.machines]
        n = len(self.view.objects)
        if action.machine not in machines or machines.count(action.machine) != 1:
            raise ValueError("Action machine is not one visible identified instance")
        if not (0 <= action.a < n and 0 <= action.b < n):
            raise ValueError("Action objects are not visible")
        return machines.index(action.machine)

    def _action_vector(self, action):
        m = self._check_action(action)
        tokens = torch.stack((self.view.machines[m]["token"], self.view.objects[action.a]["token"],
                              self.view.objects[action.b]["token"]))
        return self.action_encoder(action.op, tokens[None])

    @torch.no_grad()
    def execute(self, actuator, action, read, *, goal=None, presses_done=0, verification=None,
                candidates=None):
        """Check models, references, task limits and the pinned read set; press once;
        publish receipt + post frame (+ verifier record) as one observation event."""
        self.check()
        m = self._check_action(action)  # dangling references fail before anything runs
        if goal is not None and (action.task_id != goal.task_id or self.remaining(goal, presses_done) < 1):
            return dict(status="refused", evidence=None)
        if not self.is_ready(read):
            return dict(status="stale", evidence=None)
        pre_view = self.view
        record = self.record_for(pre_view, (m, action.a, action.b))
        pre = actuator.frame()
        self.check()  # nothing may change between the checks and the side effect
        receipt = actuator.press(record)
        post = actuator.frame()
        _, result = self.observe(post, transitions=((pre, record, receipt.status, post),),
                                 attribute_with=pre_view, action=action, verification=verification,
                                 goal=goal, candidates=candidates)
        return dict(status=receipt.status, record=record, evidence=result["evidence"][1])

    @torch.no_grad()
    def run_task(self, actuator, goal, *, request=None, verifier=None, candidates=None):
        """Dispatch, plan with the shared core within the task limits, execute, verify
        externally, replan. Every actuator call counts against the goal budget."""
        self.check()
        trace = []
        verification = None if verifier is None else (lambda t: verifier.verify(goal, t))
        presses, option = 0, None
        for _ in range(self.contract.budget + 3):
            dispatch = self.dispatch(request, goal)
            if dispatch.kind != "plan":
                option = dispatch.kind
                trace.append(dict(option=option, reason=dispatch.reason))
                break
            limit = presses + self.remaining(goal, presses)
            decision, read = self.plan(self.view, dispatch.goal, presses_done=presses, budget=limit)
            step = dict(option=decision.option, action=decision.action, value=decision.value)
            trace.append(step)
            if decision.option != "press":
                option = decision.option
                break
            result = self.execute(actuator, self.action_for(decision, goal.task_id), read, goal=goal,
                                  presses_done=presses, verification=verification, candidates=candidates)
            step.update(status=result["status"], evidence=result["evidence"])
            if result["status"] == "refused":
                option = "refused"
                break
            if result["status"] != "stale":
                presses += 1
        else:
            option = "abstain"
        if goal is not None and option in ("stop", "abstain", "refused"):
            if verification is not None and presses == 0 and self.remaining(goal, 0) > 0:
                self.observe(actuator.frame(), verification=verification, goal=goal, candidates=candidates)
            tx = self.memory.begin("internal", payload=dict(
                declared=option, task_id=goal.task_id, goal_hash=goal.digest()))
            self.memory.commit(tx)
        return dict(option=option, presses=presses, trace=trace,
                    outcome=self.task_outcome(goal) if goal is not None else "unknown")

    def task_outcome(self, goal):
        """Verifier evidence for this goal from the LATEST observation event only: any
        later source observation (an action, or a world that may have changed on its
        own) makes an older verdict unknown. Declared completion never counts."""
        view = self.memory.view()
        observations = [e.id for e in self.session.store.events() if e.kind == "observation"]
        found = [e for e in view["evidence"].values()
                 if e.modality == "verification" and e.id not in view["retracted"]
                 and e.data.get("goal_hash") == goal.digest() and e.data.get("task_id") == goal.task_id]
        found = [e for e in found if observations and e.event_id == observations[-1]]
        return max(found, key=lambda e: e.id).data["status"] if found else "unknown"

    # restart ------------------------------------------------------------------------
    def save(self, directory):
        self.check()
        self.session.save(directory / "session.pt")
        self.memory.save()

    @torch.no_grad()
    def resume_view(self):
        """Live view of the latest observation that retained a camera frame, rebuilt
        from that frame and the session's persisted decisions (other modalities or
        verifier-only events are skipped). Missing memberships are derived."""
        self.check()
        for event in reversed([e for e in self.session.store.events() if e.kind == "observation"]):
            live = self._event_view(event.id)
            if live is not None:
                self._memberships(live)
                self.view = live
                return live
        return None
