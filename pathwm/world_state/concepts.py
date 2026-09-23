"""Frozen-weight runtime concept memory and agent for the R1 reference integration.

`ConceptMemory` owns one WorldStore, hash-verified frame blobs and read sets.
`ConceptAgent` is the inference owner (like WorldSession): it perceives frames,
creates instances, proposes concepts by appearance, verifies them behaviourally,
creates concepts, recomputes codes from active evidence, plans and executes.
It never receives scene structure, rules, kinds or labels; only frames, action
records/receipts, goals and source correction messages.
"""

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F

from pathwm.io import atomic_json
from pathwm.models.slots import pointer, slot_coordinates
from pathwm.models import latent_core as lc
from .records import Limits
from .retrieval import ExactRetriever, Query, RetrievalBudget
from .store import WorldStore

SOURCE = "rule_world_camera"
R1_LIMITS = Limits(
    entities=4096,
    components=32768,
    relations=8192,
    evidence=16384,
    events=8192,
    values=8_388_608,
    bytes=268_435_456,
)


@dataclass(frozen=True)
class ReadSet:
    components: tuple[tuple[str, int], ...]  # (component id, revision) pairs
    versions: tuple
    # Live dependencies a caller pins besides stored heads (e.g. the observation,
    # identity decisions and object references an action was chosen on).
    live: tuple = ()


class ConceptMemory:
    """Durable store + blobs. Derived tensors leave this object only as clones.

    Standalone (R1): owns its store and clock. Client (`session=`): no store of its
    own; reads `session.store`, uses the session clock and publishes internal and
    correction records through `session.commit`. Source observations enter only
    through `WorldSession.observe`.
    """

    def __init__(self, directory, *, versions, limits=R1_LIMITS, store=None, session=None):
        if session is not None and store is not None:
            raise ValueError("A session client has no store of its own")
        self.directory = Path(directory)
        (self.directory / "blobs").mkdir(parents=True, exist_ok=True)
        self.versions = dict(versions)
        self.session = session
        self._store = None if session is not None else store if store is not None else WorldStore(limits)
        events = self.store.events()
        self.clock = max((e.available_at for e in events), default=0.0)
        self._cache_key, self._cache = None, None

    @property
    def store(self):
        return self.session.store if self.session is not None else self._store

    # blobs -------------------------------------------------------------
    def put_blob(self, frames):
        frames = np.ascontiguousarray(frames, dtype=np.uint8)
        stream = io.BytesIO()
        np.save(stream, frames, allow_pickle=False)
        data = stream.getvalue()
        sha = hashlib.sha256(data).hexdigest()
        ref = f"blobs/{sha}.npy"
        path = self.directory / ref
        if not path.exists():
            temporary = path.with_suffix(".partial")
            temporary.write_bytes(data)
            temporary.replace(path)
        return ref, sha

    def get_blob(self, ref, sha):
        data = (self.directory / ref).read_bytes()
        if hashlib.sha256(data).hexdigest() != sha:
            raise ValueError(f"Blob hash mismatch for {ref}")
        return np.load(io.BytesIO(data), allow_pickle=False)

    # transactions ------------------------------------------------------
    def begin(self, kind, *, payload=None, occurred_at=None):
        if self.session is not None:
            if kind == "observation":
                raise ValueError("Source observations enter through WorldSession.observe")
            self.clock = self.session.time
        else:
            self.clock += 1.0
        return self.store.begin(
            f"{self.store.revision + 1:06d}-{kind}",
            occurred_at=self.clock if occurred_at is None else occurred_at,
            available_at=self.clock,
            kind=kind,
            payload=payload,
        )

    def commit(self, tx):
        if self.session is not None:
            # Derived records at the current clock: no belief filter step.
            return self.session.commit(tx, advance_belief=False)
        return self.store.commit(tx)

    # read sets ---------------------------------------------------------
    def read_set(self, component_ids):
        """Pin each read component's id AND revision plus model versions."""
        components = self.view()["components"]
        return ReadSet(
            tuple((c, components[c].revision) for c in component_ids),
            tuple(sorted(self.versions.items())),
        )

    def is_current(self, read):
        """Same versions; each read component active, same revision, still the latest."""
        if read.versions != tuple(sorted(self.versions.items())):
            return False
        view = self.view()
        components = view["components"]
        return all(
            c in components
            and components[c].active
            and components[c].revision == revision
            and view["latest"][components[c].entity_id, components[c].name] == c
            for c, revision in read.components
        )

    def tensor(self, component_id):
        return self.view()["components"][component_id].tensor().clone()

    def view(self):
        """Detached copies of all records, cached per committed revision."""
        store = self.store
        if self._cache_key != (id(store), store.revision):
            components = self.store.components()
            latest = {}
            for c in sorted(components, key=lambda c: (c.valid_from, c.revision, c.id)):
                latest[c.entity_id, c.name] = c.id
            self._cache = dict(
                latest=latest,
                components={c.id: c for c in components},
                evidence={e.id: e for e in self.store.evidence()},
                entities={e.id: e for e in self.store.entities()},
                relations=self.store.relations(),
                retracted=self.store.retractions(),
            )
            self._cache_key = (id(store), store.revision)
        return self._cache

    # persistence -------------------------------------------------------
    def save(self):
        if self.session is None:  # a client's store is persisted by its session
            self.store.save(self.directory / "store.json")
        atomic_json(self.directory / "memory.json", dict(versions=self.versions, clock=self.clock))

    def disk_bytes(self):
        """Durable bytes: authoritative frame blobs and the store log, separately."""
        blobs = [p for p in (self.directory / "blobs").glob("*") if p.is_file()]
        store = self.directory / "store.json"
        return dict(
            blob_bytes=sum(p.stat().st_size for p in blobs),
            blob_files=len(blobs),
            store_bytes=store.stat().st_size if store.exists() else 0,
        )

    @classmethod
    def load(cls, directory, *, versions):
        """Restore with the SAME model versions only.

        R1 rejects any version change: appearance keys and codes live in the saved
        models' latent coordinates. An evidence-backed migration (re-encode, rebind,
        recompute) is a deferred architecture contract, not a flag.
        """
        directory = Path(directory)
        saved = json.loads((directory / "memory.json").read_text())["versions"]
        if saved != dict(versions):
            raise ValueError(
                f"Memory model versions {saved} differ from {dict(versions)}; "
                "migration is not implemented in R1"
            )
        store = WorldStore.load(directory / "store.json")
        memory = cls(directory, versions=versions, store=store)
        memory.verify_blobs()
        return memory

    @classmethod
    def attach(cls, directory, session, *, versions):
        """Client restart over a restored session: same versions, every blob verifies."""
        saved = json.loads((Path(directory) / "memory.json").read_text())["versions"]
        if saved != dict(versions):
            raise ValueError(
                f"Memory model versions {saved} differ from {dict(versions)}; "
                "migration is not implemented"
            )
        memory = cls(directory, versions=versions, session=session)
        memory.verify_blobs()
        return memory

    def verify_blobs(self):
        for e in self.store.evidence():  # restart integrity: every referenced blob verifies
            if e.content_ref:
                self.get_blob(e.content_ref, e.content_hash)


# ---------------------------------------------------------------- runtime agent


@dataclass
class AgentSettings:
    tau: float = 0.5  # appearance-only binding threshold (calibrate on validation)
    lam: float = 0.05  # new-concept log-likelihood margin
    min_check: int = 4
    support_limit: int = 128
    proposals: int = 3
    seed: int = 0
    loops: int | None = None
    # Feedback from the agent's own successful presses/claim tests joins concept
    # support below min_check only with confident appearance membership.
    feedback_support: bool = True
    tau_feedback: float = 0.8
    # Disposable derived caches are bounded (least recently used eviction).
    percept_cache: int = 2048
    token_cache: int = 8192
    code_cache: int = 512


class BoundedCache(OrderedDict):
    """LRU cache for disposable derived tensors; eviction only forces recompute."""

    def __init__(self, limit):
        super().__init__()
        self.limit, self.evictions = limit, 0

    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.move_to_end(key)
        while len(self) > self.limit:
            self.popitem(last=False)
            self.evictions += 1


@dataclass
class SceneView:
    frame_sha: str
    percept: object
    machines: list  # left->right dicts: slot, xy, token, lamp, instance, concept, binding
    objects: list  # dicts: slot, xy, token
    persisted: bool


def to_uint8(rgb):
    return (rgb.detach().cpu().clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).numpy()


def from_uint8(array, device):
    return torch.from_numpy(np.ascontiguousarray(array)).to(device).permute(2, 0, 1).float() / 255


class ConceptAgent:
    def __init__(self, perception, core, memory, contract, settings=None, *, device="cpu"):
        self.perception, self.core, self.memory = perception, core, memory
        self.contract, self.settings = contract, settings or AgentSettings()
        self.device = torch.device(device)
        self.rng = random.Random(self.settings.seed)
        s = self.settings
        self._percepts = BoundedCache(s.percept_cache)
        self._tokens = BoundedCache(s.token_cache)
        self._codes = BoundedCache(s.code_cache)
        self.retriever = ExactRetriever()
        with torch.no_grad():
            self.empty_code = core.induce(
                torch.zeros(1, 0, core.width, device=self.device),
                torch.zeros(1, 0, dtype=torch.bool, device=self.device),
                loops=self.settings.loops,
            )[0]
        self.core_version = memory.versions["core"]

    # perception --------------------------------------------------------
    @torch.no_grad()
    def percept(self, rgb):
        array = to_uint8(rgb)
        sha = hashlib.sha256(array.tobytes()).hexdigest()
        if sha not in self._percepts:
            frame = from_uint8(array, self.device)[None]
            self._percepts[sha] = self.perception(frame).detach()
        return sha, self._percepts[sha]

    def cache_stats(self):
        return {
            name: dict(entries=len(cache), limit=cache.limit, evictions=cache.evictions)
            for name, cache in (("percepts", self._percepts), ("tokens", self._tokens), ("codes", self._codes))
        }

    def _layout(self, percept):
        kinds = percept.kind[0].softmax(-1)
        coords = slot_coordinates(percept.alpha)[0]
        machine_slots = kinds[:, 1].topk(2).indices.tolist()
        machine_slots.sort(key=lambda k: float(coords[k, 0]))
        objects = [
            k
            for k in range(len(kinds))
            if k not in machine_slots and int(kinds[k].argmax()) == 2
        ]
        objects.sort(key=lambda k: float(coords[k, 0]))
        machines = [
            dict(
                slot=k,
                xy=tuple(coords[k].tolist()),
                token=percept.slots[0, k],
                lamp=percept.lamp[0, k],
            )
            for k in machine_slots
        ]
        objs = [dict(slot=k, xy=tuple(coords[k].tolist()), token=percept.slots[0, k]) for k in objects]
        return machines, objs

    def _key(self, token):
        return self.core.key(token[None])[0]

    # retrieval and binding --------------------------------------------
    def _propose(self, key):
        concepts = [
            e for e in self.memory.view()["entities"].values() if e.kind == "concept"
        ]
        if not concepts:
            return {}
        context = self.retriever(
            self.memory.store,
            Query(
                key=key.detach().cpu(),
                space="machine-key",
                model_version=self.core_version,
                name="key",
                entity_kind="concept",
            ),
            RetrievalBudget(entities=self.settings.proposals, components=64, values=100000),
        )
        return dict(context.scores)

    def _latest(self, entity, name):
        components = [
            c
            for c in self.memory.view()["components"].values()
            if c.entity_id == entity and c.name == name
        ]
        if not components:
            return None
        latest = max(components, key=lambda c: (c.valid_from, c.revision, c.id))
        return latest if latest.active else None

    def _bindings(self, concept=None):
        latest = {}
        for c in self.memory.view()["components"].values():
            if c.name != "binding":
                continue
            old = latest.get(c.entity_id)
            if old is None or (c.valid_from, c.revision, c.id) > (old.valid_from, old.revision, old.id):
                latest[c.entity_id] = c
        return [
            c
            for c in latest.values()
            if c.active and (concept is None or c.data["concept"] == concept)
        ]

    @torch.no_grad()
    def concept_state(self, concept):
        """Current (Z, read-set ids); recompute code/key from active supporting evidence."""
        bindings = sorted(
            (b for b in self._bindings(concept) if b.data.get("supports")),
            key=lambda b: b.id,
        )
        view = self.memory.view()
        ordered = self.support_of(concept)
        parents = tuple(b.id for b in bindings)
        code = self._latest(concept, "code")
        key = self._latest(concept, "key")
        fresh = (
            code is not None
            and key is not None
            and code.evidence == tuple(ordered)
            and code.parents == parents
            and key.parents == parents
            and code.model_version == self.core_version
        )
        if not fresh:
            z = self.induce(ordered)
            appearance = [
                view["components"][b.parents[0]].tensor(device=self.device) for b in bindings
            ]
            mean = (
                F.normalize(torch.stack(appearance).mean(0), dim=-1)
                if appearance
                else torch.zeros(self.core.key_head[-1].out_features, device=self.device)
            )
            tx = self.memory.begin("internal")
            code_id = tx.put_component(
                concept, "code", z.detach().cpu(), space="rule-code",
                model_version=self.core_version, role="inferred",
                evidence=tuple(ordered), parents=parents,
            )
            key_id = tx.put_component(
                concept, "key", mean.detach().cpu(), space="machine-key",
                model_version=self.core_version, role="inferred", parents=parents,
            )
            self.memory.commit(tx)
            self._codes[code_id] = z.detach()
            code, key = self.memory.view()["components"][code_id], self.memory.view()["components"][key_id]
        if code.id not in self._codes:
            self._codes[code.id] = code.tensor(device=self.device)
        return self._codes[code.id].clone(), (code.id, key.id)

    @torch.no_grad()
    def transition_tokens(self, pre, record, post):
        """Perception-derived transition tokens from two frames and an action record."""
        _, before = self.percept(pre)
        _, after = self.percept(post)
        xy = lambda value: torch.tensor([list(value)], device=self.device, dtype=torch.float32)
        m = int(pointer(before.alpha, xy(record.machine_xy)))
        a = int(pointer(before.alpha, xy(record.a_xy)))
        b = int(pointer(before.alpha, xy(record.b_xy)))
        m_post = int(pointer(after.alpha, xy(record.machine_xy)))
        tokens = dict(
            m_pre=before.slots[0, m], a=before.slots[0, a], b=before.slots[0, b],
            m_post=after.slots[0, m_post],
            outcome=float(after.lamp[0, m_post] > 0),  # the agent's own perception
            distinct_slots=len({m, a, b}) == 3,
        )
        tokens["e"] = self.core.evidence(tokens["m_pre"], tokens["a"], tokens["b"], tokens["m_post"])
        return tokens

    @torch.no_grad()
    def evidence_tokens(self, evidence_id):
        """Tokens of one stored transition blob (cache by evidence id)."""
        if evidence_id not in self._tokens:
            record = self.memory.view()["evidence"][evidence_id]
            frames = self.memory.get_blob(record.content_ref, record.content_hash)
            action = record.data["action"]
            self._tokens[evidence_id] = self.transition_tokens(
                from_uint8(frames[0], self.device),
                _record(action["machine_xy"], action["a_xy"], action["b_xy"]),
                from_uint8(frames[1], self.device),
            )
        return self._tokens[evidence_id]

    def support_of(self, concept):
        """Ordered active support evidence ids of a concept (<= support_limit)."""
        view = self.memory.view()
        support = set()
        for b in self._bindings(concept):
            if b.data.get("supports"):
                support.update(view["components"][b.parents[1]].evidence)
        return sorted(support, key=lambda e: (view["evidence"][e].available_at, e))[
            -self.settings.support_limit :
        ]

    @torch.no_grad()
    def induce(self, evidence_ids, loops=None, permute=None):
        if not evidence_ids:
            return self.empty_code.clone() if loops is None else self.core.induce(
                torch.zeros(1, 0, self.core.width, device=self.device),
                torch.zeros(1, 0, dtype=torch.bool, device=self.device), loops=loops,
            )[0]
        rows = [self.evidence_tokens(e) for e in evidence_ids]
        if permute is not None:  # control: permuted post-state labels
            posts = [rows[i]["m_post"] for i in permute]
            e = torch.stack([
                self.core.evidence(r["m_pre"], r["a"], r["b"], p) for r, p in zip(rows, posts)
            ])
        else:
            e = torch.stack([r["e"] for r in rows])
        valid = torch.ones(1, len(rows), dtype=torch.bool, device=self.device)
        return self.core.induce(e[None], valid, loops=loops if loops is not None else self.settings.loops)[0]

    @torch.no_grad()
    def _log_likelihood(self, z, evidence_ids):
        rows = [self.evidence_tokens(e) for e in evidence_ids]
        m = torch.stack([r["m_pre"] for r in rows])
        a = torch.stack([r["a"] for r in rows])
        b = torch.stack([r["b"] for r in rows])
        y = torch.tensor([r["outcome"] for r in rows], device=self.device)
        logit, _ = self.core.apply(m, a, b, z[None].expand(len(rows), -1, -1), self.settings.loops)
        return float(-F.binary_cross_entropy_with_logits(logit, y))

    def _new_label(self):
        return f"{self.rng.getrandbits(64):016x}"

    def _bind(self, instance, appearance_id, key, transitions_id, evidence_ids, *, prior=None):
        """Propose by appearance, verify by behaviour, bind or create. Returns decision.

        `prior`: a concept this instance belonged to before an invalidation; it is
        always among the behavioural checks even when its key is no longer retrievable.
        """
        scores = self._propose(key)
        if prior is not None and prior not in scores:
            scores = {**scores, prior: 0.0}
        decision = dict(instance=instance, proposals=scores)
        if len(evidence_ids) >= self.settings.min_check:
            z_new = self.induce(evidence_ids)
            new_ll = self._log_likelihood(z_new, evidence_ids) - self.settings.lam
            checks = {c: self._log_likelihood(self.concept_state(c)[0], evidence_ids) for c in scores}
            decision.update(checks=checks, new_ll=new_ll)
            best = max(checks, key=checks.get) if checks else None
            if best is not None and checks[best] >= new_ll:
                return self._commit_binding(instance, best, appearance_id, transitions_id, evidence_ids, True, True, float(scores[best]), decision, "bound")
            concept = self._create_concept(evidence_ids)
            if best is not None and scores[best] >= self.settings.tau:
                decision["conflicts_with"] = best
            return self._commit_binding(instance, concept, appearance_id, transitions_id, evidence_ids, True, True, None, decision, "created", conflict=decision.get("conflicts_with"))
        if scores:
            best = max(scores, key=scores.get)
            s = self.settings
            if s.feedback_support and transitions_id is not None and scores[best] >= s.tau_feedback:
                # Too few transitions for a behavioural check: confident appearance
                # membership lets the observed feedback join the concept's support.
                return self._commit_binding(instance, best, appearance_id, transitions_id, evidence_ids, False, True, float(scores[best]), decision, "appearance_support")
            if scores[best] >= s.tau:
                return self._commit_binding(instance, best, appearance_id, transitions_id, evidence_ids, False, False, float(scores[best]), decision, "appearance")
        decision.update(status="unbound", concept=None, binding=None)
        return decision

    def _create_concept(self, evidence_ids):
        tx = self.memory.begin("internal")
        concept = tx.create_entity(self._new_label(), kind="concept")
        self.memory.commit(tx)
        return concept

    def _commit_binding(self, instance, concept, appearance_id, transitions_id, evidence_ids, verified, supports, score, decision, status, conflict=None):
        tx = self.memory.begin("internal")
        parents = (appearance_id,) + ((transitions_id,) if transitions_id else ())
        binding = tx.put_component(
            instance, "binding", torch.ones(1), space="binding", model_version=self.core_version,
            role="inferred", confidence=None if score is None else min(max(score, 0.0), 1.0),
            data=dict(concept=concept, verified=verified, supports=supports and transitions_id is not None),
            parents=parents,
        )
        if evidence_ids:
            tx.relate(instance, concept, "instance_of", evidence=tuple(evidence_ids))
        if conflict:
            tx.relate(concept, conflict, "conflicts_with", evidence=tuple(evidence_ids))
        self.memory.commit(tx)
        decision.update(status=status, concept=concept, binding=binding)
        if supports and transitions_id is not None:
            self.concept_state(concept)  # publish current key/code for later retrieval
        return decision

    # sessions ----------------------------------------------------------
    def _instances(self, tx, evidence_id, machines):
        ids = []
        for machine in machines:
            instance = tx.create_entity(self._new_label(), kind="instance")
            appearance = tx.put_component(
                instance, "appearance", self._key(machine["token"]).detach().cpu(),
                space="machine-key", model_version=self.core_version, role="observed",
                evidence=(evidence_id,),
            )
            ids.append((instance, appearance))
        return ids

    @torch.no_grad()
    def observe_scene(self, rgb, *, persist=True, control=None):
        """Look at a scene: layout, appearance retrieval; optionally persist instances."""
        sha, percept = self.percept(rgb)
        machines, objects = self._layout(percept)
        if persist:
            ref, blob_sha = self.memory.put_blob(to_uint8(rgb)[None])
            tx = self.memory.begin("observation")
            proof = tx.add_evidence(SOURCE, "image", content_ref=ref, content_hash=blob_sha)
            pairs = self._instances(tx, proof, machines)
            self.memory.commit(tx)
            for machine, (instance, appearance) in zip(machines, pairs):
                key = self._key(machine["token"])
                d = self._bind(instance, appearance, key, None, [])
                machine.update(instance=instance, appearance=appearance, concept=d["concept"], binding=d["binding"])
        else:
            for machine in machines:
                scores = self._propose(self._key(machine["token"]))
                best = max(scores, key=scores.get) if scores else None
                concept = best if best is not None and scores[best] >= self.settings.tau else None
                machine.update(instance=None, appearance=None, concept=concept, binding=None)
        return SceneView(sha, percept, machines, objects, persist)

    @torch.no_grad()
    def observe_session(self, transitions):
        """Demonstration session in one static scene: list of (pre, record, status, post)."""
        first = transitions[0][0]
        _, percept = self.percept(first)
        machines, _ = self._layout(percept)
        ref, sha = self.memory.put_blob(to_uint8(first)[None])
        tx = self.memory.begin("observation")
        proof = tx.add_evidence(SOURCE, "image", content_ref=ref, content_hash=sha)
        pairs = self._instances(tx, proof, machines)
        owned = {i: [] for i in range(len(machines))}
        evidence = []
        for pre, record, status, post in transitions:
            ref, sha = self.memory.put_blob(np.stack((to_uint8(pre), to_uint8(post))))
            e = tx.add_evidence(
                SOURCE, "transition", content_ref=ref, content_hash=sha,
                data=dict(action=record.as_dict(), receipt=status),
            )
            evidence.append(e)
            if status == "ok" and machines:
                owner = self._owner(machines, record.machine_xy)
                owned[owner].append(e)
        self.memory.commit(tx)
        decisions = []
        for index, (machine, (instance, appearance)) in enumerate(zip(machines, pairs)):
            transitions_id = self._transitions_component(instance, appearance, owned[index])
            d = self._bind(instance, appearance, self._key(machine["token"]), transitions_id, owned[index])
            d.update(xy=machine["xy"], evidence=list(owned[index]))
            decisions.append(d)
        return dict(evidence=evidence, decisions=decisions)

    def _owner(self, machines, xy):
        distances = [(m["xy"][0] - xy[0]) ** 2 + (m["xy"][1] - xy[1]) ** 2 for m in machines]
        return int(np.argmin(distances))

    def _transitions_component(self, instance, appearance, evidence_ids):
        if not evidence_ids:
            return None
        tx = self.memory.begin("internal")
        cid = tx.put_component(
            instance, "transitions", torch.tensor([float(len(evidence_ids))]),
            space="count", model_version=self.core_version, role="inferred",
            evidence=tuple(evidence_ids), parents=(appearance,),
        )
        self.memory.commit(tx)
        return cid

    # source corrections -------------------------------------------------
    @torch.no_grad()
    def receive_supersede(self, old_id, transition):
        pre, record, status, post = transition
        ref, sha = self.memory.put_blob(np.stack((to_uint8(pre), to_uint8(post))))
        tx = self.memory.begin("observation")
        new = tx.add_evidence(SOURCE, "transition", content_ref=ref, content_hash=sha,
                              data=dict(action=record.as_dict(), receipt=status))
        tx.supersede(old_id, new)
        self.memory.commit(tx)
        self.repair()
        return new

    @torch.no_grad()
    def receive_retract(self, evidence_id):
        old = self.memory.view()["evidence"][evidence_id]
        tx = self.memory.begin("correction", payload=dict(source=old.source), occurred_at=old.occurred_at)
        tx.retract_evidence(evidence_id)
        self.memory.commit(tx)
        self.repair()

    def repair(self):
        """Re-derive invalidated per-instance transitions and bindings.

        Evidence is re-resolved through retractions/replacements. The previous
        binding decision (concept, verified, supports) is retained with new parents;
        automatic re-verification after evidence repair is a declared later capability.
        """
        view = self.memory.view()
        retracted = view["retracted"]
        repaired = []
        for instance in [e.id for e in view["entities"].values() if e.kind == "instance"]:
            components = [c for c in view["components"].values() if c.entity_id == instance and c.name == "transitions"]
            bindings = [c for c in view["components"].values() if c.entity_id == instance and c.name == "binding"]
            if not components or not bindings:
                continue
            latest = max(components, key=lambda c: (c.valid_from, c.revision, c.id))
            binding = max(bindings, key=lambda c: (c.valid_from, c.revision, c.id))
            if latest.active and binding.active:
                continue
            evidence = []
            for e in latest.evidence:
                while e in retracted and retracted[e]["replacement"] is not None:
                    e = retracted[e]["replacement"]
                # A replacement teaches only on its own successful receipt.
                if e not in retracted and view["evidence"][e].data.get("receipt") == "ok":
                    evidence.append(e)
            appearance = latest.parents[0]
            if not self.memory.view()["components"][appearance].active:
                continue
            transitions_id = self._transitions_component(instance, appearance, evidence) if not latest.active else latest.id
            data = binding.data
            decision = dict(instance=instance, repaired_from=binding.id)
            self._commit_binding(
                instance, data["concept"], appearance, transitions_id, evidence,
                data["verified"], data["supports"], binding.confidence, decision, "repaired",
            )
            repaired.append(decision)
        return repaired

    # prediction ---------------------------------------------------------
    def _code_for(self, machine, control=None):
        concept = machine.get("concept")
        if control == "empty" or concept is None:
            return self.empty_code.clone(), ()
        z, ids = self.concept_state(concept)
        return z, ids + ((machine["binding"],) if machine.get("binding") else ())

    @torch.no_grad()
    def predict(self, view, record, *, code=None, loops=None):
        machine = view.machines[self._owner(view.machines, record.machine_xy)]
        a = int(pointer(view.percept.alpha, torch.tensor([record.a_xy], device=self.device)))
        b = int(pointer(view.percept.alpha, torch.tensor([record.b_xy], device=self.device)))
        if code is None:
            z, ids = self._code_for(machine)
        else:
            z, ids = code, ()
        logit, _ = self.core.apply(
            machine["token"][None], view.percept.slots[0, a][None],
            view.percept.slots[0, b][None], z[None],
            self.settings.loops if loops is None else loops,
        )
        return dict(
            logit=float(logit[0]), p=float(torch.sigmoid(logit[0])),
            concept=machine.get("concept"), read_set=self.memory.read_set(ids),
        )

    # planning and execution ---------------------------------------------
    @torch.no_grad()
    def plan(self, view, goal, *, presses_done, control=None, budget=None):
        """`budget`: total presses allowed for this task (default and cap: the contract's)."""
        limit = self.contract.budget if budget is None else min(budget, self.contract.budget)
        if len(view.machines) != 2:
            return lc.Decision("abstain", None, self.contract.utility("abstain", presses_done), {}, 0), self.memory.read_set(())
        codes, ids = [], ()
        for machine in view.machines:
            z, used = self._code_for(machine, control)
            codes.append(z)
            ids += used
        objects = (
            torch.stack([o["token"] for o in view.objects])
            if view.objects
            else torch.zeros(0, self.core.width, device=self.device)
        )
        decision = lc.search(
            self.core,
            torch.stack([m["token"] for m in view.machines]),
            objects,
            torch.stack(codes),
            torch.stack([m["lamp"] for m in view.machines]),
            goal,
            self.contract,
            budget_left=limit - presses_done,
            presses_done=presses_done,
            loops=self.settings.loops,
        )
        return decision, self.memory.read_set(ids)

    def record_for(self, view, action):
        m, i, j = action
        return _record(view.machines[m]["xy"], view.objects[i]["xy"], view.objects[j]["xy"])

    @torch.no_grad()
    def execute(self, actuator, view, decision, read, *, persist=True):
        """Recheck the read set, press once, store the receipt and learn from success."""
        if not self.memory.is_current(read):
            return dict(status="stale", record=None, post=None, evidence=None, feedback=None)
        m = decision.action[0]
        return self._act(actuator, view, m, self.record_for(view, decision.action), persist=persist)

    @torch.no_grad()
    def _act(self, actuator, view, machine_index, record, *, persist=True):
        pre = actuator.frame()
        receipt = actuator.press(record)
        post = actuator.frame()
        evidence, feedback = None, None
        if persist and view.persisted:
            ref, sha = self.memory.put_blob(np.stack((to_uint8(pre), to_uint8(post))))
            tx = self.memory.begin("observation")
            evidence = tx.add_evidence(SOURCE, "transition", content_ref=ref, content_hash=sha,
                                       data=dict(action=record.as_dict(), receipt=receipt.status))
            self.memory.commit(tx)
            if receipt.status == "ok":  # failed receipts stay raw evidence, never support
                feedback = self._attach(view.machines[machine_index], evidence)
        return dict(status=receipt.status, record=record, post=post, cost=receipt.cost,
                    evidence=evidence, feedback=feedback)

    def _attach(self, machine, evidence_id):
        """Same acquisition path as demonstrations: extend the instance's transitions,
        re-derive its binding (behavioural check once m>=min_check) and let concept
        support/code recompute lazily from the new active evidence."""
        instance, appearance = machine.get("instance"), machine.get("appearance")
        if instance is None or not self.memory.view()["components"][appearance].active:
            return None
        previous = self._latest(instance, "transitions")
        evidence = list(previous.evidence if previous is not None else ()) + [evidence_id]
        transitions_id = self._transitions_component(instance, appearance, evidence)
        key = self.memory.tensor(appearance).to(self.device)
        decision = self._bind(instance, appearance, key, transitions_id, evidence)
        machine.update(concept=decision["concept"], binding=decision["binding"])
        return dict(status=decision["status"], concept=decision["concept"], transitions=len(evidence))

    def _refresh(self, view, rgb):
        """Replan from current perception: machines (token, lamp, xy) and objects."""
        _, percept = self.percept(rgb)
        machines, objects = self._layout(percept)
        for old, new in zip(view.machines, machines):
            old.update(token=new["token"], lamp=new["lamp"], xy=new["xy"], slot=new["slot"])
        view.objects = objects
        view.percept = percept

    @torch.no_grad()
    def run_task(self, actuator, goal, *, control=None):
        """Perceive, plan, execute, learn from the receipt, re-perceive and replan."""
        persist = control is None
        view = self.observe_scene(actuator.frame(), persist=persist)
        presses, trace = 0, []
        for _ in range(self.contract.budget + 3):
            decision, read = self.plan(view, goal, presses_done=presses, control=control)
            step = dict(option=decision.option, action=decision.action, value=decision.value,
                        values={k: float(v) for k, v in decision.values.items()})
            trace.append(step)
            if decision.option != "press":
                return dict(option=decision.option, presses=presses, trace=trace)
            result = self.execute(actuator, view, decision, read, persist=persist)
            step.update(status=result["status"], evidence=result["evidence"], feedback=result["feedback"])
            if result["status"] == "stale":
                continue
            if result["status"] != "budget_exceeded":
                presses += 1
            self._refresh(view, result["post"])
        return dict(option="abstain", presses=presses, trace=trace, loop_guard=True)

    # testimony ------------------------------------------------------------
    @torch.no_grad()
    def receive_claim(self, rgb, record, claimed, actuator):
        """Record the claim as testimony (never support); test it with one own press.

        The tested transition enters the same feedback path as any executed press.
        """
        view = self.observe_scene(rgb, persist=True)
        ref, sha = self.memory.put_blob(to_uint8(rgb)[None])
        tx = self.memory.begin("observation")
        tx.add_evidence("testimony", "claim", content_ref=ref, content_hash=sha,
                        data=dict(action=record.as_dict(), claimed=int(claimed)))
        self.memory.commit(tx)
        result = self._act(actuator, view, self._owner(view.machines, record.machine_xy), record)
        ok = result["status"] == "ok"
        observed = self.evidence_tokens(result["evidence"])["outcome"] if ok else None
        return dict(tested=ok, evidence=result["evidence"], observed=observed, feedback=result["feedback"])

    def episodic_answer(self, rgb, record):
        """Exact instance memory: a stored own observation of this frame and action."""
        target = to_uint8(rgb)
        for e in reversed(list(self.memory.view()["evidence"].values())):
            if e.source != SOURCE or e.modality != "transition" or e.id in self.memory.view()["retracted"]:
                continue
            if e.data["action"] != record.as_dict():
                continue
            frames = self.memory.get_blob(e.content_ref, e.content_hash)
            if np.array_equal(frames[0], target):
                return dict(evidence=e.id, outcome=self.evidence_tokens(e.id)["outcome"])
        return None


def _record(machine_xy, a_xy, b_xy):
    from pathwm.data.rule_world import ActionRecord

    return ActionRecord(tuple(machine_xy), tuple(a_xy), tuple(b_xy))


# ---------------------------------------------------------------- adapters


def slot_candidates(percept, encoder, *, source, model_version, index=0):
    """Learned slots -> existing WorldSession `Candidate` records (replaces supplied descriptors)."""
    from .modules import Candidate

    keys, values = encoder(percept.slots[index])
    return tuple(
        Candidate(
            f"slot-{k}", source, "image", keys[k].detach(), values[k].detach(),
            "slot", model_version, exclusive_group="frame",
        )
        for k in range(len(keys))
    )


def concept_context(context_encoder, code, *, age=0.0):
    """Concept code [K,D] -> one ContextEncoder token for the existing Thinker path."""
    return context_encoder.project_value("code", code.reshape(1, -1), role="inferred", age=age)
