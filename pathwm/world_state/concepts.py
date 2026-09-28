"""Frozen-weight runtime concept memory.

`ConceptMemory` owns one WorldStore, hash-verified frame blobs and read sets. The
inference owner around the shared core is rebuilt in docs/shared-core-plan.md (S7).
"""

from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from pathwm.io import atomic_json
from .records import Limits
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



