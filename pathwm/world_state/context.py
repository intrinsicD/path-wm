"""Bounded internal views of external WorldStore records; no second truth store.

Retained pins have exact component identity, representation and task ownership.
Flat uses one LRU capacity; Local/Global reserve half each (odd slot to Local).
Scope and retention are caller decisions, not a claim of learned allocation.
"""
from dataclasses import asdict, dataclass

from .records import identifier
from .store import WorldStore
from .session import WorldSession
from pathwm.io import digest


@dataclass(frozen=True)
class ContextPin:
    component_id: str
    entity_id: str
    name: str
    revision: int
    space: str
    model_version: str
    task: str
    scope: str


class WorkingContext:
    def __init__(self, store, *, capacity=4, layout='flat', representations, owner='default'):
        if type(capacity) is not int or capacity < 2 or layout not in ('flat', 'local_global'):
            raise ValueError('Context needs capacity >=2 and flat/local_global layout')
        if not isinstance(store, (WorldStore, WorldSession)):
            raise ValueError('Context source must be a WorldStore or WorldSession')
        self._source, self.capacity, self.layout = store, capacity, layout
        identifier(owner)
        self.owner = owner
        self.representations = {name: tuple(value) for name, value in representations.items()}
        for name, pair in self.representations.items():
            identifier(name)
            if len(pair) != 2:
                raise ValueError('Representation needs space and model version')
            for value in pair:
                identifier(value)
        self._entries = []
        self.evictions = 0

    @property
    def store(self):
        # Session commits replace the store atomically. Never pin an obsolete
        # store object when the authoritative owner is a live session.
        return self._source.store if isinstance(self._source, WorldSession) else self._source

    def _validate(self, pin):
        current = self.store.latest(pin.entity_id, pin.name)
        if (current is None or not current.active or current.id != pin.component_id or
                current.revision != pin.revision or
                (current.space, current.model_version) != (pin.space, pin.model_version) or
                self.representations.get(pin.name) != (pin.space, pin.model_version)):
            raise ValueError('Stale or incompatible context pin; reset and retrieve again')
        return current

    def retain(self, component_id, *, task, scope='local'):
        identifier(task)
        if scope not in ('local', 'global'):
            raise ValueError('Context scope must be local or global')
        c = self.store.component(component_id)
        if c is None:
            raise ValueError('Unknown context component')
        pin = ContextPin(c.id, c.entity_id, c.name, c.revision, c.space, c.model_version, task, scope)
        self._validate(pin)
        # One component/task pin can move scopes; never duplicate its payload.
        entries = [p for p in self._entries if (p.component_id, p.task) != (component_id, task)]
        if self.layout == 'local_global':
            cap = self.capacity // 2 + (self.capacity % 2 if scope == 'local' else 0)
            matching = [i for i, p in enumerate(entries) if p.scope == scope]
            if len(matching) >= cap:
                entries.pop(matching[0])
                self.evictions += 1
        elif len(entries) >= self.capacity:
            entries.pop(0)
            self.evictions += 1
        self._entries = entries + [pin]

    def validate(self, task, *, expected_revision=None):
        """Recheck immediately before consumption; caller owns atomic emission.

        This is not a lock or external output transaction. EpisodeClient.emit owns
        checked session output commits when a consumer needs that stronger boundary.
        """
        identifier(task)
        if expected_revision is not None and self.store.revision != expected_revision:
            raise ValueError('Stale context epoch; retrieve again')
        return tuple(self._validate(p) for p in self._entries if p.task == task)

    def read(self, task, *, max_components=2, max_values=1024, expected_revision=None):
        """Return selected task values or fail explicitly; never silent unknown.

        Reading does not change LRU order: retain explicitly marks renewed use.
        A stale selected entry fails even if newer entries would fit the budget.
        """
        identifier(task)
        if any(type(x) is not int or x < 0 for x in (max_components, max_values)):
            raise ValueError('Read budget must be nonnegative integers')
        selected = self.validate(task, expected_revision=expected_revision)
        if len(selected) > max_components or sum(len(c.values) for c in selected) > max_values:
            raise ValueError('Context read budget omitted retained detail; select/reset explicitly')
        return selected

    def reset(self, task=None, *, scope=None):
        if task is not None:
            identifier(task)
        if scope not in (None, 'local', 'global'):
            raise ValueError('Unknown context scope')
        self._entries = [p for p in self._entries
                         if not ((task is None or p.task == task) and
                                 (scope is None or p.scope == scope))]

    def snapshot(self):
        result = dict(schema='working-context-v1', owner=self.owner, capacity=self.capacity, layout=self.layout,
                    representations={k: list(v) for k, v in self.representations.items()},
                    entries=[asdict(p) for p in self._entries], evictions=self.evictions)
        return dict(result, checksum=digest(result))

    @classmethod
    def restore(cls, store, snapshot, *, representations, owner='default', layout=None):
        # Checksum detects corruption, not malicious rewriting. Task tags are caller
        # metadata, not authenticated permissions. Restore only trusted snapshots.
        expected = {'schema', 'owner', 'capacity', 'layout', 'representations', 'entries', 'evictions', 'checksum'}
        if (set(snapshot) != expected or snapshot.get('schema') != 'working-context-v1' or
                snapshot.get('owner') != owner or
                (layout is not None and snapshot.get('layout') != layout) or
                snapshot.get('checksum') != digest({k:v for k,v in snapshot.items() if k != 'checksum'}) or
                snapshot.get('representations') != {k: list(v) for k, v in representations.items()}):
            raise ValueError('Incompatible working-context restart')
        result = cls(store, capacity=snapshot['capacity'], layout=snapshot['layout'],
                     representations=representations, owner=owner)
        for record in snapshot['entries']:
            pin = ContextPin(**record)
            identifier(pin.task)
            if pin.scope not in ('local', 'global'):
                raise ValueError('Invalid scope in restart')
            result._validate(pin)
            if any((p.task, p.component_id) == (pin.task, pin.component_id) for p in result._entries):
                raise ValueError('Duplicate context pin in restart')
            result._entries.append(pin)
        if len(result._entries) > result.capacity:
            raise ValueError('Over-capacity context restart')
        if result.layout == 'local_global':
            for scope, cap in [('local', (result.capacity + 1)//2), ('global', result.capacity//2)]:
                if sum(p.scope == scope for p in result._entries) > cap:
                    raise ValueError('Over-capacity scope in context restart')
        if type(snapshot['evictions']) is not int or snapshot['evictions'] < 0:
            raise ValueError('Invalid eviction count')
        result.evictions = snapshot['evictions']
        return result
