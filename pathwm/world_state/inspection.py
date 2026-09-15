"""One bounded, detached diagnostic schema for training and inference.

Use as the existing modules' trace mapping or attach temporary named forward hooks.
Summaries are readouts, not semantic explanations. Exact tensors are opt-in/capped.
"""

from collections.abc import MutableMapping
from contextlib import contextmanager
from dataclasses import asdict, fields, is_dataclass
import copy
from pathlib import Path
import numpy as np
import torch
from pathwm.io import atomic_json, file_hash
from .store import primitive


def tensor_summary(value, *, sample_values=8):
    value = value.detach().cpu()
    flat = value.reshape(-1)
    finite = torch.isfinite(flat)
    x = flat[finite].double()
    return dict(
        shape=list(value.shape),
        dtype=str(value.dtype),
        finite=int(finite.sum()),
        nonfinite=int((~finite).sum()),
        count=value.numel(),
        mean=float(x.mean()) if len(x) else None,
        rms=float(x.square().mean().sqrt()) if len(x) else None,
        minimum=float(x.min()) if len(x) else None,
        maximum=float(x.max()) if len(x) else None,
        sample=[float(v) if torch.isfinite(v) else None for v in flat[:sample_values]],
    )


class WorldTrace(MutableMapping):
    def __init__(self, *, max_records=256, tensor_values=0, max_record_bytes=32768):
        if any(
            type(v) is not int or v < 0
            for v in (max_records, tensor_values, max_record_bytes)
        ):
            raise ValueError("Diagnostic budgets must be nonnegative integers")
        self.max_records, self.tensor_values, self.max_record_bytes = (
            max_records,
            tensor_values,
            max_record_bytes,
        )
        self._summaries, self._tensors, self.records = {}, {}, []
        self.dropped = 0
        self._values = 0

    def __len__(self):
        return len(self._summaries)

    def __iter__(self):
        return iter(self._summaries)

    def __getitem__(self, key):
        return copy.deepcopy(self._summaries[key])

    def __delitem__(self, key):
        del self._summaries[key]
        old = self._tensors.pop(key, None)
        if old is not None:
            self._values -= old.numel()

    def __setitem__(self, key, value):
        if key not in self._summaries and len(self._summaries) >= self.max_records:
            self.dropped += 1
            return
        old = self._tensors.pop(key, None)
        if old is not None:
            self._values -= old.numel()
        if isinstance(value, torch.Tensor):
            self._summaries[key] = tensor_summary(value)
            if value.numel() + self._values <= self.tensor_values:
                self._tensors[key] = value.detach().cpu().clone()
                self._values += value.numel()
        else:
            self._summaries[key] = primitive(value)

    def record(self, kind, data):
        import json

        data = primitive(data)
        if (
            len(self.records) >= self.max_records
            or len(json.dumps(data).encode()) > self.max_record_bytes
        ):
            self.dropped += 1
            return
        self.records.append(dict(kind=kind, data=data))

    def parameters(self, model):
        for name, p in model.named_parameters():
            self["parameter." + name] = p
            if p.grad is not None:
                self["gradient." + name] = p.grad

    @contextmanager
    def capture(self, model, names, *, gradients=False):
        """Enclose forward AND backward. Hook removal also occurs on exceptions."""
        modules = dict(model.named_modules())
        if not set(names) <= modules.keys():
            raise ValueError("Unknown named module for inspection")
        handles, tensor_handles = [], []

        def tensors(value, path="0"):
            if isinstance(value, torch.Tensor):
                yield path, value
            elif is_dataclass(value):
                for field in fields(value):
                    yield from tensors(
                        getattr(value, field.name), path + "." + field.name
                    )
            elif isinstance(value, dict):
                for key, item in value.items():
                    yield from tensors(item, path + "." + str(key))
            elif isinstance(value, (tuple, list)):
                for index, item in enumerate(value):
                    yield from tensors(item, path + "." + str(index))

        def attach(name):
            def record_output(module, inputs, output):
                for path, tensor in tensors(output):
                    key = f"activation.{name}.{path}"
                    self[key] = tensor
                    if (
                        gradients
                        and tensor.requires_grad
                        and key in self
                        and len(tensor_handles) < self.max_records
                    ):

                        def grad_hook(grad, key=key):
                            self["backward." + key] = grad

                        tensor_handles.append(tensor.register_hook(grad_hook))

            return record_output

        try:
            for name in names:
                handles.append(modules[name].register_forward_hook(attach(name)))
            yield self
        finally:
            for handle in handles + tensor_handles:
                handle.remove()

    def summary(self):
        return primitive(
            dict(
                schema="pathwm-world-inspection-v1",
                records=self.records,
                tensors=self._summaries,
                dropped=self.dropped,
                stored_tensor_values=self._values,
            )
        )

    def export(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        arrays = {
            str(i): (t.float() if t.dtype == torch.bfloat16 else t).numpy()
            for i, t in enumerate(self._tensors.values())
        }
        array_path = directory / "world_trace.npz"
        np.savez_compressed(array_path, **arrays)
        record = self.summary()
        record["arrays"] = dict(
            path=array_path.name,
            sha256=file_hash(array_path),
            keys=dict(zip(self._tensors, arrays)),
        )
        atomic_json(directory / "world_trace.json", record)
        return record


def inspect_store(store):
    """Portable current graph plus source history; omits raw latent arrays by default."""
    components = []
    for c in store.components():
        record = asdict(c)
        record.pop("values")
        record["tensor"] = tensor_summary(c.tensor())
        components.append(record)
    return primitive(
        dict(
            schema="pathwm-world-inspection-v1",
            revision=store.revision,
            limits=asdict(store.limits),
            entities=[asdict(e) for e in store.entities()],
            canonical_ids={e.id: store.canonical(e.id) for e in store.entities()},
            components=components,
            relations=[asdict(r) for r in store.relations()],
            evidence=[asdict(e) for e in store.evidence()],
            events=[
                dict(
                    id=e["event"]["id"],
                    occurred_at=e["event"]["occurred_at"],
                    available_at=e["event"]["available_at"],
                    kind=e["event"]["kind"],
                    revision=e["receipt"]["revision"],
                    operations=[
                        dict(
                            op=o["op"],
                            value={
                                k: v for k, v in o["value"].items() if k != "values"
                            },
                        )
                        for o in e["operations"]
                    ],
                )
                for e in store.snapshot()["events"]
            ],
        )
    )


def snapshot_diff(before, after):
    a, b = inspect_store(before), inspect_store(after)
    changes = {}
    for name in ("entities", "components", "relations", "evidence"):
        old, new = {r["id"]: r for r in a[name]}, {r["id"]: r for r in b[name]}
        changes[name] = dict(
            added=sorted(new.keys() - old.keys()),
            removed=sorted(old.keys() - new.keys()),
            changed=sorted(k for k in old.keys() & new.keys() if old[k] != new[k]),
        )
    return dict(before=before.revision, after=after.revision, changes=changes)
