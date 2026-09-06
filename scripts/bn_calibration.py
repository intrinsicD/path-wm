# Claude-authored via a context-free MCP task; integrated and verified by Codex.
"""Recalibrate BatchNorm running statistics of a pretrained module.

The public entry point is :func:`recalibrate_bn_statistics`.  It works on a deep
copy of the user's module, never touches the original (parameters, buffers,
per-module training flags) and restores the global CPU/CUDA RNG states.
"""
from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import torch
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm

__all__ = ["LayerCalibrationReport", "CalibrationReport", "recalibrate_bn_statistics"]

Batch = Any
BatchFactory = Callable[[], Iterable[Batch]]
ForwardFn = Callable[[nn.Module, Batch], Any]


@dataclass(frozen=True)
class LayerCalibrationReport:
    """Statistics for one recalibrated BatchNorm layer."""

    name: str
    module_type: str
    order: int
    batches: int
    samples: int
    original_digest: dict[str, str]
    new_digest: dict[str, str]


@dataclass(frozen=True)
class CalibrationReport:
    """Summary of a full recalibration run."""

    layers: tuple[LayerCalibrationReport, ...]
    batches_per_pass: int
    samples_per_pass: int


def _raw_bytes(t: torch.Tensor) -> bytes:
    x = t.detach().to("cpu").contiguous().flatten()
    try:
        return x.view(torch.uint8).numpy().tobytes()
    except Exception:  # pragma: no cover - exotic dtypes/back-ends
        return bytes(x.view(torch.uint8).tolist())


def _digest(obj: Any) -> str:
    h = hashlib.sha256()

    def walk(o: Any) -> None:
        if torch.is_tensor(o):
            h.update(b"T" + str(o.dtype).encode() + str(tuple(o.shape)).encode())
            h.update(_raw_bytes(o))
        elif isinstance(o, (list, tuple)):
            h.update(b"L")
            for item in o:
                walk(item)
        elif isinstance(o, dict):
            h.update(b"D")
            for k in sorted(o, key=repr):
                h.update(repr(k).encode())
                walk(o[k])
        else:
            h.update(b"O" + repr(o).encode())

    walk(obj)
    return h.hexdigest()


def _bn_digest(bn: _BatchNorm) -> dict[str, str]:
    return {n: _digest(b) for n, b in bn.named_buffers(recurse=False) if b is not None}


def _model_digest(m: nn.Module) -> str:
    state = sorted(m.state_dict(keep_vars=False).items())
    flags = [(n, mod.training) for n, mod in m.named_modules()]
    return _digest([state, flags])


def _check_batch(batch: Batch) -> int:
    """Return the leading (sample) dimension of a batch; reject empty batches."""
    stack = [batch]
    while stack:
        o = stack.pop(0)
        if torch.is_tensor(o):
            if o.dim() == 0:
                continue
            if o.shape[0] == 0:
                raise ValueError("empty input batch (0 samples) supplied by batch_factory")
            return int(o.shape[0])
        if isinstance(o, (list, tuple)):
            stack.extend(o)
        elif isinstance(o, dict):
            stack.extend(o[k] for k in sorted(o, key=repr))
    return -1


class _RngGuard:
    def __enter__(self) -> "_RngGuard":
        self._cpu = torch.get_rng_state()
        self._cuda = (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available() and torch.cuda.is_initialized()
            else None
        )
        return self

    def __exit__(self, *exc: Any) -> bool:
        torch.set_rng_state(self._cpu)
        if self._cuda is not None:
            torch.cuda.set_rng_state_all(self._cuda)
        return False


def _discover_order(probe: nn.Module, batch: Batch, forward_fn: ForwardFn) -> tuple[list[str], dict[str, int]]:
    order: list[str] = []
    counts: dict[str, int] = {}
    handles = []

    def make_hook(name: str):
        def hook(_m, _i, _o):
            counts[name] = counts.get(name, 0) + 1
            if counts[name] == 1:
                order.append(name)

        return hook

    for name, mod in probe.named_modules():
        if isinstance(mod, _BatchNorm):
            handles.append(mod.register_forward_hook(make_hook(name)))
    try:
        forward_fn(probe, batch)
    finally:
        for h in handles:
            h.remove()
    reused = sorted(n for n, c in counts.items() if c > 1)
    if reused:
        raise ValueError(
            f"BatchNorm modules invoked more than once per forward are unsupported: {reused}"
        )
    return order, counts


def _make_bn_pre_hook(stats: dict[str, int], name: str, min_elems: int):
    def hook(mod: _BatchNorm, inputs: tuple[Any, ...]) -> None:
        x = inputs[0]
        if not torch.is_tensor(x) or x.numel() == 0 or x.shape[0] == 0:
            raise ValueError(f"BatchNorm '{name}' received an empty input batch")
        per_channel = x.numel() // x.shape[1]
        if per_channel < min_elems:
            raise ValueError(
                f"BatchNorm '{name}' got {per_channel} element(s) per channel "
                f"(< {min_elems}); enlarge the batches instead of dropping samples"
            )
        stats["batches"] += 1
        stats["samples"] += int(x.shape[0])

    return hook


def recalibrate_bn_statistics(
    model: nn.Module,
    batch_factory: BatchFactory,
    forward_fn: ForwardFn,
    *,
    min_bn_elements_per_channel: int = 2,
    verify_batch_determinism: bool = True,
    require_all_bn_reached: bool = True,
    restore_training_flags: bool = True,
) -> tuple[nn.Module, CalibrationReport]:
    """Return (calibrated_copy, report) with BN buffers recomputed layer by layer."""
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not callable(batch_factory) or not callable(forward_fn):
        raise TypeError("batch_factory and forward_fn must be callables")
    before = _model_digest(model)
    with _RngGuard(), torch.no_grad():
        work = copy.deepcopy(model)
        work.eval()  # freezes dropout and every BN by default
        probe = copy.deepcopy(work)
        probe.eval()
        first = None
        for b in batch_factory():
            first = b
            break
        if first is None:
            raise ValueError("batch_factory() produced no batches")
        _check_batch(first)
        order, counts = _discover_order(probe, first, forward_fn)
        del probe
        named = dict(work.named_modules())
        all_bn = [n for n, m in work.named_modules() if isinstance(m, _BatchNorm)]
        if not all_bn:
            raise ValueError("model contains no BatchNorm layer to recalibrate")
        unreached = [n for n in all_bn if n not in counts]
        if unreached and require_all_bn_reached:
            raise ValueError(f"BatchNorm layers never executed in the discovery forward: {unreached}")
        reports: list[LayerCalibrationReport] = []
        ref_digests: list[str] | None = None
        batches_per_pass = samples_per_pass = 0
        for idx, name in enumerate(order):
            bn = named[name]
            if not bn.track_running_stats or bn.running_mean is None:
                raise ValueError(f"BatchNorm '{name}' does not track running statistics")
            original_digest = _bn_digest(bn)
            saved_momentum = bn.momentum
            bn.reset_running_stats()
            bn.momentum = None  # cumulative moving average over the whole pass
            bn.train(True)  # only this layer trains; all others stay in eval
            stats = {"batches": 0, "samples": 0}
            handle = bn.register_forward_pre_hook(
                _make_bn_pre_hook(stats, name, min_bn_elements_per_channel)
            )
            digests: list[str] = []
            nb = ns = 0
            try:
                for batch in batch_factory():
                    count = _check_batch(batch)
                    if verify_batch_determinism:
                        digests.append(_digest(batch))
                    forward_fn(work, batch)
                    nb += 1
                    ns += max(count, 0)
            finally:
                handle.remove()
                bn.eval()  # freeze before moving on to the next layer
                bn.momentum = saved_momentum
            if nb == 0:
                raise ValueError("batch_factory() produced no batches")
            if stats["batches"] == 0:
                raise RuntimeError(f"BatchNorm '{name}' was not executed during its pass")
            if verify_batch_determinism:
                if ref_digests is None:
                    ref_digests = digests
                elif digests != ref_digests:
                    raise RuntimeError(
                        "batch_factory() is not reproducible: pass "
                        f"{idx} yielded different batches than the first pass"
                    )
            reports.append(
                LayerCalibrationReport(
                    name=name,
                    module_type=type(bn).__name__,
                    order=idx,
                    batches=stats["batches"],
                    samples=stats["samples"],
                    original_digest=original_digest,
                    new_digest=_bn_digest(bn),
                )
            )
            batches_per_pass, samples_per_pass = nb, ns
        if restore_training_flags:
            src = dict(model.named_modules())
            for n, m in work.named_modules():
                m.training = src[n].training
        for (pn, p), (_, q) in zip(model.named_parameters(), work.named_parameters()):
            if not torch.equal(p.detach().to("cpu"), q.detach().to("cpu")):
                raise RuntimeError(f"parameter '{pn}' changed during calibration")
    if _model_digest(model) != before:
        raise RuntimeError("the original model was mutated during calibration")
    return work, CalibrationReport(tuple(reports), batches_per_pass, samples_per_pass)
