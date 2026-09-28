"""Offline inspection of a copied model and one explicitly supplied execution.

Module containment, observed ATen tensor provenance, and the actual autograd graph
are separate records. No edges are inferred from module order. This is a debugging
snapshot, not a static proof of all possible Python control-flow paths.
"""

import base64
import copy
from datetime import datetime, timezone
import hashlib
import inspect
import json
import random
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch
from torch.overrides import TorchFunctionMode
from torch.utils._python_dispatch import TorchDispatchMode


def tensor_leaves(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from tensor_leaves(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from tensor_leaves(item)


def source_info(value):
    try:
        lines, line = inspect.getsourcelines(value)
        filename = inspect.getsourcefile(value)
    except (OSError, TypeError):
        return None
    root = Path(__file__).resolve().parents[2]
    path = Path(filename).resolve()
    try:
        name = str(path.relative_to(root))
    except ValueError:
        return {"path": path.name, "line": line, "library": True}
    return {"path": name, "line": line, "code": "".join(lines)}


def numbers(tensor):
    """Every scalar, with explicit nonfinite/int64 representations for JavaScript."""
    values = tensor.detach().cpu().reshape(-1).tolist()
    if tensor.dtype in (torch.int64, torch.uint64):
        return [str(v) for v in values]
    return [v if not isinstance(v, float) or np.isfinite(v) else str(v) for v in values]


def statistics(tensor):
    value = tensor.detach().double().reshape(-1)
    finite = value[torch.isfinite(value)]
    if not len(finite):
        return {"count": value.numel(), "finite": 0}
    return dict(
        count=value.numel(),
        finite=finite.numel(),
        min=finite.min().item(),
        max=finite.max().item(),
        mean=finite.mean().item(),
        std=finite.std(unbiased=False).item(),
        norm=finite.norm().item(),
        zeros=int((finite == 0).sum()),
    )


def fingerprint(paths):
    digest = hashlib.sha256()
    for path in sorted(set(map(Path, paths))):
        digest.update(str(path).encode())
        digest.update(path.read_bytes() if path.is_file() else b"<missing>")
    return digest.hexdigest()


def watched_sources(root, checkpoint=None):
    root = Path(root)
    paths = [
        *root.joinpath("pathwm").rglob("*.py"),
        root / "pathwm/evaluation/explorer.html",
        root / "pathwm/evaluation/explorer-system.js",
        root / "experiments/multimodal.py",
        root / "experiments/world_state.py",
    ]
    if checkpoint is not None:
        paths.append(Path(checkpoint).resolve())
    return paths


def load_explorer_weights(model, checkpoint):
    """Strictly accept a complete learner or a complete deployed-agent state."""
    checkpoint = Path(checkpoint)
    from io import BytesIO

    payload = checkpoint.read_bytes()
    before = hashlib.sha256(payload).hexdigest()
    saved = torch.load(BytesIO(payload), map_location="cpu", weights_only=True)
    state = saved.get("model", saved) if isinstance(saved, dict) else saved
    if not isinstance(state, dict):
        raise ValueError("Checkpoint must contain a complete state dictionary")
    if set(state) == set(model.state_dict()):
        # Replay population is training bookkeeping, not a neural dimension.
        # Preserve the checkpoint's complete buffer even for a two-example capture.
        if "replay_errors" in state and "replay_errors" in model._buffers:
            if state["replay_errors"].ndim != 1:
                raise ValueError(
                    "Checkpoint replay_errors must be a one-dimensional buffer"
                )
            model.replay_errors = torch.empty_like(state["replay_errors"])
        model.load_state_dict(state, strict=True)
        scope = "complete learner (agent and teacher)"
    elif hasattr(model, "agent") and set(state) == set(model.agent.state_dict()):
        model.agent.load_state_dict(state, strict=True)
        model.target.load_state_dict(state, strict=True)
        scope = "deployed agent; teacher copied from these same weights"
    else:
        raise ValueError(
            "Checkpoint does not exactly match this model configuration; no partial weights loaded"
        )
    if before != hashlib.sha256(checkpoint.read_bytes()).hexdigest():
        raise ValueError(
            "Checkpoint changed while loading; retry after its writer finishes"
        )
    return dict(
        path=str(checkpoint.resolve()),
        sha256=before,
        scope=scope,
    )


def serve_explorer(
    path, *, root, command, checkpoint=None, port=8765, initial_fingerprint=None
):
    """Loopback-only viewer: serve one file; rebuild changed sources in a subprocess.

    No model/checkpoint upload API or arbitrary file serving. A failed rebuild keeps
    the previous snapshot and exposes the failure instead of claiming freshness.
    """
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import subprocess
    import threading

    path, root = Path(path).resolve(), Path(root).resolve()
    state = dict(
        state="current",
        version=hashlib.sha256(path.read_bytes()).hexdigest(),
        error=None,
    )
    current = initial_fingerprint or fingerprint(watched_sources(root, checkpoint))
    state["source_sha256"] = current
    lock = threading.Lock()

    def rebuild(expected):
        nonlocal current
        try:
            completed = subprocess.run(
                command, cwd=root, capture_output=True, text=True, timeout=120
            )
            if completed.returncode:
                raise RuntimeError(completed.stderr[-3000:] or completed.stdout[-3000:])
            actual = fingerprint(watched_sources(root, checkpoint))
            if actual != expected:
                raise RuntimeError(
                    "Source changed again during capture; save files, then refresh to retry"
                )
            with lock:
                current = actual
                state.update(
                    state="current",
                    version=hashlib.sha256(path.read_bytes()).hexdigest(),
                    source_sha256=actual,
                    error=None,
                )
        except Exception as error:
            with lock:
                current = expected
                state.update(state="error", error=str(error))

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.split("?")[0] == "/__explorer_status":
                try:
                    latest = fingerprint(watched_sources(root, checkpoint))
                    with lock:
                        if latest != current and state["state"] != "rebuilding":
                            state.update(state="rebuilding", error=None)
                            threading.Thread(
                                target=rebuild, args=(latest,), daemon=True
                            ).start()
                        body = json.dumps(state).encode()
                except OSError as error:
                    body = json.dumps(dict(state="error", error=str(error))).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
            elif self.path.split("?")[0] in ("/", "/index.html", "/" + path.name):
                body = path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
            else:
                self.send_error(404)
                return
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(
        f"Model explorer: http://127.0.0.1:{server.server_port} (Ctrl+C stops live updates)",
        flush=True,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()


class Recorder:
    def __init__(self, model):
        self.model = model
        self.modules, self.parameters, self.buffers = {}, {}, {}
        self.module_ids, self.parameter_ids = {}, {}
        self.tensor_ids, self.tensors, self.references = {}, {}, []
        self.storage_writers = {}
        self.operations, self.stack, self.frames = [], [], {}
        self.grad_scopes, self.grad_refs = {}, []
        self.busy = False
        seen_parameters = {}
        for path, module in model.named_modules(remove_duplicate=False):
            alias = self.module_ids.get(id(module))
            self.module_ids.setdefault(id(module), path)
            direct, buffers = [], []
            for name, parameter in module.named_parameters(
                recurse=False, remove_duplicate=False
            ):
                full = f"{path}.{name}" if path else name
                canonical = seen_parameters.setdefault(id(parameter), full)
                direct.append(canonical)
                if canonical in self.parameters:
                    if full not in self.parameters[canonical]["aliases"]:
                        self.parameters[canonical]["aliases"].append(full)
                    continue
                self.parameter_ids[id(parameter)] = canonical
                self.parameters[canonical] = dict(
                    name=full,
                    scope=path,
                    shape=list(parameter.shape),
                    dtype=str(parameter.dtype),
                    trainable=parameter.requires_grad,
                    values=numbers(parameter),
                    stats=statistics(parameter),
                    aliases=[full],
                    gradient=None,
                    gradient_stats=None,
                )
            for name, buffer in module.named_buffers(recurse=False):
                full = f"{path}.{name}" if path else name
                buffers.append(full)
                self.buffers[full] = dict(
                    name=full,
                    scope=path,
                    shape=list(buffer.shape),
                    dtype=str(buffer.dtype),
                    values=numbers(buffer),
                )
            self.modules[path] = dict(
                path=path,
                name=path.rsplit(".", 1)[-1] or type(module).__name__,
                type=type(module).__name__,
                parent=path.rpartition(".")[0] if path else None,
                children=[
                    f"{path}.{n}" if path else n
                    for n in module._modules
                    if module._modules[n] is not None
                ],
                parameters=sum(p.numel() for p in module.parameters()),
                trainable=sum(
                    p.numel() for p in module.parameters() if p.requires_grad
                ),
                direct_parameters=direct,
                buffers=buffers,
                alias_of=alias,
                training=module.training,
                calls=0,
                methods={},
                config=module.extra_repr(),
                source=source_info(type(module)),
            )
        for parameter in model.parameters():
            self.tensor(parameter)

    @property
    def location(self):
        return self.stack[-1][1:] if self.stack else ("@objective", "objective")

    def profile(self, frame, event, arg):
        if self.busy:
            return
        if event == "call":
            module = frame.f_locals.get("self")
            path = self.module_ids.get(id(module))
            name = frame.f_code.co_name
            if path is None or name.startswith("__"):
                return
            if frame.f_code.co_filename.endswith("torch/nn/modules/module.py"):
                return
            item = (id(frame), path, name)
            self.frames[id(frame)] = item
            self.stack.append(item)
            record = self.modules[path]
            record["calls"] += 1
            record["methods"][name] = record["methods"].get(name, 0) + 1
        elif event == "return" and id(frame) in self.frames:
            item = self.frames.pop(id(frame))
            if self.stack and self.stack[-1] == item:
                self.stack.pop()
            else:
                self.stack.remove(item)
            self.annotate(arg, item[1], item[2])

    def tensor(self, tensor, producer=None, new=False):
        identity = id(tensor)
        if identity in self.tensor_ids and not new:
            key = self.tensor_ids[identity]
            self.tensors[key]["requires_grad"] = tensor.requires_grad
            return key
        key = f"t{len(self.tensors)}"
        self.tensor_ids[identity] = key
        self.references.append(tensor)  # Prevent object-ID reuse during capture.
        self.tensors[key] = dict(
            id=key,
            shape=list(tensor.shape),
            dtype=str(tensor.dtype),
            requires_grad=tensor.requires_grad,
            producer=producer,
            parameter=self.parameter_ids.get(identity),
        )
        return key

    def annotate(self, result, scope=None, method=None):
        old = self.busy
        self.busy = True
        try:
            scope, method = (scope, method) if scope is not None else self.location
            for tensor in tensor_leaves(result):
                node = tensor.grad_fn
                key = self.tensor_ids.get(id(tensor))
                if key:
                    self.tensors[key]["requires_grad"] = tensor.requires_grad
                # A high-level torch call can create several autograd functions
                # (e.g. attention). Attribute its new functions to that call's
                # observed scope, stopping at already-attributed input history.
                pending = [node] if node is not None else []
                while pending:
                    current = pending.pop()
                    if current in self.grad_scopes:
                        continue
                    self.grad_scopes[current] = (scope, method)
                    self.grad_refs.append(current)
                    pending.extend(
                        child
                        for child, _ in current.next_functions
                        if child is not None
                    )
        finally:
            self.busy = old


class TensorOperations(TorchDispatchMode):
    def __init__(self, recorder):
        super().__init__()
        self.recorder = recorder

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        r = self.recorder
        if r.busy:
            return func(*args, **(kwargs or {}))
        r.busy = True
        try:
            values = list(tensor_leaves((args, kwargs)))
            inputs = list(dict.fromkeys(r.tensor(t) for t in values))
            # A view can observe an in-place write through another tensor object.
            # Preserve that real dependency in addition to the view's producer.
            for tensor in values:
                writer = r.storage_writers.get(tensor.untyped_storage()._cdata)
                if writer is not None and writer not in inputs:
                    inputs.append(writer)
            written = []
            for index, argument in enumerate(func._schema.arguments):
                if argument.alias_info is not None and argument.alias_info.is_write:
                    value = (
                        args[index]
                        if index < len(args)
                        else (kwargs or {}).get(argument.name)
                    )
                    written.extend(tensor_leaves(value))
            scope, method = r.location
            result = func(*args, **(kwargs or {}))
            identifier = f"o{len(r.operations)}"
            outputs = [r.tensor(t, identifier, new=True) for t in tensor_leaves(result)]
            for tensor in written:
                token = r.tensor_ids.get(id(tensor))
                if token not in outputs:
                    token = r.tensor(tensor, identifier, new=True)
                    outputs.append(token)
                r.storage_writers[tensor.untyped_storage()._cdata] = token
            r.operations.append(
                dict(
                    id=identifier,
                    name=str(func),
                    scope=scope,
                    method=method,
                    inputs=inputs,
                    outputs=outputs,
                    mutates=bool(written),
                )
            )
            return result
        finally:
            r.busy = False


class AutogradAttribution(TorchFunctionMode):
    def __init__(self, recorder):
        self.recorder = recorder

    def __torch_function__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        if not self.recorder.busy:
            self.recorder.annotate(result)
        return result


def backward_graph(loss, recorder):
    nodes, edges, handles, objects = [], [], [], {}
    pending = [loss.grad_fn]
    while pending:
        node = pending.pop()
        if node is None or node in objects:
            continue
        key = f"b{len(objects)}"
        objects[node] = key
        scope, method = recorder.grad_scopes.get(node, ("@autograd", "unattributed"))
        parameter = recorder.parameter_ids.get(id(getattr(node, "variable", None)))
        if parameter:
            scope = recorder.parameters[parameter]["scope"]
        record = dict(
            id=key,
            name=node.name(),
            scope=scope,
            method=method,
            parameter=parameter,
            executed=False,
            gradient_shapes=[],
        )
        nodes.append(record)

        def hook(grad_inputs, grad_outputs, record=record):
            record["executed"] = True
            record["gradient_shapes"] = [
                list(t.shape) for t in grad_outputs if t is not None
            ]

        handles.append(node.register_hook(hook))
        pending.extend(child for child, _ in node.next_functions if child is not None)
    for node, key in objects.items():
        for port, (child, output_port) in enumerate(node.next_functions):
            if child in objects:
                edges.append(
                    dict(
                        source=key,
                        target=objects[child],
                        input=port,
                        output=output_port,
                    )
                )
    return dict(nodes=nodes, edges=edges, root=objects[loss.grad_fn]), handles


def capture(model, execute, *, metadata=None):
    """Execute ``execute(copy_of_model) -> scalar loss`` without optimizer updates.

    The callback must use the supplied model, not a closed-over original. It must not
    mutate its input objects or perform IO. Backpropagation targets copied model
    parameters only, so closed-over input tensors do not accumulate gradients.
    """
    if sys.getprofile() is not None:
        raise RuntimeError(
            "Model inspection requires the current Python profiler to be stopped"
        )
    if any(p.device.type != "cpu" for p in model.parameters()):
        raise ValueError("Model explorer capture is CPU-only; construct a CPU model")
    started = time.perf_counter()
    random_state, numpy_state = random.getstate(), np.random.get_state()
    handles = []
    try:
        with torch.random.fork_rng(devices=[]):
            copied = copy.deepcopy(model)
            copied.zero_grad(set_to_none=True)
            recorder = Recorder(copied)
            try:
                sys.setprofile(recorder.profile)
                with AutogradAttribution(recorder), TensorOperations(recorder):
                    loss = execute(copied)
            finally:
                sys.setprofile(None)
            if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
                raise ValueError("Execution must return a scalar tensor loss")
            if not loss.requires_grad or loss.grad_fn is None:
                raise ValueError("Execution must return a differentiable loss")
            backward, handles = backward_graph(loss, recorder)
            parameters = tuple(p for p in copied.parameters() if p.requires_grad)
            if not parameters:
                raise ValueError(
                    "Model has no trainable parameters for backward capture"
                )
            loss.backward(inputs=parameters)
            for parameter in copied.parameters():
                name = recorder.parameter_ids[id(parameter)]
                record = recorder.parameters[name]
                if parameter.grad is not None:
                    grad = (
                        parameter.grad.to_dense()
                        if parameter.grad.is_sparse
                        else parameter.grad
                    )
                    record["gradient"] = numbers(grad)
                    record["gradient_stats"] = statistics(grad)
            result = dict(
                schema="pathwm-explorer-v1",
                modules=recorder.modules,
                parameters=recorder.parameters,
                buffers=recorder.buffers,
                operations=recorder.operations,
                tensors=recorder.tensors,
                backward=backward,
                metadata={
                    **dict(
                        title=type(model).__name__,
                        initial_scope="",
                        copied_model=True,
                        created=datetime.now(timezone.utc).isoformat(),
                        torch=torch.__version__,
                        loss=loss.detach().item(),
                        seconds=time.perf_counter() - started,
                        scope="One recorded CPU execution; unobserved branches are not inferred.",
                    ),
                    **(metadata or {}),
                },
            )
            return result
    finally:
        sys.setprofile(None)
        for handle in handles:
            handle.remove()
        random.setstate(random_state)
        np.random.set_state(numpy_state)


def write_explorer(data, path):
    """Write a self-contained snapshot with lossless compressed JSON, atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(
        data, ensure_ascii=True, allow_nan=False, separators=(",", ":")
    ).encode()
    payload = base64.b64encode(zlib.compress(raw, 6)).decode()
    template = Path(__file__).with_name("explorer.html").read_text()
    template = template.replace(
        "/*__SYSTEM_SCRIPT__*/",
        Path(__file__).with_name("explorer-system.js").read_text(),
    )
    html = template.replace("__SNAPSHOT__", payload)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(html)
    temporary.replace(path)
    return path
