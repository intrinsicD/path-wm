"""Plain reproducible diagrams: module containment and values between actual calls.

No inferred tensor edges, hooks, model mutation or browser dependency. CallFlow
records the boundaries the recipe executes. Operations inside each call are opaque;
closures, implicit module parameters and unrecorded tensor transformations are not
claimed as data-flow edges. It is not an autograd or all-possible-paths graph.
"""

from dataclasses import fields, is_dataclass
from html import escape
import inspect
import json
from pathlib import Path
import shutil
import subprocess

import torch
from torch import nn


def shape(value):
    if isinstance(value, torch.Tensor):
        return str(list(value.shape))
    if is_dataclass(value):
        primary = getattr(value, "tokens", getattr(value, "values", None))
        return type(value).__name__ + (
            " " + shape(primary) if primary is not None else ""
        )
    if isinstance(value, dict):
        return ", ".join(f"{key}: {shape(item)}" for key, item in value.items())
    if isinstance(value, (tuple, list)):
        return ", ".join(shape(item) for item in value)
    return type(value).__name__


def children(value):
    if isinstance(value, dict):
        return value.values()
    if isinstance(value, (list, tuple)):
        return value
    if is_dataclass(value):
        return [getattr(value, field.name) for field in fields(value)]
    return ()


def trackable(value):
    return isinstance(value, (torch.Tensor, dict, list, tuple)) or is_dataclass(value)


def source_location(callable):
    try:
        filename = inspect.getsourcefile(callable)
        line = inspect.getsourcelines(callable)[1]
    except (TypeError, OSError):
        return None
    root = Path(__file__).resolve().parents[2]
    path = Path(filename).resolve()
    try:
        path = path.relative_to(root)
    except ValueError:
        # Installed-library absolute paths would make exports machine-dependent.
        return None
    return {"path": str(path), "line": line}


def architecture(model, depth=2):
    """Edges mean module containment, never guessed runtime data flow."""
    if depth < 0:
        raise ValueError("Diagram depth must be nonnegative")
    nodes, edges = [], []

    def visit(module, path, level, parent=None):
        identifier = f"n{len(nodes)}"
        count = sum(p.numel() for p in module.parameters())
        direct = {
            name: list(p.shape) for name, p in module.named_parameters(recurse=False)
        }
        label = (path.rsplit(".", 1)[-1] + ": " if path else "") + type(module).__name__
        detail = f"{count:,} parameters"
        if direct and not path:
            detail += "\n" + ", ".join(
                f"{name} {size}" for name, size in direct.items()
            )
        nodes.append(
            dict(
                id=identifier,
                label=label,
                detail=detail,
                kind="module",
                path=path,
                parameters=count,
                direct_parameters=direct,
                source=source_location(type(module)),
            )
        )
        if parent is not None:
            edges.append(dict(source=parent, target=identifier, kind="contains"))
        if level < depth:
            for name, child in module.named_children():
                visit(child, f"{path}.{name}" if path else name, level + 1, identifier)

    visit(model, "", 0)
    return dict(
        title="Model architecture",
        relationship="Arrows mean contains; counts include descendants.",
        direction="LR",
        nodes=nodes,
        edges=edges,
    )


class CallFlow:
    """Record object provenance at explicitly executed function boundaries.

    References are retained until the recorder is released to avoid Python ID reuse.
    Shared inputs form forks; invocation order alone never creates an edge. If a
    tensor is transformed outside recorded calls, its provenance is unrecorded:
    register it as an input, or record that transformation as a call as well.
    """

    def __init__(self):
        self.nodes, self.edges, self.producers, self.references = [], [], {}, []

    def _register(self, value, producer):
        if trackable(value):
            self.producers[id(value)] = producer
            self.references.append(value)
            for child in children(value):
                self._register(child, producer)

    def _dependencies(self, value):
        if trackable(value) and id(value) in self.producers:
            return {self.producers[id(value)]}
        return set().union(*(self._dependencies(child) for child in children(value)))

    def _node(self, label, detail, kind, dependencies=(), source=None):
        identifier = f"n{len(self.nodes)}"
        self.nodes.append(
            dict(id=identifier, label=label, detail=detail, kind=kind, source=source)
        )
        for parent in sorted(dependencies, key=lambda name: int(name[1:])):
            self.edges.append(dict(source=parent, target=identifier, kind="value"))
        return identifier

    def input(self, label, value):
        identifier = self._node(label, shape(value), "input")
        self._register(value, identifier)
        return value

    def call(self, function, *args, **kwargs):
        dependencies = self._dependencies((args, kwargs))
        output = function(*args, **kwargs)
        label = (
            type(function).__name__
            if isinstance(function, nn.Module)
            else function.__name__
        )
        source = source_location(
            type(function) if isinstance(function, nn.Module) else function
        )
        identifier = self._node(label, shape(output), "call", dependencies, source)
        self._register(output, identifier)
        return output

    def output(self, label, value):
        self._node(label, shape(value), "output", self._dependencies(value))
        return value

    def graph(self):
        return dict(
            title="Example data flow",
            relationship="Arrows mean values passed between recorded calls; call internals are omitted.",
            direction="TB",
            nodes=list(self.nodes),
            edges=list(self.edges),
        )


def mermaid(graph):
    lines = ["flowchart " + graph["direction"]]
    for node in graph["nodes"]:
        label = escape(node["label"] + "\n" + node["detail"], quote=True).replace(
            "\n", "<br/>"
        )
        lines.append(f'    {node["id"]}["{label}"]')
    for edge in graph["edges"]:
        arrow = "-.->" if edge["kind"] == "contains" else "-->"
        lines.append(f"    {edge['source']} {arrow} {edge['target']}")
    return "\n".join(lines) + "\n"


def dot(graph):
    lines = [
        "digraph model {",
        f'graph [rankdir={graph["direction"]}, bgcolor="white", pad=0.35, nodesep=0.25, ranksep=0.65, fontname="DejaVu Sans", fontsize=17, labelloc=t];',
        'node [shape=box, style="rounded,filled", fillcolor="#edf3fa", color="#7290ae", fontname="DejaVu Sans", fontsize=12, margin="0.15,0.10"];',
        'edge [color="#6b7d8e", arrowsize=0.65];',
        "label=" + json.dumps(graph["title"] + "\n" + graph["relationship"]) + ";",
    ]
    colors = dict(input="#e6f3ec", output="#fff0d9", call="#edf3fa", module="#edf3fa")
    for node in graph["nodes"]:
        label = node["label"] + "\n" + node["detail"]
        lines.append(
            f'{node["id"]} [label={json.dumps(label)}, fillcolor="{colors[node["kind"]]}"];'
        )
    for edge in graph["edges"]:
        style = "dashed" if edge["kind"] == "contains" else "solid"
        lines.append(f"{edge['source']} -> {edge['target']} [style={style}];")
    return "\n".join(lines + ["}"]) + "\n"


def write_diagrams(directory, graphs, provenance, render=True):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    executable = shutil.which("dot") if render else None
    for name, graph in graphs.items():
        if not name.isidentifier():
            raise ValueError("Diagram names must be simple identifiers")
        (directory / f"{name}.mmd").write_text(mermaid(graph))
        (directory / f"{name}.dot").write_text(dot(graph))
        if executable:
            for extension in ("svg", "png"):
                subprocess.run(
                    [
                        executable,
                        f"-T{extension}",
                        str(directory / f"{name}.dot"),
                        "-o",
                        str(directory / f"{name}.{extension}"),
                    ],
                    check=True,
                )
    record = dict(
        schema="pathwm-diagrams-v1",
        provenance=provenance,
        graphs=graphs,
        rendered=bool(executable),
        graphviz=(
            subprocess.run(
                [executable, "-V"], capture_output=True, text=True, check=True
            ).stderr.strip()
            if executable
            else None
        ),
    )
    (directory / "diagrams.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )
    return directory
