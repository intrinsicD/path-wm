"""World-state inspection panels for the existing portable report renderer."""

from html import escape
import json


def world_state_inspection(directory):
    path = directory / "world_state.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    parts = [
        '<section id="world-state"><h2>Persistent world state</h2>',
        "<p>Stored claims and their sources. Prototype links are supplied/example averages; this view does not validate discovery or semantic interpretation.</p>",
        '<label>Find entity <input type="search" id="world-filter" placeholder="Label or ID" oninput="document.querySelectorAll(\'[data-world-entity]\').forEach(e=>e.hidden=!e.dataset.worldEntity.includes(this.value.toLowerCase()))"></label>',
    ]
    for entity in data["entities"]:
        entity_id = entity["id"]
        label = entity["label"] or entity_id
        record = dict(
            entity=entity,
            canonical_id=data["canonical_ids"][entity_id],
            components=[c for c in data["components"] if c["entity_id"] == entity_id],
        )
        parts.append(
            f'<details data-world-entity="{escape((label + " " + entity_id).lower(), quote=True)}"><summary>{escape(label)} · {escape(entity["kind"])}</summary><pre>{escape(json.dumps(record, indent=2))}</pre></details>'
        )
    for key in ("relations", "evidence", "events"):
        parts.append(
            f"<details><summary>{key.title()} ({len(data[key])})</summary><pre>{escape(json.dumps(data[key], indent=2))}</pre></details>"
        )
    parts.append("</section>")
    trace = directory / "world_trace.json"
    if trace.exists():
        content = json.loads(trace.read_text())
        parts.append(
            "<section><h2>Binding, retrieval and neural diagnostics</h2><p>Activation and gradient summaries are detached measurements. Attention is not a causal explanation. Exact tensor captures are optional and bounded.</p>"
        )
        parts.append(
            f"<p>Dropped records: {content['dropped']}; exact tensor values retained: {content['stored_tensor_values']}.</p>"
        )
        for key in ("records", "tensors", "arrays"):
            parts.append(
                f"<details><summary>{key.title()}</summary><pre>{escape(json.dumps(content[key], indent=2))}</pre></details>"
            )
        parts.append("</section>")
    return parts
