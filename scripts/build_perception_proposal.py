"""Package the design document with the canonical portable-report renderer.

Run with the project Python, then pass the emitted JSON to the Data Analytics
deliver_portable_artifact.mjs script. This script does not import model code,
train, evaluate checkpoints, or rebuild the experiment dashboard.
"""
from __future__ import annotations

import base64
import hashlib
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import yaml


ROOT = Path(__file__).resolve().parents[1]
DOCUMENT = ROOT / "docs/perception-proposal-2026-09-08.md"
PROTOCOL = ROOT / "docs/proposals/perception-program-2026-09-08.yaml"
OUTPUT = ROOT / "runs/perception_proposal_2026-09-08"
INPUTS = [
    DOCUMENT, PROTOCOL,
    *[ROOT / "docs" / name for name in [
        "encoder-study-protocol-2026-09-08.md",
        "encoder-study-results-2026-09-08.md",
        "decoder-recovery-results-2026-09-08.md",
        "encoder-visual-audit-2026-09-08.md",
        "requirements-first-perception-design-2026-09-08.md",
        "decoder-inputs-and-conditioning-2026-09-08.md",
        "encoder-conditioning-2026-09-08.md",
        "claude-collaboration-workflow.md",
        "perception-data-readiness-2026-09-08.md",
    ]],
    ROOT / "world_model/paddle/models.py",
    ROOT / "world_model/curriculum/encoder_variants.py",
    ROOT / "runs/perception_data_cowork_2026-09-08/data_readiness.json",
    ROOT / "scripts/audit_perception_data.py",
    ROOT / "notebooks/perception-data-readiness-2026-09-08.ipynb",
]


def validate_proposal() -> dict:
    p = yaml.safe_load(PROTOCOL.read_text())
    assert p["status"] == "draft" and not p["runnable_trainer_config"]
    initial = p["p1"]
    assert initial["fits"] == len(initial["seeds"]) * len(initial["encoders"])
    assert initial["total_updates"] == initial["fits"] * initial["updates_per_fit"]
    assert initial["total_frame_presentations"] == initial["total_updates"] * initial["effective_batch"]
    assert sum(initial["domain_batch"].values()) == initial["effective_batch"]
    assert initial["microbatch"] * initial["accumulation_steps"] == initial["effective_batch"]
    budget = p["initial_execution_allowance_minutes"]
    assert budget["total"] == sum(budget[k] for k in [
        "P0_development", "feature_extraction_and_cache_validation", "P1_formal_fits", "final_evaluation_and_report"
    ])
    assert budget["P1_formal_fits"] == initial["fits"] * initial["fit_wall_cap_minutes"]
    for stage in p["conditional_stages"].values():
        assert stage["maximum_fits"] == len(stage["arms"]) * stage["seeds"]
        assert stage["total_fit_minutes"] == stage["maximum_fits"] * stage["per_fit_minutes"]
        assert stage["primary"] and stage["next_decision"] and stage["before_activation"]
    for name, target in re.findall(r"\[([^\]]+)\]\(([^)]+)\)", DOCUMENT.read_text()):
        if not target.startswith(("https://", "http://", "#")):
            assert (DOCUMENT.parent / target).exists(), (name, target)
    return p


def diagram(destination: Path) -> None:
    fig, ax = plt.subplots(figsize=(13.8, 6.2), dpi=170)
    fig.patch.set_facecolor("#ffffff")
    ax.set(xlim=(0, 14), ylim=(0, 6.3))
    ax.axis("off")
    blue, amber, ink = "#235b8b", "#a76512", "#27313b"

    def box(x, y, w, h, title, detail, color=blue):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.12",
                                   linewidth=1.3, edgecolor=color, facecolor="#f5f8fb"))
        ax.text(x + w/2, y + h*.69, title, ha="center", va="center", fontsize=11,
                weight="bold", color=color)
        ax.text(x + w/2, y + h*.30, detail, ha="center", va="center", fontsize=9.3,
                color=ink, linespacing=1.3)

    def arrow(start, end, color=blue, style="-", bend=0):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14,
                                    linewidth=1.6, color=color, linestyle=style,
                                    connectionstyle=f"arc3,rad={bend}"))

    ax.text(.1, 6.0, "One spatial representation, explicit time, output-specific decoders", fontsize=14,
            color=ink, weight="bold")
    box(.15, 3.8, 2.0, 1.0, "Observed image", "Available pixels at t")
    box(2.7, 3.8, 2.3, 1.0, "Encoder E", "CNN or native ViT")
    box(5.6, 3.8, 2.3, 1.0, "Spatial features zₜ", "Grids, coordinates, width")
    box(8.75, 3.8, 2.35, 1.0, "Output modules", "RGB · masks · geometry")
    box(11.7, 3.8, 2.05, 1.0, "Produced output", "Selected format and time")
    ax.text(9.92, 5.28, "Optional output / object query", ha="center", fontsize=9, color=amber)
    arrow((9.92, 5.15), (9.92, 4.9), amber, "--")
    arrow((2.23, 4.3), (2.6, 4.3))
    arrow((5.08, 4.3), (5.5, 4.3))
    arrow((7.98, 4.3), (8.65, 4.3))
    arrow((11.18, 4.3), (11.6, 4.3))

    box(2.7, 1.6, 2.3, 1.1, "Prior history", "mₜ₋₁, executed action, dt")
    box(5.6, 1.6, 2.3, 1.1, "State update U", "Current memory mₜ")
    box(8.75, 1.6, 2.35, 1.1, "Predictor P", "Candidate action + dt")
    box(11.7, 1.6, 2.05, 1.1, "Predicted state", "Future features +\naligned future memory")
    arrow((5.08, 2.15), (5.5, 2.15))
    arrow((6.75, 3.72), (6.75, 2.8))
    arrow((7.98, 2.15), (8.65, 2.15))
    arrow((7.95, 3.78), (9.55, 2.8), bend=.10)
    arrow((11.18, 2.15), (11.6, 2.15))
    ax.text(11.48, 1.2, "P then U", ha="center", fontsize=8.5, color=ink)
    arrow((12.75, 2.8), (11.15, 3.7), amber, "--", -.08)
    ax.text(12.45, 3.24, "Future decoding", fontsize=9, color=amber, ha="center")
    arrow((7.7, 2.78), (9.15, 3.7), amber, "--")
    ax.text(7.92, 3.24, "Optional\naligned memory", fontsize=8.5, color=amber, ha="center")
    ax.text(.15, .79, "Optional context may modulate E or a readout for a named consumer; compare against equally informed late processing.",
            fontsize=10, color=ink)
    ax.text(.15, .35, "Solid: observation/state flow. Dashed: declared decoder inputs. Future outputs receive no actual future observations.",
            fontsize=9.3, color=ink)
    fig.tight_layout(pad=.6)
    fig.savefig(destination, facecolor="white")
    plt.close(fig)


def portable_links(body: str) -> str:
    # The one-file report remains readable when copied away from the repository.
    # Exact repository identities remain in the source drawer; web links stay live.
    def replace(match):
        label, target = match.group(1), match.group(2)
        if target.startswith(("https://", "http://", "#")):
            return match.group(0)
        return f"{label} (`docs/{target}`)"
    return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", replace, body)


def main() -> None:
    p = validate_proposal()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    picture = OUTPUT / "architecture.png"
    diagram(picture)
    now = datetime.now(timezone.utc).isoformat()
    text = DOCUMENT.read_text()
    title = text.splitlines()[0].removeprefix("# ")
    encoded = base64.b64encode(picture.read_bytes()).decode()
    text = portable_links(text)
    diagram_body = f'<img alt="Architecture proposal: observed images become spatial features; temporal state and candidate actions produce predicted features, and typed decoders use inputs at the target time." src="data:image/png;base64,{encoded}" style="width:100%;height:auto">'
    sources = [{
        "id": "proposal", "label": "Design proposal, completed evidence and declared draft budgets",
        "path": str(DOCUMENT.relative_to(ROOT)),
        "query": {
            "description": "Manually synthesized design proposal, not new model measurements. Historical model values are quoted from the listed completed reports; current dataset counts/checks come from the read-only audit snapshot. Draft settings are labelled proposals. Python packages Markdown sections/tables without numerical aggregation and validates manifest budget arithmetic.",
            "language": "python", "executed_at": now,
            "input_files": [str(path.relative_to(ROOT)) for path in INPUTS],
            "input_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in INPUTS},
            "transformation": "scripts/build_perception_proposal.py: validate_proposal, diagram, portable_links, main",
            "filters": ["Historical populations and exploratory status preserved", "No model training or inference", "Local links shown as repository source identities for offline portability"],
            "metric_definitions": {
                "q": "Maximum of four coordinate MAEs / 8 world units and wrapped angle MAE / 10 degrees; lower is better, existing gate <= 1.",
                "p1_presentations": "2 encoder packages * 3 fresh readout seeds * 4000 updates * 64 frames = 1536000.",
                "initial_allowance": "30 + 45 + 120 + 45 = 240 minutes, proposed execution ceilings excluding engineering work.",
            },
        },
    }]
    allowance = p["initial_execution_allowance_minutes"]
    budget_rows = [
        {"phase": label, "minutes": allowance[key], "status": "Proposed ceiling",
         "scope": scope, "total_allowance_minutes": allowance["total"], "excludes": "Implementation/review effort"}
        for key, label, scope in [
            ("P0_development", "Development", "Correctness and tiny feasibility profiles"),
            ("feature_extraction_and_cache_validation", "Feature preparation", "Extraction and cache correctness"),
            ("P1_formal_fits", "Six formal fits", "6 packages/seeds at most 20 minutes each, with training validation/checkpoints"),
            ("final_evaluation_and_report", "Evaluation and report", "Selected/final readout evaluation and verified HTML"),
        ]
    ]
    budget_sql = """SELECT phase, minutes, status, scope, total_allowance_minutes, excludes
FROM proposed_execution_caps ORDER BY ordinal;"""
    with sqlite3.connect(":memory:") as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("CREATE TABLE proposed_execution_caps (ordinal INTEGER, phase TEXT, minutes INTEGER, status TEXT, scope TEXT, total_allowance_minutes INTEGER, excludes TEXT)")
        connection.executemany("INSERT INTO proposed_execution_caps VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(i, row["phase"], row["minutes"], row["status"], row["scope"], row["total_allowance_minutes"], row["excludes"])
             for i, row in enumerate(budget_rows)])
        budget_result = [dict(row) for row in connection.execute(budget_sql)]
    (OUTPUT / "budget.sql").write_text(budget_sql + "\n")
    sources.append({
        "id": "budget", "label": "Proposed execution allowance from the draft manifest",
        "path": str(PROTOCOL.relative_to(ROOT)),
        "query": {
            "engine": "sqlite", "sql": budget_sql, "executed_at": now,
            "description": "An in-memory proposed_execution_caps table is populated from initial_execution_allowance_minutes in the draft YAML. This executed SQL returns the proposed category caps and their scope; no experimental runtime measurements are included.",
            "input_files": [str(PROTOCOL.relative_to(ROOT))],
            "input_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
            "tables_used": ["proposed_execution_caps"],
            "transformation": "scripts/build_perception_proposal.py constructs rows from the four named allowance fields, then executes the preserved SQL in SQLite.",
            "metric_definitions": {"minutes": "Proposed category ceiling in minutes; total 240; excludes implementation/review effort."},
        },
    })
    datasets = {"initial_budget": budget_result}
    budget_chart = {
        "id": "initial_budget", "dataset": "initial_budget", "sourceId": "budget",
        "title": "Formal fitting accounts for half the proposed four-hour allowance",
        "subtitle": "P0 + P1 proposed caps in minutes; engineering effort excluded. These are not measured durations.",
        "type": "bar", "intent": "comparison", "layout": "full",
        "question": "How much of the initial execution allowance is reserved for training versus preparation and verification?",
        "rationale": "Four independent category caps are compared directly with a zero-baseline bar chart; no visual uncertainty is implied for proposed limits.",
        "comparisonContext": {"grain": "One proposed execution category", "unit": "minutes", "denominator": "240 minutes total proposed allowance"},
        "encodings": {"x": {"field": "phase", "type": "nominal"}, "y": {"field": "minutes", "type": "quantitative", "unit": "minutes"}},
        "xAxisTitle": "Execution category", "yAxisTitle": "Proposed ceiling (minutes)",
        "labels": {"values": "all"}, "palette": {"kind": "identity", "name": "PATH-WM blue"},
        "settings": {"sort": "none", "categoryLabelPolicy": "wrap", "showValues": True},
        "surface": {"interactiveLegend": False, "viewMode": "both"},
    }
    blocks, tables = [], []
    sections = re.split(r"(?=^## )", text, flags=re.M)
    for i, section in enumerate(sections):
        # Quantitative and specification tables use the canonical table component.
        # The bibliography stays Markdown so its primary-source links remain live.
        pattern = r"(^\|[^\n]+\n\|[- :|]+\n(?:\|[^\n]+\n)+|<!-- budget-chart -->|<!-- architecture-diagram -->)"
        parts = [section] if section.startswith("## 12.") else re.split(pattern, section, flags=re.M)
        for j, part in enumerate(parts):
            if not part.strip():
                continue
            ident = f"section_{i}_{j}"
            if part == "<!-- architecture-diagram -->":
                blocks.append({"id": "architecture_figure", "type": "html", "body": diagram_body, "layout": "full"})
            elif part == "<!-- budget-chart -->":
                blocks.append({"id": "budget_chart", "type": "chart", "chartId": "initial_budget", "layout": "full"})
            elif part.startswith("|") and "\n|---" in part:
                rows = [[cell.strip() for cell in line.strip().strip("|").split("|")] for line in part.strip().splitlines()]
                fields = [f"column_{k}" for k in range(len(rows[0]))]
                assert all(len(row) == len(fields) for row in rows), ident
                table_name = f"proposal_{ident}"
                lookup_sql = f"SELECT {', '.join(fields)} FROM {table_name} ORDER BY ordinal;"
                with sqlite3.connect(":memory:") as connection:
                    connection.row_factory = sqlite3.Row
                    columns = ", ".join(f"{field} TEXT" for field in fields)
                    connection.execute(f"CREATE TABLE {table_name} (ordinal INTEGER, {columns})")
                    placeholders = ", ".join("?" for _ in range(len(fields) + 1))
                    connection.executemany(f"INSERT INTO {table_name} VALUES ({placeholders})",
                                           [(index, *row) for index, row in enumerate(rows[2:])])
                    datasets[ident] = [dict(row) for row in connection.execute(lookup_sql)]
                section_title = section.splitlines()[0].removeprefix("## ")
                lookup_source_id = f"source_{ident}"
                sources.append({
                    "id": lookup_source_id, "label": f"Proposal lookup: {section_title}",
                    "path": str(DOCUMENT.relative_to(ROOT)),
                    "query": {
                        "engine": "sqlite", "sql": lookup_sql, "executed_at": now,
                        "description": "Markdown table rows from this proposal section are inserted verbatim into an in-memory SQLite table, then selected in document order. This is a lookup of the authored proposal/evidence summary, not an independent analysis of experimental records. Historical claims retain the cited completed reports; proposed settings are labelled as such.",
                        "input_files": [str(path.relative_to(ROOT)) for path in INPUTS],
                        "input_sha256": hashlib.sha256(DOCUMENT.read_bytes()).hexdigest(),
                        "tables_used": [table_name],
                        "column_definitions": dict(zip(fields, rows[0])),
                        "transformation": "scripts/build_perception_proposal.py parses this Markdown table, preserves its cell strings, inserts rows with ordinal positions and executes the recorded SQL.",
                    },
                })
                tables.append({
                    "id": ident, "dataset": ident, "sourceId": lookup_source_id, "title": section_title,
                    "subtitle": "Proposal settings unless explicitly labelled completed evidence; see the adjacent interpretation and source details.",
                    "layout": "full", "density": "spacious",
                    "columns": [{"field": field, "label": label, "type": "text"} for field, label in zip(fields, rows[0])],
                })
                blocks.append({"id": ident, "type": "table", "tableId": ident, "layout": "full"})
            else:
                blocks.append({"id": ident, "type": "markdown", "body": part.strip(), "sourceId": "proposal"})
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1, "surface": "report", "title": title,
            "description": "A complete, staged architecture and experiment proposal: first compare frozen perception packages, then diagnose and act on the measured limitation.",
            "generatedAt": now, "blocks": blocks, "cards": [], "charts": [budget_chart], "tables": tables,
            "filters": [], "sources": sources,
        },
        "snapshot": {"version": 1, "generatedAt": now, "status": "ready", "datasets": datasets},
        "sources": sources,
    }
    destination = OUTPUT / "proposal.artifact.json"
    destination.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps({"artifact": str(destination), "architecture": str(picture),
                      "blocks": len(blocks), "tables": len(tables), "budget_minutes": p["initial_execution_allowance_minutes"]["total"],
                      "validation": "proposal arithmetic, local links and source files passed"}, indent=2))


if __name__ == "__main__":
    main()
