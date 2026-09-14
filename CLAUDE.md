# Working on PATH-WM

Read [workflow](docs/experiment-workflow.md), then
[current work](docs/project-state.md). Read the active plan linked there.

When discussing architecture with Alex, maintain the overview's discussion colors:
red = still to discuss, blue = discussed, green = validated within a labelled scope. Follow the
[discussion checklist](docs/architecture-discussion.md); update the matching
coverage/evidence in `docs/diagrams/architecture-atlas.json` and regenerate after
substantive discussion or validation. Green requires saved passing evidence, scope
and limits; remove or revise it when relevant changes invalidate that evidence.
An assistant-only diagram does not mark a topic discussed. Keep discussion coverage
separate from validation, including for green parts still awaiting a walkthrough.

The product is a small Python library plus readable experiment recipes that Alex
can operate himself. Preserve this boundary throughout implementation. A new
experiment normally edits a recipe; it does not create another trainer, CLI,
config hierarchy or reporting pipeline. Reusable code never imports recipes or
historical runs. See [models](docs/models.md) and [experiments](docs/experiments.md).

Every completed training/evaluation run owns raw metrics, a checkpoint where
applicable, and a verified standalone `report.html` inside its output directory.
Record result completion separately from report status. A broken report must stay
visible and be repaired before calling the workflow complete.

The user authorized a fresh start. Historical source is preserved by Git tag
`archive/pre-modular-2026-09-09`; do not restore retired machinery into the active
path. Preserve source data and completed runs. Update the current plan/state as
work progresses; keep these standing rules independent of the experiment target.
