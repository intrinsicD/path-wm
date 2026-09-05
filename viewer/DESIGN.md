# Experiment dashboard design

Recovered from `eca742a:viewer/DESIGN.md`, with current ledger and metric adapters.

## Brief

- Audience: the researcher running development and frozen experiments.
- Decision: inspect learning, prediction controls and measured control outcomes,
  including failures, interruptions, population differences and unassessed gates.
- Source of truth: raw JSON/JSONL and resolved manifests under `runs/`.
- Surface: one self-contained local HTML file, refreshed after every completed seed
  or standalone evaluation; no server, network requests or sibling files required.

## Metric hierarchy

Coverage cards establish what evidence was read. Control outcomes and selected-run
learning/prediction charts precede the complete inventory and exact values.
Training completion never implies useful control. Latent errors are interpreted
against controls within a checkpoint, with embedding scale available separately.
No missing threshold is converted into a pass. Stopped runs remain visible.

## Chart map

| Chart | Question | Encoding | Interpretation |
|---|---|---|---|
| Control success | What happened on these case identities? | Horizontal bars, result and success fraction | Higher is better; counts and protocol context accompany rates |
| Training components | Did the recorded objective move? | Step/value lines, component color | Context: raw components have different weights |
| Validation controls | Did predictions beat matched controls? | Step/MSE lines, condition color | Compare within a checkpoint; latent scale changes |
| Embedding scale | Did representation scale change? | Step/standard deviation line | Context, no inferred gate |
| Prediction checks | How does one saved evaluation compare to its controls? | Condition/MSE bars | Relative error within the selection |

## Interaction and accessibility

Independent selectors control case identities, training run, prediction check and
exact record details. Case identities group navigation only; they do not prove
identical normalization, weights, sampling or evaluation budgets. The inventory
and all-control exact table stay visible across selections. Color identifies
actual metric series, with visible legends; labels and exact tables carry meaning
without color. The canonical reader owns responsive desktop/narrow layouts.

## QA and constraints

Reconcile summary counts against case evidence. Preserve actual validation and
saved-checkpoint steps. Keep full raw ledgers; chart trajectories use at most 50
deterministically spaced points per run. Reject oversized exact datasets visibly
instead of silently dropping evidence. Auxiliary mode/clone diagnostics remain
in experiment reports/raw files and are explicitly listed as outside this view.

The recovered portable builder validates the canonical artifact, source metadata,
chart bindings, exact embedded payload, rendering and source interactions. Publish
HTML only through that builder; persist its receipt beside the HTML. Browser QA
must pass for full visual verification; structural-only results disclose the gap.
