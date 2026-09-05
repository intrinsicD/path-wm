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

## Browser repair (2026-09-05)

`chromium_transport.mjs` runs the canonical browser probes unchanged over a CDP
pipe. The former virtual-time dump-DOM path could exhaust its budget before
animation-frame startup; full Chrome also included window decorations in its
requested size. The adapter uses real time and explicit viewport/media settings,
with bounded deadlines, disposable profiles and preserved negative probe results.
`PATH_WM_CHROMIUM` selects the actual browser; `CHROMIUM_EXECUTABLE_PATH` can
still override the entire transport. No browser is downloaded.

`deliver_dashboard.mjs` calls the installed canonical delivery/build APIs and
supplies the same packaged reader with `portable_layout.css`. This corrects the
header's 100vw scrollbar overflow and constrains long metric legends to wrap on
narrow screens. It does not hide document overflow or alter verification checks.
The HTML contains these fixes; they are not browser-only test overrides.
Canonical verification now passes at 1440x1000 and 390x844, including source
interaction. Run explicit transport checks with `PATH_WM_BROWSER_TESTS=1`.
