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
| Per-scalar training panels | How did each logged scalar move? | One step/value line per scalar (prediction loss, SIGReg, total, gradient norm, learning rate ×10⁻⁶) | Components have different scales; read each on its own axis |
| Validation ratios (log10) | Does prediction beat matched controls, scale-free? | Step/log10(prediction ÷ control) lines, dashed 0 line | 0 equals the control; negative is better; copy error is near zero at initialization |
| Checkpoint internals | Is the representation collapsed, Gaussian, linearly readable; does the predictor use actions? | Step/value lines per family over inspected checkpoints of the focus run | Descriptive; reference weights appear in tables, spectrum, horizon curves and panels |
| Covariance spectrum | Isotropic or collapsed? | Component/log10 eigenvalue lines, one per inspected checkpoint | Flat is isotropic; a cliff is dimensional collapse; levels follow latent scale |
| Error versus horizon | How fast does rollout error grow? | Horizon/log10(error ÷ copy baseline) lines per checkpoint, dashed 0 line | Below 0 beats copying; scale-free within each checkpoint |
| Internals panels | What does the encoder attend to; where do predicted latents land? | Embedded PNG small multiples per checkpoint (attention, patch PCA, Q–Q, predictor attention, nearest neighbours) | Real inputs and measured internals; LeWM has no decoder |

## Interaction and accessibility

The canonical reader applies a manifest filter only when it targets every dataset
used by cards, charts and tables; per-section selectors are silently ignored and
every chart then mixes all runs on a categorical axis (found 2026-09-06). The
dashboard therefore emits no filters. Each chart shows one named selection fixed
at build time (`--focus <training run>`, default: most recently modified training
run; latest control case set; latest prediction check; first ranking model/case
with its own plan) and states it in the title. Tables hold every record with a
record column for sorting. Case identities group navigation only; they do not
prove identical normalization, weights, sampling or evaluation budgets. Color identifies
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

## Simulator-grounded ranking view

A selected model/case scatter compares 20 predicted terminal costs with measured
terminal position errors; candidate family identifies replay, stationary, random
and each model's plan. Costs are compared within a single latent space and case.
Exact rows retain candidate identity, successes and angle/state diagnostics.
Separate five-step curves compare prediction with copy MSE, and predicted with
measured latent goal costs, for a selected nonrandom candidate. Five time points
are the model's complete five-block horizon, not a sparsely sampled longer curve.
The canonical palette and labeled metric/candidate legends supply distinctions.
Reconcile declared record counts, complete per-model candidate sets, matched
simulator outcomes, and selected-candidate summary values before rendering.

## Bounded checkpoint image panels (2026-09-06)

Embedding every historical PNG exceeded the canonical 3 MB artifact limit after
the first source-run inspection. The panel section now shows the focus run's
earliest/latest inspected checkpoints and at most one released inspection whose
`source_run_manifest` matches. This makes the image comparison population-aware
and bounded to at most three inspections. The dashboard states this selection;
all scalar records, covariance spectra, horizon curves and raw panel source paths
remain indexed. No exact numeric evidence is discarded or size limit bypassed.
