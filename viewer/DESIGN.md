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
instead of silently dropping evidence. Usable eval-mode diagnostics provide native prediction records and explicit
calibration manifests. Batch-coupled current-statistic probes remain separately
labeled in experiment reports/raw files; they are not single-state controllers.

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
recorded sampling and preprocessing identity matches (legacy records: same source manifest). This makes the image comparison population-aware
and bounded to at most three inspections. The dashboard states this selection;
all scalar records, covariance spectra, horizon curves and raw panel source paths
remain indexed. No exact numeric evidence is discarded or size limit bypassed.


## Goal identity and initial-state controls (2026-09-06)

Case navigation includes dataset name/path/revision and goal offset as well as
source row/episode identities. Equal integer rows in different datasets or with
different target offsets cannot merge. Action baselines inherit the case
manifest's source and budget context. Raw success remains visible; a separate
chart and exact columns report success only among cases whose initial state was
outside the goal threshold. Missing initial-state evidence remains missing, and
zero eligible cases have no conditional rate. Counts are derived from the case
records and reconciled with any declared summaries.

Oversized native datasets are partitioned into explicitly numbered chart/table
parts of at most 2,000 rows. Exact rows are conserved, and complete color series
are kept together. A single oversized series still fails with an actionable
error; no silent truncation or relaxed portable payload/browser limit is used.


## Mobile control labels and browser readiness (2026-09-06)

Horizontal chart labels use stable chart keys plus compact model/step/solver text.
The canonical reader sizes its categorical axis automatically; full source paths
previously consumed the entire390px plot area. Hover and exact rows retain the
complete run/protocol identity, with the same chart key for cross-reference.
An actual-browser regression requires a nonzero bar at least80px wide wholly
within390px. Numeric rows and canonical rendering/verification remain intact.
Screenshot readiness tolerates a temporarily absent documentElement immediately
after navigation, while preserving the readiness deadline and failure checks.


## Reference panels across training forks (2026-09-06)

Complete inspection manifests carry a derived population hash over dataset/split,
exact validation windows, probe/rollout identities and counts, history, image
size, seeds, precision and recorded panel/measurement settings. A fork with the
same evidence can reuse the matching released panels even though its training
manifest path changed. Different or partially missing identities cannot match a
fully recorded inspection. Legacy pairs without complete identity retain the
original same-manifest rule. The image bound and exact numeric inventory remain
unchanged; one new regression covers matching and exclusion together.

## Bounded provenance and exact-record identities (2026-09-07)

The static fallback copied the entire global file list into each numeric-cell
source tooltip, eventually exceeding the extractor's 16 MB ceiling. Source
`tables_used` now names the actual `json_each(:reconciled_runs)` SQL input. The
complete raw-file union remains in `query.input_files`, while the inventory
retains each full run identity and source paths. Metric/configuration rows use a
stable short record key joined to that inventory; collisions fail explicitly.
Every exact value remains present. Renderer probes and output limits are unchanged.

The actual 22-outcome dashboard and an isolated, explicitly synthetic full-size
fixture pass canonical browser/source checks at 1440/390 pixels. The fixture is
under the experiment's `qa/final_capacity`, outside the scientific ledger. Its
payload is about 2.54 MB against the unchanged 3 MB cap. Recovery skips completed
raw evaluation and rebuilds reporting. A coordinator stage is marked verified
only if its current reporting subprocess succeeds and its receipt passes; a stale
receipt cannot validate a failed build.
