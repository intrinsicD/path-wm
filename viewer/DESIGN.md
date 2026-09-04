# Experiment dashboard design

## Raw-data microscope

`python -m viewer.data <spec.yaml>` writes a separate, self-contained `runs/data_viewer.html` so the
pixels and actions consumed by a selected spec remain visually distinct from the metrics derived from
them. It presents four populations in pipeline order:

1. stored episode trajectories sampled for ordinary E1 training;
2. stored same-snapshot intervention branches, only when either counterfactual training weight is active;
3. fixed transition trajectories regenerated for action-sensitivity and transition-error evaluation;
4. held-out same-snapshot branches regenerated for counterfactual evaluation.

Each tab reports the full tensor shape, population, seed, role, and exact source. The HTML embeds only a
bounded set of evenly spaced examples and says so prominently; training/evaluation continue to use every
validated sample. The builder reads stores without changing them, fails on any selected-spec mismatch,
and makes no network requests. Raw RGB and actions are shown directly—ground-truth labels and learned
latents are deliberately absent (Invariant 11). Until E0 is frozen, `data/e0_dev` and regenerated dev
probes are the explicit stand-ins and the page remains development inspection, not experiment evidence.

## Brief

- **Audience:** the researcher running PATH-WM development specs and frozen experiments.
- **Decision:** understand whether an intervention improves the intended signal without hiding action-correctness controls, training instability, run status, or missing thresholds.
- **Source of truth:** completed directories under `runs/<spec>/<seed>/`; `metrics.json` controls final evaluation values and is reconciled against `run_summary.json` before rendering.
- **Surface:** one read-only, self-contained `runs/experiment_dashboard.html`, refreshed after each completed seed.

## Metric hierarchy

1. Scope cards establish how many runs, variants, evaluation fields, and training snapshots are present.
2. The selected-run cards show the highest-signal evaluation outcomes available in the ledger.
3. Cross-run comparisons retain all runs while the selector changes only run-level controls and training views.
4. Exact result and training tables remain below the charts for lookup and audit.

Every chart title carries a non-color direction indicator: `↑ higher is better`, `↓ lower is better`, or
`↔ context only`. The action-control comparison is explicitly mixed-direction: correct-action error should be
lower, while identity, zero-action, and shuffled-action controls should remain higher relative to it.

## Chart map

| Section | Question | Family / type | Fields | Palette policy |
|---|---|---|---|---|
| Action sensitivity | Did action separation change across runs? | Comparison / bar | run, ratio | single-root identity |
| Counterfactual evaluation | Is discrimination above its own 1/K baseline? | Comparison / grouped bar | run, accuracy, observed/chance | hard two-root categorical |
| Transition error | How did one-step and endpoint error move? | Comparison / grouped bar | run, error, horizon | hard two-root categorical |
| Correctness controls | Does the selected run beat non-semantic controls? | Comparison / bar | control, error | single-root identity |
| Primary objectives | How did total/action/rollout/inverse behave? | Trend / line | step, value, objective | four-root categorical; interactive legend |
| Auxiliary objectives | How did counterfactual/regularization/chunk terms behave? | Trend / line | step, value, objective | three-root categorical; interactive legend |
| Counterfactual training | Did minibatch discrimination move beyond chance? | Trend / line | step, accuracy, observed/chance | hard two-root categorical |
| Gradient norm | Where was optimization unstable? | Trend / line | step, gradient norm | single-root identity |
| Curriculum | Which horizon and delta-t were active? | Trend / line | step, value, signal | hard two-root categorical |
| Parameters | Did an intervention alter model capacity? | Comparison / grouped bar | component, parameters, run | categorical identity |

Unknown future evaluation metrics receive an honest, single-measure bar comparison rather than being mixed onto an incompatible scale. Context-only counts and horizons stay in the cards/tables/tooltips.

## QA and constraints

- The canonical artifact builder validates sources, chart bindings, bounded datasets, and the desktop/narrow reader before publishing HTML.
- Training series are deterministically downsampled only when needed to stay within the 2,000-row artifact limit; raw `training.jsonl` remains complete.
- Latent ratios and errors are plotted in visibly labeled ×10⁻³ units so compact formatting cannot round meaningful small values to zero; raw decimal values remain in tooltips and the exact-results table.
- Development results are labeled `DEV`; no pass/fail state is inferred while preregistered thresholds are null.
- The page makes no network requests and requires no local server or sidecar files.
