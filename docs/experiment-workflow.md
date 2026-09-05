# Experiment and development workflow

These standing instructions apply across implementation goals and experiments.
They restore the reusable process from `eca742a:CLAUDE.md` (before reset `6a1b377`).
Current architecture, datasets, hypotheses, target runs and results belong in
[project-state.md](project-state.md), experiment configs and protocol reports.

## Work in four-step vertical slices

Build the thinnest end-to-end path from data/environment to a measured result on
the instrument panel. Follow these steps in order for each substantive slice:

1. **Plan and interfaces.** Write a short plan in the development configuration or
   task document: hypothesis/problem, experiment, affected modules/interfaces,
   success measure and budget. Define interface and configuration changes first.
   Choose concrete interfaces appropriate to the active implementation; this
   workflow does not prescribe an architecture or a root contracts module.
2. **Essential tests.** Add the few structural or behavioral checks that would
   catch silent scientific failures. For new behavior, verify an informative
   failure before implementation. Do not add tests for trivial prose, formatting
   or glue, or tests that merely restate the implementation. Commit the completed
   plan/test step and clearly report any intentional red state.
3. **Minimal working slice.** Implement the simplest complete path with tiny data,
   a small explicit development configuration and one seed. Keep fast tests green
   and produce the measured result and verified HTML. Small development runs are
   development evidence, not frozen experimental results. Commit this step.
4. **Iterate against the reference.** Change one component per iteration with
   matched data, evaluation cases and compute budgets. Preserve the reference.
   Expand only after the thin path works; commit each completed iteration.

A frozen experiment is a separate decision: predeclare the hypothesis, protocol,
seeds, budgets, metrics and success thresholds, then run the required seeds.
Record code/data/configuration hashes and artifacts with the protocol. Amend a
frozen protocol explicitly; do not choose thresholds retroactively. An unset
threshold remains unset and cannot produce a passing gate.

## Implementation discipline

Use plain Python/PyTorch modules, functions, dataclasses and explicit arguments.
Design replaceable components without speculative frameworks, registries, base
classes or deferred research modules. Introduce an abstraction after a second
real implementation requires it. Keep files focused; explain what/how/why in
critical modules and annotate non-obvious scientific invariants.

Reuse existing project and pinned upstream components before inventing new ones.
Copied or adapted upstream code carries its source URL, commit, file and license.
Keep scientific choices configurable and preserve immutable reference configs.
Version changes to frozen interfaces, metric definitions and evaluation budgets.

## Evidence and essential verification

Keep source datasets under `data/` and derived checkpoints/logs under `runs/`.
Record code and data revisions, seed, resolved configuration, split/population,
normalization, sample/step counts, precision, checkpoint identity and compute
budget. Keep source, training and held-out evaluation populations explicit.
Never overwrite a completed reference run or silently change checkpoint buffers.

Raw JSON/YAML/JSONL ledgers are authoritative. Distinguish training completion,
prediction quality and closed-loop behavior; a falling loss is not evidence of a
useful world model. Report negative results, failed gates, stopped/incomplete
runs, missing evidence and protocol deviations. Compare learned latent errors
against matched controls; their scale can change with the learned encoder.

Ordinary tests are fast and CPU-based. Test silent scientific failures such as
alignment, causality, reference calculations, normalization, rollout and artifact
integrity when applicable. Mark substantive GPU runs, slow checks and threshold
evaluations explicitly. Never delete, skip or xfail a failing test to conceal an
implementation error. Experiments are evidence, not unit tests.

## Mandatory offline HTML instrument panel

After **every completed seed or standalone evaluation**, refresh and verify
`runs/experiment_dashboard.html`, with its canonical data companion
`runs/experiment_dashboard.artifact.json`. This is part of experiment completion,
not an optional final report. Include learning curves, prediction/control results
where recorded, exact values, run/protocol context, sources and visible failures.
Inspect representative input data and qualitative rollouts when the experiment's
scientific question requires them; add the resulting evidence to its report.

Use the shared wrapper for a Python module or script:

```bash
python run.py -m <experiment.module> <arguments...>
python run.py <experiment-script.py> <arguments...>
```

It runs the supplied command and then refreshes the dashboard, including after a
failed run. Multi-seed drivers must invoke it per seed or call the same dashboard
refresh after each seed. Direct low-level module entry points still require this
refresh before the agent reports completion. Backfill existing ledgers with:

```bash
python -m viewer.dashboard
```

The reader adapts current ledgers without rewriting them. Reconcile duplicated
counts against case records; show stopped runs and preserve recorded validation
steps rather than attaching old metrics to a newer checkpoint. Keep exact source
paths and metric definitions. Do not average incompatible protocols or imply
that different learned latent spaces have comparable absolute errors.

The canonical portable-artifact builder, called through the local browser/layout
adapter documented in `viewer/DESIGN.md`, produces self-contained local HTML and
validates its data and rendering. Node.js and the Data Analytics builder are
required; `PATH_WM_ARTIFACT_BUILDER` can select its installed delivery script.
A `passed` verification receipt includes browser checks. `structural_only` means
payload/structure checks passed without browser QA: disclose that limitation and
leave visual verification pending. Record the receipt beside the HTML.

Report dashboard failures visibly even when raw evaluation succeeded; preserve
raw output and repair reporting. Never silently claim a stale dashboard is fresh.
No external experiment tracking, uploads or publishing unless requested.

## Session closeout

State the active slice, hypothesis/problem and experiment at the start. At the
end, report what ran, passed, failed or was skipped, and link the refreshed HTML.
Update [project-state.md](project-state.md) and the relevant protocol/report with
measured outcomes, decisions, limitations and next steps. Commit completed steps.
Keep standing workflow changes separate from changing research goals or targets.
