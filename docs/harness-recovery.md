# Harness recovery

Source: `eca742a` (the last commit before reset `6a1b377`). This restores the
reusable harness requested by the user, while preserving the current LeWM code,
source datasets, checkpoints and completed results.

- The old `CLAUDE.md` sections on vertical slices, code discipline, essential
  tests, commits and the mandatory HTML panel now live in
  [experiment-workflow.md](experiment-workflow.md).
- Root `CLAUDE.md` is a stable entry point. The former mixed objective/status
  material now lives in [project-state.md](project-state.md).
- The dashboard's validation and portable HTML packaging are recovered from
  `viewer/dashboard.py`; its ledger adapter is updated for current run formats.
  The old `run.py` completion/refresh contract is retained in a small wrapper.
- Old E0/E1 targets, ABI/head designs, custom losses and obsolete module paths are
  historical implementation choices, not standing process requirements.

## Repair slice

Problem: resetting implementation code also removed the independent experiment
workflow and mandatory visual instrument panel.

Success: standing instructions are separate from live goals; an experiment
wrapper refreshes the panel and exposes reporting failures; current raw ledgers
produce a verified offline HTML dashboard without retraining or modifying logs.

Modules: instruction documents, `run.py`, `viewer/ledger.py`,
`viewer/dashboard.py`, focused ledger/runner integrity tests. Verification budget:
CPU tests and HTML packaging only; no training or evaluation runs.


## Delivered and checked

- Recovered standing workflow, stable `CLAUDE.md` entry point and separate current
  state. The previously approved diagnostics/reproduction preparation remain
  recorded as authorized and pending.
- Restored `run.py` refresh/failure contract, current-ledger reconciliation,
  source-backed SQLite views and canonical portable HTML packaging. Reporting
  modules import no model, trainer or dataset implementation.
- All 22 CPU tests pass, including nine harness integrity checks. No training or
  evaluation was launched; source data, model code and saved run ledgers remain
  unchanged.
- `runs/experiment_dashboard.html` and its canonical artifact cover 30 records:
  6 training runs, 20 separate control results and 4 prediction checks. The panel
  includes five charts, exact tables, case/run selectors, sources and failures.
- Artifact validation and exact HTML payload/structure verification pass. The
  receipt is `runs/experiment_dashboard.receipt.json`.

### Browser QA limitation

The installed Data Analytics `0.2.10-13ceeea1f599` delivery script does not discover
this machine's Puppeteer browser cache by default. Explicitly using its installed
headless-shell reaches a `reader_timeout` in fallback, including with a larger
startup budget. Installed full Chrome has a viewport mismatch; a temporary
measured correction lets chart extraction pass but the final desktop/mobile
probe still times out. No browser or plugin was installed or modified, and no
browser success is claimed. The published receipt is `structural_only`; the
self-contained HTML is available for visual inspection, with browser QA pending.

The wrapper reports a nonzero workflow result when visual QA has not passed,
even if raw execution succeeded. Use a compatible canonical packager/browser to
resolve this reporting-tool limitation; do not change scientific results or
relax verification to make it green. `CHROMIUM_EXECUTABLE_PATH` (or the builder's
legacy `PLAYWRIGHT_EXECUTABLE_PATH`) selects an explicit browser when needed.
