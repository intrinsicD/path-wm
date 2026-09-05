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
