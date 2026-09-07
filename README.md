# PATH-WM

The active experiment is the fixed RGB paddle E/U/P world model. Its
[commands and architecture](docs/paddle-world-model-usage.md) and
[implementation plan](docs/paddle-world-model-plan.md) describe the separate
`world_model.paddle` package. Run `.venv/bin/python -m world_model --help`.
The LeWM implementation and earlier evidence below remain preserved.

A fresh, modular world-model baseline, followed by component research.

The first reference is [LeWM](https://github.com/lucas-maes/le-wm), pinned to
`8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`. PushT is the first learning/control
benchmark; TwoRoom supplies a second trajectory dataset. TAU Urban AV and
Charades source media are retained for later passive multimodal work. The
[source data inventory](docs/source-data.md) records every dataset's origin,
acceptance check and current state.

The previous implementation and results were reset at the user's request.
[Retained ideas](docs/ideas.md) are hypotheses for later work, not evidence.
No baseline success claim is made until training and evaluation establish it.

[Broader PushT pilot](docs/pusht-broader-pilot.md): 13 CPU tests pass. The approved
128-train/32-held-out configuration split completed 1,000 updates in 25.1 minutes.
The final checkpoint modestly beats copy/shuffled prediction controls but reaches
0/20 held-out control goals; released weights reach 17/20 on the same goals,
recorded replay 19/20, and stationary actions 0/20. Intermediate checkpoints also
reach 0/20. Learned control remains unestablished; focused rollout/control
diagnosis is recommended before more training. Component research stays deferred.

[Full-source reference validation](docs/reference-validation.md) reached 45/50
with released weights on a separate case set. Both full archives are verified
and extracted. [First diagnostic round](docs/diagnostic-results.md) and
[earlier checks](docs/baseline-checks.md) retain the previous subset results.

The standing [experiment workflow](docs/experiment-workflow.md) is independent of
[current goals and status](docs/project-state.md). Every experiment refreshes the
local [HTML instrument panel](runs/experiment_dashboard.html); regenerate it from
existing logs with `python -m viewer.dashboard`. The wrapper requires Node.js and
the installed Data Analytics portable-artifact builder; set
`PATH_WM_ARTIFACT_BUILDER` to its `deliver_portable_artifact.mjs` if needed.

Use the local environment `.runtime/lewm/bin/python`, or install this project
with its `dev`, `eval` and `data` extras in an isolated environment.

```bash
python -m pytest
python run.py -m world_model.train configs/pusht_cchi_dev.yaml
python scripts/prepare_data.py configs/datasets/pusht.yaml
python run.py -m world_model.eval_pusht configs/datasets/pusht.yaml data/reference/lewm-pusht/weights.pt runs/reference_check --released --episodes 2
```

Training refuses to overwrite an existing checkpoint. Use a new `run_dir` for
an independent run; `--resume` requires an identical recorded configuration.
Long-run configurations are available under `configs/`, but are not scheduled.
[Recipe and protocol differences](docs/baseline.md) distinguish development
checks from paper reproduction. [Dataset configs](configs/datasets/) keep source,
action semantics, frame stride and history explicit for each dataset.
