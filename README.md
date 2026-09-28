# PATH-WM

A small, editable PyTorch library for an integrated latent agent: one weight-tied
**shared core** observes, predicts, thinks, induces concepts and applies them over
object slots, with runtime memory instead of runtime weight changes. Design:
[core design](docs/core-design.md). Implementation plan and status:
[shared-core plan](docs/shared-core-plan.md). Overview figure:
[architecture overview](docs/architecture-overview.html).

The earlier architecture (belief agent, thinker, latent core R1 and their recipes)
was retired on 29 September 2026 and is preserved by Git tag
`archive/pre-core-2026-09-29`. Existing runs under `runs/` are kept.

## Install and test

```bash
uv pip install --python .venv/bin/python -e '.[dev]'
source .venv/bin/activate
python -m pytest
```

## Train the shared core (Rule World, symbolic stage)

```bash
python -m experiments.core --output runs/core/my_run --family relation --rules 4 --rule-repeats 7 1 1 1 --reader evidence
python -m experiments.core --resume runs/core/my_run
```

Open **`runs/core/my_run/report.html`**. `result.json` holds ν per family for
training rules and held-out rules, with copy, empty-support and swapped-support
controls. Controls from the design: `--untied` (one block stack per operation) and
`--continuous-only` (no categorical code). `--size check` is a declared downscale
for CPU tests only.

| Location | Purpose |
| --- | --- |
| `pathwm/models/core.py` | `SharedCore`, `CoreState`: observe, predict, imagine, think, induce, apply; write rights |
| `pathwm/models/symbolic.py` | Supplied symbolic perception and readout for the first Rule World stage |
| `experiments/core.py` | Shared-core recipe: data, losses, evaluation, report |
| `pathwm/data/rule_world.py` | Rule World generator, rules, splits, rendering |
| `pathwm/evaluation/rules.py` | ν and calibration metrics |
| `pathwm/models/multiscale.py`, `slots.py` | Multiscale encoder and Slot Attention for the pixel stage |
| `pathwm/world_state/` | WorldStore, records, retrieval, concept memory |
| `pathwm/io.py` | Run records, checkpoints, resume |
| `runs/<name>/` | Raw results, checkpoint and standalone report |

Other recipes in `experiments/` (image codes, VAEs, video order, perception,
hierarchy fusion, webcam capture) are independent research lines on shared
infrastructure. Follow [the workflow](docs/experiment-workflow.md) and
[current project state](docs/project-state.md).
