# PATH-WM

A small PyTorch workbench for experimenting with encoders, output heads, memory
and action-conditioned prediction. Start with one readable Python recipe.

## Start

```bash
source .venv/bin/activate                 # use the existing environment here
python -m pip install -e '.[dev]'
python experiments/perception.py --check
python experiments/perception.py --output runs/my_first_test
```

Open **`runs/my_first_test/report.html`**. It contains the learning curves, exact
validation values, reconstructions, feature PCA and resolved settings. It works
offline, without an AI app or a report-building plugin.

The default is a short **development run**, using prepared PushT images, a fresh
CNN, an RGB decoder and a pose head. It checks ideas; it is not a trained controller.
On another checkout, create a Python 3.11+ virtual environment first and provide
prepared data as described in [the experiment guide](docs/experiments.md).

Already available in this checkout: [perception report](runs/start_here/perception/report.html),
[dynamics report](runs/start_here/dynamics/report.html), and
[ViT + COCO report](runs/start_here/coco_vit/report.html). These are tiny verified
development examples, not finished model training.

## Change something

```bash
cp experiments/perception.py experiments/my_idea.py
```

Open **`build_model()`** to replace the encoder or heads. Open **`objective()`** to
change what they learn. Run your copy with `--check` before spending compute.
Each head is independent. Use `--diagnostic-rgb` to block reconstruction gradients
into the encoder, or `--freeze-encoder` to train only output heads.

```bash
python experiments/my_idea.py --check
python experiments/my_idea.py --output runs/my_idea --steps 200
```

Pause and resume without repeating the settings:

```bash
python experiments/perception.py --output runs/resume_example --steps 100 --stop-after 20
python experiments/perception.py --resume runs/resume_example
```

For action prediction with the retained pretrained CNN:

```bash
python experiments/dynamics.py --check
python experiments/dynamics.py --output runs/my_dynamics
```

## Where things live

| Location | What you edit or find |
|---|---|
| `experiments/` | Model construction, data, loss and budget |
| `pathwm/models/` | Ordinary `nn.Module` implementations |
| `pathwm/data/` | Prepared images and consecutive sequence windows |
| `pathwm/training/` | Two explicit loops: perception and dynamics |
| `pathwm/evaluation/` | Local reporting and candidate-action scoring |
| `pathwm/io.py` | Run records, checkpoints and resume |
| `runs/<name>/` | One run’s results and report |
| `tests/` | Small CPU checks for meaningful failure cases |

Read [models and tensor flow](docs/models.md), [running experiments](docs/experiments.md),
or [current work](docs/project-state.md). Standing development rules are in
[the workflow](docs/experiment-workflow.md).

Historical implementations and protocols are preserved in Git tag
`archive/pre-modular-2026-09-09`. Existing datasets and completed runs remain on
disk; the old aggregate dashboard is historical. [Migration record](docs/migration.md).
