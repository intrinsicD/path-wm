# PATH-WM

One small, editable multimodal world model: image/video/audio/text adapters,
structured latent state, episodic memory, internal computation, imagined futures,
action planning and a bounded learning-update gate. All parts are ordinary PyTorch
modules constructed in [one recipe](experiments/multimodal.py).

The default is a 325,704-parameter development model with freshly initialized
weights. Running it verifies the architecture and training path; it does not produce
a generally capable language, audio or physics model.

## Start

```bash
source .venv/bin/activate
python experiments/multimodal.py --check
python experiments/multimodal.py --output runs/my_multimodal
```

Open **`runs/my_multimodal/report.html`**. It contains training/validation values,
generated media, attention, latent activity, memory provenance and planning scores.
The report is self-contained and works offline. The run also owns a checkpoint,
raw metrics, source snapshot, `inspection.pt` and `inspection.json`.

On a fresh environment: Python 3.11+ and `python -m pip install -e '.[dev]'`.
With uv, use `uv pip install --python .venv/bin/python -e '.[dev]'`.

## Real observations and resume

```bash
# Existing prepared PushT images/actions; audio/text are absent.
python experiments/multimodal.py --dataset pusht --check
python experiments/multimodal.py --dataset pusht --steps 4 --improve-every 0 --output runs/my_real_multimodal

# Start a separate run, pause after 3 of its 8 main updates, then finish it.
python experiments/multimodal.py --steps 8 --stop-after 3 --output runs/my_resumable_multimodal
python experiments/multimodal.py --resume runs/my_resumable_multimodal
```

The synthetic default generates moving-ball observations, short impact waveforms
and byte descriptions. It needs no downloads. The real path defaults to
`data/pusht_world_model/cchi_v1`; supply `--data-root` on another machine.
Resume checks code, settings, data, modules, optimizer and runtime identity.

## Discuss or change a part

Generate diagrams directly from the current model and a tiny recorded execution:

```bash
python experiments/multimodal.py --diagram
```

Open [architecture](docs/diagrams/architecture.svg) or
[data flow](docs/diagrams/data_flow.svg). The first shows module containment and
parameter counts; the second shows values passed between example calls, with tensor
shapes. Increase `--diagram-depth 3` to expand the module hierarchy. SVG/PNG rendering
uses Graphviz (`dot`); Mermaid, DOT and JSON sources are always written.
This CPU command needs no training or downloads. [Details](docs/multimodal.md#diagrams).

The input pyramids have their own executed diagrams:
[image](docs/diagrams/image_scales.svg), [video](docs/diagrams/video_scales.svg),
[audio](docs/diagrams/audio_scales.svg), [text](docs/diagrams/text_scales.svg).
Every scale finishes conditioned residual/attention processing before the next scale
or the latent state reads it. [Control and inspect the features](docs/multimodal.md#multiscale-inputs-and-feature-control).

Read [the concrete architecture](docs/multimodal.md), then edit `build_model` and
`objective` in the recipe. It exposes every encoder, decoder, latent group,
observation updater, thinker, dynamics module, memory store and action head.
To make a variation, copy the recipe and edit its constructors/losses.
There is no registry or configuration framework to learn.

| Location | Purpose |
| --- | --- |
| `experiments/multimodal.py` | Construction, objective, data, training and inspection |
| `pathwm/models/agent.py` | Observe, think, imagine, decode and intervene |
| `pathwm/models/agent_state.py` | Explicit state and episodic memory |
| `pathwm/models/modalities.py` | Modality adapters and attention |
| `pathwm/models/multiscale.py` | Input pyramids, conditioned residual blocks and feature controller |
| `pathwm/evaluation/agent.py` | Bounded candidate-action planning |
| `pathwm/training/improvement.py` | Measured update acceptance and rollback |
| `pathwm/io.py` | Existing checkpoints, provenance and resume |
| `runs/<name>/` | Raw results, media, inspection and standalone report |

Run `python -m pytest` before committing shared-code changes. Follow
[the workflow](docs/experiment-workflow.md) and [current project state](docs/project-state.md).
The earlier perception/dynamics recipes remain focused references. Historical
source is preserved by Git tag `archive/pre-modular-2026-09-09`; existing data and
completed results are retained.
