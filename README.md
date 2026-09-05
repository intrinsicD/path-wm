# PATH-WM

A fresh, modular world-model baseline, followed by component research.

The first reference is [LeWM](https://github.com/lucas-maes/le-wm), pinned to
`8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`. PushT is the first learning/control
benchmark; TwoRoom supplies a second trajectory dataset. TAU Urban AV source
media are retained for later passive multimodal work.

The previous implementation and results were reset at the user's request.
[Retained ideas](docs/ideas.md) are hypotheses for later work, not evidence.
No baseline success claim is made until training and evaluation establish it.

[Current diagnostic results](docs/diagnostic-results.md): ten essential tests
pass, and local control matches upstream on ten paired cases using released
weights. Capped learning checks use small official PushT and TwoRoom subsets;
PushT exposed a BatchNorm evaluation mismatch. Useful control by our trained
models and full benchmark reproduction remain unestablished. Long training is
stopped pending user review. [Earlier checks](docs/baseline-checks.md) are retained.

Use the local environment `.runtime/lewm/bin/python`, or install this project
with its `dev`, `eval` and `data` extras in an isolated environment.

```bash
python -m pytest
python -m world_model.train configs/pusht_cchi_dev.yaml
python scripts/prepare_data.py configs/datasets/pusht.yaml
python -m world_model.eval_pusht configs/datasets/pusht.yaml data/reference/lewm-pusht/weights.pt runs/reference_check --released --episodes 2
```

Training refuses to overwrite an existing checkpoint. Use a new `run_dir` for
an independent run; `--resume` requires an identical recorded configuration.
Long-run configurations are available under `configs/`, but are not scheduled.
[Recipe and protocol differences](docs/baseline.md) distinguish development
checks from paper reproduction. [Dataset configs](configs/datasets/) keep source,
action semantics, frame stride and history explicit for each dataset.
