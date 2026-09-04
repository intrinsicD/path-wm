# Conformance tests

Every module implementation must pass these before it enters an experiment (PATH-WM_v0.3.md §16.1):

- shape and dtype against `docs/abi/abi_v1.yaml`
- action-sensitivity ratio ≥ `action_sensitivity_min` on the fixed probe set
- transition error ≤ `transition_error_max` on the same set
- adapters: §6.5 losses on held-out data
- planners: valid actions within the declared budget; predictor and critic calls accounted separately

## Two layers (CLAUDE.md §4)

**Structural** — one `test_<module>.py` per module: ABI shape, dtype, layout (token order, per-token
LayerNorm, rows of a batch independent), action range, the stateless predictor contract, and, once a
planner exists, calls within budget and counted separately. Random weights, CPU, seconds; no marker, so
plain `pytest` runs it.

**Threshold** — `test_threshold.py`: s(w), transition error, later the §6.5 adapter losses, on the fixed
probe set. Takes the checkpoint under `--run-dir` (default `runs/dev/first_slice/0`, written by
`python run.py configs/dev/first_slice.yaml`), carries the `threshold` marker (`pytest -m threshold`),
records the number in `<run_dir>/threshold_record.json`, and while its threshold in `abi_v1.yaml` is
null skips with reason `threshold_unset`. The same record includes identity/zero/shuffled-action errors
and counterfactual discrimination accuracy as diagnostics without post-hoc thresholds.

## How a test gets an implementation

`tests/conftest.py` holds `cfg` (the dev spec `configs/dev/first_slice.yaml`), `abi`, and `build(family, ...)`,
which imports the family's `build_<module>(cfg)` (DDR §13, step-2 additions) when the test body calls it. A missing
builder is a failed test naming what to write; a broken one raises its own traceback. `conftest.py` here adds the
synthetic inputs (`obs`, `W`, `W_next`, `gen`) and the `per_sample` check.
