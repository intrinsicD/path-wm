# E1 Development Baseline — 2026-09-03

- **Implementation commit:** `4556ea8`
- **Run:** `runs/dev/first_slice/0/`
- **Device:** NVIDIA RTX 3050 8 GB, CUDA
- **Training:** 2,000 optimizer steps on 20,480 deterministic E0 transitions
- **Probe set:** 256 fixed samples, rollout horizon 4

| Metric | Value |
|---|---:|
| Action-sensitivity ratio | 0.0022763058077543974 |
| One-step transition error | 0.0003451230877544731 |
| Four-step transition error | 0.0027015761006623507 |
| Final inverse loss | 0.3485717475 |
| Final rollout loss | 0.0630286559 |
| Final total loss | 0.5393426418 |

## Validation record

- Final fast suite: 55 passed, 2 threshold tests deselected.
- Threshold-only suite: 2 skipped explicitly with `threshold_unset`; 55 deselected.
- Ruff: all checks passed.
- Git diff whitespace check: clean before commit.
- Peak observed GPU memory during the run: approximately 5.7 GB.

## Raw artifacts

- `runs/dev/first_slice/0/metrics.json`
- `runs/dev/first_slice/0/run_summary.json`
- `runs/dev/first_slice/0/training.jsonl`
- `runs/dev/first_slice/0/threshold_record.json`
- `runs/dev/first_slice/0/checkpoint.pt`
- `runs/dev/first_slice/0/spec.yaml`

Thresholds were intentionally left null in the ABI spec; the recorded values are a
baseline, not a pass/fail declaration.
