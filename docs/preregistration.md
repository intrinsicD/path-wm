# Pre-registration ledger

A spec is frozen by recording its git hash here together with the thresholds that decide it. Nothing in a frozen spec is edited afterwards; changes are appended as numbered amendments.

| Experiment | Spec file | Commit hash | Frozen on | σ_pilot | Thresholds (from PATH-WM_v0.3.md §11) | Amendments |
|---|---|---|---|---|---|---|
| E0 | experiments/E0_causal_world.yaml | — | — | — | throughput ≥ 1e4 transitions/s; ≥ 2 homotopy classes; save/restore exact | — |
| E1 | experiments/E1_reference.yaml | — | — | — | five seeds; σ_pilot recorded here | — |
| E2 | experiments/E2_stitching.yaml | — | — | — | success ≥ 90% native at ≤ 10% params; fail < 70% at 25% | — |
| E4 | experiments/E4_belief_state.yaml | — | — | — | updater or causal encoder beats K-frame window by > 2σ at some horizon | — |
| E6 | experiments/E6_planner_baselines.yaml | — | — | — | three budgets spanning 10×; curves | — |
| E7 | experiments/E7_path_space.yaml | — | — | — | ≥ 2σ over best baseline on ≥ 3/4 environments at equal predictor calls | — |

## Amendments

(none)
