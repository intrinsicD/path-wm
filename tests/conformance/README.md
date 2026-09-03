# Conformance tests

Every module implementation must pass these before it enters an experiment (PATH-WM_v0.3.md §16.1):

- shape and dtype against `docs/abi/abi_v1.yaml`
- action-sensitivity ratio ≥ `action_sensitivity_min` on the fixed probe set
- transition error ≤ `transition_error_max` on the same set
- adapters: §6.5 losses on held-out data
- planners: valid actions within the declared budget; predictor and critic calls accounted separately
