# Claims

## C01: The Step 3 inverse anchor remained at the zero-action solution
- **Statement**: On checkpoint `4556ea8`, held-out inverse MSE is 0.338520 versus a 0.336418 zero-action baseline, per-axis action correlations are -0.00285/-0.02412, and token-aware probes on frozen W remain at chance although raw frame pairs support 0.235068 MSE; the baseline did not anchor recoverable action information into W.
- **Status**: supported
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Falsification criteria**: A held-out probe on the same frozen checkpoint and split recovers action materially below the zero baseline with stable positive correlation.
- **Proof**: [ara/evidence/tables/e1_dev_token_pair_2026-09-04.md, runs/dev/first_slice/0/checkpoint.pt]
- **Dependencies**: []
- **Tags**: E1, inverse-dynamics, action-anchor, latent-ABI
- **From staging**: O05

## C02: Inverse decodability did not imply predictor action correctness
- **Statement**: In the Step 4 token-pair run, held-out inverse MSE improved to 0.210544 with action correlations 0.63874/0.63642, yet correct-action predictor MSE (0.000791805) did not beat zero-action (0.000791497) or shuffled-action (0.000792345) controls.
- **Status**: supported
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Falsification criteria**: A deterministic rerun of the committed checkpoint and fixed probe shows a material, repeatable transition-error gain for correct actions over both controls.
- **Proof**: [ara/evidence/tables/e1_dev_token_pair_2026-09-04.md, runs/dev/first_slice_token_pair/0/checkpoint.pt, commit:1148cc6]
- **Dependencies**: [C01]
- **Tags**: E1, predictor, action-conditioning, diagnostic-control
- **From staging**: O06

