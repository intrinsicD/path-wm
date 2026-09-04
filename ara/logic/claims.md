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

## C03: The κ=0.1 counterfactual anchor was effectively inactive at E1's latent scale
- **Statement**: With non-vacuous K=4 save/restore branches, the hardware-matched counterfactual run ended at 0.250977 discrimination accuracy (chance 0.25), correct-action error 0.000749412 versus zero-action 0.000748925, and counterfactual loss 1.386298 (approximately log 4). Its mean distance-row span was only 0.000597, which κ=0.1 compressed to a 0.00597 logit span; counterfactual-only gradient norm was 0.000273 at κ=0.1 versus 1.048 at κ=0.001.
- **Status**: supported
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Falsification criteria**: A same-seed rerun at κ=0.1 on the committed paired data produces a material correct-action advantage and held-out discrimination above chance, with a non-negligible counterfactual-only gradient.
- **Proof**: [ara/evidence/tables/e1_dev_counterfactual_2026-09-04.md, runs/dev/first_slice_counterfactual/0/checkpoint.pt]
- **Dependencies**: [C02]
- **Tags**: E1, counterfactual, temperature, action-conditioning
- **From staging**: O09
