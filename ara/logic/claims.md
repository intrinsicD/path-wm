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

## C04: Calibrated contrastive ranking is insufficient when it overwhelms absolute dynamics
- **Statement**: With κ=0.001 and weight 1, held-out K=4 discrimination rose to 0.649414 and correct-action error became 12.31% lower than shuffled-action error, establishing action-semantic ranking. However, correct-action error was 0.0191331 versus 0.00810761 for zero action and 0.00187854 for identity, while four-step error was 8.81 times the matched control. The first active raw gradient norm was 178.74 before clipping and the stage-2 mean was 30.94 versus 2.54 in the control; this calibrated but over-weighted anchor did not yield a transition-compatible reference.
- **Status**: supported
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Falsification criteria**: A same-seed rerun of the isolated κ=0.001, weight-1 spec retains above-chance branch discrimination while producing correct-action error below zero-action and identity controls without materially worsening matched-control transition error.
- **Proof**: [ara/evidence/tables/e1_dev_counterfactual_2026-09-04.md, runs/dev/first_slice_counterfactual_scaled/0/checkpoint.pt]
- **Dependencies**: [C02, C03]
- **Tags**: E1, counterfactual, loss-balance, action-conditioning, transition-error
- **From staging**: O11

## C05: Paired absolute anchoring improves the E1-a Pareto frontier but cannot beat identity
- **Statement**: An informed grid found κ=0.002 and contrastive weight 0.04 as the best pure-InfoNCE balance, but correct-action error remained 51.06% above identity. More than 99% of the control-checkpoint InfoNCE squared gradient norm was outside the predictor; routing it only to the predictor still learned ranking while exploding absolute error. Adding paired diagonal MSE at weight 6 yielded 0.315430 held-out discrimination, correct-action gains of 6.22%/14.04% over zero/shuffled, and one-/four-step improvements of 25.04%/21.28% over control, yet remained 25.16% above identity. None of the tested E1-a parameter or objective variants passed all controls.
- **Status**: supported
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Falsification criteria**: A rerun inside the tested E1-a grid or remedy family produces statistically above-chance discrimination and correct-action error below identity, zero, and shuffled controls at no material matched-control transition-error cost.
- **Proof**: [ara/evidence/tables/e1_dev_counterfactual_2026-09-04.md, runs/dev/first_slice_counterfactual_solution_absolute_w6/0/checkpoint.pt]
- **Dependencies**: [C02, C03, C04]
- **Tags**: E1, counterfactual, objective-geometry, gradient-routing, identity-shortcut
- **From staging**: O12

## C06: A single RGB frame is not a sufficient Markov state for E0 dynamics
- **Statement**: E0 snapshots with identical rendered RGB and actions but opposite hidden agent velocities produce different next RGB in every tested world, so no deterministic single-frame representation can fully specify the transition. Across 3,840 fresh random transitions, identity, action-only, two-frame, and full-velocity next-position MSE were 0.000215142, 0.000148172, 0.000002944, and 0.000001315; two frames remove 98.63% of identity error. History-bearing E1-b is therefore the justified next architecture test.
- **Status**: supported
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Falsification criteria**: The committed alias test fails under unchanged E0 semantics, or a matched diagnostic shows that two-frame history does not materially improve physical transition prediction over identity/action-only baselines.
- **Proof**: [tests/unit/test_observability.py, ara/evidence/tables/e1_dev_counterfactual_2026-09-04.md]
- **Dependencies**: [C05]
- **Tags**: E0, E1, observability, hidden-velocity, belief-state, updater
- **From staging**: O13

## C07: E1-a's late semantic onset is a scheduling and bootstrap problem, not gradient starvation
- **Statement**: The default counterfactual term is disabled until step 1,000. At initialization W has unit RMS but only 0.007058 across-example variance, while the default predictor's paired MSE is 0.310718 versus 0.001732 for identity; auxiliary encoder/adapter/predictor gradients are all nonzero. Scaling the residual readout to 0.03 and using a 200-step joint dynamics warm-up moves fixed-probe discrimination above the three-standard-error threshold at step 1,000 rather than 2,000 and reaches 0.535156. By contrast, direct full-strength training at default initialization remains at chance, and encoder-only SIGReg+inverse pretraining inflates paired identity MSE to 0.1611 by step 200. Early learning is possible, but the current encoder-only curriculum is not suitable.
- **Status**: supported
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Falsification criteria**: A matched rerun shows absent initial auxiliary gradients, no residual-scale mismatch, no earlier onset under the scaled short-joint schedule, or predictive/smooth geometry from the tested encoder-only phase.
- **Proof**: [ara/evidence/tables/e1_dev_learning_onset_2026-09-04.md, runs/dev/first_slice_onset_short_joint_warmup/0/learning_curve.jsonl]
- **Dependencies**: [C05, C06]
- **Tags**: E1, curriculum, initialization, learning-onset, representation-bootstrap
- **From staging**: O14
