# Heuristics

## H01: Pair action sensitivity with action-correctness controls
- **Rationale**: Opposite-action separation can grow even when the predictor maps actions to the wrong transitions. Report identity, zero-action, and deterministically shuffled-action transition errors beside s(w), and require the correct action to improve over the controls before calling a model action-conditioned.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Sensitivity**: high
- **Code ref**: `evaluation/metrics.py`; design in `docs/design-decisions.md` §13.18
- **From staging**: O07

## H02: Bootstrap residual dynamics jointly at transition scale
- **Rationale**: For a residual world model, calibrate the initial readout against measured identity/transition error and keep one-step dynamics active while the representation forms. Delay a strong contrastive branch-ranking term only for the measured bootstrap window. SIGReg plus inverse alone can increase scene/action variation while destroying temporal neighborhood geometry, so it is not sufficient encoder pretraining.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Sensitivity**: high
- **Code ref**: `predictors/transformer.py`; `losses/e1.py`; `configs/dev/first_slice_onset_short_joint_warmup.yaml`
- **From staging**: O15

## H03: Balance coupled guardrails by module-specific gradient scale
- **Rationale**: When one anti-collapse term prevents the degenerate solution of another, calibrate their coefficients from gradients on the affected encoder/adapter parameters, not scalar loss magnitude alone. Covariance weight 0.01 contributed only 0.081 to the initial total but its weighted encoder gradient was 4.8 times the variance-floor gradient and the model reduced covariance by shrinking feature spread. Preserve an independent held-out panel to detect that shortcut.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Sensitivity**: high
- **Code ref**: `training/representation.py`; `configs/dev/common_base_rank.yaml`; `configs/dev/common_base_rank_balanced.yaml`; `docs/design-decisions.md` §23
- **From staging**: O17

## H04: Preserve learned functions across a time-reference handoff
- **Rationale**: Changing timestamps alone changes a trained Fourier coordinate map. Rotate its time-axis sine/cosine coefficients and translate the linear bias, online and in EMA, at fresh handoff only; preserve all other weights/RNG and test the same physical windows. Keep mathematical equivalence separate from bf16 output rounding.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Sensitivity**: high
- **Code ref**: training/common_base.py; tests/unit/test_common_resume.py; commit:5d58418
- **Proof**: [ara/evidence/tables/e1_common_base_overnight_2026-09-05.md]
- **From staging**: O19

## H05: Checkpoint the consumed batch stream while prefetching
- **Rationale**: A background worker draws from a private generator state; publish that state to the checkpointed stream only when its batch is consumed. Bound prefetch to one batch and compare actual model/optimizer/EMA/random-state trajectories and interruption recovery, not just batch equality.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Sensitivity**: high
- **Code ref**: training/common_base.py; tests/unit/test_common_resume.py; commit:74f87f7
- **Proof**: [ara/evidence/tables/e1_common_base_overnight_2026-09-05.md]
- **From staging**: O25
