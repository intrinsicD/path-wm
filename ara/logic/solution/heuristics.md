# Heuristics

## H01: Pair action sensitivity with action-correctness controls
- **Rationale**: Opposite-action separation can grow even when the predictor maps actions to the wrong transitions. Report identity, zero-action, and deterministically shuffled-action transition errors beside s(w), and require the correct action to improve over the controls before calling a model action-conditioned.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Sensitivity**: high
- **Code ref**: `evaluation/metrics.py`; design in `docs/design-decisions.md` §13.18
- **From staging**: O07
