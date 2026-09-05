# Heuristics



## H01: Diagnose reference integration before long training
- **Rationale**: Replay recorded actions and compare a released checkpoint through upstream/local control before capped batch-128 learning and second-dataset checks; separate integration failure from undertraining before spending a long-run budget.
- **Provenance**: user-revised
- **Crystallized via**: verbal-affirmation
- **Sensitivity**: high; requires explicit data, normalization and sampling scope.
- **Code ref**: [replay](../../../scripts/check_alignment.py), [paired evaluation](../../../scripts/compare_evaluators.py), [learning checks](../../../configs/diagnostics/README.md)
- **From staging**: O02
- **Evidence of adoption**: N05; user said “yes please do exactly that”. This affirms the diagnostic sequence, not subsequent capability claims.


## H02: Complete the capped schedule and full-source positive control
- **Rationale**: Finish the fixed 400-update cached schedule, evaluate untouched weights and separate normalization probes, then validate released control with verified complete-source normalization before changing the baseline or starting long training.
- **Provenance**: user-revised
- **Crystallized via**: verbal-affirmation
- **Sensitivity**: high; subset prediction, diagnostic clones and full-source released control have different evidential scopes.
- **Code ref**: [cached schedule](../../../configs/diagnostics/pusht_cached_learning.yaml), [checkpoint control](../../../scripts/check_checkpoint_control.py), [full-source reference](../../../scripts/evaluate_reference.py)
- **From staging**: O05
- **Evidence of adoption**: N10; current user requests continuation of the accepted sequence, with the original acceptance recovered from the interrupted session. This adopts the work sequence, not new capability claims.
