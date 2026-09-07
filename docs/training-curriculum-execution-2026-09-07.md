# Accepted curriculum execution — 2026-09-07

The user authorized execution of the proposed curriculum, thorough final
evaluation, and perception/internal-state visualizations. Task-first ordering,
the three-arm comparison and its prospective physical selector are accepted.
No additional private Claude transfer is implied or attempted.

## Slice 1: portability and curriculum interfaces

Problem: move the frozen evidence to the home runtime, add image-only E/D
pretraining without fictitious task labels, and compare adaptation fairly.

Runtime migration: Python 3.12.9, PyTorch 2.9.0+cu128 and RTX 3050 8 GiB,
using a new .venv with system-site PyTorch and project dependencies. This is
a new experimental runtime; no bitwise continuation equivalence to work's
Python 3.14/Torch 2.14 is asserted.

Raw archive restoration passed all 13,870 file hashes in a separate directory.
Full source verification initially rejected the different NumPy version label
despite exactly equal recomputed split/normalization values. Add a regression
that permits only that provenance difference after actual values match and
still rejects altered split results. Preserve both versions in the receipt.
The initial test suite collection hit a third-party tests-package collision;
make the repository tests package explicit. Restored report paths require
installing verified missing task files at their canonical paths.

New modules:
- world_model/curriculum/data.py: source-backed image-only frames, deterministic
  transform, exact/canonical and radius-four dHash duplicate components, grouped
  COCO split, and immutable task frame/source-row identities.
- world_model/curriculum/training.py: E/D-only or labelled E/D/H objective,
  scalar-weighted ragged accumulation, identical initialization/fresh H,
  private phase samplers, physical selector, atomic optimizer/RNG resume.
- world_model/curriculum/inspection.py: raw per-frame errors, train-only PCA
  feature visualizations, attention maps, activation statistics/spectrum,
  held-out linear probes and reconstruction/error panels.
- viewer/curriculum.py: reconcile raw curriculum ledgers and publish native
  charts and bounded qualitative figures through the existing canonical builder.

Essential tests precede implementation: absent-label routing, physical
selection, ragged gradient equality, warmup/fresh-head boundaries, sampler
resume, duplicate-group split isolation, and compatible checkpoint resumption.
The existing objective and architecture remain unchanged.

Budgets remain the accepted draft: 100-update/3-minute profile,
64-frame 500-update/10-minute diagnostic, then 4,000 updates per arm
(2,000 warmup + 2,000 adaptation for B/C), subject to a common preflight
reduction frozen before the scientific screen. 60-minute training/validation
cap per arm. Initial seed4107; repeat promising comparisons with4108/4109.

The final report will distinguish software completion, perception readiness,
prediction gates and control. It will include selected/final and matched
supervised-exposure comparisons, exact denominators, test/validation separation,
raw/decoded observations, fine/coarse feature PCA, measured cross-scale
attention, spatial activation/variation, spectrum and task-readout errors.
Attention is descriptive, not causal attribution. All learned visualization
bases/probes fit training data only. Held-out test results cannot change
checkpoint selection, gates, objectives or the continuation choice.
