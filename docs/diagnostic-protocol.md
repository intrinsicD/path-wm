# Bounded baseline diagnostic protocol

User authorization: 2026-09-05, “yes please do exactly that”, referring to replay
alignment, ten paired checkpoint-control cases, a capped batch-128 learning
check, and a second-dataset check. This does not authorize long training or
research extensions.

## Sources and scope

The full pinned LeWM PushT and TwoRoom archives are still downloading. Diagnostic
subsets contain complete episodes extracted from received archive prefixes.
Every consumed HDF5 data chunk is checked to lie inside the downloaded decoded
byte range. Sparse views of truncated originals are never training inputs.
Receipts record source revision, prefix range, prefix hash and verified extent.
Full archive checksums remain pending.

- PushT: first 8 episodes, 872 frames. These are closely related trajectory
  variants. Suitable for integration/overfit diagnosis; not a representative
  generalization or benchmark sample.
- TwoRoom: first 32 episodes, 2,875 frames. Actions are 2D directions clipped to
  [-1,1], with speed 5 and collision handling. No label enters the objective.

## Paired evaluation

The native side uses the unmodified `World.evaluate`, `WorldModelPolicy` and
`CEMSolver` from stable-worldmodel commit
`6f1e499e9cc0c898d326112f485c1062c3d20f24`, and native LeWM `JEPA` from
`8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`. The script only wires their current
APIs and enforces equal reset seeds. The original CLI targets older APIs.
The local side calls `world_model.eval_pusht.evaluate_case`, also used by the
normal evaluator, so there is one local control loop.

Both use the same recorded initial/goal images, initial state, finite-action
StandardScaler statistics, CEM generator seed, horizon 5, action block 5, 300
candidates, 30 iterations, 30 elites, goal offset 25 and budget 50. The local
solver additionally scores its final mean; that check does not alter actions.
Statistics are fitted on the supplied subset, not the complete training archive.
This is paired integration validation, not exact paper benchmark reproduction.

Initial exploratory pairing required nontrivial goal displacement. It is kept
separately. The final comparison follows the authors' valid-window sampling
rule on the subset, without a goal-distance filter. Independent per-case CEM
and reset seeds make the paired single-environment executions reproducible;
this differs from a single fifty-environment published benchmark invocation.

## Learning checks

Full baseline architecture and two-term objective, batch 128, random weights,
seed 3072, train-only normalization, episode split. At most 400 updates and an
840-second wall-clock limit per run; neither is a full training reproduction.
Encoder recomputation preserves full-batch BatchNorm and SIGReg computation.

The first PushT attempt uses disk-backed data and worker restart each short
epoch. The TwoRoom check caches its small dataset under an explicit byte limit,
with zero workers. This changes I/O only; cached/uncached windows are tested for
identity. Final diagnostics report prediction, persistence, shuffled actions,
open-loop error and representation spread on training/held-out windows. A
separate discarded clone uses batch statistics with dropout disabled to inspect
BatchNorm mismatch. A second clone resets and re-estimates BatchNorm buffers
from at most 512 fixed training windows at frozen weights, then evaluates with
those buffers. Neither clone changes the saved checkpoint; neither diagnostic
is adopted as a baseline method change.

Commands (repository root, isolated environment):

```bash
.runtime/lewm/bin/python -m pytest
PYTHONPATH=. .runtime/lewm/bin/python scripts/check_alignment.py data/pusht_prefix/episodes.h5 runs/diagnostics/alignment_official
.runtime/lewm/bin/python scripts/compare_evaluators.py data/pusht_prefix/episodes.h5 runs/diagnostics/evaluator_final --cases 10
.runtime/lewm/bin/python -m world_model.train configs/diagnostics/pusht_prefix_learning.yaml
.runtime/lewm/bin/python -m world_model.train configs/diagnostics/tworoom_prefix_learning.yaml
PYTHONPATH=. .runtime/lewm/bin/python scripts/check_training_modes.py runs/diagnostics/pusht_prefix_learning
PYTHONPATH=. .runtime/lewm/bin/python scripts/check_training_modes.py runs/diagnostics/tworoom_prefix_learning
```

Use new run directories for repetition. Source acquisition and dependency setup
are prerequisites; these commands do not launch any long training automatically.
