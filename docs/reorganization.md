# A repository for running your own experiments

Status: proposal requested by the user on 9 September 2026. No code migration,
model change or training run has started. The objective is human usability:
understand a model, replace a component, choose its training signal, run/check/resume
an experiment and inspect its results without needing an agent to navigate history.

## What is wrong now

The repository grew around consecutive experiments instead of a stable user API.
The inspected tree has 81 Python scripts, 44 modules in `world_model/curriculum`,
and 76 Markdown/YAML documents. File count alone is not the cause; responsibilities
and dependencies make the common path difficult:

- `world_model/__main__.py` dispatches to the Paddle CLI, while PushT, LeWM and
  perception have other entry points. The README still names Paddle as active.
- `curriculum/perception_training.py` contains the elapsed overnight deadline;
  `perception_cache.py` owns a dated run root and retrieves previous studies'
  manifests. Reusable training and historical orchestration are entangled.
- Reusable-looking model/data code imports experiment setup/report functions.
  Constructing a backbone can require manifests under a previous experiment.
- `paddle/types.py` fixes two grid sizes, width 64 and memory width 128. New
  perception heads have additional fixed-grid assumptions. Replacing an encoder
  requires knowledge of those hidden assumptions.
- `run.py` refreshes the aggregate dashboard through an external builder. The
  current full workflow depends on Node and an installed reporting plugin path.
  Raw training and the ordinary local inspection workflow should be portable.

## Proposed everyday layout

```text
README.md                  start here: install, run, modify, inspect
pathwm/                    reusable, ordinary Python/PyTorch
  models/                  encoders, output heads, memory, dynamics
  data/                    prepared image/sequence adapters
  training/                explicit perception and dynamics loops; losses
  evaluation/              readouts, rollouts, control and reports
  io.py                    run records and checkpoint save/load
experiments/               small executable Python recipes you copy and edit
  perception.py
  dynamics.py
notebooks/                 inspect data/features/rollouts through the same API
data/                      existing source/prepared data and pinned model assets
runs/<name>/               one experiment's settings, metrics, checkpoints, report
docs/                      short architecture and experiment guides
archive/                   historical code/recipes/notes, moved only when compatible
tests/                     behavior, scientific invariants and migration checks
```

Start the new package beside the old implementation during migration. The final
normal path should have one implementation per supported component. Historical
code is explicit reference material, not a dependency of the new library. Existing
data and immutable run artifacts keep their identities and locations. Physical
archival comes after path/import/checkpoint compatibility is checked; it is not a
bulk move at the start. Agent records and old protocols stay outside the everyday
usage path, with an archive index where useful.

## The public interface

Use plain `nn.Module` implementations with documented inputs/outputs. Encoders
produce named feature tensors plus small metadata describing their native shapes
and sources. Consumers select the levels they need and own necessary projections.
Do not make two scales, one channel width or one memory size universal constants.
Unsupported combinations should fail early with a readable explanation; the
interface does not promise every module works with every other module.

Keep observation features, temporal state and imagined transitions distinct in
the actual computation. Actions, task context and time must be explicit inputs.
Gradient destinations belong to the training recipe: frozen encoder, partial
fine-tuning, auxiliary reconstruction and fully trainable components remain
possible. Reorganization does not select a new learning objective or neural
architecture. Port current modules faithfully before adding new capabilities.

Recipes show model construction, data/splits, initialization, loss computation,
trainable parameters and training settings in one readable file. Modules can be
customized beside the recipe and imported directly. Reuse standard library
functions; do not import one experiment from another. Avoid recursive YAML
inheritance, component registries, a base Experiment class, generic dependency
injection and a universal trainer. Shared helpers stay ordinary explicit functions.

Illustrative API only; these names/commands are not implemented yet:

```python
encoder = DinoEncoder(weights=weights_path)
heads = {"mask": MaskDecoder(encoder.feature_spec)}
data = CocoMasks(root=data_root, split_manifest=split_path)

train_perception(
    encoder=encoder,
    heads=heads,
    data=data,
    objective=mask_objective,
    train_encoder=False,
    steps=4_000,
    seed=42,
    output="runs/my_test",
)
```

The objective function is visible or directly imported in that same recipe.
Swapping a supported encoder changes its constructor; a new architecture is an
editable module, not a new command/coordinator/reporting pipeline. A reconstruction
gradient ablation should be a recipe with explicit cases, not several new trainers.

Proposed commands:

```bash
python experiments/perception.py --check
python experiments/perception.py --output runs/my_test
python experiments/perception.py --resume runs/my_test
```

`--check` uses one real batch and shows module/feature shapes, parameter counts,
trainable modules, loss terms and gradient recipients before substantial compute.
It does not prove temporal causality. Use known-index fixtures for temporal
alignment and future-input independence tests: hold observed history and candidate
actions fixed, vary inaccessible future observations, and verify predictions stay
unchanged. An action-shuffling score alone is a behavioral diagnostic, not proof
that leakage is absent. Recipes run only from an explicit main function, with no training,
data writes, downloads or GPU allocation merely from importing model definitions.

## Runs and inspection

Each run records serializable resolved settings, its recipe/source identity,
dataset/split/checkpoint identities, package versions, device type, determinism
flags, metrics and result/report status. Write result completion before invoking
the reporter, so a reporting failure cannot hide successful training/evaluation.
This combines code identity with explicit settings; it does not claim to serialize
arbitrary Python closures automatically. Comparing two runs should expose the
changed modules, losses, data and budgets directly.

Resume saves/restores model parameters and buffers, optimizer/scheduler state,
progress, sampling position/state and relevant RNG. Refuse incompatible resumes
and accidental overwrites; exact replay is limited to declared supported loader
and precision settings. Normal new work starts with live features. Add caching
only for a measured need and frozen inputs; preserve verified existing cache
semantics when comparing with a historical cached reference.

Generate a compact self-contained local HTML report with loss curves, examples
and relevant task metrics using repository-owned dependencies. Preserve automatic
reporting and visible verification/failure status without making the common path
depend on an AI app or private plugin installation. Raw results/checkpoints survive
reporting failures. The rich existing aggregate dashboard remains a historical
viewer during migration; rebuilding every historical report is not the cost of
trying one new model. This reporting change requires an explicit update to the
standing workflow during implementation, not a silent relaxation of validation.

Keep the user documentation to a short README, a model/interface guide and an
experiment guide. An inspection notebook follows after the recipe API works; it
calls the same components and does not own a second implementation. A GUI,
distributed training framework, new cache system and a universal multimodal API
are deferred until a concrete need exists.

## Migration and acceptance

1. Extract only the modules/adapters necessary for one established perception
   recipe. Move experiment dates, paths, reference hashes and budgets into its
   recipe/run settings. Do not remove historical constraints from archived recipes.
2. Compare unchanged-reference forward outputs, losses, gradient/freeze behavior,
   a few updates and save/resume under declared tolerances and precision. Keep
   original checkpoints, raw evidence and commands usable. Passing this check is
   scoped migration parity, not proof of universal scientific equivalence.
3. Deliver the readable recipe, one-batch check and portable per-run report. A
   person should be able to copy it, change a supported encoder or loss, run it,
   resume it and find its outputs without reading historical protocol documents.
4. Migrate memory/dynamics and planning only after that first path is usable;
   preserve action/episode timing and existing controls. Consolidate remaining
   scripts/docs and archive implementations after dependencies are accounted for.

The first completion criterion is this working user path, not an empty old folder
or a renamed import. Model redesign and new scientific experiments remain separate
work. The proposal does not promise a complete rewrite or prescribe a count of
new abstractions before the first usable path has been demonstrated.

## Review record

Three public-only Claude exchanges reviewed the generic design. Exact briefs,
responses and execution receipts are in `runs/reorganization_review_2026-09-09/`;
repository code/history/data were not shared. The local audit was performed
independently. The review tightened the resolved-record and resume contracts,
favored named tensors with small metadata, and deferred notebook polish and a
new caching framework. Reconciliation retained compact local HTML and existing
cache semantics for reference parity, and clarified that one-batch checks alone
do not prove causality. Claude acknowledged the correction that action-shuffling
scores do not prove absence of leakage; targeted alignment and future-input tests
only establish their declared invariants. Its final suggestion to include both
plausible and extreme future perturbations remains a test-design detail to assess
when the temporal path is migrated. Peer review is design criticism, not implementation
validation. No code migration, model tests or new experiments ran in this turn.
