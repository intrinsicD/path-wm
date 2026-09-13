# Running and editing experiments

The current starting point is `experiments/multimodal.py`; see the
[multimodal architecture and training guide](multimodal.md). Run
`python experiments/multimodal.py --check`, then use a fresh `--output` directory.
`--dataset pusht` uses the retained real data at `data/pusht_world_model/cchi_v1`.
The older `data/pusht64` shortcut is not present in this checkout.
`--dataset instructions` adds synthetic operation/modality/completion supervision
to the same training loop. [Task contracts and evaluation](tasks.md).

The following perception/dynamics recipes remain focused references.

`experiments/conditional_image.py` trains the optional multiscale image generator
on frozen memory contexts, with direct-regression and flow-matching objectives.
The [commands, comparison and limits](image-output-plan.md#conditional-generator-results)
include its pure-noise weighting option, strict checkpoint loading and resume.
Use `--decoded-image-weight 10 --decoded-image-path endpoint` for the image loss
on the endpoint estimate, or `--decoded-image-path sample` for full sampler gradients.
`--steps` sets the fit budget; full-sampler training costs more. See the [decoded-image
comparison](image-output-plan.md#decoded-image-supervision-results) for exact commands.
The completed comparisons fail capability gates; the original renderer remains
the default. This is a controlled64px experiment, not general image generation.

The [hierarchy/fusion comparison](hierarchy-fusion-plan.md) tests additional per-scale
transformer depth and final attention over all scales on real COCO RGB/foreground:

```bash
.venv/bin/python -m experiments.hierarchy_fusion --arm deep_fusion --seed 7401 --device cuda --check
.venv/bin/python -m experiments.hierarchy_fusion --arm deep_fusion --seed 7401 --device cuda
```

Use a fresh output for a new run, or `--resume RUN` to continue that exact configuration.
The general perception recipe also accepts `--encoder pyramid --stage-depth 2
--fusion-depth 2`, with `--levels` and `--width`. All modality encoder constructors
accept `fusion_depth`; the multimodal Python constructor exposes `feature_depth`
and `fusion_depth`. Default fusion remains0, preserving previous model layouts.

The same recipe supports [direct numerical weights](hierarchy-weights-plan.md):
`--arm deep_fusion --weight-method constructed` writes the handwritten candidate;
`--weight-method ridge` additionally fits132 final-readout coefficients from training
pixels. Both use zero optimizer updates and save strict-loadable `weights.pt`, the
ordinary checkpoint and report. The completed screen did not beat the trained models;
these are color-transport diagnostics, not pretrained semantic weights.

The same recipe now supports [training from handwritten initialization and frozen
component diagnosis](hierarchy-training-plan.md). Reproduce the repaired RGB-only
training with a fresh output directory:

```bash
.venv/bin/python -m experiments.hierarchy_fusion \
  --arm deep_fusion --seed 7501 --device cuda \
  --initial-weights runs/hierarchy_weights_v1/constructed_7401/weights.pt \
  --loss-mode rgb --decoder-learning-rate 0.000003 \
  --output runs/my_handwritten_training
```

The encoder rate remains0.0003. `--train-part encoder` or `decoder` freezes the
other component; RGB-only training always freezes the mask head. The optional
`--open-residual-branches` seeds initially inactive RGB residual weights while
preserving the initial predictions; it is a separate initialization variant.
It was not needed to stabilize the repaired comparison. Keep the smaller decoder
rate explicit: the old0.0003 decoder rate collapsed the handwritten runs.

`--stop-after 16` pauses a run after16 additional updates. Resume it using
`--resume runs/my_handwritten_training`; the stored initial-file identity, rates,
freeze settings, sampler and optimizer are restored and conflicting settings fail.
The deep_fusion recipe has a fixed384-update target; resuming an already complete
run rechecks/reports it, rather than extending its training budget.

Each completed adaptation exports a CPU `weights.pt` (model state) and `last.pt`
(full resume state). To load the completed exact-handwritten run for inference:

```python
import torch
from experiments.hierarchy_fusion import build_model

model, _ = build_model(7501, "deep_fusion")
path = "runs/hierarchy_training_v1/decoder_rate_repair/seed_7501/hand_both/weights.pt"
payload = torch.load(path, map_location="cpu", weights_only=True)
model.load_state_dict(payload["model"], strict=True)
model.eval()
# model(rgb_float_tensor) accepts B×3×64×64 in [0,1].
```

A new run can use this exported file as `--initial-weights` with a fresh output;
that warm-starts the model but starts a new optimizer and sampler. Use `--resume`
for exact continuation of an interrupted run. The trained mask output remains
unvalidated: its head is frozen, while the shared encoder can change.

For the finite single-observation entity/location control, use the same recipe:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python experiments/multimodal.py --dataset facts --check
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python experiments/multimodal.py --dataset facts --output runs/my_fact_control
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python experiments/multimodal.py --resume runs/my_fact_control
```

Activate the installed project environment first (this checkout provides
`.venv/bin/python`). Facts defaults to 512 updates, seed 23, 96 training and 32
held-out entity/location pairs, and a cumulative 450-second active CPU cap. It
trains the text encoder and direct heads from scratch and evaluates the final
checkpoint. Passing extraction gates enables an untrained two-record selector
using cached predictions. The standalone report includes both stages; raw logits,
split manifest, metrics and checkpoint remain beside it. Read the
[declared plan and results](fact-learning-plan.md) for gates and interpretation.

To run the same fact task through the existing agent event/task reader, add
`--fact-reader event` to both check and training commands. It starts a fresh agent
session for every batch, commits one observation, interprets a constant instruction,
and reads working tokens after two thinking rounds. The source encoder and final
heads match the direct control's initialization. Evaluation keeps categorical
sampling with fixed batch seeds; it is one repeatable realization. The
[event-control plan](event-fact-plan.md) declares the comparison, limits and results.

Initialize only the event reader's trainable text encoder from a direct-fact run:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python experiments/multimodal.py --dataset facts --fact-reader event --fact-encoder-weights runs/fact_grounding_v1/reference/last.pt --check
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python experiments/multimodal.py --dataset facts --fact-reader event --fact-encoder-weights runs/fact_grounding_v1/reference/last.pt --output runs/my_initialized_reader
```

The donor heads, optimizer and state are not loaded. The recipient retains a fresh
optimizer and trainable encoder. Run settings bind the donor file/component hashes;
resume restores recipient progress and rejects changed donor contents. Keep the donor
available at its recorded path. The [initialization plan](warm-encoder-plan.md)
separates donor exposure from recipient training and labels reused development data.

Start with `experiments/perception.py`. `build_model` shows the modules; `objective`
shows exactly what is minimized. Copy the file for a new idea and edit it directly.
Custom modules can live beside your recipe until they are useful enough to share.

## Data and initialization

This checkout has prepared, provenance-checked data:

- `data/pusht64` points to the existing RGB64 CCHI episodes. Images, pose labels and
  actions retain their original grouped train/validation/test splits.
- `data/coco64` points to the existing grouped COCO RGB64 frames.
- `data/assets/coco_masks` contains the previously prepared non-crowd foreground
  union and valid-pixel masks; it is not instance segmentation.
- `data/assets/cnn_reference.pt` and `rgb_reference.pt` preserve the paired depth-2
  CNN and RGB decoder. `data/assets/dinov2/` holds verified official source/weights.
  Asset identities and upstream license/revision are in `data/assets/migration.json`.

Data and weights are intentionally outside Git. On another machine copy the prepared
assets, including manifests, or pass another prepared root with `--data-root`.
Paths resolve relative to the shell working directory; run the examples from the
repository root. Files are checked against their manifests at initial loading.
No download or full-dataset conversion happens automatically.

Other datasets can implement the small interface used by `Frames`: `len(dataset)`,
`batch(indices, device)` and an explicit `identity`. Batches contain RGB floats in
[0,1] and any targets consumed by your objective. Sequence batches keep observation
history separate from future targets. Preserve episode/group boundaries when
constructing training and evaluation splits.

Install the optional ViT dependency with `python -m pip install -e '.[vision]'`
when using DINO on a new environment.

Useful examples:

```bash
# Change residual depth and isolate task-head learning on the retained encoder.
python experiments/perception.py --depth 2 --encoder-weights data/assets/cnn_reference.pt --freeze-encoder --output runs/frozen_heads

# Pretrained ViT with native local/context RGB decoding.
python experiments/perception.py --encoder dino --freeze-encoder --check
python experiments/perception.py --encoder dino --freeze-encoder --output runs/vit_heads

# COCO foreground + reconstruction on their fixed prepared population.
python experiments/perception.py --dataset coco --masks data/assets/coco_masks --check

# Keep task learning active, while RGB becomes a detached diagnostic output.
python experiments/perception.py --diagnostic-rgb --output runs/diagnostic_rgb
```

The defaults use small fixed prefixes of the existing splits and short budgets for
quick development. They do not establish generalization. Raise the population and
budget deliberately when declaring an actual experiment. The `--check` command
shows shapes, trainable counts, losses and gradient recipients; it does not prove
causality, data coverage or learning quality.

## Use a trained encoder in the next experiment

A new recipe can load one component from `last.pt` with
`load_component(encoder, checkpoint, "encoder")`. Constructor shapes must match.
For the default CNN perception recipe (depth zero):

```bash
python experiments/dynamics.py --encoder-weights runs/my_first_test/last.pt --encoder-depth 0 --decoder-weights runs/my_first_test/last.pt --output runs/my_next_stage
```

The optional decoder loads `heads.rgb` from the same checkpoint. Omit it to inspect
latent predictions without rendering them. Use paired encoder/decoder weights when
you do render; an unrelated decoder can make a valid state look broken. Starting a
new stage loads model weights, not the previous stage's optimizer or training RNG.

## Runs and resume

Each output directory is new. An existing path is never silently overwritten.
`--stop-after N` pauses after N additional updates while keeping the original total
budget. `--resume runs/name` restores the saved arguments; conflicting scientific
CLI overrides are rejected. Changed library/recipe code, initialization files,
data identity, modules, optimizer or runtime settings require a new run.

A checkpoint contains model parameters/buffers, optimizer state, progress, raw metric
rows and Python/NumPy/Torch/direct-sampler RNG. The atomic checkpoint is authoritative
if a ledger write is interrupted; resume rebuilds the ledger from committed progress.
Exact replay is supported for the same software/device with FP32, workers=0 and
deterministic direct-batch processing. It does not promise identical results across
hardware, precision or different kernels. Load only trusted checkpoints.

Each run contains `run.json`, recipe/library source snapshots, installed package
versions in `environment.txt`, `last.pt`, `metrics.jsonl`,
`status.json`, and a self-contained `report.html`. Saved examples and PCA axes allow
inspection without repeating training. Loss terms and exact validation values are
visible; PCA colors are projections, not semantic labels or information-content
measurements. Independently fitted PCA colors are not comparable across runs.
For sequence runs, the report compares the final true future frame
to a rendering of predicted features; no true future features enter that rendering.

Rebuild a report without model inference:

```bash
python -m pathwm.evaluation.report runs/my_first_test
```

Result status and report status are separate. A report failure preserves checkpoints
and raw metrics. Structural verification runs automatically; browser QA receipts
identify the exact HTML checked. A changed or rebuilt report requires fresh browser
QA if its visual verification is to be claimed.

Run `python -m pytest` before committing shared-code changes. Keep the standing
[workflow](experiment-workflow.md) and the current plan small.

### Controlled entity learning

`--dataset entities` uses three observations of two supplied candidates, with fresh
split-specific appearance descriptors. It asks for cued identity, remembered binary
states and a proposed toggle's effect. Half the final views hide identity features;
those targets are exact conditional distributions. This is a recurrent baseline,
not yet the learned entity graph or visual object discovery.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python experiments/multimodal.py --dataset entities --check
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python experiments/multimodal.py --dataset entities --output runs/entity_learning_v1/reference
.venv/bin/python experiments/multimodal.py --resume runs/entity_learning_v1/reference
```

The fixed first screen uses 256 updates and at most 450 active CPU seconds. Its
`entity_results.json`, prediction cache and standalone report retain both cohorts,
proper log losses, paired-history scores and selection costs. Development is read
only at the final checkpoint; test data is reserved. See
[the declared gates and graph prerequisite](entity-learning-task.md).

To bypass synthetic descriptor matching while retaining learned state updates, use
`--dataset entities --entity-association observed`. Correspondences are computed
from visible descriptors only. This diagnostic supplies association; it does not
learn recognition or graph structure. Use a fresh output directory, for example
`runs/entity_alignment_v1/observed`, and the same fixed defaults as the raw baseline.

`--dataset entities --entity-association observed --entity-reader shared` uses
shared per-object recurrent updates and a coherent mixture over unresolved final
assignments. It retains two fixed object slots; it is not learned graph allocation.
Use `--output runs/entity_shared_v1/reference` for the declared 256-update screen.

Use `--dataset entities --entity-reader shared --entity-association learned` to
train a pair scorer from answer loss instead of exact descriptor matching. It
retains both possible action-time assignments and mixes answer probabilities;
fixed slots and the two-object hypothesis set remain supplied. The declared first
run uses `--output runs/entity_learned_v1/reference`. Its report includes initial
and final matching accuracy on distinct development descriptor groups.

Add `--entity-noise 0.2` to the shared/learned condition to perturb descriptors at
every observation within a guaranteed identity-separation margin. Zero is the
unchanged stable baseline; allowed values are finite and below 0.25. The declared
single run uses `--output runs/entity_variation_v1/reference`. Matching remains
learned; the nearest-descriptor oracle is used only for evaluation and tests.

`--dataset entity-matching` tests whether a unit query matches one of two stored
unit descriptors or neither. It trains a shared pair scorer and a new-entity logit,
without allocating records. Use `--output runs/entity_novelty_v1/reference` for the
fixed 256-update screen. Reports separate false merges, false splits, identity,
NLL and selection coverage. Known/new distances are deliberately separated; this
is not general open-world novelty detection or calibrated graph allocation.

## Full current-capability baseline

The [capability baseline](capability-baseline-plan.md#completed-baseline) records34
behavioral/mechanism checks across13 separate checkpoint roles, plus the complete
software suite and explicit missing capabilities. Use `--dataset capabilities`
with a fresh output; `--baseline-reference runs/capabilities_v1/reference` produces
metric deltas only when the protocol/populations match. `--capability-weights`
accepts explicit compatible checkpoint overrides. No training occurs. Separate
component successes do not constitute one fully trained agent.

Score an existing selected-object memory export on a fresh procedural test without
training or changing its configuration:

```bash
OMP_NUM_THREADS=2 .venv/bin/python -m experiments.memory_output \
  --weights runs/writer_reader_v1/seed_8501_joint/weights.pt \
  --evaluate-only --test-seed 7793 --device cuda:0 \
  --output runs/memory_export_evaluation
```

Use a new output directory. The export's original `run.json` must accompany it;
new results record that provenance separately from evaluation code and data. No
optimizer or training curve is created. Training overrides are rejected, and a
report failure leaves completed metrics available. See the [active recall plan](recall-repair-plan.md).


For a frozen-state image-output diagnostic, use the same native live mixed recipe
with `--writer-learning frozen --image-only`. This freezes the factual head too;
only image feature production learns, while its reconstruction backend stays fixed.
Without `--image-only`, the existing head/writer/thinker policies are unchanged.
Standalone exports restore the selected freeze policy. Compare to joint continuation
with `--writer-learning trainable --train-thinker`, explicit source checkpoints and
fresh declared validation/test seeds. Both restart AdamW; they do not restore the
source optimizer. See the active plan for the fixed-budget comparison and limits.

Evaluation temporarily clears all parameter trainability flags, then restores them
in a `finally` block. In this GPU environment, `no_grad` alone left small numerical
differences between otherwise identical frozen/joint states. This normalizes the
recipe's inference contract without changing checkpoints or optimizer state. Direct
model calls should use matched inference/trainability settings for bitwise comparisons.

For fixed-checkpoint brightness sensitivity, add `--input-offset -16` (or `16`)
to `--evaluate-only --test-seed ...`. Every observed RGB pixel is shifted in8-bit
units; canonical targets and labels stay unchanged. Values that would clip are
rejected. Data identity records the original image hash and exact transform. This
option is rejected for training, and offset0 preserves prior data identities.
Use predeclared offsets/seeds and report neutral versus shifted gates separately;
these backgrounds do not test new semantic combinations or real-world robustness.

For a frozen-state brightness repair, use `--repair identity --readout-stage native
--readout-context mixed --train-input-offsets -16 -8 0 8 16`. Only native factual and
image feature heads learn from cached working tokens. Ordered variants preserve
whole histories and canonical targets; `--train-input-offsets 0 0 0 0 0` provides
matched neutral presentations. Use an explicit source, seed and fresh validation/
test seeds. This option rejects writer learning and output standardization.

For scene learning on frozen states, use `--repair identity --readout-stage native
--readout-context mixed --center-input --train-scenes neutral background-tint
background-cool`. The three ordered blocks retain canonical targets and paired
histories; three `neutral` entries give a matched presentation control. This option
rejects live writer learning, output standardization and simultaneous training RGB
offsets. Additional presets include `background-warm-mild` and
`background-cool-mild`. Choose explicit source/optimizer/validation/test seeds.

Training centering uses neutral training inputs before augmentation. Its fixed RGB
reference, version and calibration identity are saved with the weights. Ordinary
`load_model()` and `--evaluate-only` restore this preprocessing automatically;
adding `--center-input` again does not refit or double-wrap it. Buffer/settings
mismatches are rejected, including after training checkpoint restoration. Training
preflights all train/validation/test histories and requires complete range coverage.
Cache manifests record preprocessing and source-weight provenance. This remains a
chosen preprocessing policy; a checkpoint trained with it depends on it.

For a separate zero-update diagnostic, add `--center-input` to `--evaluate-only`.
The recipe reconstructs and verifies the original neutral training observations,
fits a fixed per-channel reference median, then centers each observed frame before
encoding. The evaluation manifest records the reference and calibration identity;
the checkpoint remains unchanged. The operation has no access to labels, targets,
shift metadata or future frames and rejects out-of-range values. This task-specific
preprocessing can remove meaningful absolute intensity; it is not a learned
invariance or a default deployment policy. GPU median uses deterministic sorting.

To challenge this policy with scene and illumination changes, add an evaluation-only
`--scene temporal-offset` (or `background-tint`, `background-texture`,
`background-bright`, `foreground-large`, `clutter`, `gain-dark`, `local-shadow`,
`channel-offset`, `neutral`). The recipe declares each bounded transform in
`SCENE_CHALLENGES`; data construction preserves canonical targets and paired
histories. This requires the relocation curriculum saved in the source checkpoint
and cannot be combined with `--input-offset`. `--center-input` remains optional.
Background masks are used only by the renderer, never as model inputs. Range
rejection is a failure of supported coverage; do not omit rejected examples from
claims of accuracy. Centered exports preflight every frame and write
`input_coverage.json`. If any complete history fails the range check, the recipe
finishes a coverage-failed report with `task_scored: false`; it runs no model
queries and produces no predictions or accepted-subset accuracy. Valid populations
retain the normal evaluation path. The encoder's strict forward guard still
rejects out-of-range values; this reporting behavior is not a fallback policy or
learned confidence estimate.

Scene-augmented native readout runs also save `training_scene_fit.json` and display
training joint/shape accuracy per ordered scene block in their ordinary report.
Repeated scene blocks remain separate. These diagnostics do not change held-out
metrics or capability gates; target/source alignment is checked before scoring.

For the controlled single-object shape-loss diagnostic, add `--image-only
--image-weighting box` to native raw mixed scene training. This uses cached frozen
states and trains only the image feature producer; the factual head and image backend
stay frozen when `--image-only` is supplied. Weight10 covers the target foreground's whole bounding rectangle, including
empty shape corners; weight1 applies outside. Blank targets use uniform weights. Each
policy normalizes by its own weight sum. The default remains `foreground`; explicitly
use `--image-weighting foreground` for a matched control. Target masks affect only the
training loss, never inference. Evaluation retains the original foreground-weighted
metric, and the box run logs `foreground_rgb_loss` alongside its actual training loss.
This task-specific option rejects live writer learning and output standardization;
existing `--writer-learning frozen --image-only` remains available with the original loss.

For a controlled image-producer capacity comparison, add `--refine-image` to the
cached native mixed image-only path. This adds one residual MLP after attention at
each image feature scale, before feature projection. Its last projection starts at
zero, preserving the source's outputs and RNG state initially. Both existing producer
weights and the added layers then train; factual/state/backend components stay frozen.
Keep the same weighting policy and budget in the control, omitting `--refine-image`.
This compares the chosen capacity and parameterization, not an intrinsic capacity limit.

Refinement is opt-in and recorded as `producer_refinement` in settings. Standalone
loading restores it strictly; a continuation inherits an already refined source and
does not reinitialize its layers. Evaluation cannot add or remove it. It is currently
scoped to cached native raw mixed image-only training, without output standardization
or live writer training. General image generation is still a separate capability.

To train factual and image readouts together under box weighting, omit `--image-only`
and add `--separate-readout-clipping` to the same cached native raw mixed path. Each
readout then clips its own gradient norm to1. This prevents large factual gradients
from changing the image producer's step size. Include `--image-only` in the matched
control while retaining separate clipping; it matches global clipping when only the
image producer trains. Logs record each pre-clip norm and clipping indicator.

This option rejects live writer/thinker learning, stored readout, output standardization
and evaluation overrides. Saved continuations inherit the policy. Default global
clipping remains unchanged. These parameter-disjoint readouts share frozen inputs;
joint training adds no semantic connection between them. A passing implementation
check does not establish improved model quality—retain held-out scene/causal criteria.
