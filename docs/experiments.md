# Running and editing experiments

The current starting point is `experiments/multimodal.py`; see the
[multimodal architecture and training guide](multimodal.md). Run
`python experiments/multimodal.py --check`, then use a fresh `--output` directory.
`--dataset pusht` uses the retained real data at `data/pusht_world_model/cchi_v1`.
The older `data/pusht64` shortcut is not present in this checkout.
`--dataset instructions` adds synthetic operation/modality/completion supervision
to the same training loop. [Task contracts and evaluation](tasks.md).

The following perception/dynamics recipes remain focused references.

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
