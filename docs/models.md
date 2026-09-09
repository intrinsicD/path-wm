# Models and tensor flow

These are ordinary PyTorch modules. A recipe constructs the pieces and selects
the losses. There is no model registry or hidden experiment coordinator.

## Perception

```text
RGB image → encoder → named feature maps → independent output heads
                         fine/coarse          RGB, mask, pose, your own head
```

An encoder exposes `feature_spec: dict[str, FeatureSpec]`. Each entry declares its
channel count, grid size and source. `forward(rgb)` returns a dictionary of tensors
shaped `[batch, channels, height, width]`. Widths and scale counts are not universal
constants. A consumer declares which levels it needs and validates their shapes.

The retained `CNNEncoder` has two processed grids: fine at image_size/4 and coarse
at image_size/8. `width`, `depth` (residual blocks per grid), and `exchange` are
constructor arguments. Cross-scale exchange reads both original incoming grids
before either update is applied. It is a reference architecture, not a claim that
two scales are optimal. Add another encoder as an ordinary `nn.Module` with the
same dictionary interface; no other framework registration is required.

`DinoEncoder` loads explicitly supplied local official ViT-S/14 source and weights.
It provides local patch embeddings, final contextual features, and pooled final
features. Pooling does not constitute another transformer stage. The preserved
reference transform resizes the full RGB64 view to 224 and normalizes it. Keeping
a pretrained architecture does not mean a newly constructed output head is trained.

Output choices:

- `ReconstructionDecoder`: the small original equal-width fine/coarse RGB decoder.
- `DenseHead`: choose levels, native widths, output channels and activation. Each
  level gets its own projection. `retain_statistics=True` carries token mean/scale
  alongside normalized content, preserving the local/context decoder computation.
  Instantiate separate heads to keep RGB and mask parameters independent.
- `PushTPoseHead`: a task-specific example, producing pusher XY, object XY and
  sin/cos orientation. Coordinates are normalized; this does not force every
  encoder to internally represent those six values.

`Perception(encoder, heads, detached_heads=(...))` makes gradient routing explicit.
Detached RGB inputs let that decoder learn without changing E. Other heads can
still train E. Freezing parameters also needs evaluation mode to preserve buffers
and dropout behavior; the training loops handle that for fully frozen submodules.

## Memory and dynamics

```text
observed images + previous actions → encoder + MemoryUpdater → current features/memory
current features/memory + candidate action → Predictor → imagined features
imagined features + same action + previous memory → MemoryUpdater → imagined memory
```

`observe` accepts H images and exactly H-1 connecting actions, plus the action
preceding the first image. The data adapter supplies the start marker only at the
actual beginning of an episode. Short windows initialize memory to zero; this is
not full-episode memory evaluation.

`imagine` receives no future images. Future observations are encoded separately as
detached targets in the dynamics loop. Frozen U parameters still transmit gradients
through imagined activations. Candidate branches never mutate the caller's state.

The retained temporal modules require equal-width input levels; `ProjectFeatures`
is an explicit adapter for other widths. Memory size, action width and predictor
depth are constructor arguments. Predictor residual output is initialized to zero,
so the first forward copies the current features and the first gradient primarily
reaches its output projection. Memory gradients appear after that path learns.

The dynamics recipe freezes E. Joint E/dynamics training needs a separately explicit
stable-target policy; it is not silently enabled by toggling a flag. Auxiliary
physical losses, conditioning and new temporal architectures belong to future
recipes with their own tests.

`choose_sequence` evaluates explicit candidate actions using a supplied cost. The
application owns action bounds, goals and terminal behavior. This is not a migrated
claim of successful PushT/Paddle closed-loop control.
