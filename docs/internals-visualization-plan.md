# Readable learning curves and model-internals instrumentation

Authorized on 2026-09-06 after a literature survey (see the references below).
This is development instrumentation, not a frozen experiment: no training is
repeated, no checkpoint is modified and no new success threshold is introduced.

## Problem

The offline dashboard shows total, prediction and SIGReg losses on one linear
axis, so the prediction term is flat; validation errors are dominated by the
growing latent scale; gradient norm, learning rate, rollout error and the
action-effect term are logged but never charted. Nothing about the encoder,
predictor, action encoder or projectors is recorded, so a falling loss cannot
be related to representation health or to how the predictor uses actions.

## Hypothesis and questions

1. Scale-free ratios (prediction error over copy, shuffled-action and zero-action
   controls) make the recorded validation evidence interpretable without
   comparing absolute errors across latent spaces.
2. Standard representation diagnostics (covariance spectrum, effective rank,
   Gaussianity), predictor diagnostics (AdaLN gate magnitude, action versus
   state sensitivity, attention entropy) and a latent-to-state linear probe,
   measured on the saved pilot checkpoints and the released weights, will show
   where the pilot differs from a working model. This is descriptive evidence
   for the next decision, not a causal attribution.

## Slices

1. **Readable curves** (`viewer/experiment_results.sql`, `viewer/dashboard.py`):
   one panel per training scalar (prediction loss, SIGReg, total, gradient norm,
   learning rate); a validation ratio chart with a reference line at 1.0 and the
   action-effect share; rollout error added to the absolute chart. Same ledgers,
   no new evidence.
2. **Checkpoint internals** (`world_model/introspection.py`,
   `scripts/inspect_checkpoint.py`, `viewer/ledger.py`, `viewer/dashboard.py`):
   inspect saved checkpoints in eval mode on the pilot's fixed held-out
   validation windows and write `internals.json`, `manifest.json` and PNG panels
   under `runs/diagnostics/pusht_internals/<label>/`. The dashboard indexes the
   scalars over checkpoint step, the eigenvalue spectrum, error versus rollout
   horizon and the panels (embedded images inside `html` blocks).
3. **Training-time capture** (`world_model/train.py`, opt-in `introspect_every`):
   append the cheap scalar subset as `kind: internals` rows to `metrics.jsonl`
   at validation steps so future runs carry the same signals. Off by default;
   existing configurations and checkpoints are unaffected.

## Measurements

| Group | Measurement | Reading |
|---|---|---|
| Representation | covariance eigenvalues (log), effective rank, RankMe, participation ratio, per-dimension std | collapse or anisotropy; SIGReg targets an isotropic Gaussian |
| Representation | per-dimension Shapiro–Wilk W, skewness, excess kurtosis, Q–Q quantiles | how Gaussian the embeddings are |
| Representation | ridge linear probe from latent to agent position, block position, block angle (sin/cos) and agent velocity; held-out R² | whether physical state is linearly readable |
| Encoder | last-layer CLS-to-patch attention per head, PCA of patch tokens as RGB, attention entropy per layer | what the encoder looks at on real PushT frames |
| Predictor | causal attention over the three-frame history per layer/head, entropy; AdaLN gate magnitudes per block; action embedding norm | how much history and action conditioning are used |
| Predictor | mean directional derivative norm of the prediction with respect to actions and to the state input | action sensitivity relative to state sensitivity |
| Dynamics | rollout error versus horizon against encoded observations, with copy error; nearest-neighbour frames for predicted latents | multi-step drift, qualitative sanity without a decoder |
| Optimization | parameter L2 norm per module group; gradient norm per module group on one validation batch in eval mode | which parts of the model the objective actually moves |

Gradient norms are measured without an optimizer step, in eval mode so that
BatchNorm buffers stay untouched; gradients are cleared afterwards and the
checkpoint hash is verified after inspection.

## Essential checks

Effective rank and RankMe on isotropic versus rank-one synthetic data; normalized
attention entropy of uniform versus one-hot attention; recomputed predictor
attention is causal and row-normalized; AdaLN gates are exactly zero at
initialization; the ridge probe recovers an exact linear map; directional
sensitivity matches finite differences; inspection leaves the state dict and
buffers unchanged; ledger rejects an `internals.json` with a missing panel; the
dashboard artifact emits per-scalar training datasets, the validation ratio
dataset and embedded panel blocks.

## Budget and success measure

Five checkpoints (pilot 0/250/500/1000 and released weights) on 512 held-out
windows, plus 512 training windows for the probe; minutes on the local GPU.
Success means the refreshed dashboard passes browser verification and answers,
from exact values, whether the pilot's representation is collapsed, whether its
predictor uses actions, and how its rollout error grows with horizon, each next
to the released weights measured the same way.

## References

LeJEPA (arXiv 2511.08544): eigenvalue spectra, Gaussianity tests, Q–Q plots and
the loss-plane view. RankMe (Garrido et al. 2023): effective rank as a training
monitor. DINO (Caron et al. 2021): CLS-to-patch attention maps. Zhai et al.
2023 (arXiv 2303.06296): attention entropy collapse. DINO-WM (arXiv 2411.04983):
separate decoder used only for visualization; not adopted here. Per-layer
gradient/weight/update monitoring as in TensorBoard, Weights & Biases `watch`
and TorchExplorer; implemented with local hooks and ledgers instead.
