# Handoff: jointly trained visual base and per-application residuals

Updated 18 September 2026, following `2beb97f` and `95a239a` in `/home/alex/Documents/path-wm`,
branch `main`. The repeated user hash `2beb97f2beb97f` resolved to that existing
commit, which was HEAD on resume. This is a design discussion, not a training run.
Main document: [visual-adapter-design.md](visual-adapter-design.md).

## Current continuation: modality input hierarchy

Read [multiscale-modality-design.md](multiscale-modality-design.md) first. Alex
specifies information-preserving operations → **learnable filter bank** →
post-processing → compression per scale, repeated a few times, then a multiscale
representation shared with arbitrary downstream modules. Every module uses a
transformer loop followed by its own layers. Filters are explicitly learnable;
do not substitute a fixed filter bank or persistent feature-slot memory.

The design recommends pre-compression scale exports and documents strict versus
measured preservation, modality/time/geometry metadata and consumer loop choices.
Those details remain proposals where the user has not selected them. Joint/equal
training remains compatible; a separate residual per application is no longer
the starting requirement for describing the input path. Readout/PCA options below
are background. No new model or training run was created.

## Latest user direction — takes precedence over the old phase order

Alex wants to try **one shared trainable base plus residuals per application,
trained jointly from the start, treating all applications equally**. The main
open question is **feature residuals at each scale, weight residuals at each
scale, or both**. The expectation that the base learns common features and the
branches learn missing task-specific parts is a research hypothesis.

The previous handoff required frozen/unfrozen separate-task Phase A before joint
Phase B. That order is superseded. Phase A remains an optional diagnostic and its
frozen RGB anchor is not an extra loss in the equal-status joint study.

## Latest open alternative: specialist decomposition, 18 September

Alex asks whether residuals can correct wrong features and grow arbitrarily large,
and whether separate equal-architecture application VAEs could be decomposed with
PCA to obtain a frozen shared base plus set/learned residuals. See §10 of the design.
This question does not replace the earlier joint-training preference or select a run.

An additive branch can cancel/replace features if its function class and available
information suffice; reliable fitting/generalization is not guaranteed. The current
proposal has no hard magnitude bound. Rank, norm, capacity and information access
are distinct. Separate specialists are expensive references, not a guaranteed
quality upper bound; application-specific objectives/heads must be defined.

Centered PCA across compatible, aligned checkpoint vectors describes their mean
plus between-model variation. Largest components are not automatically common
features or a working VAE. Alignment must preserve each encoder/decoder interface.
With two checkpoints, one centered component fits both exactly regardless of useful
sharing. Cross-task PCA rank differs from per-layer weight-matrix rank. Count the
mean, basis, coefficients, private heads, buffers and materialized weights.

Full weight deltas `theta_t-theta0` can recover each compatible specialist exactly
around any frozen base, without PCA, but largely retain specialist storage/compute.
Small deltas require approximation and task-quality checks. Feature differences
are input-dependent functions that must be learned; a frozen information-poor base
may make them impossible to recover. Aligned PCA/SVD initialization and equal-task
specialist distillation remain possible comparison routes, not adopted winners.

## Current recommendation and its limits

- First hypothesis: one shared feature pyramid, with small nonlinear feature
  residuals at declared scales feeding each task head. The corrected features do
  not feed the next shared encoder stage. All modules learn jointly. This permits
  one base pass for multiple applications on the same input; it is an engineering
  preference, not a measured winner or an additional user constraint.
- Interleaved feature residuals and weight deltas (e.g. low rank) can alter later
  processing, usually requiring task-specific downstream activations/compute.
  Small private parameter storage does not imply one shared forward pass.
- At one matched linear operator/input, `(W+UV)x = Wx+U(Vx)`. Nonlinearity, spatial
  context, normalization and insertion before/after compression distinguish the
  actual mechanisms. Neither residual type universally dominates the other.
- Start without a hybrid. Compare J0 shared+heads, JF feature residuals+same heads,
  and JH matched expanded heads with identical taps/strides. Then investigate
  selected interleaved/weight changes and hybrids only for a concrete reason.
- Fine/middle/coarse allocation depends on labels and failures. RGB detail and
  small text motivate fine evidence; segmentation needs context and boundaries;
  geometry needs local cues and suitable context. These are placement hypotheses.
- A feature pyramid supplies more information than the compact final latent.
  Count bytes, activations and head/projection capacity. Observed-image output
  quality does not validate generation from a compact predicted world-model state.

## Equal-status training contract

Each logical update averages an equal example budget of task-normalized losses,
with fixed positive calibration scales and one optimizer step. Task-specific
batches can accumulate at the same parameter snapshot. Aligned tasks can share an
image forward pass; different images/corruptions cannot be assumed to do so.
Missing labels use valid masks and declared denominators; no fabricated targets or
silent weighting by label availability. RGB, if selected, is one equally weighted
application with a trained head, not a privileged frozen-decoder anchor.

Keep shared normalization/buffers explicit. Log per-task raw/scaled loss and shared
base gradient norms/conflict. Equal objective weight/exposure does not guarantee
equal influence or outcome. Small private capacity encourages sharing but cannot
identify common versus private semantics. Adapter-off is supplementary reliance
evidence; frozen-feature probes test accessibility under their own capacity limit.
Evaluate every task using the same final common checkpoint and all-task floors.

## Read first and local implementation boundary

Read `CLAUDE.md`, [workflow](experiment-workflow.md), the [design](visual-adapter-design.md),
[current state](project-state.md), and [Claude rules](claude-collaboration-workflow.md).
The [codec review](visual-codec-review.md) and its parameter audit remain background;
do not restart the broad literature review. Existing small-codec quality limits
remain in [v2 findings](spatial-vae-v2-plan.md) and [video findings](shared-video-vae-plan.md).

Existing code: `pathwm/models/spatial_vae.py`, `spatial_vae_v2.py`, `video_vae.py`,
and the shared `experiments/spatial_vae.py` recipe. The v2 C specimen has122,979
encoder+decoder parameters; this is a count, not fidelity evidence. Current
`encode(x)` returns a posterior and geometry; v2 trace snapshots are detached.
Joint multiscale branches need explicit differentiable taps, not those traces.
The optional codec is not already the categorical agent's visual path.

No adapter module or joint comparison recipe exists yet. Future filenames in the
design are proposals. Existing image/video runs, checkpoints and data are preserved.
No training/evaluation process was running at the initial resume inspection.

## Next bounded implementation work

1. Inspect status/running work. Audit actual aligned labels and source-disjoint
   splits; choose a small useful task set and random or named checkpoint start.
   Do not invent depth/semantic labels from raw photos. Keep teacher tracks separate.
2. Fill §7's joint run contract: exact taps/strides, private capacity/head controls,
   losses/masks/calibration, seeds, task exposure, initialization, optimizer groups,
   meaningful gains/all-task floors, and compute/memory/disk caps. Preserve the
   distinction between equal exposure and equal compute. Three paired seeds before
   confirmatory architecture claims; one tiny run proves mechanics only.
3. Implement one readable recipe and small ordinary modules with essential checks:
   one base instance/pass per same-image multihead call; correct private routing;
   differentiable taps; zero correction and later learnability; geometry/masks;
   correct averaging/optimizer step; reproducible checkpoint/resume.
4. Preserve per-arm raw metrics, checkpoints and verified standalone reports. Use
   the same final shared checkpoint for all tasks. Do not require the obsolete
   separate-task Phase A before this joint path.

## Claude review record

The earlier design review is in `runs/reviews/visual-adapter-design-20260917/`.
This continuation used actual Claude with tools disabled and a generic public-only
brief, followed by explicit corrections: `runs/reviews/visual-residual-joint-20260917/`.
Private code/data/measurements and full documents were not exported. The reviewer
withdrew several categorical claims; see §9 of the design for the exact material
corrections and remaining scope differences. In particular, development placement
search is allowed with separate fixed confirmation, and equal task priority can
coexist with different input/compute costs. Peer agreement is not validation.

## Ready-to-paste continuation

> Continue from the updated visual-adapter design/handoff. Alex chose joint training
> of one shared trainable base plus equally prioritized application residuals from
> the first update. Feature versus weight residual placement remains an experiment.
> Audit labels/data and fill the joint J0/JF/JH contract; do not impose the old
> frozen/unfrozen Phase A as a prerequisite. Keep taps and information budgets
> explicit, RGB equally weighted, private branches small and per-task outcomes
> visible. Use actual Claude only for bounded public-methods review. Preserve the
> existing codec, checkpoints and reports; no broader capability has been validated.
