# Handoff: shared visual encoder and residual task adapters

Prepared 17 September 2026. Repository: `/home/alex/Documents/path-wm`, branch
`main`. This is a documentation handoff; no new task/session or training job was
created. The design is [visual-adapter-design.md](visual-adapter-design.md).

## What Alex wants

A scalable, compact image codec for large/variable image sizes, useful world-model
features and high-quality application/debugging outputs. Compare both training
without pretrained teachers and training with them. There is no fixed parameter
ceiling: as low as useful, as high as necessary.

The latest proposal is **one shared encoder plus small residual corrections per
application**, compared with the encoder frozen and trainable. Alex asked for
this design document and a handoff to continue in another session. No dataset,
task pair, checkpoint, run budget or implementation has been selected by that
request. Do not interpret the earlier “Ok” as empirical validation.

## Read first

1. [Repository instructions](../CLAUDE.md) and [experiment workflow](experiment-workflow.md).
2. [Design](visual-adapter-design.md), especially phases A/B and the pending run contract.
3. [Codec literature review](visual-codec-review.md), its application matrix and
   [parameter audit](visual-codec-parameter-audit.json).
4. [Current state](project-state.md), [v2 findings](spatial-vae-v2-plan.md) and
   [video-codec findings](shared-video-vae-plan.md) for actual limitations.
5. [Claude review rules](claude-collaboration-workflow.md) before final scientific choices.

## What exists, and what does not

- Existing: `pathwm/models/spatial_vae.py`, `spatial_vae_v2.py`, `video_vae.py` and
  the shared v1/v2 recipe `experiments/spatial_vae.py`.
- Existing v2 C specimen: 122,979 encoder+decoder parameters, four latent channels,
  stride four. Counts and small shape checks are not fidelity evidence.
- Existing codec API: `encode(x)` → posterior with mean, variance and geometry;
  `decode(z, output_size)` → image. V2 trace snapshots are detached diagnostics.
- Proposed only: application residual adapter, task-head comparison recipe,
  shared multi-task training and connection of this optional codec to the
  categorical agent's current visual path. Future filenames in the design are
  not runnable commands.
- Existing video temporal refinement is a different adapter use case. Preserve
  its recorded failures/limits; do not mix it into the initial image comparison.

Relevant discussion commits before this handoff: `c0b34bb` broad codec review,
`4578f15` cross-task representation sources, `473b597` residual-adapter proposal.
Inspect `git status` and recent commits at resume; the handoff/design commit will
be later than those references.

## First work in the next session

1. Check local changes and running work. Read the design before editing shared
   code. Preserve current checkpoints, datasets and completed reports.
2. Audit available aligned labels and source-disjoint splits. Choose two useful
   tasks plus RGB retention only when real data supports them; label synthetic
   mechanics evidence honestly. Identify and hash a viable source checkpoint.
3. Fill the design's run contract: metric/gain and retention gates, head/adapter
   capacity, exact seeds, optimizer/schedules, anchor and compute/memory/disk caps.
   Keep missing decisions visible; a missing gate cannot pass. If baseline codec
   learning is inadequate, make its repair a separate stage. Formal Phase A needs
   at least three paired adaptation seeds; before a residual-placement claim,
   include the matched-parameter expanded-head control. Use online paired
   augmentations and the declared checkpoint-selection rule.
4. Reconcile material changes with actual Claude using only a generic public
   methods brief. Existing design-review receipts are in
   `runs/reviews/visual-adapter-design-20260917/`. Do not export private code,
   data, checkpoints, measurements or the full handoff. The remaining review
   disagreement is whether the first diagnostic needs a second anchor strength;
   resolve its budget/scope in the protocol. One-anchor results must remain
   conditional; fixed-budget rankings also include convergence-rate differences.
5. Implement one bounded Phase A path and essential failing contract checks,
   reusing the library, existing recipe conventions and report helpers. Record
   a short development run separately from a scientific comparison. Run the
   finalized comparison only within its declared scope/budget.
6. Preserve every arm's metrics/checkpoint/report. Move to the shared Phase B
   only after evaluating Phase A; adapter benefit and multitask sharing benefit
   are different questions.

## Interpretation traps to preserve

- Phase A's unfrozen per-task copies are specialization references, not a single
  shared deployed encoder. Phase B must train/evaluate one common encoder across tasks.
- Final-latent corrections permit one encoder pass; interleaved corrections
  generally require separate downstream passes. Count actual deployment cost.
- The initial mean-only diagnostic freezes the variance head/base RGB decoder;
  feature updates can still affect variance outputs. It validates no sampled
  posterior or KL claim. Keep later stochastic-codec tuning separate.
- All four arms train identical task heads. Frozen means weights and buffers.
  Preserve the declared common-latent reconstruction anchor and frozen-consumer
  checks; head retraining can hide latent drift.
- A final adapter cannot recover discarded evidence. Improvements from unfreezing
  alone do not prove irreversible loss; optimization/capacity are alternatives.
- Keep teacher tracks separate. Teacher agreement, residual norms, CKA and pretty
  reconstructions do not certify geometry, semantics or causal world-model utility.

## Ready-to-paste continuation prompt

> Continue the shared visual encoder / per-application residual adapter work in
> `/home/alex/Documents/path-wm`. Read `CLAUDE.md`,
> `docs/visual-adapter-handoff.md` and `docs/visual-adapter-design.md` first.
> Start with a source-checkpoint and labeled-data audit, then make Phase A's
> frozen/trainable × adapter/no-adapter comparison concrete. Preserve the existing
> codec and distinguish separate task-specific fine-tuning from shared multitask
> training. Keep teacher-free and teacher-assisted tracks separate. Follow the
> repository's small-slice workflow and public-only Claude methodology review;
> record metrics, thresholds, preservation checks and a bounded compute budget
> before fitting. Do not restart the literature survey or treat proposed modules
> and untrained parameter counts as existing validated capabilities.

## Completion at this handoff

Design and continuation documents created; links and repository interfaces checked.
Documentation/atlas notes only: no adapter implementation, model/default/checkpoint
change or new training. Current scientific questions remain open.
