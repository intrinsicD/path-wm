# Modular restart

Authorized 9 September 2026: keep useful components, migrate the workflow, remove
obsolete active code, and make experimentation possible without an agent.

Reference: Git tag `archive/pre-modular-2026-09-09`, commit
`e95b6a6252cae72402de0dd93f419e8e93d25d12`. Original data and completed runs remain
in place. Recover historical source with `git archive` into a separate directory;
the new package must never import it. This is a refactor, not a model-quality claim.

## Keep

- CNN reference including optional branch residual blocks and cross-scale exchange.
- Explicit local DINOv2 weights/source; native contextual and local feature outputs.
- Independent RGB, dense mask and spatial PushT pose heads.
- Attention/GRU observation memory and action-conditioned residual prediction;
  causal observed/imagined recurrence and finite-candidate planning checks.
- Prepared COCO/PushT data and their existing split/label provenance.
- Scientific integrity: source identities, freeze/gradient checks, raw metrics,
  full-state resume, verified local reports, scoped conclusions and Claude criticism.

## Remove from the active tree after verification

Historical experiment coordinators, duplicate trainers, fixed-date deadlines,
old configs/protocols, task-specific CLIs, aggregate/plugin report machinery,
unused LeWM/SWM vendoring and old tests for retired systems. Git preserves these;
no dataset, completed checkpoint or result directory is deleted. Agent research
records remain outside the user entry path.

## Delivery sequence and acceptance

1. Adapt standing instructions and commit the plan plus essential failing tests.
2. Extract the CNN and heads. On identical CPU FP32 inputs/state, compare forward
   outputs, loss, gradients and three AdamW updates to the preserved source.
   Predeclared tolerance: atol 2e-6, rtol 2e-5. Compare baseline and depth=2.
3. Deliver editable perception recipe with real PushT data, optional COCO and local
   DINO; single-batch check, training, pause/resume and standalone HTML. A short
   development run validates plumbing only. Resume parity uses deterministic CPU,
   direct batch sampling, workers=0, no mixed precision: exact tensor equality.
4. Port causal memory/prediction, verify original FP32 forward/rollout parity with
   the same tolerances, then deliver a short sequence-training recipe using the
   same run utilities. Test timing, future-input independence and gradients through
   frozen modules. Keep planner goal cost explicit, without claiming old controller
   scores are reproduced by the new general candidate scorer.
5. Remove obsolete active files, simplify packaging/docs, run the new CPU suite,
   install/import check, real recipe checks and browser QA of local reports. Confirm
   the new package works without the old source available on its import path.

One current plan, one short project-state page, one model guide and one experiment
guide. No experiment registries, universal trainer, recursive config trees, dated
constants in reusable code, or new abstraction without a concrete consumer.

## Verification record

The working library/recipes were committed at `74270c8`. The retired active tree
is now removed; root data-ignore patterns are anchored so the new `pathwm/data`
package is included in Git. The active interface has 18 library Python files,
two recipes and four focused test modules (16 tests), rather than the prior
experiment-specific source/script stack.

- Reference comparison: 500 CPU FP32 checks of CNN depths 0/2, RGB, loss,
  gradients, three AdamW updates, dense/pose outputs, memory and three-step rollout;
  maximum observed difference 0. DINO local/final/pooled features and separate
  local/context RGB/mask decoding: five more checks, maximum difference 0.
- Tests cover explicit feature selection, diagnostic gradient isolation, causal
  action timing and future-target independence, frozen weights/buffers, finite
  candidate scoring, same-runtime full/resumed equality for both loops, masked
  metric denominators, safe component loading, source snapshots, overwrite/resume
  refusal, interrupted checkpoint writes and separate reporting failures.
- Perception checkpoints can initialize the next dynamics experiment directly.
  Actual short CPU perception/dynamics runs and GPU ViT/COCO runs have trained,
  paused and resumed. GPU access required host execution; the sandbox could not
  expose CUDA. This was an execution-environment issue, not a training result.
- Installed-package import works from outside the repository; the old world_model
  package is unavailable. A wheel contains only pathwm and distribution metadata.
  A final clean-Git snapshot check follows the retirement commit.

Raw migration receipts, reference comparison programs, three actual public-only
Claude reviews and browser QA program are preserved under
`runs/modular_migration_2026-09-09/`. Claude acknowledged the optimizer grad=None
correction and withdrew claims that area-weighted metrics or cross-device resume
were mandatory. Per-image metrics and strict same-runtime resume are explicit
contracts; source/package inventories are not a complete environment recreation.

These are scoped engineering checks. No new architecture superiority, recovered
controller quality or completed long training curriculum is claimed. Existing
scientific results belong to the preserved historical source and run records.

## Recover historical source

Use a new empty directory:

```bash
mkdir /tmp/pathwm-history
git archive archive/pre-modular-2026-09-09 | tar -x -C /tmp/pathwm-history
```

Historical documentation paths referenced by agent research records resolve against
that Git reference. Do not reintroduce the retired package as a new-library dependency.
