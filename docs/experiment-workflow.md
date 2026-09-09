# Experiment and development workflow

The user-facing entry point is a readable file in `experiments/`. Shared code is
in `pathwm/`. This workflow replaces the retired dated scripts and aggregate
report wrapper while preserving scientific integrity and the four-step process.

## Work in small complete slices

1. **Plan.** Update the active plan linked from project-state.md: concrete problem,
   affected interfaces, success criteria and compute budget. Prefer one real user
   path. Do not create another framework or speculative module catalog.
2. **Essential checks.** Write the few tests that catch meaningful failure. For new
   behavior, demonstrate an informative failure before implementation. Tests cover
   data alignment, frozen gradients/buffers, temporal causality, numeric references
   and resume where relevant. Commit the completed plan/check step; label red state.
3. **Implement.** Make the recipe work with a tiny real batch and short explicit
   development run. Verify its local report. A smoke run proves the workflow, not
   model quality. Commit the working slice.
4. **Compare.** Change one scientific factor per comparison with matched populations
   and budgets. Preserve reference settings/results. Broaden only after the first
   user path works. Commit completed steps and update project state.

A scientific experiment separately declares hypotheses, seeds, populations,
metrics, thresholds and budgets before execution. An unset threshold cannot pass.
Record deviations and stopped/failed runs; never choose gates after seeing results.
Do not add tests for prose or trivial glue, or conceal failures by deleting tests.

## Keep it understandable

Use ordinary Python/PyTorch modules, functions, explicit arguments and small data
records. Recipes show construction, data/splits, losses, freeze rules and budgets.
Keep model inputs/outputs documented; readers must be able to trace their tensors
and gradients. No module registry, base Experiment class, universal trainer,
recursive configuration hierarchy, import-time training/downloads or dated paths
inside reusable code. New abstractions require a concrete consumer.

Do not edit shared source during an active run; finish it or use an isolated copy.
Dependencies point from recipes to the library. New experiments may copy a short
recipe, not import another recipe. Helpers and model files cannot import the
historical package, old run manifests or reporting services. Dataset/weight paths
are explicit inputs. Copied upstream code retains its source, revision and license.

## Trustworthy runs and reports

Keep source data/assets under `data/`; each new run owns `runs/<name>/`. Never
overwrite an existing run. Explicit resume must check compatible code, modules,
data/splits, objective, optimizer and precision. Checkpoints include model buffers,
optimizer/scheduler, progress and RNG/sampler state. Document exact replay limits.

Record resolved settings, code identity, source and split identities, initialization,
seed, sample/update counts, precision and package/device information. Raw JSON and
JSONL are authoritative. Result completion must be written before report generation.
A reporting failure preserves raw results and has a distinct visible status.

Every completed run generates a self-contained local `report.html` using repository
code and declared dependencies. Include curves, exact metric values, representative
examples and source/settings context. Structural checks always run. Browser QA is
required for a new or changed report renderer; its receipt applies only to the
checked content/version. Disclose structural-only validation when browser QA is
unavailable. No AI-app or private plugin is needed to train or read a report.

Do not confuse reconstruction quality, latent loss, prediction and closed-loop
control. Avoid comparing absolute losses across differently scaled latent spaces.
Preserve negative results and distinguish observed results from explanations.

For consequential scientific choices use the adopted
[Claude review](claude-collaboration-workflow.md): independent criticism, verify
claims, reconcile disagreements, preserve receipts. Respect the current public-only
export boundary; no private repository content or measurements in external briefs.
Peer agreement cannot authorize scope changes or replace empirical checks.

## Finish a slice

Run the relevant CPU tests and the documented check/train/resume path. Update the
active plan and the short project-state page with what works, evidence and limits.
Report the usable commands and report location. Commit completed work. Do not let
new experiment history grow into the normal user interface again.
