# Executable modality capability suites, first slice

16 September. Implements the requested test maps through the existing
`experiments.modality_readout` recipe and portable report. No new trainer or model
architecture. This slice evaluates frozen checkpoints and fits diagnostic ridge
readers only; it does not train general language, speech or natural-video ability.

## Contract and fixed budget (before implementation)

- Add `--stage capabilities --core RUN`. Reuse actual Model loading, stage capture,
  train-only standardized/validation-selected probes, positive and shuffled-label
  probe controls. Existing `diagnose` behavior stays available.
- Run the existing four symbolic inputs separately, redundant-all and complementary
  conditions. Report seen versus withheld combinations independently, with actual
  state-to-factor predictions across three explicitly named categorical RNG draws.
  These draws are not independent trained-model seeds. Screens require at least
  0.8 accuracy for EACH factor and joint correctness in EVERY draw. These are small
  descriptive development gates, not statistical evidence of general understanding.
- Source-use controls must NOT use the withheld combination split: its color and
  location are dependent. Add an evaluation-only complete Cartesian intervention
  population (18 combinations,4 views) with balanced factors conditional on the
  others. Complementary image/audio/video omission tests require full-source factor
  accuracy>=0.8, omitted-factor accuracy<=chance+0.1 and drop>=0.2 in every draw.
  No claims of full-source necessity outside this controlled distribution.
- Diagnostics expose encoder, posterior, sampled code, observed and working state;
  record unequal probe dimensions, readout choices, factor-specific errors and
  failed examples. Never automatically identify erased information or a unique
  broken layer. Random-encoder comparison remains unmeasured, so accessibility is
  not attributed to representation learning.
- Broad text/image/audio/video tasks, long-memory, streaming, calibrated uncertainty,
  action consequences and general fusion remain visible unimplemented cases.
  Decoder quality stays a separate not-run case for this core-only evaluation.
  Static arrow orientation can solve video factors; no natural-motion pass follows.
- Missing/nonfinite metrics cannot pass. Keep implementation, execution and
  assessment separate. Failures must not disappear when summarizing coverage.
- Frozen source comparison: existing native cores from
  `runs/modality_readout_v1/formal/seed7201/core` and `seed7202/core`, each with its
  corresponding dataset seed. Existing inspected populations are development
  regression data, not untouched confirmation. One exact repeat of seed7201 for
  reproducibility; no selection among checkpoints and no additional efficacy fits.
- CPU2 threads, maximum180s per frozen run,<=200MiB new artifacts,>=500MiB free
  disk. Smoke uses an existing tiny checkpoint or a reduced mechanics test, no
  additional neural training. Ridge fitting is diagnostic and its cost is logged.
- Required checks: missing-score nonpromotion, input/target shape/class validation,
  all-factor intervention balance, visible gap coverage, raw score recomputation,
  exact repeated predictions/probes, frozen source/weights and RNG preservation,
  existing readout/core regressions. New report section requires browser QA.

## Sequence

1. Red tests for result honesty and intervention data; commit plan/tests.
2. Small evaluation functions, recipe branch and existing-report integration.
3. Focused tests and real frozen-checkpoint execution; fix implementation failures.
4. Two fixed checkpoint evaluations plus exact repeat, raw audit and report QA.
5. Save scoped results, limitations and next diagnosed gap; no model promotion.

## Claude review

Actual Claude reviewed a generic hypothetical methodology brief with no repository
content or measurements. Accepted: explicit symbolic scope, selection/evaluation
separation, sampling-versus-model uncertainty distinction and visible missing cases.
Its suggestion to defer observing other modalities until one passes is not adopted:
this suite must expose failures in all requested branches. The dependent-combination
omission pitfall is addressed with the separate Cartesian intervention population.
Reconciliation and remaining issues are recorded after its response.

Reconciliation received: Claude accepts evaluating all branches while retaining
their separate failures. Exact Cartesian enumeration and conditional-count tests
establish the control balance. Missing versus unrun states are fixed by the catalog,
not decided after scores. No split-resampling uncertainty is claimed. Its claim
that accessibility is wholly uninterpretable without a random encoder is too
strong: recoverability under this reader is measured, while learning benefit is
not identified. No private material was exported; two compact CLI receipts retained.

## Implementation preflight

Seven new checks plus26 existing capability/readout checks pass. The red test found
an existing bug: `case_record` could pass a nonempty gate set containing an unmeasured
required value. It now reports `not_measured` unless an actual failure takes priority.
The real frozen two-update source smoke completed in4.82s diagnostic compute;
all15 scoped screens failed,14 broader tests are unimplemented and4 output tests
unrun, as appropriate for this source. Positive probe control100%, source unchanged.

Browser QA was attempted through the supported browser interface. Its URL policy
blocked the local report and explicitly forbade workarounds. Preserve structural
report validation and inspect saved plots; browser-interaction verification remains
blocked, not passed. This does not block local model evaluation.

## Completed baseline evaluations

Both prescribed original native core checkpoints completed; seed7201 was repeated
exactly. Each report has33 catalog entries:1 passing scoped case,14 measured failed
cases,14 broader tests not implemented and4 decoder cases not run on these sources.
The passing case is image-source color dependence on the Cartesian intervention
population. All12 complete factor/joint screens and both other source controls fail.
These are the original saved reference models, not a claim to have re-evaluated
every later repair or found the best current checkpoint.

Five-stage diagnosis makes the failure actionable without over-attribution. For
held-out audio inputs, all three factors are100% readable from encoder features in
both source models. At the posterior, location becomes0%/8.33% and direction
62.5%/43.75%; working-state probes give location0% in both and direction45.83%/52.08%.
The primary head also fails these tasks. Shared-core paths need investigation;
this does not identify an irreversible loss or justify replacing the encoder.
Probe dimensions differ, and their one fixed capture uses different categorical
RNG draws from the three-draw deployed-head screen. Sampling and probe fit therefore
remain possible contributors to discrepancies; matched interventions are required.

Formal diagnostic times4.78s/4.81s, repeat5.00s; smoke4.82s. No neural weight updates.
Each run fits61 ridge readers with185 alpha solves, using training/validation only.
Saved source/model weights match exactly. Full run/probe/output/data replay passes
15005 checks; total independent raw audit27583 checks.124 comparisons confirm the
original four datasets did not change when the intervention split was added.
80 focused tests pass. Four standalone reports structurally verified; saved stage
plots visually inspected. Browser interaction remains blocked. Artifacts111.4MiB,
below the200MiB budget; no historical data removed.

CLI (use a new output directory):

```bash
.venv/bin/python -m experiments.modality_readout \
  --stage capabilities --device cpu --seed 7201 \
  --core runs/modality_readout_v1/formal/seed7201/core \
  --output runs/my_modality_capabilities
```

This stage performs evaluation and diagnostic fitting, not neural training. The
older `--stage suite` still runs the distinct output-adapter training comparison;
it is not the new capability command. Evaluation does not support resume: exact
repetition uses a fresh output. Source compatibility is checked by strict loading.

Artifacts: `capability_suite.json` contains cases/scopes/gates/coverage and diagnostic
hints; `capability_predictions.npz` contains all actual logits, targets, IDs and
draws; `stage_probes.npz` stores stage features and fitted readers;
`capability_inputs.npz` and `capability_examples.json` preserve input/failed examples;
`stage_probe.json` stores selection/shuffled/positive controls. Ordinary Run records,
source snapshots, checkpoint and `report.html` retain provenance. Independent
verification: `runs/modality_suites_v1/verification.json` and `data_regression.json`.

Next work is to add selected real-input capability cases and evaluate later chosen
checkpoints with this fixed contract, then isolate one failing shared-core transition
with matched controls. Do not silently convert the catalog's unimplemented natural
tasks into passes from these symbolic measurements. The current natural-video goal
and earlier codec tasks remain open.
