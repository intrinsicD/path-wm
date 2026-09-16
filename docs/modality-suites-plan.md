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
