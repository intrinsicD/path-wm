# Retaining old outputs during grounded adaptation

16 September 2026. Suite-directed continuation of
[grounded readout](grounded-readout-plan.md). The previous task fit could learn
video order but regressed old outputs in both initializations. Test a training-only
anchor to the untouched source model before another architectural change.

## Preregistered comparison

- Two original joint-native checkpoints, seeds7201/7202. Four fresh fits: existing
  core+text-decoder QA/replay control and the same fit plus source-output
  distillation. Encoders and nontext decoder weights remain frozen. Architecture,
  inference, source weights and existing defaults remain unchanged.
- Reuse the previous16 full-profile VID.order calibration training examples and
  original symbolic TRAIN population.768 updates:384 QA and384 replay, batch8,
  Adam0.001, gradient clipping5, identical initialization/minibatch order per pair.
  No validation/test-label training, checkpoint selection or coefficient sweep.
- Candidate replay loss = existing supervised replay +10 * teacher discrepancy.
  QA loss is unchanged. The fixed10 is an engineering regularization choice, not
  an optimized/universal value. Retain this limitation if the attempt fails.
- Teacher is a frozen source copy reconstructed before student resume. On replay
  inputs only, average text KL on ground-truth and teacher greedy prefixes,
  normalized by log(vocabulary). Include first EOS, exclude padding and later
  continuation. Add per-modality image/audio/video MSE against the teacher,
  divided by the existing target-variance normalizers. Log all terms separately.
- Preserve RNG around teacher evaluation, then run the student with the same
  underlying random noise. Equal categorical noise does NOT imply equal sampled
  categories after distributions change. Verify equality at identical weights.
  Teacher forward must not alter the student's RNG, sampler, weights or buffers.
- Log weighted anchor/replay gradient norms on first replay and every128 updates;
  save teacher/frozen checks and training overhead. Inference parameter count must
  remain unchanged. Teacher errors may be reinforced; finite soft regularization
  does not impose a mathematical ceiling on student accuracy.
- Evaluate final checkpoints on all25 quick tasks and full-profile VID.order,
  seed9401 and the same three draws. Source/empty/last-frame controls, separate
  free generation and symbolic seen/heldout output cells remain mandatory.
  Compare per-example suite wins/losses as well as aggregate deltas.
- Adoption gates unchanged: target passes both source seeds in both profiles;
  no other quick task loses a pass or accuracy/pair/source gain>10pp; each old
  symbolic cell loses at most5pp in any factor/joint accuracy and increases
  CE/MSE at most10%; last-frame-only<=60%. A retention-only gain is partial,
  not an adopted repair. Compare paired controls and each untouched source.
- Budget: four fits <=600s each, each evaluation<=600s, peak allocation<6GiB;
  <=550MiB new formal artifacts, >=500MiB free disk. Restart smoke<=8 updates.
  No downloads. One fixed comparison; failures lead to saved diagnostics, not
  an unregistered search. Later per-modality ablations require a separate plan.

This identifies the effect of adding the specified regularizer at equal optimizer
updates. It does not establish superiority to stronger supervised replay or at
equal training compute. Log that extra cost. Two seeds and a repeatedly used
development suite cannot establish general understanding or retention outside the
replay/evaluation distributions.

## Implementation and review

Reuse `experiments/modality_readout.py`, Run, existing losses, suite and reports.
First commit failing checks for masked KL, paired random noise, no teacher
gradients, frozen encoder boundaries and zero discrepancy at identical weights.
Implement an opt-in nonnegative `--retention-weight` (default0). Verify exact
restart and unchanged default/control trajectory, then freeze code during fits.
Actual Claude critique/reconciliation is stored in
`runs/reviews/grounded_retention_v1/`; public methodology only, no private code,
data or measured results exported. Its requests for separate prefix losses,
gradient diagnostics and per-example regressions are included. Its proposed
compute-matched replay control is deferred with the narrower claim above.

Claude explicitly accepted both corrections (same noise is not same category;
finite distillation imposes no hard accuracy ceiling), confirmed no methodological
blocker for this scope, and requested symmetric reporting of failed/positive
results and actual overhead. Both are included. Local numeric/gradient checks
and the focused regression set pass. A4-update GPU continuous run and2+2 resume
match exactly across1180 tensor checks, including optimizer/RNG/sampler, with
identical logged loss rows. The teacher is reconstructed from the original source
before loading resumed student weights. Smoke reports are structurally verified.
