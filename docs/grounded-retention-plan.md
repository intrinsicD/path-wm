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

## Exploratory failure localization (no training or gate changes)

After the first retention fit, saved free answers include an old multi-word
symbolic response instead of the requested single word. For all four final fits,
separate the existing candidate score into content-byte likelihood and EOS
likelihood, with the same three evaluation draws. Also report first generated word
accuracy from the already saved free responses. This distinguishes a possible
answer-content failure from an answer-format/termination failure. These are
post-hoc diagnostics: neither dropping EOS nor accepting a first word can convert
a failed task into a pass. No altered scorer is adopted, and no follow-up fit is
chosen from these test labels.

## Results and decision

All four fresh fits and eight evaluations completed. Controls exactly reproduce
the previous core+decoder fit's weights and every training-loss row for each seed.
No source checkpoints, encoder/nontext-decoder weights or teacher weights changed.

| Source / arm | Larger target accuracy, worst draw | Both pair members, worst draw | Free exact answer | Old-output cells failing /72 | Quick passes /25 |
|---|---:|---:|---:|---:|---:|
|7201 control|50%|0%|8/16|32|0|
|7201 retention|50%|0%|0/16|20|0|
|7202 control|100%|100%|14/16|23|1|
|7202 retention|93.75%|87.5%|6/16|6|1|

**Partial retention improvement, no adoption.** Both seeds reduce old-output
regressions, but neither meets all preservation gates. Worst old factor losses
shrink47.9→16.7pp and66.7→25pp; worst CE/MSE ratios shrink1.69→1.22 and2.24→1.15.
Source7202 retention preserves every image/audio cell within the registered
tolerances, while three text and three video cells still fail. Its VID.order
answer/pair gates pass in both profiles; source/empty/last-frame controls remain50%.
The result does not replicate in source7201, which stays at chance.

The broader suite also matters: source7202 retention regresses four real agreement
families (text+image, plus audio, plus video, and all four modalities), by16.7–33.3pp
in worst accuracy/source gain relative to its source. These are different failures
from the control's four single-scene source-gain regressions. Across the quick
suite's paired example/draw comparisons, retention gains91 and loses96 correct
answers versus control in7201; in7202 it gains81 and loses87. These units reuse
examples across draws/tasks and are not independent statistical observations.
Fewer failing old symbolic cells is therefore not overall multimodal improvement.

### What the diagnostic localizes

Source7202's first generated word is correct14/16 in BOTH arms; eight otherwise
correct retention answers append old location/direction text instead of stopping.
That accounts for its exact-generation fall from14/16 to6/16. Content-only choice
accuracy is93.75/100/100%; standard EOS-inclusive scores are checked against the
saved suite arrays before decomposition. This supports an answer-format/termination
problem in addition to remaining content errors. Do not shorten output artificially
or drop EOS to declare success. Source7201 still has only50% correct first words;
its failure cannot be explained just by termination.

Full-target probes for retention: encoder100% in both sources; posterior68.75/100%,
working56.25/93.75%. These are fitted diagnostics, with the already disclosed target
calibration overlap, not proof of irreversible information loss. No evidence here
justifies enlarging the modality encoders.

**Next bounded target:** task-dependent content selection and answer format from
the shared latent state, with same-scene/different-request contrasts. First audit
the existing TaskInterpreter/request path: this small recipe currently supplies
the question as an observation and calls `think` without a separate task goal;
it does not instantiate the library's optional task interpreter/metadata encoder.
This is a concrete integration gap to test, not proof that adding the module will
solve either seed. Keep old-output and all25-task regressions mandatory. Stable
temporal access in the weaker core remains a separate open problem. No extra fit,
new inference architecture or task-specific gate was introduced this turn.

### Verification, cost and use

64 focused software checks pass. GPU identical-weight discrepancy is about-1.2e-10
(floating-point KL roundoff), with unchanged CUDA RNG. Restart is exact over1180
tensor checks. The independent audit recomputes gates, source omissions, frozen
weights, source identities and individual regressions:4951 checks across15 reports;
saved report media is decoded and verified. Browser QA remains unavailable under the local-file policy;
structural/media verification and direct chart inspection are the reported scope.

Four training loops total329.27s; all12 formal commands take716.85s. The retention
loops take99.21/98.07s against57.10/74.90s controls: about1.49× combined training
time at equal updates. Max training allocation89.79MiB (not total process VRAM).
297,013 inference parameters and133,896 trainable parameters are unchanged; only
training retains an extra frozen copy. Gradient checks show weighted anchor/replay
norm ratios2.97–5.02 in7201 and0.71–4.22 in7202. These measurements do not establish
an optimal weight or unique cause. Artifacts remain within550MiB; no downloads.

The recipe's optional switch is `--retention-weight 10`; default0 keeps the previous
trajectory. For a new output directory:

```bash
.venv/bin/python -m experiments.modality_readout --stage grounded \
  --core runs/modality_readout_v1/formal/seed7202/joint_native \
  --understanding-suite data/understanding_v1/full --grounded-scope core \
  --retention-weight 10 --steps 768 --seed 7202 --device cuda \
  --output runs/my_retention_trial
```

This opt-in candidate remains experimental; it is not the new model default.

- [Comparison report](../runs/grounded_retention_v1/report.html)
- [Raw comparison](../runs/grounded_retention_v1/comparison.json)
- [Answer-format diagnostic](../runs/grounded_retention_v1/answer_format_diagnostic.json)
- [Per-example changes](../runs/grounded_retention_v1/per_example_changes.json)
- [Verification](../runs/grounded_retention_v1/verification.json)
