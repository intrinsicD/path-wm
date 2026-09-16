# Grounded text readout from the shared latent core

16 September2026. Follow the recurring understanding suite rather than introducing
another encoder. Target `VID.order`: opposite-order red/blue clips share the last
frame and question. Saved diagnostic readers recover the target from encoder and,
for one source, working state; the actual text answers fail. This is a narrow
learnable temporal-order task, not natural video understanding.

## Fixed comparison (before training)

- Continue the two existing joint-native sources7201/7202. Arm `decoder` trains
  the existing text decoder/adapter; arm `core` additionally trains the existing
  shared core, excluding encoders, factor head, action head and monitor. No new
  module, answer classifier, soft-state substitution or direct encoder bypass.
- Use the16 full-profile **calibration** examples of VID.order as explicitly
  supervised training data (8 opposite-answer pairs). Validation/test and other
  suite task labels never enter training. Repeated suite use is development;
  no untouched/general video claim. Subsequent diagnostic calibration probes for
  this task are no longer independent of agent training; label that overlap.
-768 optimizer updates, alternating384 QA and384 original-symbolic replay updates,
  batch8, Adam0.001, same batch/sampling seeds within source/arm pair. QA uses
  normalized byte CE including EOS; replay reuses the existing summed normalized
  multimodal objective and original training distribution. Replay is included
  from the start, equally in both arms, to reduce forgetting. Encoder and other
  decoder weights stay frozen. More trainable parameters in core is a disclosed
  confound: superiority does not prove core adaptation is uniquely necessary.
- Evaluate fixed final checkpoints only. Both source seeds, all25 quick tasks,
  then full profile for target confirmation and regression scope. Evaluation seed
  9401, same scorer/data contracts and three paired categorical draws. No fitting
  on validation/test, no learning-rate search, no winner checkpoint selection.
- Engineering adoption requires target pass in both seeds on quick and full;
  no lost task passes, no accuracy/paired/source-gain loss>0.10 on other full tasks;
  original symbolic seen/heldout per-input/output accuracy loss<=0.05 and raw
  reconstruction CE/MSE increase<=10%. Compare to each source, never across seeds.
  Original symbolic evaluation uses the same recipe/data seeds before and after.
  Prefer decoder-only if both qualify. Otherwise preserve variants and defaults.
- Evidence controls: whole source/empty omission, paired correctness and explicit
  last-frame-only evaluation on this task (must remain at chance); ordered clips
  contain the same colors, so class-frequency/answer-length priors cannot solve
  balanced pairs. Unequal answer byte lengths remain a scoring limitation.
- Budget: four fits <=600s each,6GiB peak allocation; each full evaluation<=600s;
  total new artifacts<=500MiB, free disk>=500MiB. No download. Smoke/restart checks
  <=8 updates. One bounded scientific comparison; failure triggers diagnosis and
  a documented next step, not an unplanned hyperparameter search.

## Implementation / review sequence

1. Commit meaningful red checks for calibration isolation, label-free inputs,
   frozen gradient boundaries and byte-target loss. Reuse UnderstandingData, Run,
   existing recipe, output objective and standalone report renderer.
2. Add explicit `grounded` recipe stage with `decoder/core` trainable scope;
   preserve source architecture/weights and exact restart. Save training IDs,
   losses, source lineage, frozen hashes, output regressions and generated examples.
3. Verify smoke/report, restart and focused checks. Actual Claude public-only
   critique/reconciliation; no private code, measurements or data exported.
4. Run fixed comparison and suite; inspect failures, repair software if needed,
   retain all completed evidence. Update this plan/project state and commit.

Claude's first critique: report unequal trainable capacity/compute; a failed
decoder-only fit is a finite-budget learning failure, not proof of missing latent
information. Do not adopt its wording that a nonlinear decoder fit measures
linear accessibility. Keep symbolic preservation at the same final checkpoint.
Replay ratio is fixed above before results. Last-frame controls are explicit.
No capacity-matched extra adapter is added: that would test a different hypothesis.

Reconciliation: Claude explicitly withdrew the linear-accessibility wording and
accepted the scoped comparison. Its remaining requests for tolerances and prior
last-frame registration are already covered above. Two seeds are an engineering
replication, not a powered statistical effect estimate. Keep byte-length scores
inspectable, and report free generation as a separate diagnostic so a choice-score
improvement cannot silently stand in for successful text emission.
