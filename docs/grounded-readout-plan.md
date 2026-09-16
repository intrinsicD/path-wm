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
  then the full-profile **target task only** for the larger cohort check. Evaluation seed
  9401, same scorer/data contracts and three paired categorical draws. No fitting
  on validation/test, no learning-rate search, no winner checkpoint selection.
- Engineering adoption requires target pass in both seeds on quick and full;
  no lost task passes, no accuracy/paired/source-gain loss>0.10 on other quick tasks;
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

Pre-fit efficiency refinement: full-profile expansion is restricted to the
preregistered target, while the recurring quick battery still evaluates all25
families. This avoids six redundant full real-scene passes for a small readout
study. Selected fixtures receive a distinct comparison identity/profile; they
cannot silently compare to complete-suite reports. No formal fits/results preceded
this refinement; the4-update workflow check is not a quality comparison. The quick
preservation screen is correspondingly low-powered and is explicitly not a full
real-data preservation certificate.

Post-fit diagnostic, specified before measuring it: if a grounded answer remains
at chance, compare the already saved working tokens with their actual text-decoder
key normalization and value projection. Fit the same calibration/validation ridge
protocol to each representation (flattened and token-averaged); neural weights stay
unchanged. This checks whether normalization discards accessible signal or whether
the downstream read has simply not learned it. Failure of a reader cannot prove
irreversible loss. These development diagnostics neither select weights nor alter
the registered capability gates/budgets. No extra neural training is authorized by
this diagnostic addition.

## Results

All four fixed768-update continuations completed. The frozen encoders and source
checkpoints are unchanged. Each fit used384 QA updates and384 replay updates;
16 unique QA examples were presented3072 times in total per fit. Training scope:
22,531 parameters for decoder-only,133,896 for core+decoder,297,013 total.

| Source seed / training | Larger-cohort answer accuracy, worst of3 draws | Both members correct, worst draw | Free generated answer,16 examples / one draw | Quick suite passed |
|---|---:|---:|---:|---:|
|7201 / decoder|50%|0%|50%|0/25|
|7201 / core+decoder|50%|0%|50%|0/25|
|7202 / decoder|81.25%|62.5%|87.5%|0/25|
|7202 / core+decoder|100%|100%|87.5%|1/25|

The7202 core+decoder candidate passes **VID.order** in both quick and larger
cohorts. Every answer/pair is correct across the registered three draws; removing
video or all evidence yields50%. Last-frame-only accuracy is50% in all four
continuations. This is real evidence-dependent improvement on the tiny controlled
task, not general video understanding. Free generation is a different endpoint:
14/16 correct under its separately recorded common draw, so the choice-scoring
pass is not a claim of100% free text accuracy.

**No default replacement:** the effect fails to replicate in7201. All four variants
also exceed the previous-output preservation tolerances:17/32/18/23 of72 symbolic
input/output/split cells respectively. Worst per-factor accuracy losses reach
35.4/47.9/64.6/66.7 percentage points, and worst CE/MSE ratios are1.72/1.69/1.73/2.24.
Decoder-only preserves nontext outputs exactly. For7202 core+decoder, four real
single-scene suite families also lose33.3 pp of source gain; video scene accuracy
loses33.3 pp. These coarse cells are tiny, but still violate the declared guardrail.
Replay at the tested schedule therefore does not adequately preserve old behavior.

The saved-state diagnostic does **not** support removing decoder normalization:
7202 baseline raw working-token probe is87.5%, normalized93.75%, and projected
values87.5%; core adaptation reaches100% at all three. For7201 the same readers
remain50–62.5%. These are calibration-fitted diagnostic readers, not deployed
answers; the target calibration labels also trained the continuations. No unique
irreversible-loss location or architecture necessity follows.

The next justified target is **stable temporal information in the shared core and
retention of the requested output behavior during adaptation**. A candidate next
comparison is a training-only paired semantic objective at the working state,
with unchanged inference and explicit old-task preservation. This is proposed,
not implemented or authorized as an unbounded search by the present result.
Do not enlarge the image/video encoders or remove normalization on this evidence.

## Verification and artifacts

561 whole-repository software tests pass; the pre-formal focused set passed61.
Restart gives1180 exactly equal tensor checks across weights, optimizer and RNG;
the4-update continuous and2+2 resumed runs have identical loss rows. The actual
four-fit training time is224.34s, with at most88.06MiB PyTorch allocated during
training (not total device/process memory). Each fit and evaluation stays within
the declared per-command600s budget;14 formal commands complete without failure.
No downloads, altered thresholds or post-result hyperparameter search.

The raw/media audit independently recomputes acceptance gates from saved scores,
checks paired omitted-input equality, frozen weights, source hashes and embedded
media. Every result/report completes. Reports are structurally verified; browser
URL policy prevents interactive local-file QA. A representative reconstruction
panel was visually inspected. A run-local audit initially lacked the repository
on its Python import path; its receipt is retained and the helper was repaired.
This did not affect training, checkpoints or evaluation.

- [Comparison report](../runs/grounded_readout_v1/report.html)
- [Passing scoped task candidate](../runs/grounded_readout_v1/seed7202/core/target/report.html)
- [All25 tasks for that candidate](../runs/grounded_readout_v1/seed7202/core/quick/report.html)
- [Raw comparison and preservation failures](../runs/grounded_readout_v1/comparison.json)
- [Audit](../runs/grounded_readout_v1/verification.json)
- [Decoder-interface diagnostic](../runs/grounded_readout_v1/decoder_diagnostic.json)
