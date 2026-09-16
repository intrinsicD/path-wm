# Repeatable multimodal understanding regression

16 September2026. User requests repeatable understanding tasks for every current
modality and combinations, with visible changes rather than just software tests.
Reuse the existing readout recipe, Run and report; no new trainer/model framework.
This is an executable, scoped regression battery, not a certificate of unrestricted
understanding. Preserve the older factor/codec tests as distinct endpoints.

## Implementation plan and fixed protocol

1. Versioned prepared fixtures and strict source/pair validation. Ten controlled
   families: text roles/negation/correction, image spatial relation/count, audio
   order/duration, video motion/order/last-observed history. Opposite-answer pairs
   share prompts/choices; removing the designated whole evidence source produces
   identical remaining inputs. Video order/history pairs share final frames.
2. Real TAU urban audiovisual data already local: bus/park/street_traffic scene
   labels. Four single-evidence tasks and all11 combinations of2–4 modalities.
   Cross-modal task is agreement of scene CATEGORY, not synchronization/identity.
   Positive image/video/audio sources come from different recording locations;
   negative swaps match city. Calibration Barcelona, validation Helsinki, regression
   Vienna, with disjoint location identifiers. Device/time nuisance labels unavailable.
   One-second excerpts and64px frames limit what is observable; preserve these limits.
3. Fast and full profiles retain ALL25 task families; vary examples, not meaning.
   Quick uses first3 selected locations/class/city and one rotation; full4 locations
   and all4 rotations. Controlled quick2/1/2 pairs per calibration/validation/test;
   full8/4/8. This is development data once inspected; future generalization claims
   require separate untouched source groups. Freeze manifests/content hashes.
4. Actual existing text-decoder answer-choice mean log likelihood is one deployed
   endpoint, with decoder training lineage reported. No agent fitting during eval.
   Never treat language-scoring failure as proof of no latent understanding. Keep
   the scoring callback versioned/replaceable; existing structured outputs remain
   independently measurable. Diagnostic frozen linear readers are shown separately.
5. Three paired categorical draws on test. Full evidence, designated source omitted,
   all evidence omitted. Same pair/condition sampling randomness. Primary per task:
   minimum-draw accuracy and (for contrast tasks) both-partners-correct rate. Gates
   >=0.80 each, full-minus-required-omitted accuracy>=0.15, omitted accuracy<=chance+0.10.
   Full-minus-all-evidence-omitted accuracy must also be>=0.15 on every draw.
   These are preregistered engineering acceptance targets, not empirical constants.
   No omnibus understanding score. Report coverage and all missing capabilities.
6. Detached encoder/posterior/working-state probes use fixed16 bins/scale for encoder,
   train-only scaling and validation-selected ridge. Pooling/probe capacity can lose
   information; probes NEVER determine capability pass. Save stage arrays, scores,
   source identities, prediction/control errors, per-task timing and examples.
7. Strict comparison requires identical fixture/protocol/profile/readout contracts.
   Save deltas, improved/regressed/unchanged tasks and baseline lineage. Repeating
   checkpoints gives exactly zero metric deltas. Incompatible comparisons must fail.
8. Recipe adds explicit prepare/evaluate stages and opt-in post-training evaluation.
   Fit source models unchanged; auto suite only after completed core/joint training,
   never partial training. One command can run it and compare with a saved baseline.

## Checks and execution budget (before fits/results)

Red checks before implementation: contrast labels/omission identity/final-frame
controls; strict source-disjoint data/hash guards; missing/partial scoring never
passes; frozen diagnostic separation; reference mismatch rejection; evaluation
preserves outputs/RNG/model; full checkpoint repeat is exact.

Commit plan/checks, then implement and run focused regression checks. Prepare local
fixtures without downloads. CPU2 threads or CUDA within6GiB process budget; each
quick evaluation capped600s, preparation600s, new artifacts<=500MiB and free disk
>=500MiB. First baseline: trained full `formal/seed7201/joint_native` from the existing
modality readout study, then seed7202 equivalent. Same fixture/evaluation seeds, no
causal architecture comparison between these training seeds. Exact-repeat audit
on first source. No new neural training or parameter search. Missing sources fail
explicitly rather than becoming zero-scored fake examples.

## Claude review

Two actual public-only method exchanges. Adopt strict readout/fixture versioning,
source-group separation, city balancing, language-prior controls and separate probe
sections. Omission is deletion of a WHOLE source, with bitwise-identical controls;
not an ambiguous deletion of just duration. Retain raw deployed predictions rather
than silently replacing inference with prior-subtracted likelihood. Fixed acceptance
targets need not be derived from a weak baseline. Repeated test feedback is development,
not untouched confirmation. No claim that remaining nuisance confounds are eliminated.
The same location partition is used across every modality, task and omission condition.
Uncontrolled device/time confounds block broad generalization claims. Paired correctness
means BOTH opposite-answer items are correct for the same checkpoint and matched draw;
it is not an aggregate cohort comparison or a claim of statistical significance.
Controlled cohorts vary neutral text context, visual nuisance and tone phase/frequency;
they do not claim natural-language, natural-object or speech generalization.

## Execution extension before the larger profile

After the quick integration run completed, add one **full-profile workflow check**
on the same first checkpoint, with the predefined full counts, identical thresholds
and no retraining/tuning. It verifies the larger data path; it is not independent
confirmation, since it reuses cities/source families and inspected tasks. Cap this
run at600s and6GiB; retain the same total500MiB artifact budget. Compare checkpoints
and exact repetition on quick only. A full result must never be numerically compared
against quick through the regression comparator.

## Review repair

An added adversarial comparison check found that unchanged headline metrics could
hide a newly failed no-evidence gate. Repair comparison status to include lost/gained
gates and overall task pass, then rerun the focused checks and quick comparison/replay
under the final scorer contract. No thresholds, source fixtures, weights or answer
inference change. Retain the initial reports as development evidence.

The RNG integration check passed before a change: the existing `evaluation_mode`
already restores Python/NumPy/Torch RNG, module modes and buffers. Reuse that guard;
no duplicate RNG wrapper was retained. Comparison status now includes lost/gained
gates and task pass changes.70 focused software checks pass after this repair;
555 whole-repository checks passed before this final comparison-only repair.

## Results and what this does / does not establish

- **Quick:**334 records (118 calibration,98 validation,118 test),27 real recording
  locations,25 task families. **Full:**1336 records (472/392/472),36 locations,
  the same25 families. Real positives combine distinct locations of the same scene
  category; neither identity nor audiovisual synchronization is claimed.
- Both existing384-update joint-native sources (training seeds7201/7202) pass
  **0/25** quick task gates. The first source also passes **0/25** full gates.
  All21 quick paired families have minimum paired success0. Full has20 at0 and
  audio+video scene agreement at0.167, still failing. Two quick headline
  scores differ between checkpoints, but neither reaches acceptance; these are
  descriptive seed differences, not evidence of an architectural improvement.
- On full, eight of ten controlled tasks have encoder-probe accuracy>=0.80
  (count0.9375; the other seven1.0); audio order/duration do not. Actual answers
  still fail. This motivates testing core retention and output conditioning,
  without diagnosing one uniquely destructive layer. Small supervised probes,
  spatial pooling and a linear reader especially limit cross-modal attribution.
- The answer interface is a **specific byte-decoder choice scorer**. These source
  decoders learned short symbolic reconstructions, not German question answering.
  A failure therefore combines grounding, task learning and readout limitations.
  This battery is not an instruction-tuned general-model benchmark and not a claim
  that every latent representation lacks the information.
- Quick execution is about one minute; full254.79s. Peak PyTorch allocated memory
  is269.10MiB (not total device/process peak). No agent weights were trained or
  changed. Ridge readers fit only calibration examples with validation-selected
  regularization. Three categorical draws are not three independent training seeds.

The final quick [baseline](../runs/understanding_suite_v1/final_baseline/report.html)
and [comparison](../runs/understanding_suite_v1/final_comparison/report.html) are the
reusable reference reports. The larger [full-profile report](../runs/understanding_suite_v1/full/report.html)
was generated before the comparison-display repair; it has no reference comparison,
and its scoring/inference is unchanged. Strict code contracts deliberately require
rerunning it before using it as a reference under the repaired evaluator.

The recurring checks are opt-in after completed core/joint training, with standalone
prepare/evaluate commands documented in [experiments](experiments.md#recurring-understanding-checks).
Keep the original factor/codec tests too. This extension measures grounded semantic
contrasts and coarse real scene agreement. Speech/dialogue/music, natural tracking
and action consequences, long-lived memory/tools, fine visual recognition/OCR,
open-ended generation and fresh-confound-controlled confirmation remain explicit
unimplemented domains. No general understanding or high-level green promotion.

Report rendering was structurally verified and every embedded image/audio decoded.
Representative real frame strips were inspected. Browser URL policy blocked local
file navigation; no workaround was used and interactive browser QA remains unavailable.

Final verification:6448 independent raw/omission/probe-score/source-hash/media checks
across8 saved evaluations. Final quick baseline/comparison/repeat preserve every
saved score and feature from their pre-repair counterparts. Repeat reports25
unchanged tasks with exactly zero metric deltas and no lost/gained gates. Every
run is result-complete/report-structural-verified; no report failures or stopped
fits. One intentionally failing gate-regression test is preserved with its repair;
RNG protection was already provided by the existing context manager. Total new run
artifacts60MiB and fixtures5.3MiB; no downloads or agent fitting.

[Audit receipt](../runs/understanding_suite_v1/verification.json),
[full software checks](../runs/understanding_suite_v1/full-tests.txt),
[post-repair focused checks](../runs/understanding_suite_v1/final-focused-tests.txt),
[exact replay](../runs/understanding_suite_v1/final_repeat/report.html).
