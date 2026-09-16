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
