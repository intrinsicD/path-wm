# Paired request-boundary learning

17 September 2026. The saved completion diagnosis selects this slice: strong-source
answers often contain the correct first color but end for sequence requests.
Previous micro-optimizations failed; do not rerun them. Test whether explicit
supervision at the first-word boundary improves requested completion using the
existing interpreter and frozen model. No inference rule, new parameter or cache.

## Fixed comparison

Sources: request_meaning_v1/seed7201/balanced and seed7202/balanced. Four fresh
512-update fits: baseline ordinary byte CE versus the SAME loss plus coefficient1
full-vocabulary CE at the first target word's boundary (EOS or space), both losses
normalized by log259. Coefficient1 is the first fixed choice, not tuned. Reuse one
decoder forward. Byte targets alone determine the boundary; no target, format label
or expected length enters model/core metadata or free generation. Weight0 exactly
preserves the existing objective. Empty/space-first or malformed boundary targets
are rejected when this objective is enabled.

Both arms: existing interpreter only, Adam0.001, batch8, clip5. Learning seeds9801
and9802 match the respective two sources. Each step samples four calibration clips
and one calibration wording pair per clip, yielding paired first/sequence requests.
Same init/sampler/RNG and512 steps in both arms. Both use neutral observed question
text with actual task instruction intact. This shared routing/sampling change is
not the experimental factor; compare treatment to its matched CE control and also
to the untouched source under the same evaluation route. No attribution of common
baseline improvement to the additional boundary loss.

Training: unchanged16 VID.order calibration clips and24 calibration requests.
No validation/test wording enters fits. Fresh evaluation replaces test prefixes
with these six precommitted families, composed with the existing two equal-byte-
length opposite payload pairs: `Antworte mit {}.`, `Gewünscht: {}.`,
`Gib als Lösung {} an.`, `Teile mir {} mit.`, `Ich möchte {} erfahren.`,
`Bitte liefere {}.`. These are new prefix compositions with known payload lexicon,
not a claim about broad unseen language or syntax. Preserve historical test corpus
as the default and separate development evidence. Existing phrase-order stress is
retained and labelled historical. Freeze the corpus source/hash before any scored
execution; no result-driven prefix or coefficient editing.

Primary quality gate per source: worst-of-three-draws exact joint success for both
opposite requests on fresh full-evidence prefix compositions >=80% AND improvement
>=10 percentage points over matched CE control. Preserve first-word and first-only
worst-draw accuracy within2pp of untouched source and matched control. Source-omitted,
last-frame and constant-request joint success must remain<=55%; order stress is
reported separately, not substituted for primary. Also require no>2pp regression
in each of25 broader quick understanding tasks versus untouched source, and all
request-free legacy outputs bitwise unchanged. Both sources must pass to promote
this learning recipe. No checkpoint/seed selection; evaluate final512-step weights.
Three draws are repeated stochastic measurements of the same fixtures, not three
independent models. Conditional boundary probabilities do not replace free answers.

Fixed evaluations: two untouched sources and four final fits, fresh corpus with
neutral observation questions, same16 test clips and seeds19701+pair_index*10+draw.
Full evidence for all direct requests, omitted/last-frame/constant-request controls
on first fresh family, historical order stress. Decode greedily16 tokens, unchanged
EOS policy. Save per-row strings, first-word/form/EOS diagnostics and working states.
Evaluate all six on the existing quick334-record understanding fixture with the
ordinary full observation route; no new generalization or natural-video claim.

Budget: formal six evaluations/four fits/audits <=1500 process seconds, <6GiB peak
allocated GPU memory, <1GiB artifacts. Development smoke/resume <=90s; full software
suite separately<=1200s. Stop/record overruns; do not relax gates or populations.
Preserve source checkpoints, all completed runs, raw outcomes and full cost receipts.
Each completed run owns checkpoint/source snapshots/metrics and an existing-renderer
standalone report. Structural checks and figure inspection; disclose browser limits.

## Implementation and review checks

Extend the existing readable recipe and request corpus only: optional boundary
weight, paired profile, neutral training route and explicit fresh evaluation profile.
No new trainer, CLI hierarchy or report renderer. Snapshot profiles/objective/corpus
hashes in run identity so resume cannot change the experiment. Test independent
UTF-8 boundary indexing/full-vocabulary CE/gradients, zero-weight exact behavior,
single decoder call, calibration-only paired exposure, neutral evidence preservation,
real interpreter gradients with every other tensor frozen, and exact short resume.
Commit informative red checks before implementation. Run full software suite after
source settles. No shared source edits during formal measurements.

Actual isolated Claude public-only methodology review/reconciliation under
runs/reviews/request_boundary_v1. Its concerns distinguish common routing changes,
known-lexicon composition, finite model/draw coverage and teacher-forced scores
from a task-general repair. No private repository content/results exported.
