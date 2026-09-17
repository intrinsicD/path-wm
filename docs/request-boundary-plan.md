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

## Implementation checkpoint

31 focused checks pass. Actual six-update CUDA pause/resume matches848 tensors
and1,321 primitive fields, including model, Adam, CPU/CUDA RNG, sampler and loss
rows;24.03s development process budget. Both paused development runs own reports.
Claude acknowledges the conditional estimand and fixed coefficient; two sources
and known-lexicon composition remain strict limits. Broader-suite sampling seed is
fixed9901 for all six runs. The final frozen source snapshot will be used for
formal execution and full tests while independent explorer work continues in the
main checkout. Paired inputs share clips, not forcibly identical categorical draws
within each training batch; draws are matched across treatment/control arms.

## Resource correction and bounded artifact recovery

The first source consumes843.04s for its fixed six phases, substantially more than
the initial estimate. Preserve the original1500s formal execution cap; do not
rewrite its receipt or claim that cap passed if interrupted. The CPU software
suite was paused291.11s for resource priority, then completed640 tests successfully
in853.82s elapsed; this pause is included in its855.45s process receipt.

If the formal cap interrupts evaluation AFTER a final512-update checkpoint exists,
allow a separate <=360s recovery to finish only its missing fixed fresh/broader
evaluations in new run directories. No extra optimizer updates, changed cohorts,
seeds, coefficient, thresholds or candidate selection. The original run stays
stopped/failed and gets a truthful report; recovered evaluation has its own source
binding, checkpoint and report. Record total cost including recovery, and mark the
original resource screen failed. This completes inspectable artifacts without
turning a budget failure into a passing adoption result. If final weights are not
available, report that comparison incomplete; do not train past the cap.

## Rejected loss term; retained interfaces

The matched control changes the interpretation. On source7202, ordinary CE reaches
76.5625% joint exact versus74.4792% with the extra term (untouched56.7708%). Source7201
is37.5% in every arm. The extra boundary loss fails its incremental-benefit gate;
remove its duplicate CE computation, weight flag and feature-specific rejection
checks from the current recipe. Preserve its exact source/tests in commit2d43133,
the isolated checkout and `runs/request_boundary_v1/retired-boundary-tests.py`.
No failing scientific gate is hidden by removing that abandoned feature.

Keep paired calibration sampling, neutral training input routing, explicit fresh
request evaluation and the independent ordinary-CE/gradient/frozen-state checks.
The current loss is the original normalized full-byte CE, one decoder call. All30
final focused checks pass after removal. The640-test full run belongs to the frozen
implementation snapshot; later changes remove the rejected loss and reject unused
CLI profiles. Neither learned checkpoint is promoted; first-only preservation,
full task gates and the original resource cap remain binding.


## Final outcomes and verification

| Fixed source | Untouched joint | Matched CE joint | Boundary CE joint | Boundary minus CE |
| --- | ---: | ---: | ---: | ---: |
|7201|37.5000%|37.5000%|37.5000%|0.0000pp|
|7202|56.7708%|76.5625%|74.4792%|-2.0833pp|

These are worst-draw joint exact values on fresh full-evidence prefix compositions.
Source7202 first-only accuracy is98.4375%,93.2292%,95.8333% respectively; both fitted
arms fail source preservation. First-word joint accuracy stays62.5%/100% for the two
sources in every arm. Boundary7202 also regresses broader REAL.scene.image accuracy
and source-gain minima by33.3333pp. Broad task coverage remains0/25 and1/25 with seven
unimplemented domains. No learned checkpoint is promoted. Shared paired sampling,
neutral routing and continuation may explain common changes; this experiment only
identifies the extra loss's difference versus matched CE.

Original execution stops at1500.4597s after all four512-update weights were saved.
The7202 CE parent retains stopped status and a truthful standalone report. Recovery
uses its final weights in separate `recovered_requests`/`recovered_quick` directories,
no optimizer updates,213.8327s total. All six fresh/quick comparisons now exist;
`runs/request_boundary_v1/recovery.json` is the authoritative mapping. Original
resource gate fails. Available three training receipts total140.10s and peak
91,292,672 allocated bytes; the stopped fit's training-only/peak receipt is missing.
Artifacts before the summary report total~342MiB; final inventory is saved with QA.

Independent audit reproduces11,520 rows and verifies8,986 tensors,2,216 arrays,
2,208 legacy arrays and94,940 primitives, including exact frozen weights, source
hashes, row populations, RNG/sampler matching and physical-state equality. First
attempt failed at an artifact-key lookup (directory instead of last.pt); preserved
failure log, no tensor/prediction assertion failed. Corrected full audit process
11.9359s. The first failed audit lacks separate timing: recorded execution/recovery/
audit sum1726.2283s is a lower bound; exact all-process cost is unavailable.

640 full tests pass on immutable2d43133 in853.82s (855.45s process, including recorded
291.11s pause). Final30 focused tests and Ruff pass after implementation63b7321.
The six-update pause/resume and frozen-control/current-CE development comparisons
each exactly match848 tensors and1,321 primitives; combined42.5800s within90s.
Removed feature tests remain archived with the passing formal implementation.
Eighteen reports pass structural/hash verification and the comparison figure is
inspected; existing renderer unchanged, no interactive browser QA.

## Current commands

These are opt-in experimental interfaces, not promoted model checkpoints. Choose
a new output directory for each run; these examples are not executed by this note.

```bash
.venv/bin/python -m experiments.modality_readout --stage grounded \
  --core runs/request_meaning_v1/seed7201/balanced \
  --understanding-suite data/understanding_v1/full \
  --grounded-scope interpreter --request-contrasts --request-profile paired \
  --observation-question neutral --request-evaluation fresh \
  --steps 8 --stop-after 6 --seed 9893 --device cuda \
  --output runs/paired_ce_new_smoke

.venv/bin/python -m experiments.modality_readout --stage request-evaluate \
  --core runs/request_meaning_v1/seed7201/balanced \
  --understanding-suite data/understanding_v1/full \
  --observation-question neutral --request-evaluation fresh \
  --seed 9801 --device cuda --output runs/fresh_request_new_evaluation
```

Future comparisons need a fresh predeclared evaluation population; this corpus is
now inspected development evidence. Keep content access and stopping/continuation
separate, and keep preservation controls before another fit. Do not rerun this
rejected coefficient or the earlier failed performance mechanisms without new evidence.
