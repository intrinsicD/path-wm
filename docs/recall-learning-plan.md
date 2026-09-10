# Next slice: establish factual recall before changing memory learning

Authorized by Alex, 11 September 2026: send the saved brief to Claude, then implement
the proposed points while conserving both assistants' weekly limits. The immediate
slice is the diagnostic and its capped pilot. Later memory-learning changes remain
conditional on that evidence and a separately declared comparison.

## Why this comes first

The saved eight-update run selected update 7 and scored 0/12 on seen locations,
including 0/4 recent cases. All 15 operational decisions abstained. This tiny
execution check neither demonstrates useful recall nor locates failure in distant
compression. The existing all-query factual loss, source/text grounding and KL
terms provide the first reference; adding a new loss now would obscure that check.

The broader [reviewed learning proposal](remaining-decisions-2026-09-10.md) remains
the direction after this diagnostic. Its architecture does not need another review.

## Smallest next deliverable

Add an explicit current/recent diagnostic configuration to the existing
`experiments/multimodal.py` recipe and `RecallEpisodes`. Today the generator requires
recent, compressed and consolidated cohorts, so shortening `--history` alone is
not a valid diagnostic. Preserve the default historical task and its saved runs.

Use complete four-event sessions, the existing small-memory capacity (recent=2,
block=2, compressed blocks=2), full four-event gradients, the same text/query
encoding, factual head, two retrieval rounds and unchanged objective. A current
query targets event 4; a recent query targets event 3 followed by an unrelated
event. Include earlier contradictory observations and vary the queried entity so
copying the final episode event cannot solve every case. Balance the four seen
locations and report never-observed separately. The query still reaches the model
only after all historical writes; labels and the raw evaluator archive stay out
of model inputs. Reset session state between episodes.

Pilot: 40 fixed training episodes (20 per cohort, four per factual class)
and 100 independently generated diagnostic-development episodes (50 per cohort,
ten per class). Use different bindings/sequences within the supported vocabulary,
deduplicate across splits, and retain exact split manifests. This is a small-set
fit and fresh-example check, not a transfer experiment or final test population.

One CPU training run, seed 17, batch size 4, at most 256 optimizer updates and
15 minutes of training/evaluation wall time, whichever comes first. No sweep,
extra seeds, GPU run or automatic extension. Keep other model/optimizer defaults
and record their resolved values before execution. Log training factual metrics
at initialization and every 16 updates; use the final checkpoint for the one
diagnostic-development evaluation. Fit no new calibration transform and keep the
existing final calibration/test populations out of selection. A timeout or numeric
failure is recorded as stopped/failed, with available raw evidence preserved.

Routing gates, fixed before running:

- Tiny-set fit: at least 95% seen-location training accuracy in each cohort and
  seen-example mean factual NLL at most 0.35. Otherwise inspect encoding, entity
  binding, gradient flow and optimization before adding another learning mechanism.
- Fresh examples: at least 80% seen-location diagnostic-development accuracy in
  each cohort and seen-example NLL below the uniform four-location reference
  (`ln(4)`). Otherwise improve data exposure/grounding and declare another bounded
  pilot; tiny-set overfitting does not establish generalization.
- Report absence accuracy, per-class/cohort counts, raw NLL, task loss and coverage
  even when the gates pass. These are pragmatic pilot gates, not significance tests
  or calibrated deployment claims. A miss does not refute the architecture.

Resolved pilot defaults: width 32, AdamW learning rate 0.0003, its existing default
weight decay 0.01, FP32, two CPU threads, full four-event gradients and the existing
loss weights (factual CE 1, text/source grounding 0.05 each, dynamics KL 0.01,
representation KL 0.001). Factual absence uses the fifth argmax class and NLL;
operational answers require `1 - max_probability < 0.25`, with ties abstaining.
Absence is descriptive, with no separate advancement gate or fitted calibration.
Fresh-example evaluation is fixed at the end even after a failed fit gate.

The wall-time budget covers active training and evaluation, accumulates across
pause/resume, and is checked before updates and each evaluation episode. An in-flight
update/episode may finish before stopping; checkpoint/report writing is outside this
cooperative limit. A stopped run does not acquire a fresh budget on resume. Normal
checkpoint resume remains exact apart from the recorded wall-time counter.

Write the essential label/cohort and information-boundary checks before changing
behavior, plus a tiny forward/backward check of the actual diagnostic path. Reuse
the existing Run/checkpoint/report machinery. Deliver an editable recipe, raw
metrics, checkpoint and standalone verified report. Test pause/resume or report
behavior if those paths change; browser-check any changed renderer. Do not repeat
unaffected expensive checks or create a second trainer/reporting framework.

## What follows a useful diagnostic

The next experiment is the same architecture with grounded local memory loss off
versus on, preserving the baseline. Sample one compression/consolidation update,
detach its inputs, and supervise retained facts/corrections using a restricted
reader without a raw-history or current-belief shortcut. A local absence target
must refer only to the covered information. Match episode exposure, initialization
and declared update budgets, and report the extra probe computation explicitly.

Before that comparison, declare loss weight, probe mixture, seeds/populations,
minimum useful old-seen improvement, recent-recall guardrail and compute ceiling.
Those are not decided by this diagnostic. Recent success is an economical routing
preference, not a logical prerequisite for joint learning. Failure after compression
still does not isolate compression from retrieval, interference or truncated credit.
Train recurrent/recent-only and fixed-compression controls before making hierarchy
superiority claims; evaluation-only ablations measure reliance.

Freeze the selected procedure before independent final calibration/test. Delay
learned marking until marginal reader signals are informative. Prefer controlled
document revisions as the later economical application, subject to Alex's application
priority. Language interpretation, sensing, ensembles and adaptive computation are
separate later decisions.

## Coordination and evidence

One implementing assistant owns a slice end to end. Claude reviews a concise
conceptual decision or an ambiguous result once; use a short correction exchange
only for a material unresolved disagreement. No duplicate agent investigations,
automatic recurring reviews, or unchanged architecture re-review. Save a compact
handoff containing changed files, results, unresolved issue and next command.

The planned continuation review uses one isolated, tool-free Claude Sonnet call
with low effort and a response request below 400 words. Its intended export is only
an abstract hypothetical brief. Exact brief, response and execution receipt belong in the ignored
`runs/reviews/continuation_2026-09-11/` directory. `local-checks.json` records the
saved-result and source checks; no test suite or model run was repeated.

Claude review disposition: the initial request timed out (177.682 seconds,
`is_error=true`, empty model usage, reported API-equivalent cost zero). A network
retry was rejected before execution by automatic approval review: the brief was
considered potentially sensitive project-derived content requiring explicit export
approval. No alternative route was attempted. This plan is Codex's recommendation,
informed by the previously completed Claude reviews; it is not a new joint agreement.
The exact proposed export is `next-step-retry-brief.txt` in the review directory.
Alex explicitly approved this export in the next message. The retry and one short
conceptual reconciliation both completed with actual Claude. Claude accepted the
diagnostic-only scope, withdrew its conflation of factual absence with abstention
and its calibration prerequisite, and narrowed the claim that any memory-corruption
null result falsifies reliance. It accepted a terminal fresh-example diagnostic
after fit failure and acknowledged that a finite-budget fit failure need not be a
bug. Exact responses and model usage are in `next-step-retry-*` and `reconcile-*`.
No more architectural reviews are needed for this slice.

## Implementation and pilot outcome

Implemented in `experiments/multimodal.py`, with diagnostic sections in the existing
report renderer. `--dataset recall --recall-mode current-recent` runs the declared
defaults. Plan/red checks were committed as `9eceb36`; implementation as `dbdaf64`.
The initial three contract tests failed because the historical generator required
15-episode populations. The final suite passes all 79 CPU tests, including exact
pause/resume apart from elapsed time, no calibration/test access, cached final
evaluation, report-failure status and cumulative budget exhaustion. Ruff checks
and the real width-32 diagnostic forward/backward check pass.

The single pilot completed 256 updates / 1,024 sampled episode presentations in
144.58 active seconds. It used exactly the declared populations, settings and final
checkpoint. Training seen-location accuracy: current 11/16 (68.75%), recent 12/16
(75%); combined NLL 0.852505. Development: current 5/40 (12.5%), recent 9/40 (22.5%);
combined seen NLL 2.146645. Factual absence scores 8/8 in training and 7/20 in
development. Overall development accuracy is 21%, coverage 12%, task loss 0.31.
Both gates fail; no local memory objective, marking, calibration or further run
was launched. This is the declared stopping outcome, not an incomplete training run.

Read-only inspection confirms byte encoding preserves the whole canonical entity/
location strings, all saved parameters are finite, and gradients reach source,
state, memory and factual readout. Training factual NLL falls from 1.646278 to
0.727872 overall, but the final model does not fit the seen training facts to the
declared threshold and performs poorly on fresh episodes. These checks rule out
an entirely disconnected objective; they do not identify a particular representation
or optimization defect. Next propose a narrowly budgeted source-fact/query-binding
readout control before changing retention or adding losses. Do not retune on this
development population and present it as a new held-out result.

Evidence: `runs/recall_diagnostic_v1/{tests.xml,check.json,verification.json}` and
`pilot/{run.json,last.pt,metrics.jsonl,recall_diagnostic_manifest.json,
recall_diagnostic_predictions.pt,recall_diagnostic.json,report.html}`. The report
passed structural verification and browser inspection at 1280x720 with no broken
images or horizontal overflow; normal-viewport screenshots are retained. Completed
historical evidence remains unchanged. Only two short successful Claude exchanges
were used; their receipts report Claude Sonnet 5 with no thinking tokens, plus
the CLI's small Haiku overhead. Reported cost fields are API-equivalent estimates,
not subscription charges or a measurement of weekly quota consumed.
