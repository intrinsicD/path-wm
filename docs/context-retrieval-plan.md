# Bounded context and learned retrieval continuation

24 September 2026. Continue the architecture handoff with actual Claude Opus 5.5,
medium effort. Public-only briefs and model/session receipts are retained under
`runs/reviews/architecture_continuation_20260924/`. Private implementation review
stays local. This is one learning path, not completion of the integrated agent.

## Measurable slices

1. **Contracts:** add a small optional context-ranking head under TaskPolicy;
   retain versioned component references in bounded internal flat or Local/Global
   views of the existing WorldStore. Check eviction, per-task isolation, explicit
   reset, portable restart, stale/corrected sources and representation mismatch.
   Never make a second authoritative store or infer absence from capacity omission.
2. **Learning:** ordinary recipe, structured entity association plus two-word
   attribute requests, supervised ranking of canonical attribute keys and null.
   Template copying preserves exact strings/numbers. This learns lexical selection,
   not language generation, identity discovery, credibility or concept induction.
3. **Comparison:** fixed lexical, strong synonym-rule, random and no-read controls;
   flat and Local/Global use identical scan/read/reference caps. Report quality,
   omission/stale errors, retention, actual latency and memory separately. Local/
   Global allocation is deterministic in this slice, not a learned retention claim.

## Before-run protocol

Use two independent training seeds 17 and 29, at most 1200 Adam updates each,
64 examples/update, float32, CPU, two threads. Ten-minute training ceiling per seed;
full suite budget 20 minutes. One tiny train/report/resume workflow check first.
At most one diagnosed development repair, declared before running it. Never tune
against the final set. Training worlds derive from the checkpointed sampler;
validation and final population seeds are 624925 and 624927, respectively.

Four first-field and four second-field canonical words each have two aliases.
Train on canonical/canonical, canonical/alias and alias/canonical requests. Final
queries include alias/alias combinations never used in training; all words are
seen in training. This is compositional lexical generalization, not unseen-word or
natural paraphrase generalization. Random candidate permutations, random exact
payloads independent of keys, fresh entity IDs, near-miss unknowns and supplied
entity matching prevent answer/position shortcuts. Rank inputs exclude payloads,
source IDs, gold labels and future events. Current-head validity is deterministic.

Training has at most 32 candidate keys, one null output, and candidate supervision.
Evaluation uses 64 independent worlds per split, stratified delayed, distractor,
corrected, resumed and unknown queries (>=64 each). Exact copies and source validity
are checked separately from ranking correctness. Each query scans <=32 keys,
reads <=2 selected payloads, and retains <=4 component refs. Local/Global splits
those four refs 2+2; flat uses the same four with task tags. The same external store
is available to every arm; storage bytes and scanned metadata are accounted for.
No hidden reliability task or budget-unmatched unbounded reader is a competitor.

**Capability gates on each seed and each layout:** macro exact match >=95%, each
stratum >=90%; unknown precision and recall >=95%; zero accepted stale reads;
100% exact selected payload preservation; interrupted/reset/restart parity.
Canonical-query and held-out alias-combination metrics are separate. Strong-rule
baseline must achieve 100%; an adequate fixed baseline is a useful result.
**Separate hypotheses allowed to fail:** learned superiority over the strong rule;
Local/Global gain >=3 percentage points on resumption; learned end-to-end p95
latency <=2x the strong rule. A failed hypothesis does not invalidate other metrics.
Record exact trainable/total parameters, optimizer bytes, saved activation bytes
(not claimed as peak RSS), peak process RSS, wall step time, and selector/end-to-end
latency. Do not call differing FLOPs equal merely because caps are equal.

Reconstruction remains an independent frozen-codec regression: existing held-out
RGB16 direct versus recalled-code decoding parity and scoped reconstruction gate.
No visual retraining or assertion that lexical retrieval improves representations.
Preserve earlier transfer failures and all new failed runs.

## Standing principles and completion

Prepare keys once; distinguish exact source payloads from learned selection;
keep broad keys and selectively read details. References have explicit owner,
version and lifetime. Deterministic validity checks verify learned proposals.
Retention layouts remain ablatable, and quality/compute tradeoffs remain visible.
No summarization/compression claim is made for exact copying.

Commit the plan and informative red tests, implement and review locally and through
public generic Claude contracts, run focused checks and tiny exact resume, then
freeze code for the full CPU suite and registered comparisons. Save raw metrics,
checkpoint, source/split identities and standalone reports. Update this plan,
project state and scoped atlas; record research findings with existing workflow.

## Design reconciliation before implementation

Claude explicitly withdrew hidden-reliability, oversized budget, fixed-baseline
kill criterion, conflation of latency with validity, and dropping Local/Global
based on a tiny null result. Add a counted token-to-factor table trained from the
same gold labels; compare learned selection against this stronger non-gradient
baseline. Held-out pairing demonstrates per-token mapping transfer only.
Unknown candidate lists have the same 32 keys / 8 entity-matched keys as answerable
lists; missing keys have one-factor near misses. Use eight queries of each stratum
per world (512 unknowns per split); report Wilson intervals as uncertainty beside
the fixed point-estimate gates. Corrections/stale rejection are store/contract
checks, not learned correction reasoning. Context swap/reset ablations measure
whether retained views affect answers; zero effect prohibits a layout utility claim.
The four active refs count only retained component pins; query, null and <=32
transient key metadata are separate fixed costs. Payload read cap includes cached
reads. Exact copy conditional on chosen ref is a separate invariant from correctness.
No unseen-vocabulary, ambiguous-token, natural-language or allocation learning claim.

Plan/check step: `tests/test_working_context.py` fails collection because the planned
`ContextSelector` does not yet exist (saved `red-tests.log`, exit 2). These tests
specify masking/null gradients, bounded eviction, task isolation, exact payload,
restart, stale/version rejection and explicit budget omission before implementation.

Local contract review added owner-bound/checksummed portable snapshots, strict
schema/layout checks and explicit epoch validation before synchronous consumption,
including null decisions. This conservative epoch rejects even unrelated concurrent
writes; it is not an atomic external emitter. Existing EpisodeClient commit-time
pins remain the external/session output path. Trusted caller task labels and checksums
are not security authentication. Restore constructs a new object or fails entirely.

Development reports evaluate train-type (IID) phrases only. No early stopping,
learned threshold selection or tuning on final two-alias pairings. Fixed 1200-step
schedule; the final seed-624927 population and source/settings identities are frozen
before evaluation. Interval uncertainty is descriptive per seed, not pooled evidence.
Each normal/corrected answer agrees with the current source assertion, not established
world truth. Unknown and answerable queries draw the same pre-query candidate set.

## Development and local repair record

`runs/context_dev_s17_v1/` passed its IID development gates at 1200 updates:
learned and counted selection both 100%, canonical lexical baseline 36.17%.
This is not a final/generalization result. No quality setting was changed afterward.
Local review tightened the event schedule: delayed queries now reference the first
eight facts (24–31 later source events); resumed queries revisit those same facts
after interleaving/correction and restore before the resumed phase.

The first live-session integration check failed because WorldSession atomically
replaces its WorldStore on commit, while context held the previous object. The
context now accepts the authoritative session and resolves its current store on
every use. Preserve `focused-v3.log`; repaired 20-check suite: `focused-v4.log`.
Added a distinct result-completed/report-failed check. No failed artifact is a pass.

## Completed learning comparison

Both final seeds pass all declared source-agreement gates, separately on IID and
previously unseen two-alias pairings, in both layouts. Each seed evaluates 64 worlds
and 2560 queries per split, including 512 unknowns. All five strata reach 100%;
unknown precision/recall are 100% (Wilson intervals retained in raw results).

| Held-out pairing, flat view | Seed 17 | Seed 29 |
| --- | ---: | ---: |
| Learned source agreement | 100% | 100% |
| Counted-table source agreement | 100% | 100% |
| Generator-rule reference | 100% | 100% |
| Canonical-only / no-read | 20% / 20% | 20% / 20% |
| Learned selection + validated-copy p95 | 227.31 µs | 227.89 µs |
| Counted-table selection + validated-copy p95 | 229.52 µs | 228.13 µs |
| Rule selection + validated-copy p95 | 152.32 µs | 152.47 µs |

The latency column is the recipe's `end_to_end` metric: selection, context mutation,
validation and synchronous payload copying. It excludes offline ingestion, candidate
metadata preparation, simulated episode protocol and report instrumentation. It is
not full agent latency. Both seeds satisfy the <=2x rule-reader latency gate here;
no GPU or general efficiency claim follows. Matrix arithmetic in the neural ranker
is approximately 8960 multiply-accumulates/query (two 16-wide projections over one
query and 32 keys, then their dot products), excluding embeddings/masks and Python.

**Negative findings remain visible:** learned-minus-counted agreement =0; Local/
Global-minus-flat =0. Each split checks 1024 actual reset interventions and 1024
task-label swaps on resumed queries; answers remain identical. Complete bounded
search does not depend on retained context. The experiment therefore establishes
neither useful learned retention nor a layout benefit. Removing supplied identity
eligibility reduces agreement to39.49% on both splits/seeds; entity association is
not learned. No ambiguous words, unseen vocabulary or free language are tested.

The selector has705 trained /1213 total parameters; the remainder are frozen
coarse TaskPolicy heads. Adam tensors occupy5656 bytes; saved autograd tensors
325636 bytes (not allocator peak). Mean training step is1.39 ms, 1200 updates/seed;
whole train/evaluation invocation23.20 s. Process peak RSS is711–712 MiB and includes
Python/PyTorch plus evaluation. Largest serialized per-world store43851 bytes;
active-context snapshots approximately1.2 KiB. Raw rows meter32 candidate keys,
<=2 payload reads and <=4 retained references; payload/metadata bytes are saved.

### Correction, episode and reconstruction scope

Each split rejected512 deliberately superseded retained reads. Exact copies,
store/context restart, and query-budget checks pass. A separate **actual
WorldSession** demonstration saves session and context, restores both, rejects an
interrupted output and a stale dependency, rederives the corrected assertion and
records a checked response. All nine episode checks pass on both trained models.
Generated responses create no source evidence. Population store/context restart
and this small neural-session demo are separate scopes, not a claim that every
population query exercised a trained conversation loop.

Frozen existing codec checkpoints retain bit-identical direct versus recalled-code
outputs over512 original held-out RGB16 textures and all four familiar views.
Mean MSE: seed17 **0.00024258**, seed29 **0.00023893**; every pose is below0.003.
Weights are unchanged, training updates0. This is regression evidence, not an
improvement in reconstruction or concept learning.

### Artifacts and reproduction

- Learned comparisons: `runs/context_final_s17_v1/`, `runs/context_final_s29_v1/`.
  Each contains `last.pt`, `run.json`, source snapshot, raw `metrics.jsonl`, per-query
  `iid-predictions.jsonl`/`pairing-predictions.jsonl`, `result.json`, episode snapshots,
  `episode-checks.json`, and standalone `report.html`.
- Frozen reconstruction: `runs/context_codec_regression_s17_v1/` and
  `runs/context_codec_regression_s29_v1/`; raw predictions/targets, checkpoint and report.
- Claude: four substantive actual Opus5.5 medium rounds, session
  `5af6c291-64b3-40fc-b95b-1d9019a806e5`. Exact public briefs, replies, returned model
  usage and failures remain in the review directory. No private code/results exported.
  Claude reviewed generic contracts; private implementation was reviewed locally.
- The initial full-suite attempt was intentionally stopped after38.38 s for a
  report-completeness repair; exit-15 is retained as `full-suite-attempt1-exit.json`
  and is not a pass. Reports now embed representative saved predictions for every
  stratum. Historical run reports were repaired from existing raw rows only by
  `enrich_reports.py`, with a repair receipt; no metric/model/selection changed.
  The ordinary recipe now includes these examples without a postprocessing step.
- Report verification uses the existing unchanged renderer and structural checks;
  no new browser-QA claim. The tiny exact-resume check compares weights, count buffers,
  sampler, update count and deterministic metrics. Report failure preserves completed
  raw results and marks report failure separately.

```bash
.venv/bin/python -m experiments.context_retrieval --output runs/my_context_s17 --seed 17 --final
.venv/bin/python -m experiments.context_retrieval --output runs/my_context_s29 --seed 29 --final
# Tiny workflow; resumption requires identical steps/worlds/seed/final settings and source.
.venv/bin/python -m experiments.context_retrieval --output runs/my_context_check --steps 6 --worlds 1 --stop-after 3
.venv/bin/python -m experiments.context_retrieval --output runs/my_context_check --steps 6 --worlds 1 --resume
.venv/bin/python -m pytest tests/test_working_context.py tests/test_context_recipe.py tests/test_episodes.py
```

## Remaining architecture work

Keep this selector optional; counting already solves this task. Next context-control
experiment should separately preregister a workload exceeding the scan budget,
with a strong indexed baseline, candidate discovery/retention interventions and a
measurable dependency on retained context. Do not silently turn this complete-search
result into evidence for that capability. Goal-conditioned allocation, concept/
instance induction, learned association, noisy correlated evidence, harder visual
alignment, shared-depth quality/compute, audio dialogue and optional planning remain
owned by the continuation handoff. The full integrated architecture is not complete.
A perceptual detail cache (prepared encoder tokens/K/V retrieved by scene, place,
time or instance keys; see [World Labs review](worldlabs-review.md)) is a different
workload from lexical selection and is not planned. If planned, reuse this plan's
version, staleness, budget and restart contracts as the template.

## Final verification

Full `.venv/bin/python -m pytest`: **848 passed in622.19 seconds**, exit0.
The durable supervisor independently confirms source unchanged across all model,
recipe and test files. Log/receipt: `runs/reviews/architecture_continuation_20260924/`
`full-suite.log` and `full-suite-exit.json`. The earlier stopped attempt is separate.
The full suite includes10 new focused checks alongside the838 prior tests.
Independent raw-row audit recomputes both seeds' per-arm agreement and checks
budgets, delayed-source ages and context-intervention counts (`raw-audit.json`).
`artifact-audit.json` binds checkpoint, result/report hashes and unchanged shared
source; saved recipe differences are exclusively the report-example addition.
YAML/session counts, affected local documentation links, regenerated atlas and
`git diff --check` pass. Reports remain structurally verified with the unchanged
renderer; no browser-QA claim. Completed bounded slice; remaining architecture
questions are explicitly retained above and in the continuation handoff.
