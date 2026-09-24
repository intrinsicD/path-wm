# Scan-limited discovery: indexed feasibility screen

24 September 2026. Continuation of [context retrieval](context-retrieval-plan.md)
and the [handoff](architecture-continuation-handoff.md). Status: preregistered;
implementation and results pending. Actual Claude Opus 5.5, medium, reviews public
hypothetical briefs; private implementation and evidence remain local.

## Decision and slices

1. Register an exact indexed feasibility screen and failing correctness checks.
2. Implement a readable recipe using WorldStore and WorkingContext, review its
   contracts, then run a tiny software check and bounded development population.
3. Freeze source, evaluate two fresh populations, independently audit raw rows,
   run the full suite, record limits and choose the next research question.

Do not train a selector unless the strong index leaves a diagnosed discovery
failure. An exact-key screen can falsify the need for learned complexity on this
workload; it cannot establish that learning never helps. It also cannot establish
semantic representations or entity association. Model weights and RGB16 codec
behavior are unchanged; reconstruction and prediction quality are not retested or
claimed improved. Useful representations remain the next research priority if the
index passes. Existing visual and transfer failures remain applicable.

## Fixed workload and arms

Development seed 924301; final seeds 924317 and 924329, each a fresh empty store.
No training population or optimizer: stage zero tests whether training is warranted.
A tiny seed 924300 is software-only. Identifiers and payloads are population-namespaced;
256 initial records, four components per entity, and 64 additional distractors.
Random exact two-token descriptors are source-provided retrieval metadata, not
inferred identities. Payloads preserve arbitrary whitespace, Unicode and numbers.
Queries never receive component IDs or gold labels. Ground truth comes from the
causal event generator, separately from the index; unknown queries have no match.

128 queries per population: eight repeated cycles of 16 with first eight delayed,
next four corrected, next two restart probes, and final two unknown queries.
Each cycle's delayed queries revisit old sources after distractor ingestion.
Correction supersedes in-stream evidence and updates the index before the query;
a separately retained old reference must be rejected. Restart serializes and
restores the authoritative store and context; compare the same query immediately
before/after restoration. This is logical restart, not crash-atomic durability.

Arms: composite-key exact index; same index with four LRU pins; same cache reset
before every query; sequential first-16 scan (diagnostic only); complete scan
(diagnostic upper reference). Same event prefix and exact validated-copy readout.
No learned-vs-weak-baseline claim. Full candidate ranking exceeds the declared
16-candidate budget, so the full scan is ineligible as a bounded contender; it
remains physically available and is measured, not hidden. Index receives every
source descriptor at ingestion. Rebuild cost after corrections is included.

## Budgets, gates and interpretation

Indexed arms must each attain 100% exact current-source agreement, zero stale
answers, identical restart answers, at most 16 candidate probes, one returned
payload and four pins per query. Indexed p95 query latency <=20 ms; each complete
population <=120 s wall time and peak process RSS <=2 GiB. Index footprint <=1 MiB.
Timing includes context validation/copy. Record end-to-end process CPU and wall
cost, source ingestion, index construction/rebuild, query time, snapshot/replay,
serialized store/index/cache bytes, returned payload bytes, and authoritative
validation scans/materializations. The store's latest-head validation scans
historical component headers and copies selected records: the candidate limit is
**not** a global storage-work bound. Do not claim sublinear end-to-end retrieval.

Causal cache-reset effects and scan-minus-index differences are descriptive.
No benefit is claimed without an observed intervention effect. If the index
passes both final populations, stop learned work for this exact-key workload.
If correctness fails, repair it before final evaluation; if resource gates fail,
diagnose actual costs before choosing a learned remedy. Two development repair
attempts; total local experiment budget 15 CPU minutes, including failures;
full regression suite has a separate 20-minute wall ceiling. No external training.

## Required evidence

Red checks: a target beyond scan limit is recovered by index; corrected/retracted
heads never resurrect; stale index epoch fails closed; exact restart preserves
answer and index; cache reset cannot change indexed answers; unknown versus
budget-omitted is explicit; raw rows recompute gates; report failure preserves
completed results. Save per-query answers, source IDs, budgets and latency, exact
settings, source identities/snapshot, environment, restart snapshots, result and
standalone report. Existing unchanged renderer; structural report QA only.

Public Claude briefs/responses/usage receipts:
`runs/reviews/scan_discovery_20260924/`. Initial review recommends the index screen
before training. Numerical scale and limits above are this local preregistration,
not Claude's larger suggested experiment. No architecture-wide capability claim.

## Review reconciliation before development

Four public-contract rounds requested from actual Opus 5.5 medium. Round two
withdraws the million-record and crash-kill proposals for this narrow screen.
Round three explicitly corrects the earlier suggestion that skipped invalidation
must emit stale text: mandatory validation must instead fail closed, separately
from ABSENT. Wrong valid source IDs are rejected by exact descriptor readout.

All source mutations and reads are synchronous under one writer. H is **all**
historical component headers at query start; indexed validation must visit <=4H.
This is a sanity bound, not constant-time access. Record actual H and visit/H.
Version ties follow existing WorldStore `(valid_from, revision, id)` ordering.
Duplicate descriptors reject the entire narrow index explicitly; scalable per-key
conflict handling is outside this input contract. Unknown and budget-omitted
outcomes are separate and correctness checks their status, text and source ID.
No silent scan fallback. Returned payload count excludes internal materialization,
which is measured separately in serialized bytes through authoritative read APIs.

Added adversarial checks compare exact 2H/3H lookup/correction visits, reject
poisoned IDs and stale epochs, preserve inactive-head semantics, reject byte-near
and reversed keys, and compare future answers of an uninterrupted store with a
snapshot-resumed store under reversed component enumeration. Both sources receive
the same subsequent corrections; deterministic answers, contexts and stores match.
The population repeatedly corrects one slot (32 corrections in128 queries), while
saved rows expose cost versus history. Logical restarts compare cache/index snapshots
and answers; they do not establish crash durability. No training/checkpoint-resume
path is applicable: `last.json` holds store/index/context state instead of weights.

Red collection failed because the recipe did not exist (saved log). First green
attempt exposed a test-fixture retraction lacking source authorization; fixed to
use the original `person` source. No scientific data/gate changed. Fourteen focused
checks now pass, including eight new screen checks. No development result had been
inspected when these contract checks and accounting amendments were made.

## Development decision and source freeze

Development v1 passes every gate: indexed exact agreement100%, p95<1.3ms,
32 stale cached references rejected, all16 logical restarts equal. Full scan also
answers100% but visits up to320 candidates. First-16 scan omits every query in
this deliberately tail-targeted population; it is only a diagnostic, not evidence
of a learned advantage. Source validation still scans history (index2H, stale
cached recovery3H). No query expects OMITTED; it never counts as ABSENT.

Local pre-freeze review namespaces source event/component IDs by population too
(the descriptor/entity/payload namespaces were already disjoint), adds explicit
stale-context restore refusal, and mechanically asserts unique descriptors.
This uses the first allowed development repair/recheck; v1 is preserved. No final
population has been inspected. Final Claude round acknowledges the declared tie
and duplicate-error semantics, accepts the corrected fail-closed contract, and
requires no additional mechanism before this narrow screen.
