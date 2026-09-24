# Scan-limited discovery: indexed feasibility screen

24 September 2026. Continuation of [context retrieval](context-retrieval-plan.md)
and the [handoff](architecture-continuation-handoff.md). Status: complete bounded screen; both frozen final populations and856 tests pass. Actual Claude Opus 5.5, medium, reviews public
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


## Frozen final result

Implementation commit `00abd78`; red plan/check commit `014cc40`. No library/model
source changed. Both independently seeded final populations pass every registered
gate on the same frozen implementation. Each has128 queries ×5 arms,32 source
corrections,16 logical restarts, and256 initial+64 distractor source components.

| Metric | Seed924317 | Seed924329 |
| --- | ---: | ---: |
| Index / cached / reset exact status, text and source agreement | 100% /100% /100% | 100% /100% /100% |
| Full scan diagnostic agreement | 100% | 100% |
| First-16 scan diagnostic agreement | 0% (128 omitted) | 0% (128 omitted) |
| Cache-reset answer changes | 0/128 | 0/128 |
| Stale cached references rejected | 32/32 | 32/32 |
| Logical restart checks | 16/16 | 16/16 |
| Index / cached p95, including validation and instrumentation | 0.950 /1.340ms | 0.950 /1.293ms |
| Complete invocation wall /CPU, including report | 7.884 /7.834s | 7.845 /7.805s |
| Index build plus32 rebuilds | 0.214s | 0.213s |
| Logical snapshot/replay and comparison | 5.666s | 5.638s |
| Index serialized bytes | 20,482 | 20,474 |
| Peak process RSS | 523.11MiB | 523.64MiB |

Bounded contenders probe at most1 candidate (index/reset) or2 (stale cached
recovery), return at most1 payload, retain <=4 pins. Full scan reaches320
candidate probes. The weak scan deliberately cannot reach old tail targets, and
cannot certify absence after truncation; this is expected diagnostic failure,
not a learned-method comparison. The full scan is also physically fast here
(p95 about6ms), so no physical necessity for the16-candidate cap is established.

**Negative result:** neither learned retrieval nor useful retention is needed to
solve this exact-descriptor task. No selector was trained. Cache reset leaves all
answers unchanged. This closes the predeclared stage-zero screen, not semantic
retrieval, identity inference, retention learning or the integrated architecture.

**Cost finding:** indexed validation still performs2H header visits per successful
lookup, or3H after stale-cache rejection, where H includes all historical source
components. Maxima704/1056. Whole measured harness597,072 header visits per seed,
about63.1MB of serialized record materialization through the public read APIs.
This includes all arms, rebuilds, fixtures and logical recovery. These counters
exclude private transaction/replay loops; their time is included in whole-invocation
CPU and wall measurements. These are instrumentation costs/serialized bytes,
not hardware bandwidth or peak allocator bytes. Indexing saves candidate work;
it does not make this reference store globally sublinear. Actual total invocation
cost includes reporting in `invocation.json`; result timing ends before rendering.
No GPU, energy or general efficiency claim.

### Evidence, review and repeatability

- Final runs: `runs/scan_discovery_final_924317_v1/` and
  `runs/scan_discovery_final_924329_v1/`; each contains source snapshot/identities,
  environment/settings, per-query `predictions.jsonl`, causal final `last.json`,
  all logical-restart snapshots, metrics, result, invocation timing and standalone
  `report.html`. Report QA is structural using the unchanged renderer.
- Preserved development runs: `runs/scan_discovery_dev_924301_v1/` and `_v2/`.
  V2 rechecks the source-ID namespace improvement; no threshold changed.
  All four complete invocations use about32 CPU seconds, below900-second budget.
- Four actual Claude Opus5.5 medium rounds, session
  `65ffe535-c0ac-49ce-9cd1-7fbc89abfca3`; exact briefs, responses and usage receipts
  under `runs/reviews/scan_discovery_20260924/`. Public-only contracts; private
  implementation and measurements reviewed locally. Resumed usage is not summed.
- Independent `audit.py` reconstructs each query's gold from saved source events,
  without importing the recipe/index. All1,280 raw rows agree with metrics;
  the two test streams and development have disjoint descriptors, all historical
  payloads and component IDs. Counterfactual stale payloads, false unknown answers
  and OMITTED-as-ABSENT each fail the independent checker. All final source hashes
  and report/checkpoint/result hashes are bound in `raw-artifact-audit.json`.
- Eight new focused tests plus six existing context tests pass. Deterministic
  future answers/store/context after restart match uninterrupted execution under
  reversed record enumeration. This is no process-crash durability claim.

```bash
.venv/bin/python -m experiments.scan_discovery --output runs/my_scan_dev --seed 924301
.venv/bin/python -m experiments.scan_discovery --output runs/my_scan_tiny --seed 924300 --records 32 --queries 16
.venv/bin/python -m pytest tests/test_scan_discovery.py tests/test_working_context.py
.venv/bin/python -m pytest
```

These final seeds are now consumed. A changed method needs newly preregistered
populations; rerunning these commands provides regression evidence only. No
training/resume CLI is invented for an index-only screen; portable `last.json`
and saved restart snapshots cover the applicable state contract.

## Next priority and remaining limits

Stop adding learned context machinery to this exact-key task. The next scientific
priority remains **useful representations and concept/instance transfer** from the
handoff: distinguish source-supplied descriptors/association from learned access,
compare remembered examples with inferred shared structure, and retain the earlier
R1 transfer/CI1 retention failures. Consult their owning plans before registering
fresh populations, reconstruction/transfer gates and a separate compute budget.
This screen does not pick an untested concept architecture or spend that budget.

The measured linear head-validation path is a concrete engineering follow-up if
larger stores need it: preserve latest-before-active semantics and exact replay,
then compare actual whole-pipeline cost. Its sub-millisecond-to-low-millisecond
cost here is not a reason to preempt the representation question. Noisy evidence,
harder visual alignment, shared depth, grounded dialogue/audio and learned
context control remain open in the continuation handoff.


## Completion verification

Full `.venv/bin/python -m pytest`: **856 passed in662.14 seconds**, exit0.
Supervisor elapsed663.68 seconds, below1200-second ceiling, with all recipe,
library and test source hashes unchanged across execution. Saved log and receipt:
`runs/reviews/scan_discovery_20260924/full-suite.log` and `full-suite-exit.json`.
Independent causal/artifact audit,109 affected local links, research YAML/session
counts, regenerated atlas and `git diff --check` pass. Project state stays <=8KiB.
Research records bind N577 and staged O401; no general claim is promoted.
The selected screen is complete; the next scientific priority and remaining
architecture limitations above are preserved explicitly.
