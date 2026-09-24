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
