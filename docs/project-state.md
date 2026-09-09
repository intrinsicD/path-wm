# Current work

**Implemented:** task-conditioned operation/output proposals, required/disabled/
automatic output controls, separate control author/requester/producer attribution,
and tagged generated-content reflection. Read [the task interfaces](tasks.md),
[the model guide](multimodal.md), and [the implementation record](multimodal-plan.md).

One editable [recipe](../experiments/multimodal.py) constructs the model and its
losses. The default has 359,188 parameters, 30 world-state tokens and four computed
task tokens. All modules remain directly replaceable PyTorch components. No external
language model, registry or second trainer was added.

The task interpreter reads shared multiscale text features plus exact metadata
encodings. Its learned policy proposes think/recall/imagine/act/emit/ask/finish.
A bounded step consumes that proposal; actions and clarification requests return to
the caller. Discrete controls are enforced separately from raw scores. Emission
prevalidates all requests, preserves partial successes on decoder failure, and only
successful answers fulfill requirements. Tasks can abort with requirements pending.

Generated loopback updates working/reasoning without advancing world/observational
clocks or counts. Persistent ancestry prevents those states entering observational
memory, including after later real observations. Keep a separate clean observational
branch. This is a trusted-caller metadata contract, not cryptographic authentication.

**Verification:** 51 CPU tests pass. The instruction recipe completes 80 updates at
width 16, seed 42, batch 8, on 112 train/112 held-out synthetic episodes. An exact
pause-at-31/resume comparison matches weights, optimizer, RNG, training rows and all
final task decisions. A two-update real PushT path also completes. No extra proposals
or GPU job ran. Reports pass structural, media, provenance and source checks;
**browser visual QA remains blocked** by the earlier local-URL policy.

**Learning is not yet reliable:** held-out operation accuracy is 28.57%, versus
92.86% for a lexical baseline and 21.43% with mismatched instructions. Output-format
errors on emit examples are 50–56.25%. Completion labels are synthetic declarations,
not measured task success. Raw/enforced decisions and this negative result are
preserved, with no claim of general instruction following or learned compliance.

Results: [instruction report](../runs/task_outputs_v1/instructions_full/report.html),
[all held-out decisions](../runs/task_outputs_v1/instructions_full/task_decisions.json),
[real PushT report](../runs/task_outputs_v1/pusht/report.html), and
[verification receipt](../runs/task_outputs_v1/verification.json).

**Diagrams:** [architecture](diagrams/architecture.svg),
[world flow](diagrams/data_flow.svg), [task flow](diagrams/task_flow.svg), and four
individual input-scale diagrams regenerate with
`python experiments/multimodal.py --diagram`. All 29 files reproduce exactly on this
runtime. Task flow shows learned proposals and explicit user emission as separate
actual call paths.

**Claude review:** the user explicitly approved the previously blocked final brief,
and the third review completed. Claude accepted the stated core contracts and
withdrew its blanket objection to post-mask violation metrics. These are conceptual
findings; implementation conformance rests on local tests. Exact briefs, responses
and receipts are retained in `runs/reviews/task_outputs/`, including
`approved-final-response.json` and `approved-final-receipt.json`.

The earlier conditioned multiscale inputs, episodic memory, stochastic dynamics,
candidate planning and bounded update gate remain available. Their development
results are preserved under `runs/multiscale_v1/` and `runs/multimodal_v1/`; old
checkpoints require their source snapshots. The new latent schema is v2 and fails
closed when ancestry is absent. No data or completed runs were removed.

**Current discussion:** the negative learning result is acceptable for now. Finish
the model architecture before designing or running new experiments. The concrete
[state and hybrid-memory specification](state-memory-design.md) separates directions
accepted in conversation from newly proposed interface details and illustrative sizes.

Accepted directions include recent detail plus compressed history and protected
marks, both user and agent mark proposals, separate perception/prediction/thinking
reads, action/time-conditioned short-step prediction, and bounded thinking that
preserves world time and evidence boundaries. Marking should learn from delayed
benefit. The representation should learn domain-relevant content rather than reserve
fixed semantics for physical objects or places; document work is equally in scope.

The specification proposes explicit token shapes, residual attention/gating,
chronological compression staging, a bounded consolidated session state, write
ordering, protected-detail admission and gradient boundaries. Consolidation details,
numerical capacities, uncertainty family and exact training choices remain proposals.
No model implementation, checkpoint migration or new training has occurred.

**Design review completed:** three actual public-concepts-only Claude exchanges
reviewed the complete proposal and reconciled overstatements. Read the
[review and next-discussion agenda](state-memory-review.md). Both reviewers favor
discussing whether memory should retain separately addressable observation features
alongside inferred belief snapshots, under the same total budget. This is a new
proposal, not an adopted model change; evidence encodings remain learned and lossy.
Then settle executed-action propagation/observation correction, persistent derived
knowledge, conflicting memory reads, learning targets and action/objective adapters.
Exact briefs, responses and receipts are in `runs/reviews/state_memory_design_2026-09-09/`.

The user wants memory kept within reasonable bounds while being large enough for
useful recall. The specification distinguishes storage capacity, attention/read cost
and learned recall quality. Keep its illustrative counts as a starting proposal;
event cadence and required recall horizon must inform the actual capacity choice.
Bounded indexed reads remain optional; no such implementation or scaling benchmark
exists, and the two-view choice remains open.

The user now requires independent short-term and long-term resets. The design
recommends separately accessible source-evidence and belief views, with counted
capacity, and proposes short/long/all controls. Long reset preserves recent evidence
but invalidates beliefs and working state that may contain recalled history; short
reset preserves existing long-term records. Protected marks belong to long-term.
Ownership, invalidation and generation-barrier details remain proposals;
the reset requirement does not imply implementation or adoption of every detail.
Three further abstract Claude exchanges reconciled reset semantics: retain source
records, invalidate shared derived state, and use normal future reads rather than
historical replay. Review follow-up also specifies protected admission and late
external-result handling. No implementation or reset execution occurred.

Physical understanding, useful language generation, calibrated progress, semantic
feature controls, scalable memory and broad self-improvement remain research
questions. Real data still use `data/pusht_world_model/cchi_v1` directly.
