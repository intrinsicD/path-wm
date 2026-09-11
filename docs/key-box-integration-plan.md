# Key-and-box integrated slice

Authorized 11 September2026. First operational integration, not learned action
dynamics, general graph search or visual discovery. Existing frozen descriptor
matcher and binary state cell feed an actual BeliefAgent workspace through a
trainable projection. Content heads read working tokens only. Neutral delivered
text events advance the belief; semantic box contents are available only through
retrieved entity latents. Entity store owns content updates; imagination never writes.

Two opaque boxes and one key, or no key. Supplied unit descriptors; independent
session IDs and stores. Observe reset/toggle/idle content events, query after delay.
Four conditions: remembered, moved with delivered correction, invalidated contents
requiring inspection, absent. Opposite initial key positions have identical final
neutral observation. Entity query resolves through descriptor matching, not array ID.
Inspect reveals one box; open changes its latch; retrieve succeeds only if open
and key present. Inspect cost0.25, open/retrieve1; four external actions maximum.
Explicit expectimax horizon4 scores goal reward10 minus costs with stop0; conditional
inspection outcomes are weighted, not chosen optimistically. Replan after each real
observation. Terminal success is externally verified. Record false stop vs budget
exhaustion. Planning transitions are supplied task mechanics, not BeliefAgent.imagine.

Train only projection, workspace thinker and readout; freeze perception/belief dynamics,
matcher and state-cell donors.256 updates, batch32, lr0.003, seed2301; reset/toggle/idle
histories of length2..8; no gradient through decisions or environment. Evaluation
16 fresh descriptor pairs seed2401, four conditions and opposite placements:128
sessions per policy. Policies learned integrated, no-history-memory (clear historical
store access and agent workspace before action loop; retain new observations), and
supplied-state planner control. Freeze final checkpoint; no threshold tuning.
Readout accuracy>=95% on known initial contents; reachable goal success>=90%; absent
correct stops>=90%; utility advantage over no-history>=0.01 across all sessions;
supplied-state reachable/absent outcomes100%. Utility=success(correct stop for absent)
minus0.05*external cost. These gates are conjunctive.240CPU seconds training/evaluation.
No claim that same templates with fresh descriptors establish broad task transfer.

Tests: planner known/unknown/absent action choices, no optimistic chance branches,
branch nonmutation, readout gradient through actual thinker, no label forward input,
entity retry and joint snapshot restore, bounded action loop, donor hashes and resume.
Use existing multimodal recipe/Run/reports, new small task/model modules. Structural
report QA under existing browser restriction. Preserve failures rather than tuning.

Claude reconciliation: no oracle actions in training; new external observations are
retained in the no-history control. Deterministic search order is box0 then box1,
inspect/open/retrieve; a candidate replaces the incumbent only above1e-9, with stop
initially preferred at equal value. Budget exhaustion without verified success is
failure (utility minus incurred cost), recorded separately from false stop. Snapshot
covers entity/belief/known/receipt state at call boundaries; this slice does not yet
resume a running external environment halfway through an action. Cached run resume
is separately tested. Review receipts: key-box-contract and key-box-reconcile under
runs/reviews/continuation_2026-09-11. Smoke uses a declared exact matcher substitute
for transaction plumbing; formal evaluation uses the actual frozen learned matcher.
Fixed text event shape to the encoder's [B,T] contract after smoke failure.

First integration (8f00d8d): full screen fails; known readout82.29%, reachable75%,
absent100%, integrated utility0.738672 vs no-history0.630469. Supplied-state100%.
Independent execution/search replay passes. In remembered/moved known pairs the
first queried box is100% correct, second43.75%/50%; training's repeated same-entity
second read differs from deployment's switched-entity second read. This is a
training-contract hypothesis, not proof that every failure has that cause.

One targeted iteration: keep original seed2301, batches, initialization, steps256,
first read and optimizer unchanged. Second read receives batch-reversed entity latents
and batch-reversed targets instead of the same entity. Test alignment explicitly.
Final candidate and saved first-model weights evaluated on fresh descriptor families
seed2411 (16 families, same conditions), all existing gates unchanged. Preserve first
run; no further tuning this slice.240CPU seconds including both evaluations; no repeat
baseline training. Bind baseline checkpoint hash to candidate run. No clearing working
state to hide interference; candidate training directly exercises query switches.

Claude reviewed query-switch intervention. Batch indices are not model inputs and
attention operates within each example, so reversing iid batch examples supplies
new within-example query pairings rather than an exposed index shortcut. Retain
its narrower concern: improvement would not uniquely distinguish learning to switch
from benefits of decorrelated repeated supervision. Do not claim that mechanism.
No new hyperparameters or labels introduced. Added full comparison-path resume smoke.

Run the integrated slice through the existing editable recipe:

```python
from experiments.multimodal import train_key_box
train_key_box(
    'runs/entity_variable_v1/frozen_adapted.pt',
    'runs/entity_noinfo_v1/frozen_state.pt',
    'runs/my_key_box',
    query_switch=True,
    eval_seed=2411,
)
```

Use identical arguments plus `resume=True` for cached resume. A reference checkpoint
can be passed as `reference_weights=...` for the matched frozen comparison. The
entity-memory snapshots preserve semantic content; the generic agent's episodic
stores receive neutral event packets in this slice. Thus success would validate
external entity retrieval into the workspace, not long-term semantic compression
inside the generic belief memory. Goal/action semantics and invalidation notices
remain supplied by the task harness. No full mid-action environment restore claim.

Second iteration (2b2fb61), fresh seed2411: known initial content100%; reachable
92/96=95.8333%; absent32/32=100%. Frozen first model on identical cases: known
82.8125%, reachable73/96=76.0417%. Candidate utility0.891015625 remains below
no-history0.9046875 (required advantage0.01), so full screen FAIL; no promotion.
Four candidate failures are false stops (families2/3, remembered/uncertain). Traces
show confidence falling on later reads despite an initial correct read or positive
inspection. No extra fitting or threshold changes after this result. Next proposal:
train/test readout stability across the actual observe/read/act interleaving, not
only two reads. Attribution to a specific internal mechanism remains unproven.

29 distinct relevant tests pass (28-test integration suite plus query-switch test);
comparison-path smoke/resume, formal cached resumes, independent search/physical
execution replay, initial-model equality, training-sampler equality, frozen donor/
non-thinker equality and frozen-reference identity all pass. Original result:
`runs/key_box_v1/verification.json`; second result:
`runs/key_box_v1/switched_verification.json`. Reports are structural-only under
prior browser restriction. Both failed capability screens remain available.

History stability iteration: train four read pairs with a neutral observation
between each pair, keeping the recurrent workspace. Same seed2301, batch32,
256 updates, optimizer and donors; query switching stays enabled. Average all
eight supervised read losses. This changes sequence exposure and compute, not
architecture; it is not a matched-compute attribution test. Fresh evaluation
seed2421,16 families, frozen switched checkpoint comparison; all gates unchanged.
Budget240 CPU seconds. Check read/target alignment, event ordering, gradients,
report and cached resume. No controller-derived supervision or oracle inputs.
This approximates action interleaving; changing entity contents within that
unroll and on-policy training remain outside this bounded comparison.
