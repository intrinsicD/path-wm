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
