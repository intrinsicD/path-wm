# Modular World State foundation

15 September 2026. The user authorizes the complete basic code structure, modular
replacement and debugging during training/inference. This implements the reviewed
[hybrid proposal](world-state-proposal-review.md), not a claim of general entity
discovery, concept learning or world prediction.

## Implementation slice

1. Add ordinary records and a bounded, single-writer event/transaction store under
   `pathwm/world_state/`. Entity, shaped latent/readable component, relation and
   evidence records share one revision. Append operations, reject stale/conflicting
   writes, publish atomically, restore/replay exactly. Existing task-specific entity
   modules remain usable references; do not change their behavior.
2. Implement replaceable exact retrieval and binding, supplied-candidate feature
   encoding, recurrent updates and context projection using normal PyTorch modules.
   Add an explicit session adapter to the existing BeliefAgent. Persistent commits
   detach; functional neural forwards support training. No new universal trainer.
3. Provide concrete extension records/functions for concepts, self/control,
   feedback, predictions and action selection. Optional learned policy/regulation
   blocks are replaceable and off by default; no autonomous tools/actions.
4. Add one bounded diagnostic schema: tensor/gradient/attention summaries, optional
   capped tensor values, binding and retrieval decisions, attribution, event history
   and interventions. Reuse the current Run/checkpoint/report infrastructure.
5. Add an editable CPU recipe demonstrating the entire base path, tiny supervised
   training, restart/correction and report inspection. Run focused regression tests
   and independent artifact/replay checks; document remaining learned capabilities.

## Contracts fixed before implementation

- Candidate proposals are supplied regions/mentions, not inferred ground-truth IDs.
  A vector's representation name/version must match before similarity is meaningful.
- Store IDs persist independently of learned keys. All post-merge writes retain the
  original matched entity. Accepted identity links change only a canonical read view.
  Revoking a link never unmixes vectors; explicit reattribution corrects stored claims.
- Evidence is stored even for unresolved bindings. Predicted/inferred claims cannot
  advance `last_seen`. Observation and availability time are distinct. Every read
  is pinned to one snapshot and explicit temporal cutoffs; future corrections cannot
  leak into historical reads. Initial records are bounded, without silent eviction.
- A transaction uses an exact base revision. A completed identical retry returns its
  old receipt without reapplying; a conflicting retry or new stale transaction fails.
  Entity/core state publication occurs only after both updates succeed. This is a
  single-process single-writer boundary, not a distributed database transaction.
- Components/relations retain evidence and origin. Relation endpoints can reference
  an entity or a component; current-snapshot component references follow corrected
  attribution. Historical revisions remain reconstructible from the operation log.
- Retrieval bounds entities, components, relations, events and numeric payload; a
  context projection has explicit accepted dimensions and token cap. Unknown or
  oversized representations are reported, never silently treated as valid features.
- Model fingerprints gate runtime restore. Training checkpoint/RNG ownership stays
  in Run. Debugging must not change outputs, consume RNG or retain autograd graphs.

## Essential tests and budget

Red tests first, then CPU tests for atomic rejection, stale writes, retries, invalid
references, merge/write/split, late evidence and historical queries, restored state,
capacity, gradient boundaries, replaceable modules and actual reasoner consumption.
Test debug neutrality/caps/gradient hooks and inspectable interventions.

One small CPU development fit (default 96 updates, batch16, two supplied entities;
under two minutes expected) checks learning/gradients and exact training resume.
No real-world capability gate or sample-efficiency claim. The recipe must save raw
metrics/checkpoint, result status and standalone report. Mechanical checks must pass;
loss must decrease on the fixed development task. No tuning on a test population.

## Review and progress

Claude's first conceptual review requests merge-time attribution, exact staleness,
atomic temporal reads and unresolved-evidence retention; these contracts are above.
We keep the user-requested extension interfaces but make them concrete small clients,
not a new directory tree or a claim that optional behaviors already learn. One debug
schema serves console/raw/report views. Direct store writes reject attached tensors;
the session deliberately detaches after neural computation.

## Implemented and verified

Code: `pathwm/world_state/`, [API and guide](world-state.md),
[recipe](../experiments/world_state.py). Main commits: `13c27ac`, `567d47a`, `54825d2`.
Core records/store, scorer/policy, updater, query/context/relation modules, prediction,
optional prototype/self/feedback clients and training/inference inspection all have
working consumers. No new framework or backend dependency.

Final evidence: [report](../runs/world_state_foundation_v1/final/report.html),
[independent checks](../runs/world_state_foundation_v1/verification.json),
[81 focused tests](../runs/world_state_foundation_v1/focused-tests.xml).
All new world-state tests and selected Run, event, task, entity and multimodal
regressions pass. Two broad test attempts were stopped to address newly discovered
issues after 215 and 173 passing tests; no complete all-repository suite is claimed.

The fixed 96-update CPU development task reduces loss 6.156751 -> 0.00448038;
all four supplied identity/property combinations are decoded correctly. Two histories
with identical final input produce the correct distinct answers through the persistent
store and actual thinker. Two identities survive hidden-property updates, merge/link
revocation works, and zeroing retrieved state changes readout logits. This is one
tiny development task, not real-world recognition, sample-efficiency or concept proof.

Continuous 96 updates and 48+48 resumed updates have exactly identical complete
checkpoints: model, optimizer, sampler, RNG, progress and metric rows. Corrected
session snapshots also replay exactly, including the internal commit clock.
3293 independent checks pass. 60,934 total parameters, 3,812 trainable in this recipe.
Final direct command 4.10s; pause/resume commands 3.42s + 3.51s, including evaluation and
reporting. All work is CPU; all development artifacts together approximately 15 MiB.

### Failures found and fixed

- The four-update preflight appropriately left one candidate unresolved; no false
  claim that an untrained scorer already distinguishes entities.
- The first 96-update run learned the task but failed exact traced/nontraced thinker
  replay (maximum token difference 9.54e-7). `need_weights=True` selected a different
  attention kernel. Native attention now always uses its normal path; a detached
  diagnostic computes probabilities separately. Tests prove equal outputs, parameter/
  input gradients and RNG in training/eval, plus masked-weight agreement.
- Reattribution now invalidates transitive dependent states; explicit replay and
  preserved evidence prevent silent reuse of tainted latents.
- Internal/correction transactions now publish store and neural clock/RNG together;
  direct ahead-of-core store mutation fails visibly.
- Debug hooks now traverse nested feature outputs. Exact float64 captures preserve
  dtype; reports use the same schema. The audit's initial expectation of two `img`
  elements was corrected: responsive charts use one `img` plus one `source`.

Reports are structurally checked, embedded assets verified and untrusted labels
escaped. Browser interaction QA remains unavailable under the established local-file
URL policy; no alternate browser/localhost workaround was used. Original diagnostic
runs and their failing gates remain saved.

Claude review: two actual isolated conceptual rounds under
`runs/reviews/world_state_foundation_v1/`; no private code/data/results sent. Clarified
that completed retries return old receipts without replaying writes after correction;
new transactions still require an exact base revision. Transitive invalidation,
pinned snapshots, unresolved evidence and version boundaries are exercised locally.

Next capability slice: fixed candidate granularity and real image/text observations,
confusable negatives, held-out identity/state/history tests, abstention calibration
and flat/no-history/oracle controls. General concept induction, useful exploration,
large spatial maps and reliable generation remain separate milestones.
