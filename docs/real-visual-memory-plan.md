# Real visual-memory connection

24 September 2026. Alex authorizes Codex and actual Claude Opus 5.5 at medium
effort to plan, implement, review, test and fix the agreed first actual-model slice.
Claude has the same repository permissions for this task; this explicit instruction
supersedes the older public-only review boundary. Preserve existing architecture,
checkpoints and historical evidence. Other rerun families remain separately open.

## Objective and working plan

Make the existing native R2 visual path rerunnable through persistent memory:
actual RGB64 -> shared MultiScaleImageEncoder -> retained complete FeaturePyramid
-> existing SlotPerception.from_pyramid -> existing BroadcastDecoder. No new codec,
encoder or learned representation. Slot compression is an existing consumer, not
a reason to discard the full source pyramid at the storage boundary.

First establish exact persistence, ownership/provenance, source correction,
unaffected-record preservation, restart and output equivalence to the live path.
Use the existing J perception/identity checkpoint and native width64, seven slots,
three iterations and decoder width32. Do not count agreement with a lossy live
reconstruction as proof of pixel fidelity; report source reconstruction error
separately. Train existing exercised components only if the declared quality task
requires it, preserving existing abilities and reporting failures.

Trace and reconcile the exact persistence/correction contract with Claude before
implementation. Granularity must follow actual feature dependencies: global
attention makes arbitrary spatial token replacement unsafe. Frame/source-level
correction must not be reported as object-part correction or unseen-operator
transfer. Pin model/source versions and reject invalidated retained reads.

## Measurable slices

1. Joint code-grounded plan, exact APIs and red regression tests.
2. Small missing connection in the existing library/session path, full-size tests.
3. Existing-checkpoint evaluation of actual live versus recalled output, source
   correction, independent unrelated source, restart and negative controls.
4. If needed, bounded training of existing modules and held-out reevaluation.
5. Independent Claude review, fixes, focused and full CPU suite, retained evidence,
   standalone report and updated continuation state.

## Initial budget and gates

Planning/software checks: native CPU configuration, two threads. One initial
checkpoint evaluation on fresh synthetic scenes, <=5 minutes CPU or GPU. No
scientific tuning on held-out examples. Training, if needed, gets its exact
objective, disjoint streams, preservation gates and bounded update count registered
before execution. Maximum initial training allocation: one GPU, <=6GiB, <=20 minutes;
no automatic sweep or architecture addition. Source stays fixed during every run.

Persistence gates: all scales and metadata survive exactly; same consumer output
within declared execution precision; stale/retracted/model-mismatched reads fail;
independent records remain unchanged; restart retains the same results; no weight
updates during memory acquisition/correction. These establish the connection,
not fine-detail quality or the full agent. Quality gates will be specified before
the first new quality evaluation after tracing the existing task and checkpoint.

## Status

Planning with actual Claude; no implementation or new quality result yet.
Exact briefs/replies: `runs/reviews/real_visual_memory_20260924/`.
