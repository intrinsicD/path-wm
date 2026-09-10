# Selective recall implementation and overnight handoff

Authorized by Alex on 10 September 2026: implement the reviewed historical recall
task, then work out proposals for remaining decisions with Claude until 09:00
Europe/Berlin today. Follow [the task contract](recall-task-design.md), preserve the
Gaussian/multimodal paths, and use the existing recipe, optimizer/Run/resume and
standalone report pipeline. No broad scientific comparison is authorized here.

The usable path is `experiments/multimodal.py --dataset recall --state-model belief`.
Add exact query/decision/verification records beside task consumers, five factual
logits, complete canonical text episodes, all-query supervision, fixed two-round
retrieval and 0/1/0.25 costs. Distinct train/development/calibration/test episode
identities; development NLL selects a checkpoint, calibration fits one bounded
positive temperature, and cached test logits support raw/adjusted metrics and cost
sensitivity. Independent history verification must not become model input.

Bound training graphs by retaining gradients only in the final configured replay
segment, while executing the entire earlier prefix with unchanged weights/state.
Forward memory is never reset at the boundary. Final-segment text grounding and
categorical KL accompany final factual NLL; earlier source encodings and historical
compressor graphs are detached. This is an explicit initial truncated learning
scheme, not proof of full-horizon credit assignment. No fake image/audio/actions.

Essential red tests: inclusive historical label and never-seen versus forgotten;
hidden-world independence; exact cost/tie/invalid/zero-coverage behavior; calibration
preserves labels and uses only provided held-out data; full replay/truncation forward
equivalence; no label/query leakage into history; gradient routes, bounded memory,
safe result snapshots and exact pause/resume including selected weights/calibration.
Commit the plan and informative red failures before implementing behavior.

Development budget: CPU only, existing suite plus focused checks, at most 8 updates
on 15-episode training and 15 each development/calibration/test with a 16-event
small-memory fixture; a 2-update pause/resume comparison; one 256-event default-memory
forward/backward check. A real recorded PushT batch regression may be run if shared
behavior changes. No quality pass threshold: report negative results and raw metrics.
Each completed run retains source/checkpoint/metrics and a structurally/browser
verified offline report. Numerical failures remain visible; no test-data selection.

After implementation: maintain `docs/remaining-decisions-2026-09-10.md`. Review
bounded groups of unresolved decisions with actual isolated Claude, using public
conceptual briefs and independent reasoning. Reserve time after 08:30 for synthesis;
deliver the decision agenda by 09:00 and pause the thread heartbeat
`overnight-agent-design-proposals`. It runs every half hour with a deadline in its
prompt. Do not start new model experiments or adopt those proposals overnight.

## Implementation and evidence

Implemented in `pathwm/models/recall.py`, `pathwm/evaluation/recall.py`, the existing
recipe and report renderer. See [the user guide](recall-task.md). The plan and red
contract checks were committed as `95e849e` before implementation; initial failure
was the missing recall module. Seven focused tests now cover labels/cutoffs,
independent verification, safe query/decision loading, forward truncation invariance,
hidden/evaluator-field isolation, gradient routes and exact replay/finalization.
The complete suite passes 76 tests.

The first attempt to retain pytest artifacts failed because the parent output
directory did not exist (two fixture setup errors, no failed learning run). After
creating it, all seven focused checks passed. Their exact resume checks retain two
completed two-update runs. The final suite repeated the resume comparison after
adding separate latency logging and restricting the cost views to the three declared
values. Completed earlier runs and source snapshots are preserved.

Eight CPU development updates completed under `runs/recall_v1/development`, on
15 training and 15 development/calibration/test episodes each, 16 records per
episode, recent=2, block=2, compressed blocks=2, final gradient segment=8, batch=1.
Initialization plus every update was eligible for development-NLL selection;
update 7 won with NLL 1.547959. Calibration fitted T=20 at its upper bound.
Test factual accuracy is 20% (all top-class predictions are not-observed), raw
NLL 1.611440, adjusted NLL 1.608746. Both views abstain on all 15 cases, with loss
0.25; the eight seen-old cases also all abstain and have 0% factual accuracy.
This is negative capability evidence from a tiny development check, not a trained
memory demonstration. No thresholds were retuned or extra training launched.

The independent one-episode check at 256 records used default capacities and the
default final 32-record gradient segment. It completed forward/backward with finite
losses and gradients through the outcome head, state update, text adapter, readers,
compression/consolidation; the selected-model copy has no gradients. The full suite
preserves the Gaussian and ordinary multimodal paths. No further real-data run was
needed because those numerical paths are unchanged and their regression tests pass.

Raw receipts: `runs/recall_v1/final_checks.xml`, `default_memory_check.json`,
`development/{last.pt,metrics.jsonl,recall_results.json,recall_predictions.pt,
recall_performance.json,report.html}`, and retained exact-resume runs under
`contract_checks/` and `final_checks/`. All five retained recall reports passed
browser checks at 1280×720, with no broken images or horizontal overflow. Summary
and confidence-bin counts were improved after training; raw results and captured
training source remain unchanged, while QA records the renderer actually used.
Use `python -m pathwm.evaluation.report RUN` to rebuild reports without a training
resume/source-compatibility check. The verification receipt is
`runs/recall_v1/verification.json`. The implementation phase is complete; the
seven overnight topic reviews and combined Claude reconciliation are complete in the
morning agenda. Its final scheduled handoff remains due before 09:00 Berlin; the
broader proposals are not implemented or adopted.
