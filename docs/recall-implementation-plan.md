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

Status: implementation in progress; red checks next.
