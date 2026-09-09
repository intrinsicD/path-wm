# Decisions for the 10 September discussion

Status: preparation in progress overnight, authorized by Alex through 09:00 Berlin.
The historical-recall implementation is being verified first. Entries below are
proposals to discuss, not adopted model changes or permission for experiments.
Actual isolated Claude reviews and reconciliation will be attached as completed.
Stop new reviews at 08:30 and use the remaining time for the morning synthesis.

The established architecture remains: learned world-state semantics, separate
observed evidence/current belief/task workspace, bounded recent/compressed/protected/
consolidated session memory, and separate perception/prediction/thinking readers.
A fresh agent session replaces selective memory reset. The first decision consumer
is historical recall with explicit abstention and independent verification.

## Proposed discussion order

| Priority | Decision | Why it matters now | Review status |
| --- | --- | --- | --- |
| 1 | Long-horizon learning and useful compression | A bounded forward memory is insufficient if distant write operations receive no useful learning signal | Pending |
| 2 | What to mark and how to spend fixed memory | Retention should serve future tasks without seeing future queries at write time | Pending |
| 3 | Instructions, exact objectives and verification | The current controlled query adapter does not interpret arbitrary user requests | Pending |
| 4 | Hypothetical observation updates and sensing | Planning an inspection needs observation-conditioned continuations without contaminating live history | Pending |
| 5 | Model uncertainty and calibration | Latent variability is not an estimate of model error; selected actions may exploit prediction mistakes | Pending |
| 6 | Planning and thinking budgets | Search and retrieval costs need concrete caps and honest stopping signals | Pending |
| 7 | Transfer and evidence for world understanding | Canonical recall alone cannot establish visual mapping, document understanding or general competence | Pending |

Each review should supply a preferred initial design, a viable alternative, the
interface/gradient consequences, failure cases, required evidence and the precise
choice for Alex. Separate choices that unblock implementation from empirical
questions that need a preregistered comparison. Do not turn every tunable number
into a user decision. Keep proposed experiments bounded and unexecuted here.

## Review boundaries

Only public conceptual briefs go to Claude, using the existing isolated CLI with
tools disabled. No private source, dimensions, measurements or user history are
exported. Independent notes precede each response; consequential claims require
local checks or public primary sources. Preserve objections and corrections rather
than treating model agreement as validation. Receipts remain under
`runs/reviews/state_memory_design_2026-09-09/`.
