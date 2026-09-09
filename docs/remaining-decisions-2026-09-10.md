# Decisions for the 10 September discussion

Status: preparation in progress overnight, authorized by Alex through 09:00 Berlin.
The historical-recall implementation is complete in commit `32e9f99`. Entries below are
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
| 1 | Long-horizon learning and useful compression | A bounded forward memory is insufficient if distant write operations receive no useful learning signal | Reviewed and reconciled; proposal ready |
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

## 1. Teach individual memory updates, then measure long-delay usefulness

Proposed next step: keep the complete-session recall task and add directly grounded
questions at compression and consolidation boundaries. The existing multimodal
recipe already has memory-related surrogate objectives; the proposed change makes
the historical task's local storage supervision explicit and tests whether it helps.
This extends the learning signal, not the persistent memory capacity.

Imagine the agent observes a key on a shelf, then later sees it on a desk. A
compression probe asks where it was last observed within its declared source
interval. A consolidation probe tests both an older fact that should survive and
the newer correction. Labels come from the training harness's original delivered
records. They never come from a reconstructed memory being tested or the hidden
current state. An absent record in a local window cannot establish whole-session
absence.

The local student reads only the new compressed or consolidated tokens, the question,
and legitimate time/scope metadata. It cannot access the original records, the old
summary as a second read path, current recurrent belief, or a prior task workspace.
Reuse the ordinary learned query/read components where practical, with an explicit
restricted input contract. The deployment reader still receives its normal context;
the restricted view is an auxiliary training/diagnostic path with a different scope.
The question is legitimate reader input when that read occurs. It remains unavailable
to earlier write operations. Consolidation receives its normal previous consolidated
state and incoming compressed block; it does not gain raw-history access from the
training harness. Recompute the prefix under current weights for each sampled replay
initially. Caching old states across parameter updates is a later efficiency option,
with a separate staleness check.

| Operation | Available information | Trainable path |
| --- | --- | --- |
| Rebuild old prefix | Original training observations, current weights | No gradient; preserve complete state and memory |
| Local compression | Recomputed source records from the selected interval | Compressor and restricted reader; an encoder receives this loss only if its permitted source encoding was recomputed in the local graph |
| Local consolidation | Detached previous consolidated state and incoming compressed block | Consolidator/gate and restricted reader; compressor too only when explicitly included in this differentiable local chain |
| Grounded query loss | Student probabilities versus exact scoped record label | The permitted student path; labels do not enter its forward inputs |
| Optional richer-history distillation | Detached teacher distribution versus student distribution | Student storage/read path; teacher receives no gradient and its guesses remain inferred targets |
| Final session query | Full causal belief/memory and late query | Existing bounded suffix and read path; no claim of gradient into detached ancient writes |

For the first local-loss comparison, choose exactly one trainable compression or
consolidation boundary per probe example, with its input records/states detached.
No gradient crosses preceding consolidation transitions. Existing observation losses
train encoders; the local loss trains storage and its restricted reader. Recomputing
an encoder/compressor chain inside that local graph is an explicit later variant,
not an unspecified default. All query-conditioned addressing occurs in the reader.
Cached content-only keys are compatible with this rule; future-question-conditioned
keys prepared during an earlier write are not.

Start with exact scoped labels and ordinary observable grounding. Add teacher
distillation only as a separate comparison if useful labels are unavailable or
too narrow. A teacher can reinforce its own mistakes; a low distillation loss is
not factual validation. Jointly learning a storage code and its reader is allowed;
held-out question families and fresh bindings test whether that code transfers.

Sample consolidation probes from both older retained content and the incoming block,
including latest-visible-value corrections. Probing only newly entering facts could
reward an updater that overwrites the old summary every time. Record how many
consolidation transitions a tested fact has crossed. Local supervision can teach
individual preservation/correction operations; success after repeated updates still
needs evidence.

A local update cannot recover a fact already absent from both its old state and
incoming inputs. Sampling earlier write boundaries supplies separate training
opportunities for upstream retention; it does not reconstruct lost facts by fiat.
The relevant risk is learning a plausible answer from dataset priors while memory
remains uninformative, which fresh bindings and memory interventions must expose.

Keep the probe distribution explicit. A future question may select a training loss
without becoming writer input. However, if the queried entity is sampled independently
and uniformly, a relevance predictor cannot know which entity that particular future
question will choose. Do not add such a predictor on the assumption that it creates
information. The initial objective should preserve expected usefulness across the
declared task/question distribution, including questions not predictable from the
current task.

The simpler alternative is fixed compression/selection with a learned reader. It
is a meaningful reference for whether learning the writer earns its cost. A later
finite-horizon recomputation/checkpointing comparison can extend direct task credit.
It changes training compute and activation memory, not online memory capacity.
Checkpointing can preserve the corresponding finite graph's gradients when
dependencies and random draws match; explicit detachment is the operation that
cuts credit. Neither method guarantees unlimited-horizon learning.

Local auxiliary compression has precedent in
[Compressive Transformers, section 3.2 and Algorithm 2](https://arxiv.org/pdf/1911.05507).
That paper's attention-reconstruction objective and gradient choices motivate an
alternative; they do not validate our recursively consolidated factual memory or
make frozen readers universally necessary. The recomputation tradeoff and random-state
requirements are documented in [PyTorch's checkpoint interface](https://docs.pytorch.org/docs/2.9/checkpoint.html).
The proposed scoped factual probes and combination above are design inferences.

Before any experiment, distinguish two claims. For the new objective, compare the
same architecture with and without the local grounded loss and account for extra
queries/updates/replay computation. For the hierarchy itself, train suitable
recurrent/recent-only and fixed-compression controls on the same episode populations.
Removing memory only at evaluation measures reliance, not superiority to a trained
alternative. Recurrence can retain a fact after its original record is evicted, so
eviction alone cannot certify that a test requires the external hierarchy. Declare
the primary resource constraint and report bytes, parameters, computation and latency;
one scalar match does not equalize all of them.

**Decision for Alex:** make grounded local memory supervision the next implementation
proposal, with fixed compression as a reference; retain longer finite-horizon
checkpointed learning as a later comparison. Numerical loss weights, probe mixtures,
budgets and success thresholds need a concrete experiment plan after the design is
chosen. No new experiment is authorized or run by this proposal.

Review receipts: `overnight-credit`, `overnight-credit-reconcile` and
`overnight-credit-clarify` (actual Claude responses via the isolated CLI; the primary
review used Opus 5). Claude explicitly accepted corrections to checkpoint-bias,
invented provenance-loss, mandatory gradient-barrier, reader-only distillation,
random future-query prediction and attribution claims. A final clarification
corrected its overbroad exclusion of questions from readers and confirmed the
normal consolidation input contract. Its remaining depth/addressing questions are
resolved by the one-boundary default and reader-only query-conditioned addressing
specified above. Independent notes preceded each response. Grounded versus
distillation-first learning remains an empirical tradeoff; the recommendation here
prioritizes grounded labels and retains a fixed-compression reference. Agreement
does not establish effectiveness or authorize an experiment.

## Continuation for the overnight work

Next: review marking and fixed-budget allocation, then the remaining rows in order.
Avoid reopening the settled causal/gradient distinctions unless new evidence changes
them. For each group prepare an independent note and public conceptual brief before
reading Claude, verify consequential claims, reconcile errors, and append a concrete
recommendation here. Keep all implementation and experiment proposals unexecuted.
By 08:30 stop initiating reviews, then produce a compact decision sheet and identify
which choices need Alex versus a separately declared experiment. Deliver by 09:00
Berlin and pause `overnight-agent-design-proposals`.
