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
| 2 | What to mark and how to spend fixed memory | Retention should serve future tasks without seeing future queries at write time | Reviewed and reconciled; proposal ready |
| 3 | Instructions, exact objectives and verification | The current controlled query adapter does not interpret arbitrary user requests | Reviewed and reconciled; proposal ready |
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

## 2. Score a concrete protection decision within the existing capacity

Keep the current store capacities and one shared protected pool. Preserve complete
event envelopes initially; their source encoding is already lossy. A user saying
“remember the location” supplies a retention preference, not proof that the model
encoded that location correctly. Protecting an envelope preserves its stored values;
it does not guarantee raw-frame reconstruction or accurate factual recall.

User priority stays exact. A user request can displace an agent mark; if every
protected slot contains a user mark, report that the new mark was not admitted.
Do not silently evict a user mark, grow the store, or reintroduce selective memory
reset controls. Initially mark the latest committed observation, matching the current
interface. Addressing older still-retained events is a later extension; reconstruction
from a compressed trace cannot recreate the original detailed envelope exactly.

For agent marks, replace a permanent event-importance ranking with a proposed
**admission-action score**. For one new candidate, enumerate the permitted actions:
keep the current bank; insert if there is space; or replace each eligible agent
record. Score each change relative to keeping the same current bank. Inputs are the
current causal state/task, candidate, proposed displaced record and bounded bank
context. Select the largest estimated positive gain; ties keep the bank unchanged.
User records are never eligible eviction candidates for an agent proposal.

This makes the scores comparable at that decision and accounts for the named
replacement. It does not optimize every possible set of memories or guarantee that
the score is accurate. Complementary clues can have low individual value but high
joint value. Novelty may generate candidates; it is not the definition of usefulness.
Use exact record identity for duplicate handling. An approximate latent-similarity
gate can merge distinct updates and is unnecessary in the first version.

The current implementation has a scalar event scorer and a batched mean in its
admission path. The proposal requires independent admission per session and explicit
replacement context; it is not a description of that code already working. Recompute
action scores at admission from current context. Do not compare a fresh candidate
to an old scalar from a different task or model version. Scores may change as the
task changes; source timestamps and observed facts do not become fresh evidence.
Keep weights fixed within a live session initially.

For offline supervision, replay from the proposed admission point under matched
weights, visible observations and random draws. Compare keeping the original bank
with the capacity-respecting insertion/replacement. Both branches retain ordinary
history, compression and belief updates. Initially suppress further mark admissions
in both continuations, then ask the same late query. The label is the difference in
factual NLL, detached before training the score predictor. This is the controlled
effect of **one admission**, not the value of an entire adaptive marking policy.
The future question is reader input only when it arrives and is never earlier
scorer/writer input. The harness may evaluate rejected as well as admitted candidates;
online exploration is not inherently needed for this passive replay fixture.
The first score-regression loss updates only the score predictor over detached causal
features; the paired replay labels and discrete admission do not backpropagate into
the world model. Grounded state/read learning remains separate. A pool full of user
marks has no eligible agent action and supplies no positive/negative discrimination
label. Record this as capacity-blocked, not as measured zero usefulness.

A zero gain when ordinary memory or recurrence already answers correctly is a valid
result: extra protection supplied no improvement under that comparison. It does not
mean the original observation was unimportant. Nor can a small score reveal which
latent path retained the fact. Use declared source-position groups and separate
read interventions as diagnostics, with their limited interpretation.

NLL provides a dense initial training target, while actual answer cost remains a
separate evaluation. For example, increasing the true-class probability from 0.30
to 0.70 improves NLL substantially but still abstains at the 0.75 threshold. Moving
from 0.74 to 0.76 gives a smaller NLL gain but changes a correct decision from
abstention to an answer. The arithmetic is retained in
`overnight-marking-arithmetic.json`; it is not a learned-agent experiment. Avoid
adding an unexplained byte penalty to NLL. Equal-sized slots already impose an
opportunity cost through the replacement; variable-sized records would require an
explicit budget and unit convention later.

Compare to fixed recency/novelty and random protection policies with the same user
priority and capacity. Report actual occupancy and compute as well as old-fact
recall, task loss and wrongful answers. Keep user-marked, agent-marked and unmarked
conditions separate. A user mark conveys information about retention preferences;
it is not an oracle usefulness or factual-truth label. If the underlying reader
cannot use detail in either replay, utility targets may be uninformative; the
grounded-memory work in decision 1 is a dependency.

**Decision for Alex:** retain the existing user-priority capacity policy and propose
learning insertion/replacement value for agent marks, beginning with a controlled
single-admission replay task. The simpler alternative is fixed recency/novelty
protection while learning the reader. Broader adaptive-policy credit and selective
detail encoding remain separate later comparisons.

Reviews: `overnight-marking` and `overnight-marking-reconcile`. Claude accepted the
replacement-context critique and withdrew treating redundant protection's zero gap
as an underestimated value, latent scores as causal explanation, mandatory live
exploration, exact detail certification, semantic near-duplicate merging, new
subquotas and release/reset controls. It retained the substantive limits: complementary
sets, rare candidate/evictee combinations, changing bank distributions and a proxy
loss can defeat a one-admission learner. FIFO avoids a learned ranking but its
downstream quality is still affected by the encoder and reader; no universal
drift-invariance or superiority claim is adopted.

## 3. Resolve natural instructions into a small exact objective

The existing instruction path proposes operations/output formats; the historical
query adapter supplies an exact controlled objective. Neither demonstrates general
language-to-goal understanding. Propose a supervised interpreter for one small task
family first: natural paraphrases of last-observed questions, including references,
scope, temporal distinctions, ambiguity and unsupported requests. It can share
learned task/text components; no external runtime LLM or universal parser framework
is required by this design.

Keep the original request and a small proposed record:

| Field | How it is resolved |
| --- | --- |
| Request/channel, task ID and revision | Caller-owned identity and history |
| Objective kind, entity/payload, source and temporal scope | Learned interpretation constrained to supported schemas |
| Required outputs | User controls and declared application defaults; model proposals remain attributed |
| Capability, cost, budget and verifier-policy references | Resolve against trusted runtime definitions; the model cannot create authority or executable verifier code |
| Applied defaults and bounded supporting source references | Audit what was assumed; proposed references do not prove a faithful interpretation |
| Disposition: proceed, clarify or unsupported | Policy combines exact missing-field checks with an evaluated learned interpretation/ambiguity signal |

The record is exact once resolved, but may still encode a mistaken interpretation.
Schema validity is not intent accuracy. Retain learned task tokens for retrieval
and reasoning alongside the exact record used for execution and checking. Clear
supported requests use declared defaults automatically. Ask a focused question
when unresolved interpretations materially change the requested action or outcome;
do not add routine confirmation of every valid parse. Report unnecessary questions
and confidently wrong interpretations separately.

“Where did we last see the key?” is historical recall. “Where is the key now?”
requires a different current-world objective and possibly sensing; until that
consumer exists, do not silently answer the historical question instead. For a
document edit, “replace this heading in the current draft” can bind to an exact
document/version and check the actual changed heading. “Make this match the policy”
may require resolving which document and which kind of change. A checker for the
heading does not certify the truth or completeness of the whole document.

An explicit user request to follow a referenced procedure can supply task content
from that procedure within the user's existing authority. The referenced document
does not become a new user-authority channel. Instructions encountered incidentally
in an observation cannot silently create a new task or widen capabilities.

Each revision invalidates pending decisions while preserving executed effects and
spent costs. Replan against the revised goal. Any compensation is another action
with its own capability checks and costs; do not pretend already executed effects
were rolled back because a task record changed.

Keep delivery, predicted outcome and verified outcome separate. A verification
record names the actual candidate/output, task revision, checker/version, checked
scope and supporting evidence. Its truth status is verified, contradicted or unknown.
Unknown carries a reason such as inconclusive, skipped by policy, missing evidence
or exhausted budget. Known action/check costs remain recorded; the task contract
must specify how unverifiable outcomes affect termination and evaluation. Neither
automatic zero penalty nor automatic failure is universal. A physical placement
outside subsequent sensor coverage may be executed yet unverified.

Specify checks before inspecting the outcome. Runtime validation enforces deterministic
bounds and capability availability, while learned predictions remain estimates.
Small typed outcome consumers are the initial preference; sharing their backbone
or learning a common task-conditioned head is an empirical choice. A universal
success logit does not replace a checker. Fallible model critiques may supply
explicitly tagged auxiliary training feedback, but independent evaluation needs
authoritative references or declared human judgments.
Tag and version auxiliary judge labels so future data use can be audited or filtered.
That does not undo parameter updates from past training; correction may require
retraining or restoring an appropriate checkpoint.

Train interpretation from approved contracts and ambiguity/unsupported examples.
Split dialogue/paraphrase structures and fresh entity bindings; add separate
unseen-name and unsupported-objective challenges where applicable. Do not require
disjoint vocabularies in every fixed-head task or call difficult supported
compositions unsupported by definition. Measure field/intent agreement, disposition,
actual task loss and verification coverage, with an exact-contract control to
separate parser errors from world-state/decision errors.

Explicit learned interpretation and revision have precedent in
[Task-Oriented Dialogue as Dataflow Synthesis](https://aclanthology.org/2020.tacl-1.36/).
Testing new compositions separately is motivated by
[Measuring Compositional Generalization](https://arxiv.org/abs/1912.09713).
These support mechanisms and evaluation distinctions; they do not establish this
agent's language competence or validate the proposed authority/verification contract.

**Decision for Alex:** add a narrow supervised natural-language objective interpreter
and the minimal exact contract before expanding into general actions. Preserve
autonomous defaults, focused clarification and scoped independent verification.
The simpler alternative is to keep exact programmatic queries while improving
world-state and memory learning first. This choice changes interface breadth,
not the learned semantics of world-state tokens.

Reviews: `overnight-objectives` and `overnight-objectives-reconcile`. Claude withdrew
universal cost-free unknown status, a blanket rejection of explicitly delegated
procedure content, inevitability claims about shared heads, a ban on all weak model
labels, and compulsory unseen-entity splits. It accepted preserving executed effects
through revisions, one unknown status with reasons, and focused clarification without
routine confirmation. Remaining application choices concern the cost of checks and
the treatment of unverifiable outcomes; they must not hide unknown cases by reporting
only successful verification. The first controlled tasks can keep complete evaluator
truth while the online agent remains uncertain.

## Continuation for the overnight work

Next: review hypothetical observation correction/sensing, model uncertainty and
bounded thinking/planning; then transfer/evaluation and the cross-topic dependencies.
Avoid reopening the settled causal/gradient distinctions unless new evidence changes
them. For each group prepare an independent note and public conceptual brief before
reading Claude, verify consequential claims, reconcile errors, and append a concrete
recommendation here. Keep all implementation and experiment proposals unexecuted.
By 08:30 stop initiating reviews, then produce a compact decision sheet and identify
which choices need Alex versus a separately declared experiment. Deliver by 09:00
Berlin and pause `overnight-agent-design-proposals`.
