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
| 4 | Hypothetical observation updates and sensing | Planning an inspection needs observation-conditioned continuations without contaminating live history | Reviewed and reconciled; proposal ready |
| 5 | Model uncertainty and calibration | Latent variability is not an estimate of model error; selected actions may exploit prediction mistakes | Reviewed and reconciled; proposal ready |
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

## 4. Let planning condition on a possible observation without changing live history

The first active task should inspect once and then answer or abstain about a current
location, with the target stationary from query to answer. Historical last-observed
recall stays a separate task: inspecting now does not retrieve an earlier fact.
The action contract specifies duration, cost, finite returned categories and failure
behavior. The environment owns truth; a sensor return is evidence about it.

Create a private branch from a completed live event and frozen memory snapshot.
Advance the proposed action and duration once per acquisition candidate. Enumerate
its possible returns, correcting each in a separate branch from that same prior;
mutually exclusive returns must not accumulate as packets in one event. Then run
the bounded task reader. Share numerical correction components with live
perception, while keeping an explicit scratch wrapper and hypothetical provenance.
Branch results cannot commit, mark memory, consume real event ordinals or verify an
answer. Their temporary state and workspace are discarded after scoring. For this
first task, both actual and hypothetical terminal reads use the corrected state
and the pre-event memory snapshot, before the new observation is written to memory.
Apply any required pure state finalization consistently on both paths. The actual
event then writes normally; no branch memory hierarchy is needed. Executing a chosen
inspection uses a new ordinary live transaction and the actual return; it never
copies the simulated posterior into live history. Deliver the actual answer only
after that transaction has been handled. There is at most one inspection per attempt,
including a failed inspection; automatic retries are outside this first scope.

Planning consumes its separately declared compute budget, including failed branch
calculations. It does not consume a real acquisition quota or fabricate external
action costs. Telemetry may record that planning occurred, but cannot route private
simulator truth or hidden member identity into live belief. Expose only declared
operation counts, charged costs and remaining budget through the ledger; these are
runtime context, not world observations. Predicted branch scores reach the planner
through its explicit prediction interface and remain inferred quantities.

A return saying the inspection failed can still be evidence if failure likelihood
depends on location. It always carries the action/time/cost actually incurred.
An exact example with equally likely locations and failure likelihoods 0.8/0.2 gives
a posterior of 0.8/0.2 on failure, reversed on the other return. With wrong-answer
cost 1, abstention 0.25 and inspection 0.02, its expected cost is 0.22 versus 0.25
for immediate abstention. This is checked arithmetic, not a learned result. An
uninformative failure must not falsely increase confidence; an informative failure
must not be silently replaced by a neutral missing-data token.

For one acquisition, choose the best continuation after each possible visible
return, then average its loss and add acquisition cost. A hidden target or sampled
model identity used by the simulator is never continuation input. Skip truly
zero-probability branches only under the stated model; declare handling for actual
unsupported or invalid returns instead of manufacturing a normalized belief.

The existing planner evaluates fixed action sequences and has no correction branches.
The live event API deliberately rejects imagined states and generated source content.
A new scratch wrapper must preserve those protections; removing the guards is not
an implementation of this proposal. Check state/memory/counter invariance under fixed
random draws, exception-path isolation, live-versus-scratch numerical parity for
matching inputs and ordering, and independence
of a continuation from private labels when its permitted inputs are unchanged.

The read-only guarantee covers tensor aliasing, caches, mutable statistics and RNG,
where those exist. Shared live context is legitimate branch input; its derived
outputs remain hypothetical. This calls for ownership and mutation-boundary checks,
not a blanket ban on shared tensors or a compulsory generic taint framework. Software
errors cannot roll back an external action already executed. Preserve its actual
effects and costs, and distinguish valid sensor failure evidence from malformed or
unavailable evidence. Expected failures/no-data belong in the normalized sensor
alphabet. An actual out-of-contract return instead triggers explicit abstention or
the declared error/unknown path; it cannot silently produce a confident prior-only
answer. Preserve separately valid evidence and executed-action bookkeeping on that
path. Include such attempts and known costs in reporting. A meaningful catch-all
sensor status can be modeled; every private software fault need not become a
Bayesian observation. Classify valid returns versus contract violations by rules
fixed before execution, never according to whether the return supports a preferred
answer. Score factual probabilities across valid evaluable episodes regardless of
answer/abstain selection; report invalid/missing cases and all-attempt costs visibly.
Planning wall-clock latency is separate from imaginary time;
the first task assumes the target stays stationary throughout the attempt. A future
moving-world task must account for actual elapsed time.

**Recommended sequence:** first verify exact finite sensing algebra with a declared
known prior and sensor likelihood. Then compare a learned finite probability model
against the shared neural correction path. For the finite model, use a common
`p(Y | b, query)` and a normalized `p(O | Y, b, a)` to form the joint. The likelihood
head predicts a table across candidate target hypotheses; the actual hidden target
only indexes its offline loss. Considering a different pure inspection cannot change
the pre-return target marginal. The output categories belong to this task adapter,
not to the meaning of the world-state tokens.

Bayes conditioning is exact relative to that joint; its learned prior or likelihood
can still be wrong. For an explicit finite factorization, a proper joint loss can
train both factors without three separately supervised heads. The general neural
correction needs a grounded posterior learning signal on actual returns, alongside
prediction and existing world objectives. Start with supervised losses; the discrete
action minimum is not a differentiable training path. Imagined observations are
planning inputs, not factual labels. Complete evaluator labels and predetermined
action coverage make online forced exploration unnecessary in this first study.

The general corrector can be checked by marginalizing its predicted posteriors over
its predicted returns and comparing with its prior. Consistency alone is weak:
wrong or uninformative predictors can agree. Conversely, a uniform marginal over
sensor returns can be perfectly informative about a uniform binary target. Do not
reward entropy reduction merely to satisfy an information-gain target. Use grounded
probability scores and realized task loss, with both calibration and discrimination.

This is a staged comparison, not a requirement that every future neural corrector
match an oracle. The known-model oracle checks decision arithmetic; the learned
finite reference helps separate correction error from errors in predicted facts and
sensor behavior. Finite models can also handle moving targets or deeper plans when
their transition spaces remain tractable. Complexity is a reason to compare learned
correction, not a theorem that it becomes necessary at a particular depth.

Before deeper planning, require the causal checks, observable prior/sensor/posterior
scores, plan-versus-actual loss, acquisition rate and costs, and comparison with
answer-now/abstain and fixed-inspection references. Include uninformative/noisy
returns, failed returns and cases where current location differs from last observed
location. A known-kernel oracle and learned-head results must be labelled separately.
Set populations, budgets and any pass thresholds before executing the comparison.
At most `C*O` corrected continuations are considered for `C` acquisition candidates
and `O` returns; count their actual reader/model calls as well.

The observation-conditioned decision formulation follows the
[POMDP belief update and policy construction](https://cs.brown.edu/courses/csci2951-k/papers/kaelbling98.pdf).
The scratch transaction and training sequence above are our design proposals; that
source does not establish learned filtering or memory sufficiency in this agent.

**Decision for Alex:** use one-step inspection with an exact finite reference and
an isolated shared correction path as the next active-task proposal, after useful
historical learning. Prefer the common-prior finite factorization for the first
learned reference. The viable alternative is to train the general corrector directly
and retain finite Bayes only as an oracle diagnostic; it has fewer task-specific
components but makes prediction and correction errors harder to separate.

Reviews: `overnight-sensing`, `overnight-sensing-reconcile` and
`overnight-sensing-boundaries`. Claude accepted the common-prior factorization and
read-before-write ordering. It withdrew compulsory deep copies/taint machinery,
a blanket ban on shared live context, presumed read-side mutation, mandatory three
heads or likelihood floors, uniform-return-marginal claims, and forced online
exploration for the controlled offline design. It also corrected exclusive error
attribution to representation, calibration-without-discrimination, and claims that
finite models cannot handle later dynamics/depth. The final clarification separated
charged planning computation from external-action accounting and modeled sensor
failures from private software faults. Its remaining ledger-readback and ex-ante
return-classification concerns are addressed explicitly above. Correction quality,
action coverage and selected prediction error remain empirical. The algebra checks
in `overnight-sensing-arithmetic.json` and `overnight-sensing-consistency.json` are
finite illustrative calculations, not model experiments.

## 5. Start with observable decision risk; leave model-error estimates unclaimed

Keep one model initially. Its task head predicts observable outcomes; independent
calibration data may adjust those probabilities, and the exact task costs determine
answering, acquisition or abstention. Report factual accuracy and proper scores
alongside selected risk and coverage. Calibrating a nearly uninformative predictor
cannot make it remember facts or correct its ranking mistakes.

Three examples explain the distinction. A key could be in either of two locations
because it was not seen being moved; a good inspection can reduce that uncertainty.
A sensor may return noisy readings even when its mechanism is known. The model may
also have learned the wrong sensor mechanism or lost a relevant memory. The entropy
of one latent categorical distribution does not uniquely separate these causes or
estimate the chance that its own model is wrong. Repeated thinking adds no external
evidence, even when confidence increases.

Treat unsupported task kinds and invalid contracts explicitly. A learned novelty
or distance score is only an empirical indicator; neither low novelty nor ensemble
agreement certifies correctness. Fixed weights do not prevent input distribution
shift, and changing weights requires rechecking the calibration procedure.

Positive temperature scaling preserves each factual logit argmax, but it can change
which answers are accepted under abstention and which acquisition has lowest
predicted cost. Thus marginal calibration and calibration of the selected subset
are different questions. Freeze the complete evaluated procedure before test access,
report the resulting selected decisions, and never infer a subgroup guarantee from
an aggregate reliability curve. A specified calibration fit is allowed to choose
its parameter on calibration data; an untouched test population evaluates the final
pipeline. Repeated design tuning after inspecting test results needs new independent
evidence, rather than renaming the same test set.

A later bounded ensemble is a possible comparison. Members may have different latent
coordinate systems; combine probabilities in common observable outcome categories.
A cheaper shared-encoder/multiple-head alternative may miss shared representation
errors. Separately initialized models can also agree on the same mistake. Ensemble
variation is a model-error indicator under assumptions, not a calibrated probability
that the true mechanism is contained in the ensemble.

In sensing, mix the members' joint target/return predictions, condition the mixture
on the visible return, and choose one continuation. Averaging individually optimal
member decisions falsely gives the policy the member identity. With two equally
weighted members certain about opposite locations and identical uninformative
returns, the legitimate mixture still prefers abstention at cost 0.25. Averaging
the member-specific optimal losses gives zero, an artificial value of information.
Adding inspection cost 0.1 yields 0.35 versus the invalid 0.1. The fraction check
is retained beside the review receipts.

The finite sensing factorization should remain coherent through calibration: apply
a declared transform to its normalized factors and derive joint/posterior values,
rather than independently adjusting incompatible prior and posterior heads and
claiming they still describe one Bayesian model. For the neural path, measure any
resulting inconsistency; calibration does not automatically repair filtering.

For action selection, prefer a calibration population with declared action coverage
when the controlled environment supplies valid outcomes. Fit the predeclared
transform, freeze the resulting policy, and evaluate it on independent episodes.
The cost grid is fixed before test evaluation. An operating point chosen later
after inspecting its test curve is not independently validated by that same curve.
A calibrator trained only on outputs selected by an earlier policy may encounter
a different population if it changes that policy. This is a risk to assess, not
a universal fixed-point problem. A caller-specified abstention cost does not require
a fitted threshold or a new threshold-selection split. Off-policy or complete
counterfactual data may support evaluation when their assumptions and coverage hold;
an unexecuted real action does not acquire a label merely because we want one.

Declare the primary proper score, risk/cost metric and slices in an experiment plan;
include sample counts and suitable uncertainty intervals. Distinguish descriptive
subgroup views from confirmatory statistical tests. Binning-sensitive reliability
plots are useful diagnostics, not proof of calibration. Bind the calibration map
and claim to weights, input/outcome contract, fit population and evaluated procedure.
Exhaustive controlled tasks need no invented factual out-of-vocabulary label; open
applications must explicitly define what happens when truth lies outside the output
space. Factual absence, structural infeasibility and operational abstention differ.

A reinspection can be valuable even when the information was once available but is
now forgotten. Whether it is worth doing depends on current attainable decision
risk and cost, including the alternative of more retrieval. Error taxonomy alone
cannot decide that. Likewise, selection can favor optimistic prediction errors,
but the direction and size of the bias depend on those errors and their dependence.
Measure chosen-versus-reference predicted and realized loss instead of assuming
that every planner must over-inspect.

An ensemble comparison should follow a concrete residual-error hypothesis and a
declared budget; it does not require exhausting every calibration method or proving
that all alternatives have already lost. Compare a single-model reference and the
chosen ensemble variant under the declared primary resource constraint, and report
the other costs. Full independent recurrent members need their own beliefs and often
member-specific memory encodings, as well as weights and computation. The total
session-memory budget therefore matters, not only each member's per-stream bound.
Shared-representation heads avoid those extra stream states but cannot reveal all
shared encoding errors. Preserve current capacity unless Alex explicitly chooses a
different total budget. Zero disagreement does not establish safety or correctness.

Conditioning a model mixture is Bayesian within that assumed mixture; it need not
match the real world's uncertainty, even with many members. Worst-member penalties
or disagreement bonuses change the stated decision objective and are not automatic
upgrades to expected task cost. No calibrated epistemic-uncertainty claim is proposed
for the initial single model or promised by the ensemble alternative.

[Temperature scaling](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf) supports a
simple held-out probability adjustment while preserving factual argmax.
[Deep ensembles](https://papers.neurips.cc/paper/7219-simple-and-scalable-predictive-uncertainty-estimation-using-deep-ensembles.pdf)
and [PETS](https://proceedings.neurips.cc/paper/2018/file/3de568f8597b94bda53149c7d7f5958c-Paper.pdf)
provide predictive-ensemble precedents. These do not guarantee calibration under
this agent's memory compression, action selection or domain shift. The budget and
mixture-conditioning recommendations are design inferences.

**Decision for Alex:** keep one model with observable probability calibration and
explicit selected-risk evaluation first. Reserve shared-head or full-model ensembles
for a bounded comparison motivated by measured residual errors. No new universal
epistemic score, confidence-driven stopping rule or pessimistic penalty is needed
to complete the initial architecture.

Reviews: `overnight-uncertainty` and `overnight-uncertainty-reconcile`. Claude
withdrew universal calibration circularity, mandatory threshold splits and factual
OOV classes, on-policy-only evidence, the claim that Bayesian conditioning requires
a true ensemble member, and the claim that re-sensing after forgetting is necessarily
wasteful. It accepted total member-specific memory accounting and hypothesis-driven
ensemble comparisons. Its remaining operating-point, slice-uncertainty and resource
matching concerns are addressed by a predeclared cost grid, counts/intervals, and
one declared parity axis with the other costs reported. Coverage and useful
calibration remain empirical; agreement makes no capability claim.

## Continuation for the overnight work

Next: bounded thinking/planning, transfer/evaluation and the cross-topic dependencies.
Avoid reopening the settled causal/gradient distinctions unless new evidence changes
them. For each group prepare an independent note and public conceptual brief before
reading Claude, verify consequential claims, reconcile errors, and append a concrete
recommendation here. Keep all implementation and experiment proposals unexecuted.
By 08:30 stop initiating reviews, then produce a compact decision sheet and identify
which choices need Alex versus a separately declared experiment. Deliver by 09:00
Berlin and pause `overnight-agent-design-proposals`.
