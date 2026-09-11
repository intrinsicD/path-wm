# Agent specification questionnaire

Status: unanswered decision questionnaire, based on the
[current readiness review](model-readiness-review.md). Answers define an
implementable, testable design and its first experiment; tests must still establish
whether the implementation and learned behavior work. This is not a new architecture
or an authorization to launch all the experiments mentioned below.

## How to answer

Use the IDs, for example: `G02: verified retrieval or verified absence; uncertainty
ends as abstention, not success.` Short answers are sufficient if they settle the
listed cases. A later answer may revise an earlier answer; record the revision and
check its dependent interfaces.

For each section, specify **first experiment**, **later**, or **out of scope**.
For a deferred capability, name the first experiment's substitute and the claim
we must not make. Deferring vision might mean supplied descriptors and no visual
recognition claim. Deferring learned scheduling might mean a fixed, costed read
schedule and no adaptive-computation claim.

You can answer `Codex: propose the concrete settings` for engineering choices.
That assigns the work; it does not make an unspecified dimension, objective or
threshold complete. Codex must fill the implementation sheets below before the
corresponding work starts. Names, sizes and loss weights need not all be chosen by
you personally.

Carry forward your direction: allow learned graph structure and learned values;
support concrete identity, concepts, history, current belief and interactions;
make stored knowledge inspectable; allow internal and external actions toward a
goal. The questions below resolve how those intentions become an experiment.
Human-readable labels, stable storage IDs and learned retrieval keys are distinct
roles. The earlier design documents contain proposals, not automatic answers.

Suggested discussion order: S → G → B → D → A → T, then the supporting entity,
memory and runtime contracts. Vision, general concepts and skills can be deferred
explicitly. The final check must cover every section, including deferrals.

## S — Scope and the claim we want to test

- **S01.** What single capability should the next experiment establish: using memory
  to act, predicting action consequences, planning with learned consequences, or a
  different precisely stated capability? What observation would refute that claim?
- **S02.** What is the first environment and input format: the existing structured
  world, another simulator, a document/tool task, or raw sensory input? What are
  the agent's body/tools and the task's relevant objects?
- **S03.** Which components must be learned in that experiment, which may be fixed
  harness mechanics, and which may receive supplied labels or associations?
- **S04.** What ranges must it handle: entity count, action vocabulary, episode
  length, delays, changes, noise, task variety and simultaneous goals? Give a
  bounded first-version range for each relevant item.
- **S05.** What must transfer to new cases: appearance, object identity, starting
  state, action sequences, dynamics, tasks, or environments? Which broader
  ambitions are explicitly deferred, and what would trigger returning to them?

Deliverable: one falsifiable claim, environment definition and learned/fixed map.

## G — Goal, value, completion and verification

- **G01.** How is a goal supplied: an exact structured request, language, a target
  observation, an example, or a combination? Who resolves ambiguous intent, and
  what happens if goals conflict or the user changes a goal during execution?
- **G02.** What exactly counts as success, failure, unknown outcome, abstention,
  cancellation and budget exhaustion? In the key task, does proving absence
  satisfy the goal, or merely explain why retrieval cannot succeed?
- **G03.** Who verifies the outcome, from which actual evidence, at what time?
  When verification is unavailable or delayed, what status may the agent report?
  Keep output delivery, predicted success and verified success distinct.
- **G04.** What is the exact reward/loss table and unit for each outcome, physical
  action, inspection, retrieval, thinking step and elapsed time? Are costs summed,
  discounted, constrained or treated as secondary objectives? Give the formula.
- **G05.** Must planner scoring and the reported primary utility use that same
  formula? If there is a surrogate, what is it and how will its disagreement with
  the primary objective be tested? Resolve the existing reward10/cost1/stop0 versus
  success−0.05cost discrepancy, including the 85%-absent stopping example.
- **G06.** What happens when the goal is unreachable, insufficiently specified or
  cannot be reached within budget: stop, ask, explore, seek a subgoal or report the
  best verified partial result? What evidence permits declaring it impossible,
  rather than merely not yet finding a plan?

Deliverable: task record, outcome state machine, verifier and exact scoring rule.

## B — Observation, time, belief and uncertainty

- **B01.** What fields does an observation carry: modality/value, source, object
  association, observation time, arrival time, event/version ID and uncertainty?
  Which fields are supplied, inferred or unavailable to the online agent?
- **B02.** Which distinct meanings must the representation support: present,
  absent, unobserved, unknown, occluded, stale, contradicted and no longer existing?
  What does missing information do to an existing belief?
- **B03.** What distinguishes a new world change from late evidence about the old
  world? How are equal timestamps, reordered events, duplicates and corrections
  handled, and can a late observation revise history without replacing current truth?
- **B04.** How is uncertainty represented and updated: probabilities over states,
  multiple hypotheses, latent uncertainty plus calibrated readouts, or another
  mechanism? What dependencies between entities must be preserved rather than
  approximated independently? Can the world contain more than one key?
- **B05.** How is input reliability inferred, supervised or supplied, and how can
  reliability change by source and time? What happens when sources disagree,
  uncertainty is high, or there is too little feedback to estimate reliability?
- **B06.** What should unexpected action outcomes change: the current state,
  identity association, source reliability, dynamics belief, or several of these?
  How does the agent keep competing explanations until evidence distinguishes them?
- **B07.** What can inspect, ask or wait reveal, at what cost/delay, and with what
  failure/noise distribution? What makes the agent seek more evidence versus act,
  and how will confidence be calibrated under the information actually available?

Deliverable: evidence schema, temporal rules, belief-update and sensing contract.

## P — Perception and grounding

- **P01.** Which modalities and input shapes/rates are in scope now? Who handles
  alignment, missing modalities, normalization, coordinate frames and units?
- **P02.** How does an observation produce candidate entities: supplied candidates,
  boxes/masks, object slots, learned discovery or another method? How are background,
  parts and overlapping objects represented?
- **P03.** Which training signals establish the candidates and their properties:
  image-level labels, boxes, masks, tracking correspondences, reconstruction,
  prediction or task outcomes? Which labels are training-only?
- **P04.** Which changes should preserve identity, and which alter it: viewpoint,
  lighting, deformation, occlusion, motion, replacement or identical appearance?
  What unresolved-match output is available?
- **P05.** Which source details must remain recoverable for future questions, and
  how will we measure grounding separately from reconstruction? If image/face
  reconstruction is required for inspection, what decoder training and fidelity
  checks establish what that reconstruction can actually show?

Deliverable: encoder/candidate interface and observable grounding criteria, or an
explicit supplied-candidate substitute.

## E — Concrete entities and persistent identity

- **E01.** What may count as an entity in the chosen scope: physical object, person,
  place, event, document, group or part? How are simultaneous objects and nested
  things represented without imposing a universal taxonomy prematurely?
- **E02.** How is a current candidate matched to existing records: learned features,
  motion/history/context, explicit identifiers, or a combination? What separates
  a match, a new entity and an unresolved set of candidates?
- **E03.** Who creates stable record IDs, what is their uniqueness/lifetime scope,
  and how do mutable names and learned retrieval keys refer to them? What prevents
  a label or database address from becoming a shortcut for the desired answer?
- **E04.** How can mistaken merges and splits be corrected? What happens to earlier
  observations, edges, aliases, latent beliefs and pending plans after correction?
- **E05.** What belongs to an instance: recognition features, observations, inferred
  state, history, uncertainty, relations and learned interactions? Which items are
  explicit metadata, which are latent, and which entity/state owner may update them?
- **E06.** How do entities leave view, reappear, become inactive or cease to exist?
  What are allocation/retention limits, and what happens to references when a
  record cannot be retained or resolved?

Deliverable: candidate-to-record binding rules and reversible identity lifecycle.

## K — Concepts and learned graph structure

- **K01.** What belongs to a general concept such as “bicycle,” and what belongs
  only to one bicycle? Can an instance belong to several uncertain concepts, and
  can concepts have subtypes, shared attributes or exceptions?
- **K02.** What are node and edge records allowed to contain: latent values,
  explicit fields, evidence links, relation embeddings, direction, confidence and
  time validity? Are relations binary or multi-party, and what cardinality/cycle
  rules are fixed by the harness?
- **K03.** What graph operations may the agent propose: create, link, unlink,
  revise, merge, split or delete? How are invalid proposals rejected, and how do
  capacity limits, update order and conflicting writes behave?
- **K04.** What teaches graph structure, relation meanings and values? What task
  requires useful structure rather than letting a single large latent bypass it?
  What comparison would show that learned links contribute to performance?
- **K05.** How are human-readable labels attached, revised and checked? Does the
  agent consume text labels, learned relation representations or both? How are
  unnamed/poorly understood relations displayed without assigning false meaning?
- **K06.** How do concepts, skills and instance-specific experience influence one
  another? For another bicycle, what may transfer, what needs fresh evidence,
  and how are exceptions prevented from becoming false general rules?

Deliverable: graph operation vocabulary, record schema and structure-learning task.

## M — Memory organization, contents and lifecycle

- **M01.** What is the responsibility of each store: working state, entity belief,
  episodic history, concept graph, compressed/consolidated memory and retained raw
  evidence? Which is authoritative for each kind of information, and how are
  derived copies invalidated when their sources change?
- **M02.** What do we retain for an observation or entity: raw data, evidence
  references, cached encoder features, latent belief, readable summary or a mix?
  What metadata accompanies every retained representation?
- **M03.** What triggers a memory write, belief update, compression or consolidation?
  Which decisions are learned, fixed or user-directed, and what loss teaches them?
- **M04.** What are the bounds per store and for total storage, compute and retained
  evidence? What is evicted, compressed, protected or rejected at capacity, and
  what happens to graph references and evidence claims after eviction?
- **M05.** What survives a task ending, a new session, process restart or model
  update? How are memory versions, incompatible encoders and stale cached latents
  migrated, re-encoded or rejected?
- **M06.** What do recall queries return: whole records, selected attributes,
  subgraphs, episodes or raw evidence? How are recalled features routed into the
  focus/workspace with source, time and entity attribution, without becoming new
  external observations?
- **M07.** What must survive compression and for how long? Which exact details or
  provenance may be lost, what should the agent then admit it cannot recover,
  and which delayed tasks will measure retention independently of recent-state shortcuts?

Deliverable: storage ownership map, budgets, retrieval payload and retention policy.

## F — Focus, retrieval and internal operations

- **F01.** What is focus: a selected entity set, question, relation, subgraph,
  working-token allocation or combination? How is it created, switched and shared
  across multiple tasks, with what maximum capacity?
- **F02.** Who decides what to retrieve and what to keep in focus: the task adapter,
  a fixed schedule, a learned policy or search? What query features may it use,
  and what are the result limit, ranking and no-match behavior?
- **F03.** Which internal operations exist—think, recall, compare, imagine,
  summarize, re-encode—and what may each read/write? What computation is fixed,
  what is learned, and how are internal operations distinct from physical actions?
- **F04.** What is the unit and budget for internal work? How is another thinking
  or retrieval step selected and stopped, and what happens if the external world
  changes while the agent is computing?
- **F05.** What learning signal shows that focusing, retrieving or thinking more
  helped the task? What equal-budget fixed-schedule and no-retrieval controls
  distinguish learned selection from merely spending more compute?

Deliverable: bounded workspace and internal-operation interface with a selection rule.

## D — Learned dynamics and predicted outcomes

- **D01.** What is the predictor's input contract: belief, retrieved memory,
  executed/proposed action, action parameters, elapsed time and context? Which
  variables may affect physical transitions, and which are merely goal preferences?
- **D02.** What exactly does it predict: next latent belief, entity attributes,
  observations, action success, duration, costs or several outputs? Which
  observable readouts make each prediction testable?
- **D03.** How are stochastic outcomes, multiple plausible futures, unknown
  dynamics, no-op/rejected actions and missing observations represented? What
  remains uncertain after a prediction versus after new evidence?
- **D04.** What real transitions and targets train the model, with which horizon,
  supervision masks, losses and weights? If latent targets are learned too, how
  are target drift, collapse and trivial copying detected or prevented?
- **D05.** Which recurrent state and memory are shared between real observation
  updates and imagined rollouts? What is copied, detached, frozen or recomputed,
  and how are hypothetical writes prevented from entering factual history?
- **D06.** What prediction quality is necessary before the planner may rely on it?
  Define action-sensitive, multi-step and uncertainty checks, plus behavior when
  the model is outside its trained range or demonstrably wrong.

Deliverable: action-conditioned predictor, grounded training objective and rollout rules.

## A — Planning graph, goal search and skills

- **A01.** What is a planning node: physical-state prediction, belief distribution,
  entity subgraph, task progress or their combination? What are action edges and
  possible-observation branches? How are alternative futures represented?
- **A02.** Must the plan be an explicit DAG, or may the first version use a bounded
  tree/search trace? Physical states can recur: are nodes indexed by time/depth,
  or can states merge? What equivalence test permits merging without discarding
  relevant uncertainty, history, resource budget or task progress?
- **A03.** How are candidate actions generated and bounded: enumeration, learned
  proposals, retrieval of skills or continuous optimization? How are preconditions,
  constraints and currently unavailable actions checked before execution?
- **A04.** How are nodes and paths scored using G04/G05, including running cost,
  terminal goal loss and future observations? What are horizon, branching, search
  budget, tie-breaking, fallback and exploration rules when no goal path is found?
- **A05.** How is an executed action's actual outcome compared with its prediction?
  What observable or calibrated discrepancy counts as on track, contradicted or
  unresolved? What causes replanning, branch invalidation or a new subgoal?
- **A06.** How are complex goals decomposed, if in scope: supplied subtasks,
  learned proposals or reusable skills? For each skill, what are its initiation
  conditions, inputs, policy, expected effects, termination and failure behavior?
  Are skills stored as weights, programs, trajectories or another representation?
- **A07.** For conversation, essay writing or learning to ride, what is the
  environment interface and success signal? How do information gathering,
  practice, internal reasoning and external execution interleave? What distinguishes
  learning a reusable skill from successfully completing one attempt?

Deliverable: bounded planner/skill interface, plan validation rule and explicit scope.

## L — Training and adaptation

- **L01.** Which learning happens offline, within an episode or across sessions?
  What updates weights versus memory/state, and when are exploration and evaluation
  allowed to change either? What persists after learning a new skill or concept?
- **L02.** Where do training sequences come from, who chooses actions, and how do
  we cover the states/errors the deployed policy will actually visit? How are
  failures, recovery, rare events, no-information cases and action alternatives sampled?
- **L03.** For every learned component, what are its exact inputs, targets, loss,
  loss weight and training-only information? What prevents evaluator truth,
  generator IDs, future observations or labels from leaking into online decisions?
- **L04.** What is trained jointly or in stages, with which frozen donors, gradient
  boundaries, optimizer/settings and sequence truncation? How are stale latent
  caches and non-differentiable retrieval/graph decisions trained or handled?
- **L05.** Which information must the training history require the model to retain,
  bind, generalize or infer? Which paired examples defeat last-frame, position,
  label-name and scripted-action shortcuts?
- **L06.** How will we distinguish missing information, failed memory retention,
  incorrect retrieval, inaccurate prediction and bad planning? Which matched
  probes and component-replacement controls localize those failures?
- **L07.** What limits adaptation when evidence is weak, and how are regressions,
  forgetting and incompatible memory/model versions detected? What is the recovery
  or rollback rule if online or later-stage learning damages earlier capabilities?

Deliverable: data/credit-assignment plan and per-component learning specification.

## R — Runtime, ownership and persistence

- **R01.** What is the complete agent-step order from incoming event through belief,
  memory, focus, planning, execution and verification? Which component owns each
  state transition, and can more than one task or writer be active?
- **R02.** What exact request/result records cross each boundary, including IDs,
  schema/model versions, tensors/shapes, units, timestamps and error variants?
  Which consistency checks run at those boundaries?
- **R03.** Who actuates physical/tool actions, checks capabilities and records what
  was actually executed? How are rejection, partial execution, timeout, interruption
  and externally changed state reported back to the agent? Who arbitrates cascading
  failures or conflicting module requests, and when does the whole agent retry,
  fall back, ask or stop?
- **R04.** What is atomic, what is retryable, and how are duplicate or uncertain
  executions reconciled? If the process crashes after an external action but
  before recording its result, how does it avoid repeating the action blindly?
- **R05.** What is included in a resumable snapshot: model, RNG, environment/task
  state, entity/graph memory, belief, pending operation, plan, evidence, verification
  and budgets? At which boundaries is exact replay supported, and what happens
  when external reality has changed since the snapshot?
- **R06.** What are total memory/storage, latency, throughput and concurrency limits
  on the intended hardware, including concurrent branches, snapshots and caches?
  How do local budgets compose into those totals? What degrades or stops at a limit, and what artifacts
  are retained when execution or report generation fails?

Deliverable: one executable agent loop with versioned interfaces and recovery rules.

## I — Inspection and understanding what the agent stores

- **I01.** What must a debugger show for an entity, concept, relation and plan:
  identity hypotheses, evidence, belief changes, retrieved content, predicted
  outcomes, executed actions, costs and verification?
- **I02.** Which displays are exact stored metadata, learned readouts, generated
  labels or reconstructions? How are uncertain interpretations and unsupported
  details marked rather than presented as direct access to latent meaning?
- **I03.** Which raw examples or evidence references support an inspection, and
  what remains inspectable if they were discarded or the latent model changed?
- **I04.** Can a user rename, correct, merge, split, delete or protect records?
  How are those edits attributed, validated and propagated to beliefs, graph
  references, cached representations and pending plans?
- **I05.** What behavioral tests show that the interpretation is faithful—for
  example, changing/removing a relation affects the expected retrieval or action?
  What must remain explicitly uninterpreted if no reliable readout exists?

Deliverable: inspection contract with provenance and tests of interpretation fidelity.

## T — Experiments, thresholds and readiness

- **T01.** What is the primary hypothesis and metric for each claimed capability?
  Give numerical pass/fail thresholds, required denominators and guardrails before
  running; distinguish task success, prediction quality, calibration and efficiency.
- **T02.** Which populations are used for training, development, calibration and
  final evaluation? How are episodes, entities, tasks and seeds separated, and
  which data becomes development-only after a result informs a change?
- **T03.** Which baselines and ablations are required: reactive/copy-last,
  no-history, wrong-entity/shuffled memory, fixed versus learned graph or scheduling,
  supplied versus learned dynamics? What information and budgets must be matched?
- **T04.** Which cases must be reported separately: impossible goals, absent
  objects, ambiguous identity, unseen actions, long delays, missing/conflicting
  evidence, unannounced changes, failed tools and resource exhaustion? How are
  unknown verification and excluded/missing results counted?
- **T05.** How many independent training seeds and held-out episodes are required,
  with what uncertainty intervals or variability summaries? What limits claims of
  transfer and what would justify moving beyond the controlled task?
- **T06.** What are the data, training, inference and review budgets; stop rules;
  checkpoint-selection policy; and permitted iteration process after failure?
  What makes an experiment inconclusive rather than passed or failed?
- **T07.** What must be reproduced and archived: configuration, source and donor
  identities, raw predictions/actions, checkpoint/RNG, cost accounting, failure
  traces, resume checks and report QA? What independent replay/check can detect
  a convincing report built on an incorrect evaluator?

Deliverable: preregistered experiment plan and evidence requirements.

## Implementation sheets to fill from the answers

One row per component is enough; these are concrete outputs of the discussion,
not additional architectural choices to leave implicit.

| Component | Fixed/learned | Owner and persistent state | Input/output schemas, shapes and units | Update/gradient rule | Capacity/time limits | Failure behavior | Acceptance check |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Perception/candidates | Pending | Pending | Pending | Pending | Pending | Pending | Pending |
| Entity binding/graph | Pending | Pending | Pending | Pending | Pending | Pending | Pending |
| Belief and evidence | Pending | Pending | Pending | Pending | Pending | Pending | Pending |
| Memory and retrieval | Pending | Pending | Pending | Pending | Pending | Pending | Pending |
| Focus/internal operations | Pending | Pending | Pending | Pending | Pending | Pending | Pending |
| Dynamics/outcome reader | Pending | Pending | Pending | Pending | Pending | Pending | Pending |
| Goal/planner/skills | Pending | Pending | Pending | Pending | Pending | Pending | Pending |
| Executor/verifier | Pending | Pending | Pending | Pending | Pending | Pending | Pending |
| Persistence/debugger | Pending | Pending | Pending | Pending | Pending | Pending | Pending |

Also fill one learning row per learned component: data identity/split, allowable
inputs, target construction, exact loss/weight, initialization/freeze rules,
optimizer, batch/sequence lengths, gradients/truncation, steps, seeds and budget.
Fill one experiment row per claim: population, controls, metric formula, thresholds,
unknown/failure handling, repetitions, stopping rule and artifact paths.

## When the specification is complete

Every question has either a concrete answer or an explicit scoped deferral.
Required implementation/learning/experiment rows have no unresolved values. The
schemas agree at connected boundaries; costs/time units and ownership rules agree.
Confidence, model uncertainty and utility have explicit meanings and conversions;
unrelated scores are not silently compared as probabilities. Each claim has a test that could fail, and each failure path has a defined
runtime response. Requirements and chosen implementation should be traceable to
question IDs rather than hidden in prose or code defaults.

Check the combined answers against these example traces:

- Observe a key, lose sight of it, act, receive a late contradictory observation.
- See two similar entities, make a mistaken association, then correct it.
- Fill memory, evict/compress evidence, and later ask a question requiring it.
- Predict success, execute, observe failure, update belief and replan or stop.
- Encounter an unreachable goal or exhaust the thinking/action budget.
- Crash after an external action and resume without assuming it never occurred.
- For a scoped generalization claim: meet a new bicycle and apply a learned skill
  while keeping its instance-specific properties uncertain.

Expected outcomes for these traces must follow from the answers. If a trace needs
an unstated rule, add that rule before claiming the specification is complete.
Different capability scopes can have different completion dates; empirical success
is established only after implementation and the declared experiments.


Coverage review: Claude identified cross-module schemas, update ordering, score
compatibility, failure arbitration, total budgets, migration and complete traces
as common residual gaps. These are covered by B04/B07, M05, R01–R06 and the final
consistency check. No answers or technical settings were adopted by that review.
