# First task for learned entity memory

Alex authorized defining the initial task and learning signal on 11 September 2026.
The design requirement is to learn both graph structure and latent contents, with
evidence-based inspection. Human-readable relation names are not a prescribed ontology.
This document completes that task-definition step; no neural training or graph
implementation was launched. The existing reader diagnosis remains a prerequisite
to interpreting a new graph's learning failure.

## Task: follow and update two anonymous objects

Each short simulated episode contains two objects. They have episode-local appearance
descriptors, spatial positions and binary switch states. Descriptors are newly sampled
for each episode and stay stable while visible. The first cohort has distinct descriptors;
candidate extraction is supplied, so visual discovery is deliberately not tested.

At the first observation a cue selects one visible object. Subsequent observations
randomize candidate order, change positions and hide/reveal switch states. Visible
actions address a current spatial candidate and change its switch. The other object's
state persists. Every physical state change is caused by a shown action; there are
no hidden spontaneous changes. Simulated action semantics are task rules to learn
from examples, not proof of motor-control learning.

After the sequence, ask three questions:

1. Which current candidate is the object originally cued?
2. What switch-state pair is supported by the history, in current candidate order?
3. What would the state pair be after a proposed toggle of a specified current candidate?

The final query does not repeat the initial target's appearance or supply its simulator
identity. Simulator IDs, correspondence labels and state bookkeeping remain outside
model-visible features. Runtime memory handles are opaque routing information. Cue,
appearance, order, initial switches and actions must be balanced independently.

## Primary learning signal

Use the equal-weight mean negative log probability of the correct answer for the three
question families. Identity has two candidate answers; each state-pair question has
four joint answers. Joint state-pair probabilities preserve uncertainty about which
object has which state. These are observable task outputs, not predefined graph slots,
entity classes or edge meanings.

The primary screen has no loss assigning a semantic name to an internal node or edge.
Trainable keys, values and connection proposals are rewarded through useful answers.
If correspondence supervision is needed to bootstrap association, declare it as a
separate condition; do not mix that result with task-only structure learning.

Act versus inspect is derived from predicted probabilities and explicit costs, not
an extra arbitrary label. For target selection, a wrong choice costs 1 and deferring
for inspection costs 0.25; choose a candidate only when its predicted probability
exceeds 0.75. This is a diagnostic decision cost, not a guarantee that inspection
can resolve every ambiguity or a physical robot policy.

## Identifiability and shortcut controls

Construct paired prefixes whose final observations and final query are identical,
but whose answers differ. A final-view-only baseline gets exactly the final candidate
descriptors/positions, visibility flags and proposed action; it receives no earlier
cue, recurrent state, stored graph, prefix-derived query embedding or history replay.
Compute its optimal accuracy separately for each answer family by grouping identical
allowed inputs and counting targets. Do not use an unspecified shuffled-history
baseline or assume all answers have the same chance rate.

For genuine ambiguity, enumerate possible worlds producing the same complete visible
history and query. Score predictions against the conditional answer distribution.
Only equally likely unresolved alternatives justify uniform probabilities. Known
properties remain known. Include same-looking objects crossing while occluded as a
separately scored ambiguity cohort; do not require impossible identity recovery.

Train/development/test episodes and appearance descriptors are disjoint. Freeze their
manifests before training. Randomize candidate order and test order equivariance.
The generator must check oracle consistency, absence of target fields in observations,
paired-history construction, and non-target preservation after an action.

## First learning screen

Run an observation-history oracle, then a small recurrent baseline. Only after the
baseline learns the task should a graph comparison be interpreted as an architectural
test. A graph candidate must have the same observations, targets, episode/update budget
and declared persistent tensor budget; report extra parameters and read/compute costs.
The two-object task can fit into a small recurrent state, so graph superiority is not
assumed and may not appear.

Proposed first-screen gates, to be fixed with the implementation's manifests before
any model run:

- On identifiable development episodes: target accuracy at least 95%; each state-pair
  accuracy at least 90%; both members correct on at least 85% of paired-history cases,
  reported separately by question family.
- On the specified ambiguity cohort: excess mean log loss above the conditional oracle
  at most 0.10 nats for each answer family. Also report distributions, coverage and errors;
  do not infer general calibration from this finite cohort.
- Target-selection coverage at least 90% on identifiable episodes and error at most 5%
  among selected targets. Report action/inspection cost on both cohorts so universal
  abstention cannot pass.
- All source/provenance, permutation, episode isolation and resume checks pass.

Bound the initial baseline at 256 updates and 450 active CPU seconds. If it fails,
preserve the result and diagnose it; do not start a graph search. If it passes, allow
one matched graph candidate under the same cap: at most 900 active seconds total.
The concrete network, capacity, seed, optimizer and split sizes must be recorded in
the implementation plan before launch. No extra seeds or parameter sweep is implied.

## Minimal graph operations and inspectability

Expose bounded read, propose allocation/association, update latent, propose connection,
and commit/revise operations. Do not require one node per simulator object or assign
human-defined meanings to edges. A first differentiable bounded-slot implementation
may learn occupancy and connection weights; that is a soft-structure prototype, not
yet arbitrary discrete graph growth. Specify that limitation if chosen.

Task loss can train differentiable within-episode operations. Hard lookup, allocation
and deletion need an explicitly chosen training estimator or supervised decision
signal; a database API alone provides none. Replay/recompute bounded histories for
writer credit and keep persisted snapshots detached, as in the reviewed contract.

Inspect actual allocation, association, reads, updates, connections and evidence
references. Compare relevant versus irrelevant connection interventions, and a declared
connection-disabled control, before claiming useful structure. A readable label or a
large attention weight is not a semantic explanation. Latent readouts and reconstructions
need their own validation. High task accuracy alone does not prove a graph is necessary
or that concepts, faces, language or motor skills have been learned.

## Completed definition checks and Claude review

The finite checker `runs/entity_task_contract_v1/check_contract.py` enumerates 512
illustrative cases: initial states/cue, one visible set action, candidate permutations
and a subsequent toggle query. Its history reader exactly matches the simulator in
all cases. Final-view-only optimal accuracy is 50% for cued identity and 25% for each
state-pair family. Every final-view group requires history. A separate symmetric
occlusion construction has target posterior [0.5, 0.5].

These are specification checks, not model evaluation or validation of a future random
generator. They do not cover long sequences or learned graph behavior. Receipt:
`runs/entity_task_contract_v1/verification.json`.

Two actual public-only Claude exchanges (`entity-task*` under the existing continuation
review directory) support the narrowed task. Corrections distinguish proper log loss
from accuracy, conditional ambiguity from blanket uniformity, and derived decisions
from arbitrary action labels. Claude's remaining request for an explicit no-memory
control is addressed above and by the finite input-group calculation. No private source,
data or measurements were sent; peer agreement is not a learning result.

## Authorized implementation slice

Build `--dataset entities` in the existing multimodal recipe. Reusable controlled
candidate data, a small recurrent reader and exact outcome metrics live in `pathwm/`.
The existing Run, update loop, checkpoint/cache and report own the lifecycle.
Use three observations: initial cue/states, visible set action, final hidden-state
view with a proposed toggle. Half the episodes mask final identity features and use
an exact two-assignment posterior. This first three-event screen does not claim
long-horizon discovery, arbitrary graph growth or a motor controller.

Before training fix: seed31, width64, recurrent state128 floats, shared candidate
input projection and GRU, AdamW lr0.003/weight decay0.01, batch32, 256 updates,
512 training and256 development episodes, FP32/two CPU threads, clip1 and450 active
seconds. Evaluate training every32 updates and development only at the final checkpoint.
Reserve separately generated test data; do not load it in this development screen.
Each32-row group varies initial state/cue/action value exhaustively within identifiable
and ambiguous cohorts. Shared final views provide exact no-history controls.
If baseline gates fail, no graph training starts; preserve and diagnose its errors.
