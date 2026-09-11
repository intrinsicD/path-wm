# First task for learned entity memory

Alex authorized defining the initial task and learning signal on 11 September 2026.
The design requirement is to learn both graph structure and latent contents, with
evidence-based inspection. Human-readable relation names are not a prescribed ontology.
The initial recurrent baseline is now implemented and evaluated below; graph
implementation remains conditional on successful baseline learning. The existing reader diagnosis remains a prerequisite
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
512 training and 256 development episodes, FP32/two CPU threads, clip1 and450 active
seconds. Evaluate training every32 updates and development only at the final checkpoint.
Reserve separately generated test data; do not load it in this development screen.
Each32-row group varies initial state/cue/action value exhaustively within identifiable
and ambiguous cohorts. Shared final views provide exact no-history controls.
If baseline gates fail, no graph training starts; preserve and diagnose its errors.

## Implemented baseline result, 11 September

The recipe, three-observation generator, recurrent reader, proper scores, cached
final evaluation and inspectable standalone report are implemented. All 92 CPU tests
pass, including exact resume, conditional oracle scoring and candidate-order target
equivariance. Input-gradient checks reach the first observation. Every generated
training/development target matches the independent history oracle. Model equivariance
is not established by the data permutation test.

The fixed seed31 run completed 256 updates within the 450-second cap (1.35 active
CPU seconds including cached-resume bookkeeping, excluding checkpoint/report overhead).
On 128 identifiable development cases: identity 60.15625%, state pair 62.5%, effect
pair 54.6875%. Training values are 66.796875%, 100%, 99.21875%. Development paired
success is 45.3125%, 48.4375%, 31.25%; identity selection coverage is 9.375% with
16.6667% error among selected targets. Ambiguous development excess NLL is 0.041062,
1.111356, 1.083753 nats by head. All combined development gates fail.

No graph run or parameter search was started. Identity remains weak even on training
examples, while state/effect fitting does not transfer well to fresh descriptor groups.
This does not isolate whether association, capacity, optimization or data diversity
caused the failure. A next diagnostic should separate descriptor matching from state
updating before adding graph complexity; the earlier event-reader probes also remain
open. The 32 examples within a descriptor group are controlled variants, not independent
appearance samples (16 training groups, 8 development groups).

Raw scores, unchanged prediction/result/metric hashes after cached resume, source
snapshots and 1280×720 browser QA are bound in
`runs/entity_learning_v1/verification.json`. The report is
`runs/entity_learning_v1/reference/report.html`. Prior Claude task reviews supplied
the conceptual critique; this implementation used local tests and audits without another
external review round. No test population was loaded by the training recipe.

## Association-bypass diagnostic (authorized continuation)

Add `--entity-association observed` as a deterministic input transform: replace the
8 appearance coordinates by two exact visible-descriptor matches to the initial
candidate order, padded with zeros. Hidden descriptors receive a symmetric 0.5/0.5
assignment while their visibility flag stays false. Other features and candidate
order remain unchanged. This supplies association, not switch-state answers or hidden
simulator IDs; it is a synthetic diagnostic, not learned recognition or a graph.
Keep the same GRU, parameter initialization, data, seed31, optimizer and 256-update /
450-active-second cap as the preserved raw reference. One run, unchanged gates; no
sweep or graph run in this slice. Report the intervention explicitly. Improvements
cannot isolate the original failure's sole cause. Test input-only matching, missingness,
permutation covariance, no mutation, and exact resume for the new condition.

### Supplied-association outcome

Actual Claude reviewed the public-only diagnostic and accepted the correction that
this changes both matching and feature geometry, rather than isolating a unique cause.
Its requested learned permutation measurement is reported separately from the exact
transform covariance test. Review receipts: `entity-alignment*` in the continuation
review directory. No private source or measurements were exported.

Implementation `c64bd39` adds `--entity-association observed`. All 94 CPU tests pass,
including both association modes' exact pause/resume and cached final evaluation.
The one 256-update run has identical manifests and initial parameters to the raw
reference. Development identifiable identity/state/effect accuracy is
88.28125%/64.84375%/57.8125%, versus 60.15625%/62.5%/54.6875% raw. Every combined gate
still fails. Identity selection coverage is 85.15625% with zero observed selected
errors; ambiguous coverage is 25.78125%, costing 0.314453 versus 0.25 for abstention.

Swapping both candidate positions throughout each complete development history and
undoing the output permutation gives mean absolute probability differences
0.178131/0.106659/0.120512 across the three heads. Argmax agreement is
58.59375%/68.359375%/57.421875%; ambiguous ties make argmax agreement alone inadequate.
The probability changes establish that the learned reader is not equivariant despite
the correct transform. This does not establish a learned graph, reliable state updating,
or the sole cause of the original failure. No additional training run was started.
A subsequent architectural diagnostic should enforce shared per-entity processing
and permutation-consistent readout before interpreting graph learning.

Report and verification: `runs/entity_alignment_v1/observed/report.html` and
`runs/entity_alignment_v1/verification.json`. The earlier raw run remains untouched.

## Shared entity-update diagnostic

Authorized continuation: add `--entity-reader shared` with observed association only.
Two shared GRU streams process the initial observation and visible set-action event,
using observation-derived routing to initial object order. Each stream has width64,
so persistent latent capacity remains 128 floats. Heads share parameters: identity
score per object, binary state distribution, and binary effect distribution conditioned
on whether that object receives the proposed action. Updating/toggling is learned,
not implemented as switch arithmetic. Joint pair probabilities factor within each
known assignment; final occlusion mixes two coherent assignments with equal prior.
This assumes the supplied two-object independent-switch task, not arbitrary relations.

Fix seed31, width64, lr0.003/wd0.01, batch32, same 512/256 manifests, 256 updates,
450 active CPU seconds, final-only development and unchanged gates. One run, no sweep.
Report parameter changes; initialization cannot be tensor-matched to the different
architecture. This tests shared processing plus structured readout, not graph learning
or uniquely one causal mechanism. Verify all 8 independent per-frame permutations,
probability normalization, earliest-state gradients, no input mutation and exact resume.

### Shared-reader result

Implementation `98e36af`; all 96 CPU tests pass. Two actual public-only Claude
exchanges (`entity-shared*`) reconciled assignment mixing, independent-switch scope,
readout input boundaries and capacity versus compute. No private source/results exported.
The full model audit confirms zero maximum probability difference across all eight
independent per-frame candidate permutations on the 256 development episodes.

The fixed seed31/256-update run passes every declared development gate. All 128
identifiable development cases have correct identity, state pair and action-effect
pair; each paired-history success rate is 100%. Selection coverage is 100% with no
errors. Ambiguous identity stays exactly uniform and selection coverage is zero.
Excess NLL above the conditional oracle is 0 / 0.00043944 / 0.01034916 nats for
identity/state/effect. This is the known finite ambiguity construction, not general
calibration. Both training and development identifiable accuracy are 100% by head.

The network has 30,021 parameters versus 102,154 in the flattened reader, with the
same 128-float persistent state capacity. One run used 1.51 active CPU seconds before
cached-resume overhead, excluding checkpoint/report time; architecture and computation
are different, so this is not a tensor-matched or equal-compute causal comparison.
The dataset and training budget are matched. The changed shared updates and structured
readout jointly solve this supplied-association task under the budget.

Results, independent NLL recalculation, source snapshots, exact cached resume and
permutation audit: `runs/entity_shared_v1/verification.json`. Report:
`runs/entity_shared_v1/reference/report.html`. Report validation is structural using
the unchanged previously browser-verified renderer; no new browser receipt is claimed.
No test split or graph training was used. Exact descriptor matching, fixed slots and
two-assignment enumeration are supplied. Next proposed step is learned association
under controlled feature variation, retaining this successful reader as a reference;
success here does not establish learned graph structure or a cause for the older
agent event-reader failure.

## Learned-association screen

Authorized continuation: shared reader with `--entity-association learned`. Add a
shared pair scorer, squared descriptor difference → Linear(8,32) → GELU → Linear(32,1).
Sum scores over each of the two bijections and softmax. No exact equality or teacher
correspondence enters forward/training. Initial slots remain bookkeeping. Enumerate
both action-time assignments with separate shared recurrent states and marginalize
answer probabilities over action/final assignments. Missing final descriptors use the
known symmetric prior. No latent averaging. Descriptor noise/discovery/graph growth
are outside this first test of replacing exact matching.

Keep existing reader initialization (construct scorer afterward), seed31, width64,
lr0.003/wd0.01, batch32, 512/256 manifests, 256 updates and450 active CPU seconds.
One run; retain prior gates. Report the extra321 parameters and 256 floats across two
128-float hypothesis states, versus 128 in the reference; no equal-compute claim.
Report initial/final association accuracy separately using evaluator-only matching,
including distinct descriptor-group counts rather than treating replicated episodes
as independent evidence. Development task evaluation remains final-only; initial
association scoring is a declared fixed diagnostic and does not select training.

### Learned-association result

Implementation `cf5a506` replaces exact lookup in the shared reader's learned mode.
Two actual public-only Claude exchanges (`entity-learned*`) are reconciled. Claude
accepted that shared pair scoring over bijections guarantees probability covariance
without guaranteeing correct assignments or calibrated posteriors. The factorial
hypothesis enumeration is explicitly scoped to two objects. No private source or
results were exported. Local tests forbid exact lookup in the learned forward path
and confirm task-loss gradients reach the matcher.

The one seed31/256-update run passes all declared development gates. Identity,
state-pair, action-effect and paired-history accuracies are 100% on the 128 identifiable
development cases; coverage is 100% with no errors. On ambiguous cases identity is
uniform by the specified prior and mixture, coverage is zero, and excess NLL is
0 / 0.00023004 / 0.01106636 nats by head. This is not learned general uncertainty.

Association accuracy on eight distinct held-out descriptor groups improves from
37.5% to 100% at both visible action/final times, NLL 0.704208 to 0.000006944. The two
measurements share descriptors and are not independent trials. Training group accuracy
improves from 56.25% to 100% on 16 groups. The labels are evaluator-only and diagnostics
are cached. Actual matcher tensors change; the reference's shared initial tensors
and manifests match exactly. All eight per-frame reorderings have zero maximum
probability difference on development. The matcher sees stable synthetic descriptors;
no noisy-view or image recognition result is claimed.

The model has 30,342 parameters (321 extra) and 256 latent floats across two hypothesis
states versus 128 in the reference; 3.44 active CPU seconds before resume bookkeeping,
excluding reports/checkpoints. No equal-capacity or equal-compute claim. Raw scores,
source snapshots, exact cached resume, tests and symmetry audit are bound in
`runs/entity_learned_v1/verification.json`. Report:
`runs/entity_learned_v1/reference/report.html`; structural validation uses the unchanged
previously browser-verified renderer. No new browser receipt, test-split result, graph
allocation or learned concept/edge semantics. Next proposed: controlled descriptor
variation and unmatched/new-entity handling, preserving this passing baseline.

## Bounded descriptor-variation screen

Authorized continuation: add `--entity-noise 0.2` to the learned/shared path. For each
of three times and two physical objects, independently perturb its descriptor by a
vector of norm alpha times the original pair separation. Noise is shared within the
32-row counterfactual group; masks, actions, clean descriptors and targets stay fixed.
Use a separate deterministic noise RNG. Require finite 0<=alpha<0.25, so even with
noisy initial features same-object distance is at most2alpha D and different-object
distance at least(1-2alpha)D. The nearest-initial oracle remains identifiable. This
is controlled bounded drift, not arbitrary noise or image recognition.

One fresh seed31 run at alpha 0.2: same learned shared model, parameter initialization,
512/256 populations, lr0.003/wd0.01, batch32, 256 updates and450 active CPU seconds;
unchanged gates. No sweep or new-entity allocation in this slice. Preserve zero-noise
compatibility. Evaluate matching with a nearest-initial oracle only in diagnostics;
train using answer loss only. Report the8 development groups and paired controls.

### Descriptor-variation result

Implementation `76338f4`; all 102 CPU tests pass. Claude's first public-only review
accepts the positive separation margin and identifies the narrow single-condition
scope. The prepared reconciliation was rejected by automatic approval review before
execution because payload-specific authorization was judged insufficient. The user subsequently approved the exact payload; the follow-up completed and Claude
accepted the reconciliation with no remaining conceptual objection. The original blocked
attempt is retained as history. Local implementation and verification are complete.
Review/blocked receipts: `entity-variation*` under the continuation directory.

The one 256-update run at alpha 0.2 passes every declared development gate. All 128
identifiable cases and paired cases are correct for identity/state/effect, with 100%
selection coverage and no errors. Ambiguous excess NLL is 0 / 0.00022755 / 0.01080668;
uniform unresolved identity is the specified prior, not general learned calibration.
Matching reaches 100% on eight held-out descriptor groups (initial action 37.5%, final
62.5%). Data geometry, all 768 training/development oracle targets, unchanged labels/
nonappearance features, and exact zero-noise compatibility are verified. Predicted
probabilities have zero maximum difference across all 8 per-frame reorderings.

All initial tensors match the stable learned-association reference. Model size stays
30,342 parameters and 256 hypothesis-state floats. Training used 2.43 active CPU seconds
before cached-resume bookkeeping, excluding checkpoints/report overhead. This trains
and evaluates within the same bounded variation condition; it is not a frozen-transfer
result or arbitrary-noise guarantee. No test split or new-entity allocation was used.

`runs/entity_variation_v1/verification.json` binds source, raw-score, geometry, resume
and test checks. `runs/entity_variation_v1/reference/report.html` is structurally
verified with the unchanged previously browser-verified renderer; no new browser QA
is claimed. Next proposed is explicit unmatched/new-entity handling in a separately
defined task. Learned graph allocation and concept/edge semantics remain unimplemented.

The completed reconciliation requires no model change or additional run. Noise uses
independent draws from one separate noise RNG, not a separate RNG instance per object.
Claude reviewed the abstract construction; the existing local audits supply implementation
evidence. No broader robustness or graph capability is inferred from peer agreement.

## Novelty-matching diagnostic

Authorized continuation: add `--dataset entity-matching` to the existing recipe.
Inputs hold two unit memory descriptors and one unit query, in the existing tensor
envelope; task outputs are memory0, memory1, or new/neither. Each four-row group
contains one query for each known memory and two novel queries. Known queries use
normalized perturbations and remain within0.35 times memory separation; novel queries
are farther than0.65 times separation from both. Norms and presentation match. This
is a deliberately separated first rejection task, not boundary calibration or graph
allocation. Unused event/features remain zero; no generator IDs enter model inputs.

Model: shared squared-difference scorer Linear(8,64), GELU, Linear(64,1), plus learned
null logit; no threshold/equality oracle in forward. Train3-class CE. Seed31, AdamW
lr0.003/wd0.01, batch32, 512 train/256 development rows (128/64 descriptor groups),
256 updates, FP32/two CPU threads and450 active seconds. Development at final only;
separate test split remains reserved. One run, no sweep or actual database writes.
Gates: known identity and novel rejection accuracy each>=95%; false merge/split each
<=5%; each cohort selection coverage>=90% and selected error<=5% at confidence>0.75;
overall NLL<=0.15 nats. Test memory-order covariance, unit-norm/margin/label contracts,
episode isolation and exact resume. This new task is not directly comparable to the
three-head historical state task. Near-boundary and unknown priors are later work.

### Known-versus-new result

Implementation `365bb2b`; all 105 CPU tests pass. Two actual public-only Claude
reviews (`entity-novelty*`) support this deliberately separated sanity check. A second
margin and real novelty/calibration are deferred, not established. Memory swaps must
permute identity labels, while false-merge/split metrics remain invariant; both
probabilities and all metrics are verified identical in the swapped evaluation.
No private source/results were supplied to Claude.

The one seed31/256-update run passes all gates. On 256 development queries sharing 64
memory groups: known identity accuracy 128/128; novel rejection 128/128; false merges
and false splits both 0. Coverage above 0.75 is 100% known and 96.09375% novel, with no
selected errors. Mean NLL 0.02360923 nats. Training has one false merge among 256 novel
queries; this is preserved, not hidden by aggregate development success. The 642-parameter
model uses a learned null logit, not a generator threshold in forward. Active CPU time
was 0.5872 seconds before resume bookkeeping, excluding checkpoint/report overhead.

Disjoint train/development descriptors, exact resume, independent NLL, source snapshot,
memory-order covariance and 1280×720 browser QA are bound in
`runs/entity_novelty_v1/verification.json`. Report:
`runs/entity_novelty_v1/reference/report.html`. Selection thresholds and generator
margins were declared before training; no generalization evidence beyond this condition
or posterior calibration is claimed. This model classifies new observations; it does
not allocate IDs, persist entities or learn graph edges. Next proposed: a bounded
transactional memory lifecycle that can defer uncertain matches, allocate confirmed
new entities and recognize them on return, with explicit identity-switch tests.

### Persistent entity lifecycle: implementation contract

Add a single-writer bounded EntityMemory around a frozen descriptor scorer. Stable
IDs are bookkeeping; prototypes stay fixed, while last-seen/count metadata updates.
Confidence <=0.75 and capacity exhaustion defer. Exact retries are idempotent;
conflicting retries and non-increasing new timestamps reject without mutation.
Bounded receipts retain replay answers; expired events reject through the clock.
Snapshots bind weights, validate state and preserve identity continuity. Essential
CPU tests use controlled scores to isolate transaction correctness, plus the real
reader for variable-cardinality and snapshot checks. No new training or scientific
accuracy claim in this slice: variable-cardinality confidence remains unvalidated.
Budget: CPU tests only. Existing novelty experiment and report remain unchanged.

The runtime is available as `pathwm.models.entity_memory.EntityMemory`. It takes an
`EntityMatchReader` (including a caller-loaded checkpoint), copies/freezes it on CPU,
and exposes `observe(event_id, unit_descriptor, timestamp)`, `snapshot()` and
`EntityMemory.restore(model, snapshot)`. Persist snapshots with `pathwm.io.atomic_json`.
For example:

```python
memory = EntityMemory(reader, capacity=3)
receipt = memory.observe('camera-event-1', descriptor, timestamp=1)
atomic_json('memory.json', memory.snapshot())
```

This is an explicit Python harness component, not automatic integration into the
agent's input loop. The caller owns event IDs and latest-snapshot selection; retries
must retain the original payload and timestamp. Metadata is last observed time and
count, not a learned belief state. Stable IDs remain contiguous because this slice
has no deletion or eviction. Scorer weights and type are checked on restoration;
compatible implementation code is still the caller's responsibility.

Both short Claude reviews completed (`entity-lifecycle*`). Adopted concerns include
cardinality shift, replay expiry, immutable prototypes and deferred-record isolation.
Claude's suggestion that tie deferral requires calibrated probabilities is not
needed: two equal maxima cannot individually exceed one half of a softmax, regardless
of calibration. Snapshot rollback is explicitly caller-owned. No neural evaluation,
training, graph learning or new report was performed for this runtime-only slice.

Verification: the full collected regression suite passed (108 tests), followed by
coverage of all four lifecycle tests including the added atomic-file/failure case
(109 distinct tests overall). Ruff and whitespace checks pass. Implementation commit
`7180e2d`; red contract commit `69f9191`. No extra model training consumed the budget.

### Frozen growing-memory screen (predeclared)

Use the existing novelty checkpoint unchanged. Seed61; 32 independent descriptor
families, each shared across capacities1,2,4,8. Generate nine random unit 8D descriptors
with pairwise distance>=0.9, at most10000 proposals per family. Introduce the first N,
revisit them in reverse order with normalized additive noise of norm0.05, then present
the ninth descriptor as overflow. Fixed threshold0.75. Per capacity require >=95%
correct allocations, stable-ID revisits and capacity deferrals; uncertainty is a miss.
Require exact retry and mid-episode snapshot continuation equality on every episode.
Record all inputs/receipts, separated family dependence and donor/source identities.
No training, sweep, calibration or graph-learning claim. CPU evaluation budget60s,
excluding report generation. Implement in the existing recipe/report pipeline.

### Frozen growth outcome

The corrected screen (`cf6f4b6`) completes with a **failed scientific gate** at
capacities4/8. At capacities1/2 every event is correct. At capacity4 allocation and
revisit correctness are127/128 each, and final capacity deferrals30/32. At capacity8
these are252/256 each and26/32. There are no wrong-ID matches or false splits in the
corrected receipts. All128 episodes preserve exact retries and restored continuation.
The final-event criterion is end-to-end: one capacity4 episode and four capacity8
episodes remain underfilled after uncertain allocations, so creating a final new
record is locally valid but fails the intended complete-lifecycle sequence. The
remaining final-event misses are uncertainty (one/two respectively). This does not
indicate a capacity enforcement bug.

Two implementation defects were fixed: loading nested `agent` checkpoint weights
(preflight failed before a run existed), and scoring through stable allocated IDs
instead of assuming IDs equal simulator indices after a missed allocation. The first
screen remains at `runs/entity_growth_v1/reference` with superseded identity-offset
metrics. The corrected screen has identical inputs and model receipts; only identity
scoring changed. Donor/checkpoint weights are equal, no optimizer steps occurred,
and cached CLI resume succeeds. Both conceptual Claude reviews completed. Reported
rates are descriptive on32 shared families, not independent confidence intervals.

Full regression suite112 tests passed; the additional ID-offset regression passed
with all four growth tests (113 distinct tests total). The corrected report passed
1280×720 browser QA, with no overflow/broken images and working score details.
[Report](../runs/entity_growth_v1/corrected/report.html),
[verification](../runs/entity_growth_v1/verification.json).

Run with `python -m experiments.multimodal --entity-growth-weights PATH --output DIR`;
resume with `python -m experiments.multimodal --resume DIR`. The present next research
question is training novelty decisions across varying candidate counts, with the
failed frozen reference retained and fresh held-out families. No threshold tuning,
new training, or learned graph updates were performed in this slice.

### Variable-count adaptation (predeclared)

Train the same642-parameter matcher from seed31 for256 AdamW updates, lr0.003,
wd0.01, batch32,512 train/256 development queries; CPU450s. Candidate counts1..8,
each balanced known/new. Independent pairs of queries share a fresh descriptor family;
unit descriptors separated by>=0.9; known perturbation norm0.05 then renormalization.
Pad to8, explicitly mask absent candidate logits, final class means new. This changes
both geometry and cardinality distribution; it is not a causal isolation of count.
Compare original and adapted frozen models on identical fresh seed73 growth families
(32 families, capacities1/2/4/8), unchanged95% lifecycle gates and0.75 threshold.
One training run; no tuning after results. Separate test split remains reserved.

### Variable-count result

Implementation38db3f5;115 CPU tests pass. One256-update training run passes all
matching gates:128/128 known and128/128 novel development queries correct,100%
coverage above0.75, NLL0.02105163. Masked padding is gradient-free and leaves valid
scores unchanged. Both Claude conceptual reviews completed without unresolved
critical contradiction; neither inspected private source or measurements.

On identical fresh seed73 histories, the original model fails capacity8 (allocation
and revisit252/256, overflow28/32). The adapted model gets every allocation, revisit
and final capacity deferral correct at capacities1/2/4/8, with no uncertainty,
wrong-ID matches or false splits. All128 histories pass retry and snapshot checks.
These are descriptive results on32 shared synthetic families. Adaptation changed
geometry and candidate counts together; no causal count-only conclusion or visual
identity/calibration/graph-learning claim follows.

Training resume reserializes its checkpoint, changing file SHA despite unchanged
weights; the first dependent evaluation correctly refused that changed donor on
resume. Preserve an immutable copy before dependent evaluations. The final
`frozen_adapted.pt` and `adapted_pinned` run have exactly the same predictions as
`adapted`; their CLI resume passes. This was provenance repair, no extra training
or parameter adjustment. Raw first evaluation remains preserved.

[Training report](../runs/entity_variable_v1/training/report.html),
[matched baseline](../runs/entity_variable_v1/baseline/report.html),
[adapted lifecycle](../runs/entity_variable_v1/adapted_pinned/report.html),
[verification](../runs/entity_variable_v1/verification.json).
Reports passed browser QA. Verified frozen weights, identical evaluation inputs,
disjoint descriptors, independent NLL and cached resume. Commands add
`--entity-variable` to `--dataset entity-matching`; frozen comparisons accept
`--entity-growth-seed 73`. Next proposed: freeze this recognizer and test changing
learned state attached to persistent IDs, before broader graph learning.

### Learned state attached to persistent IDs (predeclared)

Freeze the immutable adapted matcher. Train a shared width16 GRU cell and binary
readout on observed0/observed1/toggle/no-information events. Per episode introduce
two entities with binary states, perform two targeted toggles, then revisit both
with no new state information. Six events;16 balanced state/action variants share
a descriptor family and identical final views.32 training/16 development families,
512/256 episodes, independent generator seeds101/102. Labels stay outside matcher
and recurrent state. Train seed31, AdamW lr0.003/wd0.01, batch32,256 updates, CPU450s.
End gates: >=95% complete state-pair accuracy, <=0.15 NLL per entity, and all retry,
restore and unaffected-entity tests pass. No last-view-only solution can distinguish
paired histories. State-cell training uses matcher-routed slots; deployment uses
transactional EntityStateMemory with the same cell. No general belief uncertainty,
visual discovery or learned graph claim. One run; no post-result tuning.

### Learned persistent state result

Implementation08839f3;119 CPU tests pass, including uncertain retries, unaffected
entities and rollback after a state-update exception. Both Claude conceptual reviews
completed. Exact retries precede bookkeeping advancement; uncertain retries are
also idempotent. Fixed observe-then-toggle order is retained, with broader temporal
order generalization deferred.

One256-update run passes the declared gates:512/512 training and256/256 development
state pairs correct, NLL0.00145318 per entity. All256 persistent runtime episodes
recover both states, reproduce the training-path latents and preserve retries and
restored continuation. Recognizer weights remain frozen. Independent targets/NLL,
descriptor split isolation and cached CLI resume pass. A post-run metamorphic audit
reverses creation order while retaining within-entity event order:256/256 outputs
remain correct when read through their allocated IDs. This audit was not a tuned
training condition or an additional predeclared gate.

**Scope:** both splits contain the same16 state/action templates; only descriptors
are fresh. Identical train/dev NLL is therefore expected. This establishes the small
binding task, not unseen temporal-pattern generalization. Each final-view group has
four equally likely target pairs, so last-view-only best accuracy is25%. State is a
learned16-float vector per entity with binary readout, not a general belief model.
Graph edges, visual discovery, arbitrary properties and long histories remain open.

[Report](../runs/entity_state_v1/reference/report.html) passed1280×720 browser QA;
[verification](../runs/entity_state_v1/verification.json) binds raw outputs and models.
Run `python -m experiments.multimodal --entity-state-weights IMMUTABLE_MATCHER.pt
--output DIR`; resume with `--resume DIR`. The runtime is
`pathwm.models.entity_state.EntityStateMemory(matcher, cell)`, with
`observe(event_id, descriptor, timestamp, observation)`, `read(id)`, `snapshot()` and
`restore(matcher, cell, snapshot)`. Persist the whole snapshot atomically. The four
input features encode observed0, observed1, toggle and no-information respectively.
Next proposed: held-out temporal compositions and lengths with this model frozen,
before adding richer attributes or learned graph structure.

### Frozen temporal generalization (predeclared)

Freeze the saved state cell and immutable recognizer. Fresh seed111,16 shared
descriptor families;16 balanced binary/action variants per family per condition.
Four cohorts: reference6 events; reset-order6 events (middle reset/toggle pair on
entity0, both orders with same multiset); toggle-length20 (16 middle toggles);
no-information-length20 (reference two toggles followed by14 no-information events).
All finish with identical no-information views of both entities. Compute targets by
an independent binary simulator. Require >=95% state-pair accuracy, <=0.15 NLL per
entity, >=95% persistent-runtime accuracy, exact retries/restores and latent agreement
in every cohort. Budget120 CPU seconds excluding report generation, no fitting or
tuning. Report each condition separately; shared families imply correlated scores.

### Frozen temporal result

Implementation216e587;121 CPU tests pass. Both Claude conceptual reviews completed.
The frozen screen completes with failed gates: reference256/256 correct, NLL0.001453;
long toggles256/256, NLL0.001271; reset order160/256 (62.5%), NLL0.570421;
no-information length192/256 (75%), NLL0.263887. Runtime predictions and latents agree
with the batch evaluator in every cohort; all1024 histories pass retries/restores.
Independent simulation confirms labels and balanced pair counts; every observation
routes to the correct entity. Frozen matcher/cell weights and descriptor holdout are
verified. This narrows the observed failure to learned state behavior under these
new histories, without proving a general cause or isolating length from repetition.

No training or threshold tuning occurred. Preserve the negative screen as the next
reference. The repeated-toggle condition specifically repeats a two-target pattern;
its success is not evidence for arbitrary long histories. All four cohorts share16
fresh descriptor families, so episode rows are correlated. No-information opcode
has no state-value input and appears separately from reset/toggle order pairs.

[Report](../runs/entity_temporal_v1/reference/report.html) passed1280×720 browser QA;
[verification](../runs/entity_temporal_v1/verification.json) binds raw outputs,
independent NLL/labels, frozen weights and successful cached CLI resume.
Use `--entity-temporal-cell STATE_CHECKPOINT --entity-state-weights MATCHER_CHECKPOINT
--output DIR`; `--resume DIR` reuses cached results. Next proposed: broaden the
state-update training distribution with reset order and no-information events, then
compare frozen models on fresh unseen temporal compositions. Keep the recognizer
frozen and separately inspect whether no-information should be an explicit no-op
for this deterministic task. Neither repair has been implemented or validated yet.

### Mixed-history state adaptation (predeclared)

Keep recognizer, width16 GRU/readout, seed31, AdamW lr0.003/wd0.01, batch32 and256
updates unchanged. One training run, CPU450s. Train12-event histories: two initial
values, eight random targeted operations (observed0/1, toggle, no-information), two
final no-information queries. Per family four random templates with four entity-wise
binary complements each balance all final pairs.32 train/16 development descriptor
families (seeds201/202); separate temporal seeds211/212.512/256 histories.

Compare original and adapted frozen cells on identical fresh descriptor seed121,
16 families. Keep four existing temporal cohorts and add mixed20-event histories
(seed221,16 middle operations); thus evaluation lengths6/20 differ from training12.
Same >=95% pair accuracy, NLL<=0.15, runtime accuracy and transaction/latent gates.
No hard no-information gate, extra supervision, optimizer-budget change or tuning.
The intervention is training-distribution adaptation; descriptor/template holdouts
and frozen models must be audited. Preserve failures.

### Mixed-history adaptation result

Implementatione828131;123 CPU tests pass. One256-update run reaches100% train and
development state-pair accuracy (development NLL0.003449); persistent runtime agrees.
Training/development temporal signatures and descriptors are disjoint, and evaluation
uses different lengths with fresh shared descriptors. Both Claude conceptual reviews
completed. Only the recurrent cell/readout trained; recognition stayed frozen.

Matched frozen comparison on seed121: reset order improves62.5% to100%; unseen random
compositions38.671875% to100%. Reference and repeated-toggle conditions remain100%.
Long no-information histories remain75%; NLL worsens from0.263887 to0.412956. Therefore
the adapted model still fails the overall gate. No hard no-op, threshold tuning or
extra training was introduced after seeing the results. Mixed random no-information
examples did not establish stability over long consecutive stretches.

All routing and retry/restore checks pass. Adapted runtime latents agree in every
condition. The old model's unseen-composition cohort has two numerical latent-check
failures (maximum absolute difference1.78069e-6), retained under the fixed tolerance;
its runtime accuracy is unchanged. Frozen weights, independent targets/NLL, balanced
pairs, exact matched histories and cached training/adapted resume were verified.
An immutable state checkpoint copy was made after training resume, before evaluation.

[Training](../runs/entity_state_varied_v1/training/report.html),
[baseline](../runs/entity_state_varied_v1/baseline/report.html),
[adapted comparison](../runs/entity_state_varied_v1/adapted/report.html),
[verification](../runs/entity_state_varied_v1/verification.json).
All reports passed1280×720 browser QA. Train with `--entity-state-varied`; comparisons
add `--entity-temporal-seed 121 --entity-temporal-varied`. Shared descriptor families
and complemented templates imply correlated rows, not independent confidence bounds.
Next proposed: test a state-preserving no-information update for this deterministic
task, with fresh idle lengths and the failed learned-update reference preserved.
This would be a task-specific architectural intervention, not a general rule for
belief evolution without observations. No graph-learning claim follows.

### Explicit idle preservation plan

Compare a task-specific exact `[0,0,0,1]` no-information bypass with the frozen
mixed-history cell. Persist the policy in checkpoint state; legacy checkpoints keep
learned updates. Training and transactional runtime share one update operation.
Essential red checks cover exact latent preservation, gradient identity, non-idle
updates, retries and rejection of snapshots with different update semantics.

One seed31 mixed-history training run, same512/256 examples,256 updates,32 batch,
AdamW0.003 and450-second budget. Recognition remains frozen. No tuning. Frozen
comparison uses fresh descriptor seed131,16 families, the existing five temporal
cohorts plus idle stretches of3 and31 events;256 rows per cohort. Evaluation budget
240 seconds per model. Fixed gates remain pair accuracy>=95%, NLL<=0.15, runtime
accuracy>=95%, all transaction and existing latent-tolerance checks. Require exact
idle preservation additionally. Reuse the original mixed-history immutable donor;
copy the new donor only after cached training resume. This establishes an explicit
invariant for deterministic dynamics, not learned general belief persistence.

Claude reviewed the abstract design and withdrew the claim that an exact identity
necessarily blocks gradients. The implementation uses differentiable selection;
checks compare parameter gradients with and without an inserted idle stretch.
Both training arms see the same histories. Policy is a persistent buffer only in
the new mode, preserving legacy state keys and binding new snapshots to semantics.
The first runtime test failure was a randomized recognizer fixture mismatch, fixed
by restoring with the same recognizer; no runtime compatibility check was weakened.

### Explicit idle preservation result

Implementation9e22167;128 CPU tests pass. The single256-update mixed-history run
passes development at100% pair accuracy, NLL0.006980. Matched training manifests,
settings and optimizer are verified; only the explicit idle policy changes.
Frozen recognition and both evaluated cells are unchanged during evaluation.

On fresh seed131 descriptors, all seven adapted conditions pass at100% pair accuracy.
The previous mixed-history baseline scores75% after14 idle events and25% after31;
the new cell scores100% on both. Added3/14/31 idle events preserve final logits and
persistent latents exactly relative to the reference history. This is an engineered
invariant; it is not evidence that the model learned temporal belief persistence.

Reference, reset order, repeated toggles and unseen mixed histories retain100%
accuracy. Their NLLs rise (largest0.073697 on repeated toggles), remaining below the
predeclared0.15 gate. All routing, retries, restore and latent-tolerance checks pass
for both models; maximum batch/runtime latent error is2.98e-7. Independent target
simulation/NLL, balanced pairs, matched histories and descriptor isolation pass.
Training and adapted cached resume pass; immutable new donor copied after resume.
Both Claude conceptual reviews completed, with the gradient claim corrected.

[Training](../runs/entity_noinfo_v1/training/report.html),
[baseline](../runs/entity_noinfo_v1/baseline/report.html),
[adapted](../runs/entity_noinfo_v1/adapted/report.html),
[verification](../runs/entity_noinfo_v1/verification.json) are retained. All three
reports pass1280×720 browser QA, including expanded training examples.

Train through the existing recipe with `--entity-state-weights MATCHER.pt
--entity-state-varied --entity-state-preserve --output DIR`. Evaluate using
`--entity-state-weights MATCHER.pt --entity-temporal-cell CELL.pt
--entity-temporal-seed 131 --entity-temporal-varied --entity-temporal-idle --output DIR`.
Resume with `--resume DIR`. Legacy donors omit the policy buffer and remain ungated;
new donors persist it and snapshots reject a policy mismatch.

Next proposed slice: a controlled interaction whose correct update depends on a
second entity's remembered state, with paired histories and swapped-role controls.
The present cell updates entities independently, so passing this slice alone does
not implement learned edges, concept discovery, visual identity or general graphs.
Keep the deterministic idle rule scoped to this task; time passage in an evolving
world can warrant belief changes even without observations.

### Directed interaction plan

Introduce a learned copy-state update receiving destination and source latents.
Keep the successful no-information cell/readout and recognition frozen; train only
an added interaction MLP. IDs route records but are not features. Relationship type
and endpoints are supplied, not learned. Persistent observation accepts an optional
known source ID with a zero feature vector, includes it in retry identity, stages
recognition and state together, and preserves the source. Batch/runtime share the
same interaction function. Reject absent sources and conflicting retry payloads;
injected failures must leave the whole snapshot unchanged.

Compare full source access with an identically initialized/trained source-zero
control. One256-update seed41 run per arm, AdamW0.003, batch32,450 seconds each.
Use immutable recognition and no-information donors. Train32 descriptor families
seed301, develop16 seed302; independent middle-history seeds311/312. Each family
contains16 balanced initial-state/direction/post-toggle variants. Frozen evaluation
uses16 fresh families seed303, history seed313: reference, swapped roles,31 inserted
idle steps and a second copy using the first updated entity as source. Models see
no answer or raw identity label. Fixed full-model gates: pair accuracy>=95%, NLL<=0.15,
runtime accuracy>=95%, all retries/restores and existing latent agreement tolerance.
The source-zero control should remain<=60% reference pair accuracy. Counterfactual
source changes must affect destination answers, with source latents unchanged.
Record any held-out-composition failure without tuning. These correlated variants
are a bounded interaction diagnostic, not evidence of learned graph structure.

Claude's two conceptual reviews support the latents-only interface, source isolation,
transaction checks and separately trained ablation. Its remaining concern is whether
an interaction-produced latent is compatible with a subsequent interaction. That is
exactly what the composition cohort tests; a failure will count against the declared
capability gate, not be dismissed as an artifact. The same runtime checks apply to
both copies. Swapped roles are labeled symmetry, not novel-ID generalization.

### Directed interaction result

Implementation71c6d97;132 CPU tests pass. One256-update run per arm, with matched
initialization, data and optimizer. Frozen recognition and ordinary state weights
remain unchanged. Only the interaction MLP trains. The full model passes all five
populations at100% pair accuracy: development, reference, swapped roles, idle and
second-copy composition. Reference NLL0.004632; composition NLL0.000961.

The source-zero control scores50% on reference/development/idle/swapped populations
and29.296875% on composition. It meets the declared<=60% reference control criterion;
its capability failures remain visible. Among128 counterfactual pairs whose source
history changes while destination history stays fixed, the full model answers both
members correctly in100% of pairs; the control does so in0%. Source identity labels
never enter the interaction network. Idle outputs are exactly invariant and swapped
outputs exactly exchange entity positions. Independent simulation/NLL and routing
checks pass. The composition population has two balanced final pairs; other cohorts
have four. These are correlated controlled variants, not independent samples.

Runtime retries, restore and batch-latent agreement pass for both models. Tests
cover unchanged source latents, missing source rejection, conflicting retry source
IDs and injected-transition rollback. Both cached resumes pass. An immutable full
checkpoint was copied after resume. Two abstract Claude reviews were reconciled;
no private code or measurements were sent. Both reports pass1280×720 browser QA,
including expanded examples. A transient screenshot capture failed and succeeded
on retry; no report defect was found.

[Full model](../runs/entity_interaction_v1/full/report.html),
[source-zero control](../runs/entity_interaction_v1/blind/report.html),
[verification](../runs/entity_interaction_v1/verification.json).
Use the existing recipe: `python -m experiments.multimodal --entity-interaction
--entity-state-weights MATCHER.pt --entity-temporal-cell IDLE_STATE.pt --output DIR`;
add `--entity-interaction-blind` for the control. Resume with `--resume DIR`.
The base donor must be the ordinary idle-preserving cell, not a prior interaction
checkpoint. Runtime adds `source_id=KNOWN_ID` to `observe` with four zero features.

This learns the update for a supplied directed copy relation. It does not learn
which edge to create, which source to retrieve, general concepts or visual identity.
Next proposed: source selection with a distractor entity and counterfactual routing
controls, removing the supplied source ID before claiming learned relational retrieval.

### Frozen source retrieval plan

Reuse the learned matcher and learned interaction, both frozen. Replace externally
supplied source IDs with descriptor queries over three remembered entities. Add
read-only lookup, query-bound retries, persisted selected-source receipts, unknown
source rollback and batch state allocation for an explicit entity count. IDs remain
bookkeeping. Exact retries must return their original receipt before re-resolving;
different queries that match the same record still conflict under one event ID.

No training or new optimizer updates. One seed401 frozen screen,8 descriptor families,
24 balanced destination/source/bit variants each:192 episodes per condition. Source
and distractor bits oppose each other, making wrong retrieval observable. Conditions:
oracle source IDs, learned query lookup, permuted allocation order, deliberately
wrong queries and unknown queries. Fixed gates: oracle/lookup/permuted complete-state
accuracy>=95%, NLL<=0.15 and lookup source accuracy>=95%; wrong-query source/state
accuracy<=5%; unknown rejection100%. All conditions require retries/restore,
non-target preservation and existing batch/runtime latent tolerance. Budget180 seconds
for the entire screen. Ordinary dynamics, copy update, thresholds and descriptors
are unchanged. This tests frozen learned retrieval integration, not graph discovery.

Claude confirms threshold/selectivity semantics and the query-bound transaction
contract. All unresolved cases, including ambiguity and below-threshold matches,
use the same explicit lookup error and full rollback; no fallback source is chosen.
Wrong-query cohorts are sensitivity controls, not robustness claims. Matching margins
are not treated as calibrated correctness. Source receipt validation rejects unknown
persisted source IDs while retaining compatibility with ordinary legacy receipts.

### Frozen source retrieval result

Implementationf86a0ae;136 CPU tests pass. No fitting: the matcher, ordinary state
cell and learned interaction remain frozen. Oracle, lookup and allocation-permuted
conditions each achieve100% complete-state and source accuracy on192 episodes;
NLL0.006948. Oracle and lookup logits are identical. Semantic outputs agree after
allocation permutation. Minimum accepted source confidence0.97638 exceeds the
unchanged0.75 threshold; this is a score/selectivity observation, not calibration.

All192 unknown queries reject and preserve the prior state. Deliberately wrong
queries select the opposing-state distractor, producing0% intended-source and
intended-state accuracy as required by the sensitivity control. This does not claim
robustness to wrong instructions. All retry/restore, non-target preservation and
batch/runtime latent checks pass. Independent target/NLL and source-selection
recomputation, metadata counts and frozen checkpoint comparisons pass. Cached resume
passes. The report passes1280×720 browser QA with expanded examples. Three short
Claude conceptual reviews close threshold, identity and abstention concerns.

[Report](../runs/entity_source_v1/reference/report.html),
[verification](../runs/entity_source_v1/verification.json).
Run `python -m experiments.multimodal --entity-source --entity-state-weights MATCHER.pt
--entity-temporal-cell INTERACTION.pt --output DIR`; resume with `--resume DIR`.
Runtime accepts `source_query=UNIT_DESCRIPTOR` instead of `source_id` for a copy
observation with four zero features. `memory.lookup` is read-only. Unresolved queries
raise `LookupError` with no committed change. Exact replay returns the recorded source;
changed query payloads conflict even if they resolve to that same record.

This removes externally supplied source IDs, but still supplies the descriptor query
and copy operation. It does not discover graph structure or learn what to retrieve
from a task. Next proposed: remember a source relation from an earlier cue, then use
it for a later destination-only action after the source cue disappears. Use paired
relation histories with identical final observations to test relational memory.

### Remembered relation key plan

Add one explicit relation slot per destination containing a learned latent key.
An earlier bind cue writes that key; a later destination-only recall decodes it and
retrieves the current source through the frozen matcher. Only the key encoder/decoder
trains (8→16→8 with normalization). The existing state dynamics and copy module stay
frozen. Latest-write replacement, storage and relation type are explicit machinery,
not learned graph discovery. Bind/read operations stage all changes atomically.
Read retry identity binds destination/action/time, not a newly decoded key, so exact
retries after rebind return their original result without undoing the new relation.

One seed51 training run:256 updates, batch32, AdamW0.01,450-second budget. Train96
source queries from32 families seed501; develop24 from8 families seed502. Objective:
source cross entropy through the frozen matcher. Frozen runtime screen uses8 fresh
families seed503 with24 variants each. Paired variants keep all state observations
and final action fixed while earlier relation cues select opposing-state sources.
Conditions: reference, latest-cue replacement,31-event gap, allocation permutation,
and erased relation key. Sources change state after binding, so cached old values
cannot solve recall. Fixed gates: development source accuracy>=95%, NLL<=0.15;
reference/rebind/gap/permutation source and full-state accuracy>=95%, NLL<=0.15.
Erased source accuracy<=60%. All conditions require retry/restore, non-target-state
preservation and relation-key preservation on reads. Record failures without tuning.

Claude's two conceptual reviews distinguish source-key learning from supplied storage
semantics. The encoder consumes only the cue descriptor, never candidate order or
state labels. Fresh actions read the live destination key; retries intentionally
return the old receipt and cannot undo a later rebind. Tests cover two independent
bindings. Erased keys are an experimental zero-latent ablation, not a promised
forgetting API: unresolved reads reject, while confident misrouting is measured.

### Remembered relation key result

Implementation7347bf1;140 CPU tests pass. One256-update run trains only the280-parameter
key encoder/decoder. Development retrieval is100% over24 fresh queries, NLL0.023475.
Reference, replacement,31-event gap and permuted allocation each achieve100% source
and complete-state accuracy on192 episodes, NLL0.006948. Their logits are identical.

Erased-key reads all reject: source accuracy0%, complete-state accuracy50%, NLL0.777530.
This passes the predeclared negative-control criterion, not the normal capability
gate. For all96 paired histories per normal condition, state latents before recall
are identical while correct targets differ; both sources are recalled correctly.
Erased-key paired-source success is0%. Sources toggle after binding, so successful
recall reads current state rather than a cached binding-time value.

Independent target/NLL and decoded-source recomputation, descriptor split isolation,
frozen-consumer hashes, non-target and key preservation, retries/restores and cached
resume pass. Tests additionally cover multiple bindings, missing relations, model
compatibility, failed-transition rollback and old retries after rebind. Two Claude
conceptual reviews were reconciled. Browser QA passes1280×720 including expanded
examples. The key checkpoint was copied immutably after resume.

[Report](../runs/entity_relations_v1/reference/report.html),
[verification](../runs/entity_relations_v1/verification.json).
Run `python -m experiments.multimodal --entity-relations --entity-state-weights MATCHER.pt
--entity-temporal-cell INTERACTION.pt --output DIR`; resume with `--resume DIR`.
`EntityRelationMemory` wraps the state store: ordinary `observe`, early
`bind(event_id, destination_descriptor, source_query, timestamp)`, then
`recall(event_id, destination_descriptor, timestamp)` without the source cue.
Snapshots persist learned keys and bind restoration to all model fingerprints.

This establishes learned key addressing across a stored relation, with explicit
slots, bind/read calls and latest-write policy. Persistence itself is supplied.
Next proposed: learn a relation-update gate that distinguishes a valid binding cue
from an irrelevant cue, using paired histories that require either overwrite or
preservation. No claim of learned graph topology or general concepts follows yet.

### Context write gate: transactional interface (2026-09-11)

First implement the runtime boundary: optional learned four-dimensional context
comparison, fixed probability > 0.5 write decision, full retry payloads, exact key
preservation on ignored cues, and atomic rollback on accepted unresolved sources.
Gate fingerprints must accompany snapshots. Existing ungated snapshots remain
compatible. CPU unit checks only in this slice (no training or quality claim).
The subsequent experiment must separately predeclare its populations and compare
soft training with hard decisions and always/never-write controls. Claude's public
review emphasized this mismatch and repeated-ignore drift. Squared differences
are symmetric but do distinguish opposite vectors; state bits are not gate inputs.

Runtime implementation: `RelationWriteGate` and optional
`EntityRelationMemory.consider` now provide this boundary. Claude follow-up
`entity-gate-reconcile` identified concurrency: this interface retains the
existing single-writer contract; callers must serialize access. It does not add
concurrent transactions. No gate training/evaluation run has been performed.

### Gate learning comparison (2026-09-11)

Freeze matcher, interaction cell and relation key donors. Train only the 97-parameter
context gate, seed61, AdamW lr0.01/wd0.01,256 steps,batch32,CPU450s total.
Descriptor families: train601/32, development602/8, evaluation603/8; context
seeds701/702/703 respectively. Per family use all six ordered distinct source
pairs, each with matched and unrelated contexts (12 examples). Matched context
is unit active plus0.03 Gaussian noise, renormalized; unrelated unit context has
distance>=0.9. Gate sees only squared context differences. Loss is source CE
through the frozen decoder/matcher on a sigmoid blend of old/new keys.

Evaluate hard runtime decisions on the96 held-out cases, reference/permuted/
31-repeated-irrelevant/always-write/never-write. Source states oppose, destination
starts zero. Gates: learned source and complete-state accuracy>=95%, source NLL
<=0.15, soft/hard source agreement>=95%, exact replay/restore and ignored-key/
non-target-state preservation. Controls must score50% source/state accuracy.
These are correlated synthetic context-rule tests, not graph discovery. No tuning
on evaluation results; preserve any failure. Check cached resume before report QA.

Claude review (`gate-training`, `gate-training-reconcile`) accepted the narrowed
scope after identifying the easy separated regimes. Added an unlabelled sweep
along a fixed unit-circle arc; Euclidean input distance is independent of learned
weights, although these same context coordinates are deliberately gate inputs.
This is a boundary diagnostic, not calibration or evidence of ambiguous relevance.


Result: source/state accuracy100% in reference, permuted and repeated conditions
(96 cases each); source NLL0.02357394 and soft/hard source agreement100%.
Always/never-write controls each50%, NLL7.13082. All integrity checks pass.
Development96 cases accuracy100%, NLL0.02842698. Held-out ignored probabilities
range0.01038–0.30586; accepted0.82645–0.82904. These are not calibrated beliefs.
Full145 CPU tests, cached resume, independent target/NLL/hash/split audit pass.
Implementation commit78c7f4c. No evaluation tuning or additional training.
Report is self-contained and structurally verified; browser security denied local
file navigation, so visual QA remains incomplete and no workaround was attempted.

Run `python -m experiments.multimodal --entity-gate --entity-state-weights MATCHER.pt
--entity-temporal-cell INTERACTION.pt --entity-relation-key KEY.pt --output DIR`;
resume with `--resume DIR`. Evidence: `runs/entity_gate_v1/verification.json`;
immutable donor copied after resume to `runs/entity_gate_v1/frozen_gate.pt`.
Next proposed: explicitly labelled noise-shift cohorts and threshold sensitivity,
without interpreting an unlabelled distance sweep as calibrated relevance.

### Frozen gate noise shift (2026-09-11)

No retraining. Freeze `entity_gate_v1/frozen_gate.pt`. Seed801 generates128 pairs
of four-dimensional unit prototypes; negative prototype distance>=0.9. Each pair
contains one same-prototype and one different-prototype cue. Add the same sampled
Gaussian vector to each cue at sigma0.03/0.15/0.30/0.60, then renormalize. Active
context remains clean. Labels follow generating identity, not observed distance.
All cohorts share underlying prototypes/noise;256 cases each, not independent
across severity. Evaluate decision thresholds0.4/0.5/0.6. Primary robustness gate:
at threshold0.5 each class recall>=95% in EVERY severity; retain failure. Other
thresholds, Brier score and distance overlap are descriptive, not calibration.
Budget30 CPU seconds, no optimization. This isolates write decisions and does not
retest downstream state accuracy. Save raw contexts/probabilities/labels, checkpoint,
manifest, report and cached resume. Browser QA remains unavailable after the prior
local-file policy denial; do not try an alternate route.

Claude identified noise/separation ambiguity. Added a fixed Euclidean-distance
baseline (accept when distance<0.5, not an oracle) and per-case pre-normalization
noise/separation ratios. These are descriptive and cannot establish a Bayes floor.


Frozen shift result (implementation5a241b8): robustness gate FAIL, retained.
At sigma0.03/0.15 both recalls100%. At0.30 accuracy95.3125%, accept92.96875%,
ignore97.65625%; at0.60 accuracy75.390625%, accept55.46875%, ignore95.3125%.
Distance baseline accuracy100/99.21875/84.375/60.9375%. Threshold0.4/0.5/0.6
accuracy at0.60 is79.6875/75.390625/69.921875%; no threshold was selected.
Claude accepted the diagnostic scope after adding baseline/noise-separation checks.
12 gate and multimodal training/report tests pass; cached resume and independent
probability, context reconstruction, metrics and frozen-weight audit pass. One
initial test command named a nonexistent report test file; corrected to the actual
multimodal report/resume tests before validation. No model weights changed.
Report structural QA passes; visual QA remains unavailable after prior browser
policy denial. Evidence: `runs/entity_gate_shift_v1/verification.json`.
Run `python -m experiments.multimodal --entity-gate-shift --entity-gate-weights
GATE.pt --output DIR`; cached resume `--resume DIR`.
Next candidate: noise-augmented training, fresh held-out contexts and per-class
checks, preserving this frozen reference. This is not yet an adopted repair or
proof that training can overcome observation ambiguity.

### Matched gate continuation (2026-09-11)

Continue the immutable gate donor in two arms: low-noise control and augmentation.
Both use seed71,256 AdamW updates lr0.01/wd0.01,batch32,450CPU seconds each,
identical descriptor/context rows and initial gate; matcher/key/interaction frozen.
Descriptor seeds901/902/903 with32/8/8 families; context seeds911/912/913.
Training contexts use paired same/different prototypes and shared Gaussian noise;
control repeats sigma0.03 four times, treatment uses0.03/0.15/0.30/0.60.
Same row count1536 and sampler trajectory; labels derive from generating identity.
Use source-selection CE through frozen consumers, no direct gate supervision.
Fresh stress contexts seed921,128 pairs, existing four severities and thresholds.
Each arm reports frozen-before and after on identical cases; no evaluation tuning.
Success: prior low-noise runtime gates pass, both class recalls>=95% at0.03/0.15,
and high-noise0.60 accuracy improves>=2 percentage points over frozen while ignore
recall loses<=5 points. Treatment must additionally improve>=2 points over the
matched control to support augmentation benefit. Preserve every failed criterion.
Moderate/high-noise results do not imply calibrated semantic beliefs. Visual QA
remains unavailable under prior browser policy; structural report checks required.

Claude reconciliation accepts the narrow exposure-adaptation claim. Control rows
repeat the same underlying pairs, matching treatment rows and sampler, but this
is not a comparison with additional independent low-noise observations. Per-bin
class recalls and false-write regressions remain required; no denoising claim.


Matched continuation result (0139198): acceptance FAIL, preserved. At high noise,
frozen accuracy75.78125%, control71.484375%, augmented80.078125%; treatment gain
8.59375 points over control,4.296875 over frozen. Treatment accept recall70.3125%
versus control49.21875%, but ignore recall89.84375% versus93.75%. At noise0.15,
treatment ignore94.53125% violates95% guardrail (7 false writes/128 negatives).
No threshold selected and reference donor unchanged. Both clean runtime state
accuracies100%; treatment development95.8333%, NLL0.128206. These correlated
single-seed results establish a tradeoff, not a reliable repair.
14 relevant tests pass; both cached resumes and independent donor/population/
probability/metric audit pass. Reports structural-only under prior browser denial.
Evidence: `runs/entity_gate_augment_v1/verification.json`.
Use the existing `--entity-gate` command with `--entity-gate-weights INITIAL.pt`;
add `--entity-gate-augment` for the treatment, omit for matched control. Resume
with `--resume DIR`. No candidate promoted to the immutable reference.
Next proposed: independent-seed replication with the same criteria to estimate
whether the false-write cost persists before changing loss or uncertainty policy.

### Two fresh matched replications (2026-09-11)

Add explicit nonnegative continuation replicate index; index0 preserves prior
seeds. For indices1 and2 use seed71+index and add100*index to descriptor and
context seeds (including stress). Same donor, optimizer,256steps/batch32,450s per
arm; four runs total, no tuning. Preserve all previous criteria. Replication passes
only if both treatments pass their full gates AND gain>=2 points over their paired
controls at sigma0.60. Report each replicate separately; no pooled confidence or
best-seed selection. This is conditional on one donor, not initialization variance.

Claude reconciliation: two single-donor repeats are a descriptive screen, not
statistical confirmation. Report each class/severity, no pooled promotion; even
two passes remain provisional.15 relevant tests pass, including nonzero-index resume.


Replication result (a5c7b69): BOTH full gates FAIL. High-noise accuracy gains over
matched control are8.984375 and11.71875 points; treatment accuracy80.078125%
in both. Rep1 adaptation subgate passes but clean-runtime NLL0.152775>0.15.
Rep2 clean-runtime NLL0.269396, development accuracy92.7083%/NLL0.232708,
and low-noise0.15 ignore93.75% fail. Thus augmentation gains recur but do not
qualify the model for replacement. These are two conditional single-donor runs.
15 relevant tests, four cached resumes and independent paired metrics/donor audit
pass. Structural report QA only; prior browser-policy limit remains. No threshold
changes, additional tuning, or donor promotion. Evidence:
`runs/entity_gate_replicate_v1/verification.json`. Use `--entity-gate-replicate 1`
or2 with the existing matched continuation command; index0 preserves prior seeds.
Next candidate: clean-behavior retention term during noisy continuation, with
separately declared tradeoff weights and fresh evaluation. Not implemented yet.

### Clean-retention comparison (2026-09-11)

Replicate index3 (seed74, descriptor1201/1202/1203, context1211/1212/1213,
stress1221), two noisy continuation arms, same immutable gate donor. Existing
256steps/batch32/lr0.01/wd0.01/450s per arm. Treatment adds Bernoulli KL weight1
between current and initial gate probabilities on corresponding low-noise replay
contexts; control weight0. Cache detached teacher probabilities from train only.
Same sampled rows and clean forwards in both arms; no additional sampler draws.
Keep all prior adaptation/runtime/development guards. Retention comparison passes
only if treatment passes those guards, clean runtime NLL is lower than control,
and sigma0.60 accuracy loses<=2 points versus control. No coefficient search.
Teacher consistency is not truth or calibration; errors may be preserved.

Claude corrected its initial misunderstanding: held-out ground-truth labels, not
teacher agreement, determine acceptance. Fixed coefficient sensitivity remains
untested; no broader retention claim follows from this single setting.16 relevant
tests pass including teacher detachment, noisy retention training and cached resume.
