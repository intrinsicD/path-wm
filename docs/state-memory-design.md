# Learned state and hybrid memory: proposed interface specification

Status: architecture discussion, 9 September 2026. This is a concrete proposed
contract, not implemented behavior or an approved experiment. The existing negative
learning result remains unchanged. Finish the model design before task design or
training. Ordinary PyTorch components and one editable recipe remain the boundary.

The [Claude review and discussion agenda](state-memory-review.md) identifies remaining
state/evidence, filtering, derived-knowledge, fusion, learning and action contracts.
The concrete shapes below do not settle those choices. In particular, a separately
addressable observation-evidence view is a new recommendation to discuss, not an
already accepted change to the belief-snapshot memory described here.

## Direction established in discussion

- Learn the semantic content of the world representation. Do not reserve state
  coordinates for physical objects, places, documents or other domain concepts.
- Keep current belief and task working state operationally distinct. Their token
  meanings remain learned. Factory observations and document observations can use
  the same interface without implying that one trained model already handles both.
- Keep recent latent states intact, compress older groups, and preserve extra
  detail for marked events. Both user and agent can propose marks; explicit user
  marks have priority within a declared capacity.
- Perception, prediction and thinking make separate queries over shared memory.
  Thinking does not advance world time or create observational evidence.
- Condition prediction on action and elapsed time; initially compose recorded
  transitions for longer forecasts. Future observations are training targets only.
- Preserve observation sources, inferred beliefs and imagined/generated ancestry.
  Spatial relationships may be learned, without imposing a spatial map on all domains.

The dimensions, consolidation mechanism, distribution family and exact policies
below are assistant proposals made to close interface gaps. They are not additional
user-approved choices or demonstrated architectural improvements.

## Token and record contract

`B` is batch size; each batch member is an independent session. `D` is shared token
width. Token positions carry learned slot identities, not predefined semantic roles.

| Record | Tensor payload | Meaning of the interface |
| --- | --- | --- |
| Observation features | `[B,L,D]`, valid mask `[B,L]` | Variable-length outputs from modality adapters, retaining input scale and modality tags |
| Current belief | `[B,N,D]` | Inferred current environment state; no required object/place slots |
| Task tokens | `[B,T,D]`, valid mask | Encoded task content; exact task controls remain separate records |
| Working state | `[B,W,D]` | Task-specific intermediate computation |
| Consolidated session state | `[B,G,D]`, valid mask | Lossy information carried forward as old episodic blocks leave memory |
| Action condition | `[B,1,D]` | Encoded action kind, parameters, action-presence indicator and elapsed time |
| Recent memory | `[B,k,N,D]` plus occupancy mask | Up to k most recent committed belief snapshots |
| Compression staging | `[B,b-1,N,D]` plus occupancy mask | Evicted snapshots awaiting a complete chronological group |
| Compressed history | `[B,m,C,D]` plus occupancy mask | Up to m blocks, each summarizing b source states using C tokens |
| Protected detail | `[B,p,N,D]` plus occupancy mask | At most p state-equivalent token budgets for marked material |

Records also carry content-time support, availability time/order, source/event IDs,
session ID, representation revision and origin/ancestry. A derived representation
is an inference grounded in sources, not a claim that every encoded fact was directly
observed. Generated material cannot become observational evidence through compression,
consolidation, copying or subsequent real observations.

Time metadata uses a declared common coordinate. Use measured durations when the
data supplies them, otherwise explicitly use step units. Distinguish availability
from the time of the underlying content. A chronological tie uses an event ordinal.
Do not infer positions, pose or timestamps that were never supplied or estimated.

All session tensors belong to the caller. Model parameters are separate; ordinary
inference changes session state without changing weights. A new representation
revision requires re-encoding retained source material or an explicit migration,
rather than silently reusing incompatible cached tokens.

## Concrete provisional sizes

One possible first recipe is `D=64, N=32, T=4, W=8, G=16`, with four attention heads;
`k=32, b=8, C=32, m=16, p=8`. These are bookkeeping defaults, not selected optima or
training authorization. Every module receives sizes explicitly.

Compression maps 8 x 32 source tokens to 32 tokens: an eightfold reduction in that
block's payload. The staging buffer is explicitly budgeted: up to seven older
states remain exact while awaiting compression. A full ordinary history covers
approximately 160--167 committed steps, plus lossy consolidated information and any
protected material outside that interval. Coverage is not recoverability.

The maximum memory payload, excluding current/task/working states and metadata, is
`(k*N + (b-1)*N + m*C + p*N + G)*D` scalars. Account separately for uncertainty,
training activations, raw evidence storage and source-index metadata. A protected
multi-state event consumes multiple state-equivalent slots; marking never creates
unbounded storage. Exact raw media retention is not included in this latent budget.

## Memory reads

Each consumer owns its query projections, attention and gates. Share the stores
and interface, not necessarily the weights. Read from an immutable snapshot for
the duration of an operation. An imagined branch pins the storage-version handle
for its complete rollout as well as its real-evidence cutoff; concurrent commits
or consolidation must not change that branch's memory contents.

Three read groups preserve the discussed structure:

1. Recent detail: recent states plus valid compression-staging states.
2. Older information: compressed blocks plus consolidated session tokens, with
   distinct type tags so the reader can distinguish their origins.
3. Protected marked detail.

For query tokens `q`, perform an independent cross-attention into each group. Add
relative time, duration and memory-type features to the addressing inputs. Empty
groups are masked, and an all-empty memory yields a zero update.

Per query token, a learned gate produces normalized weights for the three read
vectors and a fourth, zero-valued no-memory choice. Add the weighted read through
a residual connection, followed by a residual feed-forward block. Gate weights are
learned influence, not calibrated confidence or evidence authenticity. No rule makes
recent memory universally more relevant than older memory.

Block access to unavailable future sources and other sessions. Imagined rollouts
retain the real-memory cutoff of their starting observation; advancing imagined
time does not unlock future real evidence. Source metadata is an audit trail, not
a neural mechanism that guarantees correct inference.

## Perception update

`observe(belief, observation_features, memory_snapshot, previous_action, time)`
returns an updated belief and distribution parameters, with source metadata.

1. Previous belief tokens cross-attend to current observation features and the
   previous-action/time condition, producing preliminary belief tokens.
2. These tokens query the three memory groups using perception-specific weights.
3. Add gated memory reads residually.
4. Cross-attend again to current observation features and refine the belief.

The second observation read supplies a direct correction route; it does not prove
that fresh evidence always wins. Stale and conflicting-history examples must teach
appropriate revision. The observation branch never consumes generated working
state as clean evidence. Any optional task-conditioned sensory branch retains its
ancestry separately.

Observation packets and commit events are distinct: multiple modalities can update
one event before it is committed once. Equal-time ingestion must not silently
duplicate a committed snapshot. Future input is rejected before encoding.

## Prediction

`predict(belief, action_record, dt, memory_snapshot)` returns a next-state
distribution on a new imagined branch. The action adapter uses learned type and
parameter encodings plus a presence indicator; missing action is distinct from
explicit no-op. A continuous time encoding combines with these into one condition
token, which the state attends to before prediction-specific memory reads.

After gated fusion, a transition head predicts a residual mean and log scale with
shape `[B,N,D]`. A diagonal Gaussian is a concrete baseline distribution, not a
claim of calibrated epistemic uncertainty or adequate multimodal futures. A later
mixture or other distribution can replace this head at the same operation boundary.
The mean or a sample produces one branch state; a set of rollouts retains multiple
branches. Diagonal Gaussian samples do not solve distinct-outcome modeling by
themselves.

One step initially corresponds to one recorded transition with its associated
action and duration. Long predictions use the corresponding action sequence and
repeated steps, with no intermediate future observations. They can re-query the
same real history from evolving imagined states. They do not write observed memory.
The next real observation uses the clean observation branch for correction.

## Thinking and task outputs

`think(belief, task_tokens, working, memory_snapshot)` returns updated working
tokens. Working tokens attend to belief and task tokens, query the three memory
groups with thinking-specific weights, then receive gated residual updates.

Propose a maximum of three iterations per invocation initially; later queries can
depend on earlier working updates. The bound is an explicit caller budget. More
iterations are not assumed to help. Thinking updates neither world time nor belief
snapshots, consolidated evidence or observational memory. Generated reflections
have a separate tagged working-memory route.

Task action/output proposals consume task-conditioned working tokens with access
to current belief. The planner invokes prediction on separate branches with
explicit candidate actions and an objective. Environment actuation remains with
the caller. Answer decoders read the working state and current belief initially;
there is no additional decoder-to-memory bypass in this baseline. Existing exact
required/disabled/automatic output controls and requester/producer attribution
remain outside learned scores. Planning, instruction following and completion
quality remain unproven.

## Writes, compression and consolidation

`commit_observation(updated_belief, event_id, mark_requests)` is explicit and runs
after the perception update. A commit stores a detached snapshot at inference,
updates the recent ring, and moves each evicted state into the bounded staging
buffer. On reaching b states, compress that complete group and clear the buffer.
Readers can use staging states before compression. This preserves exactly the k
most recent snapshots without hiding extra staging storage.

The compressor has C learned query tokens. They cross-attend to the b source
states with their chronological/action/time encodings, then pass through residual
attention/feed-forward processing. Compression receives only information available
at its write cutoff. The proposed baseline compressor is task-neutral; current
tasks influence reads and marks, rather than determining all historical retention.

Append the new summary to the compressed FIFO. When a summary leaves that FIFO,
update G consolidated session tokens by attending to the departing block and
the previous consolidated state. This is lossy recurrent consolidation with bounded
capacity, not unlimited retention. The consolidated result records its new
availability and inference ancestry. It is current inferred knowledge, not a
rewritten historical observation. It can encode document relations as well as
spatial relations. This proposed addition closes the finite-FIFO persistence gap.

The proposed consolidation update is explicit:

1. Take the pre-update G session tokens and the departing C-token block, with masks,
   source content times, availability and ancestry. Read no later evidence.
2. Let the G tokens cross-attend to the departing block, then apply a residual
   self-attention/feed-forward refinement to form a candidate H of shape `[B,G,D]`.
3. Compute per-token/per-channel sigmoid gates `g` from the old tokens, the candidate
   and time metadata. Set `K_new = (1-g)*K_old + g*H`. A low gate preserves an old
   component; a high gate adopts the candidate component. The learned token slots
   still have no prescribed object/place meanings. Empty inputs leave K unchanged.
4. Record this as a new inferred session-state version, then evict the source block.
   Retain content-time support separately from update time: old information does
   not become new evidence merely because it was consolidated today.

Only observed-branch history enters this baseline update. Thinking, imagination and
mark proposals cannot directly rewrite K. Newly observed corrections immediately
affect current belief through perception; they reach K later through chronological
consolidation. Therefore K is historical context, not the authoritative latest state.
Gates do not guarantee correct conflict resolution or prevent forgetting.

Train the gate and candidate update through delayed prediction/completion/recall
and a read-preservation auxiliary objective that compares access to old K plus
the departing block with access to new K. Keep the reference targets detached and
recompute learned updates during replay. This trains useful retention without
requiring recovery of every source token. The initial gate bias and loss weights
belong to the later training recipe; no empirical preservation guarantee follows.

Consolidated tokens reset at session start in this baseline. Cross-session durable
knowledge and changes to model weights are separate designs, not implied by the
word long-term. Source records can be retained in a run audit outside the model's
attention budget; faithful recovery from a pointer requires that its source payload
actually remains stored.

## Mark proposals and protected detail

Each proposal contains author, event/source IDs, a detail query or whole-state
request, priority and requested token budget. A marker can request additional
retention from the detail still available at that point. It cannot reconstruct
discarded information merely by naming an old event.

Both user and agent proposals use the same admission function. Explicit user marks
take priority; agent scores allocate the remaining capacity. If user requests alone
exceed capacity, return an explicit capacity outcome for the caller to resolve;
do not silently promise protection. Agent entries may replace lower-scored agent
entries under the same capacity. An accepted mark protects retained latent material,
not necessarily a lossless sensory recording. Identical protected content should
share storage rather than be duplicated by repeated marks.

A learned marker sees current state/event/task context only. Proposed supervision
uses replay comparisons of protected detail versus ordinary compression: delayed
prediction/completion/answer benefit minus storage cost. Outcomes are detached
training labels, never future inputs to the live marker. Pair histories and account
for displaced memories when comparing equal-budget policies. Raw attention frequency
or surprise is not itself proof of usefulness.

## Learning and gradient boundaries

- Predict future detached target representations and observable outputs; mask
  partial inputs before encoders, pooling or memory can leak withheld content.
- Preserve representation grounding and diversity alongside predictability. An
  EMA target alone does not prevent collapse or prove semantic quality.
- Train delayed queries after recent detail expires, and retention of marked
  details. Include conflicting/stale histories and task changes.
- Train compression/consolidation through delayed consumer losses when their
  outputs remain in the training graph, plus local read-preservation objectives.
  Recompute the learned compression operations during replay when saved memories
  are detached; detached source snapshots can still train compressor parameters.
- Detached deployment storage is not a training design. Specify truncation windows
  in the eventual recipe and ensure losses cross actual compression/consolidation
  operations, rather than silently cutting every useful gradient.
- Internal working tokens need no textual chain-of-thought target; supervise
  grounded answers, decisions and predictions. Exact task controls remain enforced.

Masked online completion uses only current/past available inputs. Retrospective
completion using later evidence is a separate explicitly labeled training mode.
Neither allows withheld targets into online predictor memory. Actual objective
weights, datasets, optimizer schedule, compute and scientific thresholds are not
fixed by this interface document.

## Scope and compatibility

This changes the existing named latent slices, state schema, memory representation
and module signatures. It requires an intentional new model/checkpoint version;
existing reports and checkpoints remain bound to their source snapshots. No code
or checkpoint migration has happened in this discussion.

The specification fixes proposed shapes, ownership and ordering. It does not prove
that learned tokens discover useful semantics, that memories retain arbitrary
details, or that spatial consistency emerges. Domain-specific structure can later
be tested as an optional adapter or readout. Architecture adoption and review come
before implementation, then the separate experiment workflow supplies data,
budgets, failure checks and measurable acceptance criteria.
