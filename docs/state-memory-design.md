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
- Keep memory within reasonable bounds while allocating enough capacity for useful
  recall. This is an accepted sizing objective, not approval of particular counts
  or a finding that the illustrative budget below is sufficient.
- Give the user independent short-term and long-term memory reset controls. The
  reset requirement is accepted; the detailed ownership and invalidation rules below
  are proposals, not implemented behavior.
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

## Proposed observation and belief views

Recommend independently addressable evidence and belief views tied to the same
source event. Evidence tokens encode the available observation packet and declared
source metadata; their write path receives no recurrent belief, historical memory
read, task conditioning or generated workspace. Any encoder cache that violates
that independence must be excluded or explicitly reset. Belief tokens hold the
interpretation produced by the observation-update path. Both contents remain
learned; source-derived features are lossy and are not an assertion that the source
is true or that every source detail can be recovered.

Perception, prediction and thinking can query either view independently within their
read allowance. Protect requested evidence detail when it remains available, with
the belief view retained as useful context. Older evidence compression must read
only evidence-view inputs and declared source metadata; mixing in belief would
break the independent-source route. Separate queries/heads can share parameters,
but the data dependencies and origin labels stay separate. Consolidated interpreted
knowledge remains derived context. Exact sensory values or wording need a separately
budgeted retained payload if the latent encoding cannot recover them faithfully.

Reserve an explicit part of the total memory allowance for each view. Keep recent
belief snapshots intact at their declared shape; do not silently halve them to fit
an added evidence view. The belief-only dimensions and byte count below therefore
remain reference bookkeeping, not the cost of this proposed extension. A selected
two-view recipe must jointly specify evidence token count, belief token count and
history capacities, then recompute its total storage and read cost. No particular
split or evidence encoder has been selected.

## Concrete provisional sizes

One possible first recipe is `D=64, N=32, T=4, W=8, G=16`, with four attention heads;
`k=32, b=8, C=32, m=16, p=8`. These are bookkeeping defaults, not selected optima or
training authorization. Every module receives sizes explicitly.

Compression maps 8 x 32 source tokens to 32 tokens: an eightfold reduction in that
block's payload. The staging buffer is explicitly budgeted: up to seven older
states remain exact while awaiting compression. A full ordinary history covers
approximately 160--167 committed steps, plus lossy consolidated information and any
protected material outside that interval. Coverage is not recoverability.

Retain these counts as the starting proposal for the bounded-memory discussion.
Before fixing them, relate committed steps to the intended task's event cadence and
recall horizon. At one commit per second, 160 steps cover about 2.7 minutes; at 30
commits per second they cover about 5.3 seconds. Neither cadence is adopted here.
Do not change observation ingestion or omit state updates merely to make the nominal
memory horizon longer. Useful capacity must eventually support needed details after
recent-memory eviction, at acceptable read cost; increasing counts alone does not
establish that. Exact counts, cadence and adequacy remain open while model design
continues. Indexed retrieval remains optional.

The maximum memory payload, excluding current/task/working states and metadata, is
`(k*N + (b-1)*N + m*C + p*N + G)*D` scalars. Account separately for uncertainty,
training activations, raw evidence storage and source-index metadata. A protected
multi-state event consumes multiple state-equivalent slots; marking never creates
unbounded storage. Exact raw media retention is not included in this latent budget.

## Scaling considerations

This is a cost analysis, not a runtime or learning benchmark. Increasing the number
of retained records (`k`, `m`, `p`) can leave the learned parameter shapes unchanged;
changing token width or learned slot counts is a different model change. With the
illustrative sizes above, memory alone holds 2,032 tokens / 130,048 scalars: about
0.496 MiB per session in float32, excluding metadata, raw evidence, projections,
temporary copies and training activations.

Let `M` be the number of stored tokens and `Q` the consumer's query-token count.
Payload storage is `O(M*D)`. Dense cross-attention has `O(Q*M*D)` score/value work,
plus projections (including `O(M*D^2)` when memory projections are recomputed).
At fixed query count and width, the memory-read work grows linearly with memory.
This does not mean whole-agent latency grows by the same factor. Thinking rounds
and imagined action branches repeat reads; training also needs activation/gradient
accounting. The proposed readers do not require all stored tokens to self-attend.

Compression extends represented history per stored token at the expense of detail.
A fixed consolidated state cannot preserve arbitrarily many independent facts.
Larger banks also change retrieval distractors, delays and compression exposure;
usable recall beyond the training distribution is an empirical question.

A possible scaling extension is to separate storage capacity from a per-consumer
read budget: read bounded recent/consolidated tokens directly, and select a bounded
set of older/protected records through an index before attention. Index lookup,
maintenance and transfers still cost resources; approximate retrieval can miss the
needed record. This is a new proposal, not an adopted replacement for the dense
read contract below. [Memorizing Transformers](https://arxiv.org/abs/2203.08913)
demonstrates approximate retrieval from larger neural memory in language modelling;
its gains do not establish this design's scaling quality.

The existing `EpisodicMemory` implementation selects a fixed number of records but
scores every key and concatenates the bank on writes. Its current lookup and copying
therefore still grow with capacity; it is not an implemented scalable index. If a
separate observation-evidence view is adopted, divide an explicit total budget
between evidence and belief, or explicitly fund the extra storage/read cost. The
two views need not each duplicate the full existing budget.

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

## User-controlled memory reset: proposed contract

Expose `reset_memory(scope="short" | "long" | "all")` as an exact caller control,
outside learned marking or gating. It returns the new session state and a receipt
of cleared stores, retained source records and invalidated derived state. This
is a design requirement and proposed interface; no reset is being executed here.

| Scope | Cleared | Retained |
| --- | --- | --- |
| Short | Recent records, uncompressed staging, pending short-term marks/writes, current belief and task workspace | Existing long-term compressed blocks, consolidated state and protected records |
| Long | Compressed history, consolidated state, all protected records including user marks, pending long-term writes; all recent belief views, current belief and workspace | Independent recent/staged observation evidence and its factual execution/time records |
| All | Both groups and all memory-derived session state | Explicit caller task/control inputs and model parameters |

Treat protected details as long-term because their purpose is surviving recent
eviction. Short reset discards staging directly; it must not run normal eviction
compression or consolidation while clearing. Long reset clears any later-adopted
persistent derived-knowledge store too. Scope refers to storage lifetime, not age
of every fact: a protected record can be recent and a recent belief can recall an
old fact.

After short reset, initialize belief/workspace afresh. Subsequent ordinary reads
may consult the deliberately retained long-term memory. After long reset, initialize
belief/workspace and mark old recent belief views invalid; retain their independent
evidence views. Normal perception or thinking can read that evidence to form fresh
interpretations. Thinking still updates only working state. No special chronological
replay or rewriting of historical belief snapshots is required. Empty or invalid
views are masked; an entirely empty memory produces the existing zero read.

Independence applies to owned memory records, not continuity of shared inferred
state. Both resets invalidate current interpretations because those may mix the two
scopes. Preserving recent source evidence during a long reset depends on the proposed
evidence view; a belief-only implementation would instead need to discard dependent
recent content and expose that limitation. An action execution receipt can remain
as an explicitly retained fact about what occurred; its old rationale or predicted
consequences must not masquerade as independent input. Do not recover receipts from
an unbounded external log during reset.

Use a structural receipt containing execution ID, issued action kind/parameters,
available factual status, source/execution time and arrival time. External response
payloads enter as separately source-tagged observations. Internal rationale, cached
belief/plan vectors and predicted outcomes are not receipt fields. An action choice
or external text can still correlate with old beliefs; input-path separation does
not claim semantic erasure. Generated external content retains that origin status.

Reset does not execute actions, ingest observations, advance observation counts,
emit answers, fulfill tasks or trigger compression/marking/consolidation. Normal later
commits may compress retained short-term evidence into long-term memory again.
That preserves deliberately retained information; it does not restore deleted blocks.
Keep per-store capacities fixed across resets, within the overall bound. Clearing
one scope does not silently resize the other.

Every reset creates an atomic session-generation boundary. Invalidate retrieval and
encoder caches, active imagined branches, candidate plans and derived task tokens;
reject results or writes started before the boundary. Reset revokes pinned memory
handles, unlike ordinary concurrent commits. Stop affected background operations
before allowing subsequent reads, writes or action/output proposals. Re-encode
explicit caller-held task inputs without recovering the cleared workspace. Preserve
actual environment time and executed-action facts; do not reset event identity in
a way that makes old work appear current.

An already-issued external action is not undone by reset. A later external result
can enter through a fresh observation-ingestion call with its actual availability
time, original source/execution time and action ID. It cannot resume the cancelled
plan or publish a pre-reset latent. Stale internal retrieval, thinking or compression
results are discarded. A bounded outstanding-execution correlation table belongs
to caller control metadata, not a historical memory-replay channel.
This table survives all memory-reset scopes only until completion or expiry, within
the caller's fixed outstanding-action bound, and is not exposed as model memory.
Unmatched completions return an `unmatched_execution` outcome to the caller rather
than entering memory automatically. A stale-generation synchronous operation returns
`stale_generation`; asynchronous stale work is discarded with a recorded outcome.
No automatic retry may recover its old payload. This preserves source correlation
without preserving a cancelled plan or turning the table into durable recall.

This clears accessible session memory. It does not change learned weights, source
files or completed audit artifacts, and those artifacts must not automatically
rehydrate a cleared store. Facts deliberately retained in the other store, explicit
task input, or a new observation can still be learned/recalled. Erasing a fact across
all representations would be a different, stronger requirement. No such erasure
guarantee or implemented reset mechanism is claimed.

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

For the reset proposal, an accepted mark installs its protected record immediately;
it does not wait for normal compression. Its long-term ownership survives a short
reset, including when the implementation shares its payload with a recent record.
Pending/unadmitted requests have no protection guarantee and are cancelled at reset.
Admission and installation use the same atomic reset-generation check; an old
proposal cannot install after reset. The existing user-priority/overflow and
lower-scored-agent replacement rules still apply.
For a batch, process user requests first and preserve caller order within each
priority group. Return a per-request accepted or `capacity_exceeded` result;
rejection leaves existing protection unchanged. A reset-generation mismatch returns
`stale_generation` instead of installing or automatically retrying the old request.

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
