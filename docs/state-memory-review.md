# State and memory design: Claude review and discussion agenda

Status: conceptual review, 9 September 2026. The user requested a complete review
with Claude and an agenda for continued architecture discussion. No implementation,
new dataset, training run or model-quality evaluation was requested or performed.

The [interface proposal](state-memory-design.md) is sufficiently concrete to expose
the remaining decisions. It is not yet a complete learning or belief-update contract.
The requested hybrid memory and learned, domain-neutral representations remain the
starting point. The important gaps concern what information the states preserve,
how real observations correct predictions, and how derived knowledge can persist.

## Review scope and evidence

Actual Claude reviewed an abstract brief covering the full proposed architecture:
modality/action adapters, neutral state tokens, recent/staged/compressed/protected
memory, consolidated session state, the three consumer read paths, observation and
prediction ordering, thinking/output/planning, origin and time metadata, and the
compression/marking learning proposals. The brief omitted private source, results,
dataset details, repository identifiers and exact implementation defaults.

Claude ran from an isolated temporary directory with tools, MCP, project settings,
custom instructions, hooks and session persistence disabled. The configured main
review model is recorded as `claude-opus-5[1m]`. Exact prompts, responses and CLI
receipts are retained locally under `runs/reviews/state_memory_design_2026-09-09/`.
Cost fields are API-equivalent estimates, not subscription bills. This is conceptual
criticism, not Claude inspection of the private implementation or empirical validation.

An independent local review was written before reading Claude's first response.
Three successful exchanges are retained. The second reconciled overstatements;
the third accepted the architecture-first next question, withdrew a demand for fixed
semantic fields and exact latent retention, and supported the two-view interface as
an engineering preference. It left one useful qualification: an evidence encoder is
also lossy, so the design must specify what distinguishes its input and retention
pressure, and whether consumers can read it independently.

## Architecture that remains the starting point

- Semantic token contents are learned. Operational roles such as observation
  evidence, inferred current state and task workspace do not reserve coordinates
  for machines, rooms, documents or other domain concepts.
- Recent detail, compressed chronological history, protected marked detail and
  bounded lossy consolidation provide different retention costs and horizons.
- Perception, prediction and thinking query shared stores with their own read
  parameters. Timing, session ownership and generated ancestry stay explicit.
- Both user and agent can propose marks. Extra detail and any raw evidence access
  consume a declared budget. Exact retained latents are not lossless observations.
- Prediction uses action and elapsed time; thinking preserves world time and does
  not transform hypotheses into observations. External execution and output controls
  remain explicit caller responsibilities.

None of these choices establishes learned world understanding, useful retention,
calibrated uncertainty or effective planning. The review does not designate a new
architecture winner or override earlier scoped negative results.

## Open decisions, in recommended discussion order

### 1. What is retained before memory compression even begins?

The current proposal snapshots the inferred recurrent belief. That state has
already filtered observations through its encoder, previous state and memory reads.
A protected copy cannot restore details that were dropped before this snapshot.
The same issue applies to the existing episodic memory implementation, whose write
path stores state tokens rather than a separate original observation-feature view.

Two viable designs are:

- **Belief-only history:** retain inferred state snapshots and use observation
  grounding, completion and delayed-query objectives to preserve relevant detail.
- **Observation and belief views:** allocate the same total memory budget between
  encoded observation evidence and interpreted recurrent state. Both views have
  learned contents; their distinction is functional and about source history.

Our proposed next question is: **Should memory preserve only the agent's interpreted
state, or also an observation-based view that can be reconsidered when the task
changes?**

Both reviewers provisionally favor the second interface. It provides a route
for revisiting evidence without requiring it all to pass through the current belief
bottleneck. It does not guarantee useful separation or complete information retention;
belief-only remains viable with adequate grounding. Sharing adapters, tying features
and allocating the token budget remain open.

A concrete proposal to discuss is that the observation view encodes the available
source without conditioning its write on current task, belief or generated working
content. Consumers can query it independently of the belief view, within the same
total memory budget. Its observation/completion retention objectives and gradient
routes must be explicit. A shared backbone can still acquire training-time biases;
inference-path separation does not guarantee preservation of every detail. This is
the qualification Claude retained, and the local synthesis accepts it. Whether this
requires separate parameters or only separate conditioning/read paths is still open.

A factory example is a brief unusual sensor pattern that contributes little to
predicting the next machine position but later matters for diagnosis. A document
example is a qualifier that contributes little to predicting the next edit but
later changes the answer to a question. These illustrate retention obligations;
they are not predefined token meanings. Exact raw values or wording require enough
retained information or an explicitly budgeted source payload.

### 2. How does a predicted state become a corrected current state?

Define the relationship between action-conditioned propagation and observation
update, including intervals with no incoming observation. The current prose says
perception corrects prediction, but its signature takes the previous state rather
than explicitly naming the executed-action prior it corrects.

Viable alternatives are a deterministic recurrent estimate with a conditional
transition distribution, or a probabilistic filtering state with an explicit
prior/posterior relation. Neither is inherently contradictory. The recommendation
for discussion is to start with the simpler estimate, make executed-action
propagation and observation correction explicit, and avoid calling transition spread
epistemic confidence. A filtering distribution is appropriate if calibrated
belief uncertainty is a required model function.

Hypothetical candidate rollouts and propagation after an actually executed action
must have distinct branch roles. An action's execution receipt does not prove its
predicted consequences; the next observation supplies evidence about those effects.
Memory versions and real-evidence cutoffs stay fixed within each hypothetical branch.

[DreamerV3](https://arxiv.org/html/2301.04104v2) provides a concrete prior/posterior
option with recurrent state, observation-conditioned latent distributions, predicted
latent distributions and consistency losses. It does not make that probabilistic
interpretation mandatory for every learned token architecture.

### 3. Which useful conclusions can persist, and with what status?

The proposal protects observed history from generated content, but does not fully
specify the lifetime of useful task-derived conclusions. A document agent may need
to retain that it found two inconsistent passages, while preserving which passages
were observed and which conclusion it inferred.

Discuss a bounded tagged derived-knowledge route versus task-lifetime-only working
memory. Persistent derived records must not become observational facts. Decide how
they refer to supporting evidence, become stale, are superseded, and can be corrected.
This is separate from the accepted within-session temporal memory; cross-session
persistence is not implied.

Keep source/derivation status distinct from execution status. A file actually
written by the agent is a real external artifact; the factual claims in that file
remain generated unless supported separately. Observing or executing an artifact
does not authenticate its claims. Conservative ancestry unions can preserve
record-level dependencies but do not provide per-claim attribution.

### 4. How do reads handle overlap, contradictions and delayed corrections?

Belief snapshots contain information inherited from earlier states and previous
memory reads. Evicting a snapshot does not necessarily remove its information from
the current belief. Multiple stored representations can therefore share evidence
even when their event IDs differ.

Specify whether gates inspect read vectors as well as query, time, memory type and
source-overlap metadata. A content-aware gate is a viable baseline; an extra
cross-attention reconciliation block is an alternative. A normalized gate is not
automatically a Bayesian evidence accumulator, and no architecture described so far
guarantees that duplicate information increases confidence.

Choose what must happen when older consolidated information conflicts with a new
observation. Recency alone is not universally sufficient: a recent hypothesis can
conflict with an older direct source. Conservative evidence-status rules can coexist
with learned content reconciliation. Consolidation's delayed update remains an
acknowledged tradeoff, not a current-fact guarantee.

### 5. What do grounding, compression and marking actually optimize?

The architecture needs a content-retention objective independent of simply making
future latents easy to predict. Fixed observable targets with a trainable decoder
are a valid grounding route; a frozen external encoder is an option, not a requirement.
Grounding, target alignment, input/target gradient routes and representation revision
must be stated. Loss weights do not by themselves guarantee desired retention.

Choose the distribution of delayed queries that compression and consolidation are
trained to serve. Read preservation for current queries alone may miss future task
needs. Detached source snapshots can train a recomputed compressor and downstream
reader without a complete original-write-to-read graph; the replay still needs to
cover the intended uses and retention stages.

[Compressive Transformers](https://arxiv.org/html/1911.05507v1), algorithm 2 and
section 3.2, explicitly stops auxiliary gradients through source states, queries
and attention projections while training compression. The motivation includes long
backpropagation delays; learned compression is not necessarily nondifferentiable.

Marking utility is conditional on occupied capacity, the displaced record and future
policy. A minimal proposal is lagged, fixed-policy paired replay, with the limited
interpretation of marginal retention benefit under that replay policy. Whether
marks should be credited for changing future actions is a further choice. The
availability deadline depends on the requested detail still existing, including in
staging or protected memory, rather than recent-ring exit alone.

### 6. Where do candidate actions and success criteria come from?

Declare the action schema, validity checks, source of candidates and objective for
each domain. Candidates may come from a caller, an explicit generator or a trained
proposal policy. A shared latent core does not supply these meanings automatically.

Programmatic or observable-grounded objectives are a useful starting option. Learned
latent objective/value heads are also possible, including for document work, when
their training and meaning are explicit. There is no domain-wide prohibition on
latent objectives. Exact operation controls remain separate from learned scores.

Specify time or event-step semantics for each adapter. Unknown duration must be
represented explicitly or rejected, never silently interpreted as a measured unit
step. Unknown action remains distinct from no-op.

## Corrections reconciled with Claude

Claude explicitly withdrew or narrowed these initial claims after checking the
supplied contracts and general reasoning:

| Initial assertion | Reconciled assessment |
| --- | --- |
| Deterministic state plus stochastic prediction is contradictory | A valid family; the intended state and scale semantics remain to be specified |
| Consolidation necessarily destroys provenance | Exact conservative ancestry and time-support propagation can preserve record-level provenance; fact-level attribution is stronger |
| Learned gated reads cannot resolve supersession | Content-aware gating can learn it; extra reconciliation layers are optional and usefulness remains unproven |
| A frozen external representation anchor is mandatory | Observable grounding with a learned decoder is viable; the information-retention contract matters |
| Compression training requires a full original write-to-final-read graph | Recomputation from detached snapshots and local auxiliary objectives are valid; coverage still matters |
| Marking must occur before a record leaves the recent ring | Requested-detail availability is the relevant deadline |
| Latent objectives are invalid in software/document domains | Unsupported prohibition; explicit objective/action semantics are required across domains |
| Every slot needs persistent object identity | Content-addressed tokens need no fixed object meaning; positional correspondence or set-invariant loss rules must be explicit where losses compare slots |

These withdrawals remove unsupported objections; they do not establish that the
proposed alternatives work. The final exchange also withdrew exact retention as a
latent-memory default and the implication that a staleness feature substitutes for
a posterior or a justified refusal rule. Fixed semantic field lists are not required
to design a domain-neutral representation. Later tasks can define measurable
retention requirements without hard-coding the latent contents; such task definitions
are themselves compatible with learned representations.

## Choices that can remain adjustable

Token width and counts, buffer capacities, compression ratio, exact gate network,
thinking-round limit, replay lengths, label refresh intervals and loss weights can
remain explicit recipe parameters. Their resource cost and achievable learning
coverage eventually constrain each other. They need not be settled before the
state/evidence and belief-update questions above.

The review recommends continuing architecture discussion with item 1, then item 2.
It does not authorize tests or settle the new two-view memory proposal on the user's
behalf. Exact review receipts remain local; the active specification retains its
proposed status.
