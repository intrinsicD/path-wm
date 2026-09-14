# PATH-WM · Architecture atlas

A map of the implemented components and their interfaces, from the agent loop to attention blocks. The general categorical agent, the Gaussian photo experiment, and the entity experiments are distinct configurations. A drawn module indicates implementation, not proven general capability.

Source review: 2026-09-14, repository snapshot `5ad801d`. [Open the rendered atlas](architecture-atlas.html).

Overview (1): Red: to discuss. Blue: discussed. Green: validated within the labelled scope. [Discussion and validation checklist](architecture-discussion.md).

Detail diagrams (2–13): blue = learned modules; gray = state/mechanics; green = external I/O; purple = optional; amber = training/control; dashed = proposed or labelled training-only connections.

- [1 · The agent loop](#01-overview)
- [2 · Modality encoders and feature hierarchy](#02-encoders)
- [3 · Inside a residual attention block](#03-attention)
- [4 · Predict, correct and expose world state](#04-belief)
- [5 · Short and long session memory](#05-memory)
- [6 · Tasks, focus, thinking and output control](#06-workspace)
- [7 · How state reaches each output modality](#07-decoders)
- [8 · The current real-photo experiment](#08-photo-path)
- [9 · Latent-guided image production](#09-image-generator)
- [10 · Entity identity, state and relations](#10-entities)
- [11 · Planning and the proposed action DAG](#11-planning)
- [12 · Training signals and gradient routes](#12-learning)
- [13 · The broader entity graph we discussed](#13-target-graph)

<a id="01-overview"></a>

## 1 · The agent loop

General categorical agent · runtime data and control flow

Red = a dedicated discussion remains. Blue = discussed, with validation still incomplete. Green = validated within the explicitly labelled test scope, not general capability. Discussion coverage is retained separately, including for green parts that still need a walkthrough. Evidence refers to the recorded configurations and must be revisited after relevant changes.

Red: to discuss. Blue: discussed. Green: validated within the labelled scope. [Coverage, validation evidence and remaining questions](architecture-discussion.md).

```mermaid
flowchart TB
    world["World / user / environment<br/>Images · video · audio · text"]
    class world discussion_discussed;
    input["Observation adapters<br/>Values + time + validity + source<br/>Validated: Event mechanics<br/>Discussion still pending"]
    class input discussion_validated;
    encode["Modality encoders → §2–3<br/>Processed features at several scales"]
    class encode discussion_discussed;
    belief["Predict and correct → §4<br/>Recurrent world state + categorical belief"]
    class belief discussion_needs_discussion;
    memory["Session memory → §5<br/>Recent · compressed · protected · consolidated<br/>Validated: Storage / causal reads"]
    class memory discussion_validated;
    workspace["Task workspace → §6<br/>Read state, memory and task; think"]
    class workspace discussion_discussed;
    request["Task request<br/>Instruction + output controls + actor metadata"]
    class request discussion_needs_discussion;
    plan["Optional bounded planner → §11<br/>Candidate actions → imagined states → costs<br/>Validated: Bounded search mechanics"]
    class plan discussion_validated;
    emit["Modality outputs → §7<br/>Image · audio · text · video"]
    class emit discussion_discussed;
    act["Action proposal<br/>Environment adapter executes it"]
    class act discussion_needs_discussion;
    reflect["Generated-content reflection<br/>Re-encode + provenance → workspace only<br/>Validated: Routing / provenance"]
    class reflect discussion_validated;
    world -->|"delivered observations"| input
    input -->|"source packets"| encode
    encode -->|"features + support times"| belief
    belief -->|"store event"| memory
    memory -->|"history reads"| belief
    belief -->|"world tokens"| workspace
    memory -->|"thinking read"| workspace
    request -->|"interpreted task tokens"| workspace
    workspace -->|"conditioned state"| emit
    workspace -->|"policy selects act"| act
    belief -->|"branch starting state"| plan
    request -->|"recipe-defined goal / cost"| plan
    plan -->|"selected first action"| act
    act -->|"external action"| world
    emit -->|"optional loopback"| reflect
    reflect -->|"generated context + provenance"| workspace
    classDef discussion_discussed fill:#e6eef8,stroke:#7696bc,color:#202a36;
    classDef discussion_needs_discussion fill:#fee2e2,stroke:#b91c1c,color:#7f1d1d;
    classDef discussion_validated fill:#dcfce7,stroke:#15803d,color:#14532d;
```

[Full-size SVG](diagrams/atlas/01-overview.svg)

The caller owns state, event transactions, action execution and task lifetime. The model returns proposals and tensors; it does not execute arbitrary software tools itself.

Imagination uses the dynamics without future observations. Reflection and thinking leave the live world clock unchanged. Generated content is kept out of source-evidence writes.

The categorical recipe is the CLI default. The recent photo model uses the separate Gaussian configuration in §8. Entity storage and the key-box connection in §10–11 are optional experiments, not an invisible extra store in every agent.

Source: [experiments/multimodal.py · build_model:103](../experiments/multimodal.py), [pathwm/models/belief.py · BeliefAgent:102](../pathwm/models/belief.py), [pathwm/models/agent.py · step_task:637](../pathwm/models/agent.py).

<a id="02-encoders"></a>

## 2 · Modality encoders and feature hierarchy

Same hierarchy pattern, separately instantiated parameters per modality

```mermaid
flowchart TB
    image["Image / video<br/>RGB frames"]
    class image external;
    audio["Audio<br/>Waveform chunks"]
    class audio external;
    text["Text<br/>Byte token IDs"]
    class text external;
    patch["Image stem<br/>Patch convolution + position + modality"]
    class patch learned;
    wave["Audio stem<br/>Waveform patches → linear projection"]
    class wave learned;
    embed["Text stem<br/>Byte embedding + position + modality"]
    class embed learned;
    fine["Scale 0<br/>Scale embedding → k residual transformer blocks"]
    class fine learned;
    merge1["Merge / pool local groups<br/>Optional coarse-query attention to fine footprint"]
    class merge1 learned;
    mid["Scale 1<br/>Scale embedding → k residual transformer blocks"]
    class mid learned;
    merge2["Repeat merge and processing<br/>Until the configured number of scales"]
    class merge2 learned;
    coarse["Final scale<br/>Fully processed coarse features"]
    class coarse learned;
    fusion["Optional all-scale fusion<br/>Concatenate → f transformer blocks → split"]
    class fusion learned;
    out["FeaturePyramid<br/>Per-scale tokens + grid + time + validity + support"]
    class out store;
    consumer["Consumers<br/>Belief correction / spatial heads / reflection"]
    class consumer store;
    image -->|"B × T × 3 × H × W"| patch
    audio -->|"B × T × samples"| wave
    text -->|"B × length"| embed
    patch -->|"frame / patch tokens"| fine
    wave -->|"waveform tokens"| fine
    embed -->|"byte tokens"| fine
    fine -->|"processed fine scale"| merge1
    merge1 -->|"coarser queries"| mid
    mid -->|"processed next scale"| merge2
    merge2 -->|"coarse features"| coarse
    fine -->|"retain fine tokens"| fusion
    mid -->|"retain middle tokens"| fusion
    coarse -->|"retain coarse tokens"| fusion
    fusion -->|"same output layouts"| out
    out -->|"all scales or selected named grids"| consumer
    classDef learned fill:#e6eef8,stroke:#7696bc,color:#202a36;
    classDef store fill:#f3f4f6,stroke:#9098a4,color:#202a36;
    classDef external fill:#e7f1eb,stroke:#789887,color:#202a36;
    classDef optional fill:#efeafa,stroke:#9c87b5,color:#202a36;
    classDef training fill:#fff0db,stroke:#bd934d,color:#202a36;
    classDef proposal fill:#fafafa,stroke:#9b9b9b,color:#202a36,stroke-dasharray:5 4;
```

[Full-size SVG](diagrams/atlas/02-encoders.svg)

Each later scale consumes the fully processed previous scale. The final fusion is optional: f=0 in the general defaults, f=2 in the current photo checkpoint. k and scale count are recipe choices.

Image merging pools 2×2 spatial patches. Video also pools adjacent frames; audio and text pool adjacent sequence positions. Video has causal temporal attention and pooling within the supplied window, not a separate persistent recurrent video cell.

Masks prevent reading later support. The categorical source-evidence path uses no task, belief or feature-controller conditioning. Optional controls apply on permitted task/reflection paths.

Example for the photo configuration: RGB64 → 16×16 → 8×8 → 4×4, all width32; k=2 and f=2. The hierarchy emits 336 tokens.

Source: [pathwm/models/multiscale.py · FeatureHierarchy:207](../pathwm/models/multiscale.py), [pathwm/models/multiscale.py · MultiScaleImageEncoder:330](../pathwm/models/multiscale.py), [pathwm/models/multiscale.py · MultiScaleAudioEncoder:375](../pathwm/models/multiscale.py), [pathwm/models/multiscale.py · MultiScaleTextEncoder:436](../pathwm/models/multiscale.py), [pathwm/models/belief.py · _features:253](../pathwm/models/belief.py).

<a id="03-attention"></a>

## 3 · Inside a residual attention block

Attention reads representations; a query head is not a persistent memory slot

```mermaid
flowchart TB
    query["Query tokens x<br/>Positions / state slots to update"]
    class query store;
    context["Context tokens c<br/>Features / state / retrieved memory"]
    class context store;
    qnorm["Normalize query<br/>Project to Q"]
    class qnorm learned;
    kvnorm["Normalize context<br/>Project to K and V"]
    class kvnorm learned;
    attention["Multi-head attention<br/>Masked softmax(QKᵀ / √d) · V<br/>Concatenate heads → output projection"]
    class attention learned;
    add1["First residual<br/>u = x + attention output"]
    class add1 store;
    mlp["Normalize → MLP<br/>D → 2D → GELU → D"]
    class mlp learned;
    add2["Second residual<br/>y = u + MLP output"]
    class add2 store;
    control["Optional feature code<br/>Bias-free map → bounded scale / shift"]
    class control optional;
    mask["Attention constraints<br/>Validity · time · local footprint"]
    class mask store;
    query -->|"x"| qnorm
    context -->|"c"| kvnorm
    qnorm -->|"queries"| attention
    kvnorm -->|"keys and values"| attention
    mask -->|"allowed reads"| attention
    attention -->|"read result"| add1
    query -->|"identity path"| add1
    add1 -->|"u"| mlp
    mlp -->|"nonlinear residual"| add2
    add1 -->|"identity path"| add2
    control -->|"ConditionedBlock only"| qnorm
    control -->|"ConditionedBlock only"| mlp
    classDef learned fill:#e6eef8,stroke:#7696bc,color:#202a36;
    classDef store fill:#f3f4f6,stroke:#9098a4,color:#202a36;
    classDef external fill:#e7f1eb,stroke:#789887,color:#202a36;
    classDef optional fill:#efeafa,stroke:#9c87b5,color:#202a36;
    classDef training fill:#fff0db,stroke:#bd934d,color:#202a36;
    classDef proposal fill:#fafafa,stroke:#9b9b9b,color:#202a36,stroke-dasharray:5 4;
```

[Full-size SVG](diagrams/atlas/03-attention.svg)

Self-attention uses the same token set as query and context; cross-attention uses distinct sets. Attend uses four heads by default. Learned query tokens acquire roles through their consumer and training targets.

Q/K/V projection weights are learned parameters. Their per-input activations perform a read; memory records are stored separately. Neither attention weights nor readable slot names guarantee a semantic interpretation.

ConditionedBlock adds bounded scale/shift on query and MLP normalization; code zero is neutral. Its time/support masks are richer than the simpler Attend block. OutputBlock (§9) adds separate self-attention and context-attention residuals.

Source: [pathwm/models/modalities.py · Attend:75](../pathwm/models/modalities.py), [pathwm/models/multiscale.py · ConditionedBlock:88](../pathwm/models/multiscale.py), [pathwm/models/conditional_image.py · OutputBlock:47](../pathwm/models/conditional_image.py).

<a id="04-belief"></a>

## 4 · Predict, correct and expose world state

Categorical BeliefAgent · nominal width32 configuration

```mermaid
flowchart TB
    previous["Previous belief<br/>h: 16 recurrent tokens + sampled categorical code"]
    class previous store;
    action["Recorded action + presence flag + Δt<br/>A missing action differs from a zero action"]
    class action external;
    memory["Memory snapshot<br/>Separate prediction / perception readers"]
    class memory store;
    dynamics["BeliefDynamics<br/>Projected code + action/time → attention update"]
    class dynamics learned;
    prior["Prior<br/>New h + predicted logits over 8 groups × 8 codes"]
    class prior store;
    features["Available source features<br/>All modalities / scales + valid masks"]
    class features store;
    correct["BeliefCorrection<br/>Attend to features → add perception-memory read<br/>→ attend again → categorical logits"]
    class correct learned;
    posterior["Posterior distribution<br/>Sample code z; straight-through training gradient"]
    class posterior store;
    workspace["Previous workspace<br/>4 working + 4 reasoning tokens"]
    class workspace store;
    readout["World readout<br/>h + projection(sampled code)<br/>Concatenate the 8 workspace tokens"]
    class readout learned;
    state["BeliefState<br/>24 tokens × 32 + h + logits/code + clocks/provenance"]
    class state store;
    evidence["Source-only evidence summary<br/>8 learned queries attend to source features"]
    class evidence learned;
    commit["Seal observed event<br/>Store evidence + belief envelope once"]
    class commit store;
    previous -->|"h and code"| dynamics
    action -->|"transition condition"| dynamics
    memory -->|"prediction read"| dynamics
    dynamics -->|"h and prior logits"| prior
    prior -->|"prior h as queries"| correct
    features -->|"observation context"| correct
    memory -->|"perception read"| correct
    correct -->|"corrected logits"| posterior
    prior -->|"h; unchanged by correction"| readout
    posterior -->|"sampled code"| readout
    workspace -->|"workspace retained"| readout
    readout -->|"world + workspace tokens"| state
    features -->|"no belief/task conditioning"| evidence
    evidence -->|"source view"| commit
    state -->|"inferred view + event metadata"| commit
    classDef learned fill:#e6eef8,stroke:#7696bc,color:#202a36;
    classDef store fill:#f3f4f6,stroke:#9098a4,color:#202a36;
    classDef external fill:#e7f1eb,stroke:#789887,color:#202a36;
    classDef optional fill:#efeafa,stroke:#9c87b5,color:#202a36;
    classDef training fill:#fff0db,stroke:#bd934d,color:#202a36;
    classDef proposal fill:#fafafa,stroke:#9b9b9b,color:#202a36,stroke-dasharray:5 4;
```

[Full-size SVG](diagrams/atlas/04-belief.svg)

The observation correction changes the categorical distribution, not the recurrent h directly. Its sampled code changes world readout immediately and influences h on the next transition.

With no usable observation, the prior is retained and no source-memory record is written. Imagination runs the prior transition without correction, marks the result hypothetical and shares live memory read-only.

Events are caller-owned transactions with ordered IDs. Partial packet arrivals are corrected against one prior and one memory snapshot. Source times and state time remain distinct.

Categorical entropy is a diagnostic; it is not established calibrated confidence or model-parameter uncertainty. The Gaussian log_scale inherited for compatibility is zero here, not the belief uncertainty.

Source: [pathwm/models/belief.py · BeliefDynamics:33](../pathwm/models/belief.py), [pathwm/models/belief.py · BeliefCorrection:74](../pathwm/models/belief.py), [pathwm/models/belief.py · correct_packets:352](../pathwm/models/belief.py), [pathwm/models/belief.py · _readout:190](../pathwm/models/belief.py), [pathwm/models/belief_state.py · BeliefState:91](../pathwm/models/belief_state.py).

<a id="05-memory"></a>

## 5 · Short and long session memory

HybridMemory · bounded storage mechanics plus learned compression and reading

```mermaid
flowchart TB
    event["Committed observed event<br/>Source features + inferred belief + provenance"]
    class event store;
    record["Detached event record<br/>Separate evidence and belief views"]
    class record store;
    recent["Recent store<br/>32 records; exact stored latent envelope"]
    class recent store;
    staging["Staging FIFO<br/>Up to 7 records waiting for a block of 8"]
    class staging store;
    compress["Two learned compressors<br/>Separate attention queries for evidence / belief"]
    class compress learned;
    blocks["Compressed store<br/>16 blocks · 8 tokens per view per block"]
    class blocks store;
    consolidate["Gated consolidation<br/>Old accumulator + departing block"]
    class consolidate learned;
    long["Consolidated store<br/>8 tokens per view; bounded recurrent summary"]
    class long store;
    protect["Protected store<br/>8 marked records; user priority / agent utility"]
    class protect store;
    query["Consumer query + current time<br/>Perception / prediction / thinking"]
    class query store;
    read["Three reader branches<br/>Recent + staging / compressed + consolidated / protected<br/>Add view identity and source/state ages → attention"]
    class read learned;
    gate["Token-wise softmax gate<br/>Mix 3 reads + no-memory option"]
    class gate learned;
    result["Memory context<br/>To correction, dynamics or workspace"]
    class result store;
    event -->|"validated live evidence"| record
    record -->|"append"| recent
    recent -->|"oldest eviction"| staging
    staging -->|"full chronological block"| compress
    compress -->|"two compressed views"| blocks
    blocks -->|"oldest block eviction"| consolidate
    consolidate -->|"new summary"| long
    long -->|"old summary"| consolidate
    recent -->|"mark latest observed event"| protect
    recent -->|"recent records"| read
    staging -->|"uncompressed overflow"| read
    blocks -->|"compressed blocks"| read
    long -->|"oldest summarized history"| read
    protect -->|"retained marked detail"| read
    query -->|"attention queries"| read
    query -->|"mixing weights"| gate
    read -->|"three read residuals"| gate
    gate -->|"weighted context"| result
    classDef learned fill:#e6eef8,stroke:#7696bc,color:#202a36;
    classDef store fill:#f3f4f6,stroke:#9098a4,color:#202a36;
    classDef external fill:#e7f1eb,stroke:#789887,color:#202a36;
    classDef optional fill:#efeafa,stroke:#9c87b5,color:#202a36;
    classDef training fill:#fff0db,stroke:#bd934d,color:#202a36;
    classDef proposal fill:#fafafa,stroke:#9b9b9b,color:#202a36,stroke-dasharray:5 4;
```

[Full-size SVG](diagrams/atlas/05-memory.svg)

These capacities are general defaults, not universal constants. Source and inferred views stay separate through compression; categorical probabilities, not only sampled codes, reach belief readers.

Storage is caller-owned bounded tensors and records. Reading attends over the stores directly; this is not yet a scalable indexed external database or an unbounded raw-media archive.

Exact recent storage means exact latent copies, not exact raw pixels. Compression loses detailed source alignment; support intervals and bounded source labels are retained. Learned read/compression quality needs its own evaluation.

The photo experiment uses simpler episodic top-k snapshots instead (§8). The explicit entity store (§10) is another separate mechanism.

Source: [pathwm/models/hybrid_memory.py · HybridMemory:98](../pathwm/models/hybrid_memory.py), [pathwm/models/hybrid_memory.py · MemoryRead:28](../pathwm/models/hybrid_memory.py), [pathwm/models/belief_state.py · MemoryRecord:26](../pathwm/models/belief_state.py), [docs/belief-model.md](../docs/belief-model.md).

<a id="06-workspace"></a>

## 6 · Tasks, focus, thinking and output control

Workspace updates are internal computation; external actions return to the caller

```mermaid
flowchart TB
    instruction["Instruction<br/>Text encoder → instruction features"]
    class instruction learned;
    metadata["Task / actor / output metadata<br/>Canonical bytes → embedding → GRU → pooled token"]
    class metadata learned;
    state["Current state tokens"]
    class state store;
    interpret["TaskInterpreter<br/>4 query tokens read instruction, then state + metadata"]
    class interpret learned;
    policy["TaskPolicy<br/>Pool → MLP → operation / modality / completion logits"]
    class policy learned;
    gate["TaskSession controls<br/>Required / automatic / disabled outputs<br/>Selection + fulfillment checks"]
    class gate store;
    think["Thinker<br/>Workspace queries attend state + recall + goal + feedback"]
    class think learned;
    memory["Memory reader<br/>Task-dependent access through query state"]
    class memory learned;
    feedback["Feedback<br/>Detached state → error MLP → softplus<br/>Add uncertainty diagnostic + think count"]
    class feedback learned;
    workspace["Updated workspace<br/>4 working + 4 reasoning tokens"]
    class workspace store;
    dispatch["Bounded operation dispatcher<br/>think · recall · imagine · act · emit · ask · finish"]
    class dispatch store;
    actionhead["ActionHead<br/>Mean-pool conditioned state → linear<br/>Continuous action mean + log-scale"]
    class actionhead learned;
    output["Emission / action / hypothetical branch<br/>Return result and provenance to caller"]
    class output store;
    reflect["Optional reflection<br/>Tagged generated output → same modality encoder<br/>Valid feature pool + metadata"]
    class reflect optional;
    instruction -->|"instruction tokens"| interpret
    metadata -->|"metadata token"| interpret
    state -->|"state context"| interpret
    state -->|"monitor / uncertainty / counter"| feedback
    state -->|"memory queries"| memory
    interpret -->|"task tokens"| policy
    policy -->|"raw proposal"| gate
    gate -->|"allowed operation"| dispatch
    interpret -->|"goal context"| think
    state -->|"world + previous workspace"| think
    memory -->|"retrieved context"| think
    feedback -->|"projected feedback token"| think
    think -->|"update 8 tokens only"| workspace
    workspace -->|"conditioned state"| dispatch
    workspace -->|"conditioned state"| actionhead
    dispatch -->|"when act is selected"| actionhead
    actionhead -->|"action proposal"| output
    dispatch -->|"selected operation result"| output
    output -->|"generated-media loopback"| reflect
    reflect -->|"attributed context"| think
    classDef learned fill:#e6eef8,stroke:#7696bc,color:#202a36;
    classDef store fill:#f3f4f6,stroke:#9098a4,color:#202a36;
    classDef external fill:#e7f1eb,stroke:#789887,color:#202a36;
    classDef optional fill:#efeafa,stroke:#9c87b5,color:#202a36;
    classDef training fill:#fff0db,stroke:#bd934d,color:#202a36;
    classDef proposal fill:#fafafa,stroke:#9b9b9b,color:#202a36,stroke-dasharray:5 4;
```

[Full-size SVG](diagrams/atlas/06-workspace.svg)

Thinker uses the same residual attention primitive as §3. More thinking repeats this read/update loop; it does not advance the live world clock or by itself acquire evidence.

In emit, task tokens first condition a thinking step, then modality decoders consume the conditioned state. General decoders see all state tokens; the photo wrapper intentionally exposes only its working/reasoning tokens.

The feature controller can propose a bounded feature code for supported reads. It is excluded from categorical source encoding. Reflection uses shared encoder weights but preserves generated ancestry; it is not a new camera event.

finish is a policy/session operation, not independent proof of task success. The historical-recall experiment has an external verifier based on the delivered event log. General tool execution, open-ended decomposition and calibrated stopping remain incomplete.

Source: [pathwm/models/tasks.py · TaskInterpreter:419](../pathwm/models/tasks.py), [pathwm/models/tasks.py · TaskPolicy:442](../pathwm/models/tasks.py), [pathwm/models/tasks.py · MetadataEncoder:397](../pathwm/models/tasks.py), [pathwm/models/agent.py · think:709](../pathwm/models/agent.py), [pathwm/models/agent.py · emit:508](../pathwm/models/agent.py), [pathwm/models/agent.py · reflect:404](../pathwm/models/agent.py), [pathwm/models/recall.py · verify_recall:144](../pathwm/models/recall.py).

<a id="07-decoders"></a>

## 7 · How state reaches each output modality

General multimodal recipe · reference decoders are small learned readouts

```mermaid
flowchart TB
    state["Task-conditioned state<br/>World tokens + working / reasoning tokens"]
    class state store;
    image["Image decoder<br/>Learned spatial queries attend state"]
    class image learned;
    rgb["Patch projection → sigmoid<br/>Rearrange RGB patches into image"]
    class rgb learned;
    audio["Audio decoder<br/>4 learned queries attend state"]
    class audio learned;
    wave["Flatten → linear → tanh<br/>Fixed-length waveform chunk"]
    class wave learned;
    prefix["Text prefix<br/>BOS + previously generated bytes"]
    class prefix store;
    text["Text decoder<br/>Embedding + position → causal attention<br/>→ cross-attention to state"]
    class text learned;
    bytes["Vocabulary projection<br/>Next-byte logits; repeat until EOS / budget"]
    class bytes learned;
    trajectory["Video trajectory<br/>Explicit states at increasing times"]
    class trajectory store;
    video["Reuse image decoder per state<br/>Stack decoded frames"]
    class video learned;
    emit["Emission envelope<br/>Values + modality + requester + producer + ancestry"]
    class emit store;
    replace["Optional image replacement → §9<br/>State → generated multiscale features → RGB head"]
    class replace optional;
    state -->|"context"| image
    image -->|"spatial tokens"| rgb
    state -->|"context"| audio
    audio -->|"four tokens"| wave
    prefix -->|"causal prefix"| text
    state -->|"context"| text
    text -->|"prefix-conditioned features"| bytes
    bytes -->|"append chosen byte"| prefix
    trajectory -->|"one state per frame"| video
    rgb -->|"RGB tensor"| emit
    wave -->|"waveform tensor"| emit
    bytes -->|"generated byte sequence"| emit
    video -->|"frame sequence"| emit
    state -->|"alternative producer context"| replace
    replace -->|"optional recipe output"| emit
    classDef learned fill:#e6eef8,stroke:#7696bc,color:#202a36;
    classDef store fill:#f3f4f6,stroke:#9098a4,color:#202a36;
    classDef external fill:#e7f1eb,stroke:#789887,color:#202a36;
    classDef optional fill:#efeafa,stroke:#9c87b5,color:#202a36;
    classDef training fill:#fff0db,stroke:#bd934d,color:#202a36;
    classDef proposal fill:#fafafa,stroke:#9b9b9b,color:#202a36,stroke-dasharray:5 4;
```

[Full-size SVG](diagrams/atlas/07-decoders.svg)

An output does not require an input of the same modality: decoder inputs are internal state. Whether useful content is present depends on learned representations, memory, objectives and training data.

The text path is a byte autoregressor, not a pretrained language model. The waveform path is not established general speech synthesis. Video currently decodes a supplied state trajectory; there is no general video diffusion generator in this recipe.

Image resolution and audio length are constructor settings. Output size alone does not establish reconstructed detail. The stronger conditional image producer is an optional controlled experiment, not a common generator already implemented for every modality.

Source: [pathwm/models/modalities.py · ImageDecoder:224](../pathwm/models/modalities.py), [pathwm/models/modalities.py · AudioDecoder:249](../pathwm/models/modalities.py), [pathwm/models/modalities.py · TextDecoder:267](../pathwm/models/modalities.py), [pathwm/models/agent.py · decode_video:795](../pathwm/models/agent.py), [pathwm/models/tasks.py · GeneratedOutput:155](../pathwm/models/tasks.py).

<a id="08-photo-path"></a>

## 8 · The current real-photo experiment

MemoryOutput · Gaussian reference configuration, not BeliefAgent

```mermaid
flowchart TB
    photos["Real photo history<br/>RGB64 observations then a blank view"]
    class photos external;
    encoder["Frozen image hierarchy<br/>4×4 RGB patch: 48 values → 32<br/>16² + 8² + 4² = 336 tokens × 32"]
    class encoder learned;
    update["ObservationUpdate<br/>State queries attend features + action/time token<br/>Residual MLP + Gaussian scale head"]
    class update learned;
    state["Gaussian LatentState<br/>30 tokens × 32 + log_scale<br/>16 sensory / 4 entities / 2 context / 4 working / 4 reasoning"]
    class state store;
    bank["Episodic snapshot bank<br/>Detached copy of all 30 tokens<br/>Capacity4; read top2 by pooled-key cosine"]
    class bank store;
    start["Query starting state<br/>Final state, or fresh blank state with bank restored"]
    class start store;
    think["Native recall / Thinker ×2<br/>Read bank + current state; update 8 workspace tokens"]
    class think learned;
    working["Output context<br/>4 working + 4 reasoning tokens × 32<br/>Fixed trained-channel normalization"]
    class working store;
    generator["Conditional feature generator → §9<br/>Noise + workspace → all codec feature maps"]
    class generator learned;
    head["Frozen PatchDetailHead<br/>Predicted patch means + predicted within-patch detail"]
    class head learned;
    image["Generated 64×64 RGB image"]
    class image external;
    facts["Parallel factual head<br/>Controlled entity / attribute outputs"]
    class facts optional;
    teacher["Direct codec control / training teacher<br/>Original RGB → hierarchy + raw patch residual"]
    class teacher training;
    photos -->|"real observation input"| encoder
    encoder -->|"336 spatial tokens"| update
    update -->|"first major measured access drop"| state
    state -->|"previous state; next observation"| update
    state -->|"explicit remember; exact copies"| bank
    state -->|"ordinary query mode"| start
    start -->|"live or reset state"| think
    bank -->|"retrieved snapshots + relative time"| think
    think -->|"native working/reasoning readout"| working
    working -->|"only image-dependent generation context"| generator
    generator -->|"three scales + detail map"| head
    head -->|"pixels"| image
    working -->|"separate readout"| facts
    photos -.->|"original image, training/control only"| teacher
    teacher -.->|"detached feature targets for losses"| generator
    classDef learned fill:#e6eef8,stroke:#7696bc,color:#202a36;
    classDef store fill:#f3f4f6,stroke:#9098a4,color:#202a36;
    classDef external fill:#e7f1eb,stroke:#789887,color:#202a36;
    classDef optional fill:#efeafa,stroke:#9c87b5,color:#202a36;
    classDef training fill:#fff0db,stroke:#bd934d,color:#202a36;
    classDef proposal fill:#fafafa,stroke:#9b9b9b,color:#202a36,stroke-dasharray:5 4;
```

[Full-size SVG](diagrams/atlas/08-photo-path.svg)

Checkpoint: runs/real_photo_v1/training/weights.pt. This model has image input/output only. A new session starts from learned initial tokens; each observation updates the preceding state. Group names such as entities are token-slice labels, not an explicit knowledge graph or verified object slots.

The latest continuation trained only the 314,576-parameter generator. Encoder, state updates, recall and codec stayed frozen. The photo route bypasses the teacher's raw-detail branch; no source pixels are silently fed to the generator during recall.

The frozen diagnostic found the largest spatial-accessibility drop at encoder → first state, with another drop on recall. Memory writes themselves are exact. A failed readout does not prove all information is absent.

The direct codec sees extra within-patch residual detail. Its good reconstruction is not evidence that this smaller state/memory path retains those pixels. General photographic generation remains open.

Source: [pathwm/models/memory_output.py · build_model:496](../pathwm/models/memory_output.py), [pathwm/models/memory_output.py · MemoryOutput:340](../pathwm/models/memory_output.py), [pathwm/models/agent.py · ObservationUpdate:33](../pathwm/models/agent.py), [pathwm/models/agent_state.py · EpisodicMemory:88](../pathwm/models/agent_state.py), [docs/real-photo-plan.md](../docs/real-photo-plan.md), [docs/photo-detail-plan.md](../docs/photo-detail-plan.md).

<a id="09-image-generator"></a>

## 9 · Latent-guided image production

ConditionalFeatureGenerator · current flow configuration uses 8 Euler steps

```mermaid
flowchart TB
    context["Workspace context<br/>8 tokens × 32"]
    class context store;
    project["Normalize + project context<br/>32 → hidden width64"]
    class project learned;
    noise["Seeded noise feature maps<br/>Same layouts as every required codec feature"]
    class noise external;
    progress["Generation progress τ<br/>0 → 1; separate from world time"]
    class progress store;
    input["Per-scale input projection<br/>Feature channels → 64 + position + progress embedding"]
    class input learned;
    block["Per-scale OutputBlock<br/>Residual self-attention<br/>Residual workspace cross-attention<br/>Residual MLP"]
    class block learned;
    fusion["Joint-scale OutputBlock<br/>Concatenate scales → shared processing → split"]
    class fusion learned;
    velocity["Per-scale output projection<br/>Predict velocity in standardized feature space"]
    class velocity learned;
    integrate["Euler sampler<br/>x ← x + velocity / 8<br/>Repeat at the next τ"]
    class integrate store;
    features["Final feature maps; unstandardize<br/>32×16×16 / 32×8×8 / 32×4×4<br/>Detail: 48×16×16"]
    class features store;
    base["DenseHead<br/>Project / align / concatenate scales<br/>Residual CNN + upsampling → patch means"]
    class base learned;
    detail["Detail head<br/>1×1 projection → pixel shuffle<br/>Remove within-patch mean"]
    class detail learned;
    rgb["Combine patch means + residual detail<br/>Clamp to [0,1] → RGB64"]
    class rgb store;
    context -->|"conditioning"| project
    noise -->|"initial x"| input
    progress -->|"progress embedding"| input
    input -->|"spatial feature tokens"| block
    project -->|"cross-attention context"| block
    block -->|"all scale tokens"| fusion
    project -->|"cross-attention context"| fusion
    fusion -->|"split hidden features"| velocity
    velocity -->|"field value"| integrate
    integrate -->|"next x; repeat"| input
    integrate -->|"after final step"| features
    features -->|"three spatial scales"| base
    features -->|"predicted detail map"| detail
    base -->|"base patch means"| rgb
    detail -->|"within-patch residual"| rgb
    classDef learned fill:#e6eef8,stroke:#7696bc,color:#202a36;
    classDef store fill:#f3f4f6,stroke:#9098a4,color:#202a36;
    classDef external fill:#e7f1eb,stroke:#789887,color:#202a36;
    classDef optional fill:#efeafa,stroke:#9c87b5,color:#202a36;
    classDef training fill:#fff0db,stroke:#bd934d,color:#202a36;
    classDef proposal fill:#fafafa,stroke:#9b9b9b,color:#202a36,stroke-dasharray:5 4;
```

[Full-size SVG](diagrams/atlas/09-image-generator.svg)

Every decoder feature, including fine detail, is produced from workspace and noise at inference. Noise allows sampling; it does not supply missing facts about a particular remembered image.

The head owns no encoder or raw-image input. A direct regression variant also exists; older exports instead use StateFeatureDecoder: spatial queries attend state → project each scale → the same head.

The current image experiment uses a fixed selected-object/recall request. The architecture is conditional, but arbitrary language-to-image control has not been established. Replacing a producer later is possible if the feature interface and conditioning contract are kept compatible.

Source: [pathwm/models/conditional_image.py · ConditionalFeatureGenerator:73](../pathwm/models/conditional_image.py), [pathwm/models/conditional_image.py · OutputBlock:47](../pathwm/models/conditional_image.py), [pathwm/models/conditional_image.py · integrate:30](../pathwm/models/conditional_image.py), [pathwm/models/decoders.py · PatchDetailHead:11](../pathwm/models/decoders.py), [pathwm/models/decoders.py · DenseHead:217](../pathwm/models/decoders.py), [pathwm/models/decoders.py · StateFeatureDecoder:49](../pathwm/models/decoders.py).

<a id="10-entities"></a>

## 10 · Entity identity, state and relations

Implemented controlled experiments · supplied descriptors and relation semantics

```mermaid
flowchart TB
    observation["Entity observation<br/>Supplied unit descriptor: 8 values<br/>State features: 4 values + event/time"]
    class observation external;
    matcher["Learned entity matcher<br/>Squared descriptor differences → MLP scores<br/>Compete with learned new-entity score"]
    class matcher learned;
    records["Bounded EntityMemory<br/>Stable local integer ID + descriptor<br/>First/last seen + count + retry receipts"]
    class records store;
    decision["Confidence / capacity rule<br/>Match · create · uncertain · full"]
    class decision store;
    latent["Per-entity latent state<br/>Read the matched record's previous value"]
    class latent store;
    cell["EntityStateCell<br/>GRUCell(4 → 16) updates state<br/>Optional exact no-information preservation"]
    class cell learned;
    commit["Atomic state update<br/>New latent + identity/event receipt"]
    class commit store;
    query["Relation source cue + destination cue<br/>Optional active/cue contexts"]
    class query external;
    relation["Relation key + optional write gate<br/>Encode cue to latent key; context comparison"]
    class relation learned;
    edge["Per-destination relation slot<br/>One supplied directed relation type<br/>Stored latent key resolves a source by matching"]
    class edge store;
    interact["EntityInteractionCell<br/>Concatenate destination/source latents → MLP<br/>Update destination state"]
    class interact learned;
    read["Readout / task adapter<br/>State logits, or project entity latent into workspace"]
    class read learned;
    observation -->|"recognition descriptor"| matcher
    records -->|"stored candidate descriptors"| matcher
    matcher -->|"candidate / new probabilities"| decision
    decision -->|"admit known or new entity"| records
    decision -->|"resolved storage ID"| latent
    latent -->|"previous latent"| cell
    observation -->|"state features"| cell
    cell -->|"updated latent"| commit
    commit -->|"persist value"| latent
    query -->|"relation cues"| relation
    relation -->|"accepted key update"| edge
    records -->|"endpoint resolution"| edge
    edge -->|"resolved directed binding"| interact
    latent -->|"source + destination states"| interact
    interact -->|"destination update"| commit
    latent -->|"stored entity belief"| read
    classDef learned fill:#e6eef8,stroke:#7696bc,color:#202a36;
    classDef store fill:#f3f4f6,stroke:#9098a4,color:#202a36;
    classDef external fill:#e7f1eb,stroke:#789887,color:#202a36;
    classDef optional fill:#efeafa,stroke:#9c87b5,color:#202a36;
    classDef training fill:#fff0db,stroke:#bd934d,color:#202a36;
    classDef proposal fill:#fafafa,stroke:#9b9b9b,color:#202a36,stroke-dasharray:5 4;
```

[Full-size SVG](diagrams/atlas/10-entities.svg)

The model does not predict an arbitrary universal integer. Learned matching selects a record; the integer is the store's address. Uncertain matches can remain unresolved. Unique record IDs do not prevent mistaken identity merges.

These experiments start with supplied descriptors and state cues. They do not yet implement discovery of arbitrary people/objects from webcam pixels. Matching, state updating and relation keys are learned; storage bookkeeping is explicit.

EntityRelationMemory has one supplied relation type and one slot per destination. This is not yet learned open-ended graph structure or a general concept/instance taxonomy. The broader proposed graph appears in §13.

The key-box experiment actually connects an entity latent through projection and two Thinker steps to the general belief workspace, then a readout feeds a supplied planner. This remains a distinct experimental configuration.

Source: [pathwm/models/entities.py · EntityMatchReader:147](../pathwm/models/entities.py), [pathwm/models/entity_memory.py · EntityMemory:12](../pathwm/models/entity_memory.py), [pathwm/models/entity_state.py · EntityStateCell:10](../pathwm/models/entity_state.py), [pathwm/models/entity_state.py · EntityInteractionCell:48](../pathwm/models/entity_state.py), [pathwm/models/entity_relations.py · EntityRelationMemory:38](../pathwm/models/entity_relations.py), [pathwm/models/key_box.py · KeyBoxReader:27](../pathwm/models/key_box.py).

<a id="11-planning"></a>

## 11 · Planning and the proposed action DAG

Existing finite candidate rollouts; broader DAG search remains a target

```mermaid
flowchart TB
    live["Current live belief"]
    class live store;
    candidates["Recipe / caller<br/>Candidate action sequences + bounds<br/>Goal cost + horizon + duration"]
    class candidates external;
    branches["Temporary starting branches<br/>Categorical draws, default4; memory shared"]
    class branches store;
    rollout["Learned dynamics rollout<br/>State + action → imagined next state<br/>Repeat for the candidate sequence"]
    class rollout learned;
    score["Evaluate predicted states<br/>Discounted cost; average over samples"]
    class score store;
    choose["Choose lowest-scoring sequence<br/>Return full plan and first action"]
    class choose store;
    execute["Caller executes one action<br/>Collect actual new observations"]
    class execute external;
    correct["Correct live belief and replan<br/>Predicted success needs external verification"]
    class correct store;
    keybox["Separate key-box planner<br/>Inspect / open / retrieve / stop<br/>Expectimax, supplied dynamics, horizon ≤4"]
    class keybox optional;
    dag["PROPOSED · Reusable action DAG<br/>Nodes: predicted belief states<br/>Edges: external / internal actions and outcomes"]
    class dag proposal;
    search["PROPOSED · General graph search<br/>Expansion / merging / subgoals / contingent branches<br/>Unreachable or uncertain goal handling"]
    class search proposal;
    live -->|"starting belief"| branches
    branches -->|"scratch state"| rollout
    candidates -->|"candidate sequence"| rollout
    rollout -->|"imagined trajectory"| score
    candidates -->|"declared cost"| score
    score -->|"candidate scores"| choose
    choose -->|"first action"| execute
    execute -->|"real event"| correct
    correct -->|"next live state"| live
    keybox -->|"separate controlled harness"| execute
    live -.->|"proposed starting node"| dag
    dag -.->|"proposed graph frontier"| search
    search -.->|"expand"| dag
    classDef learned fill:#e6eef8,stroke:#7696bc,color:#202a36;
    classDef store fill:#f3f4f6,stroke:#9098a4,color:#202a36;
    classDef external fill:#e7f1eb,stroke:#789887,color:#202a36;
    classDef optional fill:#efeafa,stroke:#9c87b5,color:#202a36;
    classDef training fill:#fff0db,stroke:#bd934d,color:#202a36;
    classDef proposal fill:#fafafa,stroke:#9b9b9b,color:#202a36,stroke-dasharray:5 4;
```

[Full-size SVG](diagrams/atlas/11-planning.svg)

The general planner scores caller-provided sequences; it does not currently construct a persistent, general action DAG. Candidate generation, action units and goal cost are recipe-owned. There is no guarantee that the supplied candidates can reach a goal.

Hypothetical branches do not consume real event ordinals or write live evidence. General prior rollouts do not yet support observation-conditioned sensing branches; key-box expectimax handles sensing in a separate finite supplied model.

The task dispatcher already distinguishes internal think/recall/imagine from external act/ask/emit. General learned task decomposition, automatic state equivalence/graph merging, a universal success verifier and broad software-tool skills remain open.

Prediction quality, goal readout quality and search coverage all constrain planning. A low imagined cost is not proof of realized success.

Source: [pathwm/evaluation/agent.py · plan:23](../pathwm/evaluation/agent.py), [pathwm/models/belief.py · imagine:495](../pathwm/models/belief.py), [pathwm/models/agent.py · step_task:637](../pathwm/models/agent.py), [pathwm/models/key_box.py · plan_key:59](../pathwm/models/key_box.py), [docs/decision-design.md](../docs/decision-design.md).

<a id="12-learning"></a>

## 12 · Training signals and gradient routes

Recipe-selected objectives; no single checkpoint currently proves all capabilities

```mermaid
flowchart TB
    data["Training data<br/>Causal observations + recorded actions<br/>Targets / task labels from the training harness"]
    class data external;
    encode["Encoders + observation correction<br/>Posterior state"]
    class encode learned;
    prior["Action-conditioned dynamics<br/>Prior / future states"]
    class prior learned;
    decode["Modality decoders<br/>Observed reconstruction + prior prediction"]
    class decode learned;
    worldloss["World objectives<br/>Image/audio likelihood + byte likelihood<br/>Split prior/posterior KL"]
    class worldloss training;
    teacher["Optional partial-view teacher<br/>EMA full current view + causal prefix<br/>Detached distribution targets"]
    class teacher training;
    memory["Memory replay / compression / marking<br/>Detached stored envelopes; bounded recomputation"]
    class memory learned;
    memoryloss["Memory / task objectives<br/>Recall outcomes · compression auxiliary<br/>Mark utility · task policy supervision"]
    class memoryloss training;
    optimizer["Optimizer<br/>Update only recipe-declared trainable parameters"]
    class optimizer store;
    evaluation["Held-out evaluation and controls<br/>Reload / causality / wrong-memory / ablations<br/>Quality, identity, calibration and planning metrics"]
    class evaluation store;
    data -->|"available input only"| encode
    data -.->|"optional full-view teacher input"| teacher
    encode -->|"previous belief + actions"| prior
    encode -->|"posterior context"| decode
    prior -->|"imagined context"| decode
    decode -.->|"predictions"| worldloss
    data -.->|"supervision targets"| worldloss
    prior -.->|"prior distribution"| worldloss
    encode -.->|"posterior distribution"| worldloss
    teacher -.->|"optional partial-view objective"| worldloss
    encode -->|"observed event envelopes"| memory
    memory -.->|"readouts / compression / utility"| memoryloss
    worldloss -.->|"world gradients"| optimizer
    memoryloss -.->|"memory/task gradients"| optimizer
    optimizer -->|"checkpoint under test"| evaluation
    classDef learned fill:#e6eef8,stroke:#7696bc,color:#202a36;
    classDef store fill:#f3f4f6,stroke:#9098a4,color:#202a36;
    classDef external fill:#e7f1eb,stroke:#789887,color:#202a36;
    classDef optional fill:#efeafa,stroke:#9c87b5,color:#202a36;
    classDef training fill:#fff0db,stroke:#bd934d,color:#202a36;
    classDef proposal fill:#fafafa,stroke:#9b9b9b,color:#202a36,stroke-dasharray:5 4;
```

[Full-size SVG](diagrams/atlas/12-learning.svg)

This is a map of available learning paths, not a claim that every loss runs in every recipe. Runtime memory writes detach; training deliberately recomputes bounded windows where needed.

The separate photo training path (§8–9) uses detached codec features and real pixels as feature/flow/decoded-image targets. Only the generator learned in the latest run. Frozen encoder/state/memory parameters blocked upstream repairs; the frozen RGB head still transmitted gradients to generator inputs.

Offline probes measure what a different reader can recover at each stage. They do not become inference shortcuts. Longer training, larger width, data coverage and architecture changes need separate controlled comparisons.

The next proposed photo repair is intermediate spatial supervision of observation/state updates and recall. It is not implemented by this diagram work.

Source: [pathwm/training/belief.py](../pathwm/training/belief.py), [experiments/multimodal.py](../experiments/multimodal.py), [experiments/real_photo.py](../experiments/real_photo.py), [pathwm/models/conditional_image.py](../pathwm/models/conditional_image.py), [pathwm/evaluation/photo_detail.py](../pathwm/evaluation/photo_detail.py), [docs/photo-detail-plan.md](../docs/photo-detail-plan.md).

<a id="13-target-graph"></a>

## 13 · The broader entity graph we discussed

Proposed organization · beyond the implemented controlled entity store

```mermaid
flowchart TB
    concept["General concept<br/>Learned subgraph for shared structure<br/>Example interpretation: bicycle"]
    class concept proposal;
    instance["Concrete entity instance<br/>Stable storage ID + learned matching keys<br/>Example interpretation: this bicycle"]
    class instance proposal;
    recognition["Recognition representations<br/>Appearance / identity across observations"]
    class recognition proposal;
    evidence["History and evidence<br/>Observed events + source/time attribution<br/>Selective raw media or versioned encodings"]
    class evidence proposal;
    belief["Current latent belief<br/>Learned properties / state / uncertainty"]
    class belief proposal;
    relations["Context and relationships<br/>Other entities · places · people · events"]
    class relations proposal;
    skills["Interactions and skills<br/>Action models / preconditions / outcomes"]
    class skills proposal;
    focus["Focus / task workspace<br/>Retrieve selected nodes, values and relations"]
    class focus proposal;
    operations["Graph operation interface<br/>Propose · match · create · link · update · retrieve<br/>Bounded storage, versions and uncertainty rules"]
    class operations proposal;
    inspect["Inspection layer<br/>Aliases + supporting examples + probes<br/>Trained reconstructions where supported"]
    class inspect proposal;
    instance -.->|"learned concept membership"| concept
    instance -.->|"identity support"| recognition
    instance -.->|"observed history"| evidence
    evidence -.->|"evidence for updates"| belief
    instance -.->|"current belief belongs to entity"| belief
    instance -.->|"learned links"| relations
    concept -.->|"shared interactions"| skills
    belief -.->|"retrieved state"| focus
    relations -.->|"relevant context"| focus
    skills -.->|"possible actions"| focus
    focus -.->|"learned proposals"| operations
    operations -.->|"accepted updates"| instance
    instance -.->|"structure and values"| inspect
    evidence -.->|"ground interpretations"| inspect
    classDef learned fill:#e6eef8,stroke:#7696bc,color:#202a36;
    classDef store fill:#f3f4f6,stroke:#9098a4,color:#202a36;
    classDef external fill:#e7f1eb,stroke:#789887,color:#202a36;
    classDef optional fill:#efeafa,stroke:#9c87b5,color:#202a36;
    classDef training fill:#fff0db,stroke:#bd934d,color:#202a36;
    classDef proposal fill:#fafafa,stroke:#9b9b9b,color:#202a36,stroke-dasharray:5 4;
```

[Full-size SVG](diagrams/atlas/13-target-graph.svg)

The intended agent learns representations and which connections are useful. Stable IDs are bookkeeping; readable labels are revisable interpretations. These boxes are conceptual distinctions, not mandatory database fields or one vector each.

Only bounded matching, per-entity state, a supplied relation slot and a controlled workspace connection exist today (§10). Open-ended concept discovery, learned graph topology, scalable indexing and general skill links remain to be specified and trained.

A recognition latent cannot automatically be decoded into a faithful face or image. That requires a compatible trained decoder and retained information; a reconstruction is evidence about a readout, not a literal picture of all the agent's beliefs.

Source: [docs/entity-memory-design.md](../docs/entity-memory-design.md), [docs/entity-learning-task.md](../docs/entity-learning-task.md), [pathwm/models/entity_relations.py · EntityRelationMemory:38](../pathwm/models/entity_relations.py).
