# Tasks, output controls and generated feedback

`finished` in this implemented API means required output artifacts were delivered;
it does not verify their correctness or an external world condition. The proposed
[decision and verification extension](decision-design.md) makes those statuses
separate. Categorical state/memory details are in [the belief guide](belief-model.md);
the Gaussian-specific sizes and provenance restrictions below describe the original
task implementation.

The model now has an explicit path from instructions to operation and output
proposals. Construction and supervision remain in
[`experiments/multimodal.py`](../experiments/multimodal.py). Exact records and neural
task consumers are in [`tasks.py`](../pathwm/models/tasks.py); execution methods are
in [`agent.py`](../pathwm/models/agent.py). There is no external language model or
keyword parser in inference.

The default model has 359,188 parameters. Its 30 world-state tokens are unchanged.
Four additional task tokens are computed for each decision; they are not a hidden
second world state. `TaskSession` belongs to the caller and persists the instruction,
controls, output ledger, pending requirements and completion/failure status.

## What an instruction becomes

1. `task_tokens(state, sessions)` converts each instruction to UTF-8 byte IDs and
   runs the **same conditioned multiscale text encoder** used for observations.
   Instruction encoding does not call `observe`, advance time or write memory.
2. Four learned queries attend the finished text features, then the world state
   and an encoded record of the task, controls and prior output provenance.
   Both attention stages have residual MLP processing.
3. `TaskPolicy` reads the mean task token through LayerNorm/linear/GELU and produces
   seven operation logits, one independent output logit per configured modality,
   and a completion logit. The ask softmax probability is the clarification score.
4. `TaskPrediction.select` applies discrete controls outside the neural network.
   It returns both the original proposal and the enforced operation/output requests.
   Required pending outputs are selected on emission; disabled ones are excluded.
   Finish with pending requirements becomes an emission proposal. An empty emission
   becomes ask. These gates do not establish learned understanding.

`task_predictions` accepts one session per batch member. `decide`, `emit` and
`step_task` operate on one interactive stream so each output has unambiguous actors.

| Operation | One bounded step does |
| --- | --- |
| think | Update working/reasoning using task tokens |
| recall | Retrieve observed memory and run an additional task-conditioned thought |
| imagine | Return a separate future branch; retain the live state's clock |
| act | Return a normalized action proposal; no environment actuation |
| emit | Validate requests, condition working/reasoning on task tokens, run selected decoders |
| ask | Return a clarification operation to the caller; no implicit text artifact |
| finish | Close the task once required output artifacts are delivered; no correctness check |

The existing candidate-action planner remains explicit: callers provide candidates,
bounds and a cost. Operation selection does not invent that cost. There is no
unbounded retry/execution loop. Call `session.abort(reason)` to end an unsuccessful
task while retaining its pending requirements. Subsequent steps return `failed`.

## Who asked for each output

`Actor(role, identity)` separates a role from an exact caller-provided identity.
One `OutputControl(modality, mode, specified_by)` resolves each modality's control:

| Mode | Meaning |
| --- | --- |
| required | A successful answer in this modality is required before finishing |
| disabled | Suppress output artifacts of this modality |
| automatic | The agent may select this modality |

Unspecified modalities are automatic. Two controls for the same modality are
rejected, including contradictory modes. Unsupported controls fail before selection
or emission. Controls are caller declarations; prose-inferred preferences never
become user-authored controls silently.

Every `OutputRequest` stores `output_id`, `modality`, `requested_by`,
`in_response_to` (the task ID), `purpose` and parent IDs. `GeneratedOutput` adds
the tensor plus `Provenance`, which records an independently specified `produced_by`,
generated origin and ancestry. A user-required answer credits the control author.
An automatic agent choice credits the agent even when the user authored the
automatic setting. Local requester identity governs fulfillment; parenting an
agent request under a user task never grants it user authority.

Purposes are answer/candidate/review/clarification. Only successful answers fulfill
requirements; agent candidates do not. Distinct output IDs permit multiple images
or revisions. Reusing an output ID in a task is rejected.

`emit` validates the **entire request tuple before any decoder runs**. It then runs
decoders in tuple order. If a decoder fails, the returned `Emission` contains prefix
successes, an error keyed by the failed output ID, and the updated session with an
exact `remaining` tuple. The original session is unchanged. Caller replacement
decoders must be pure tensor modules; external side effects cannot be rolled back.

Disabled refers to an artifact modality. Disabling image suppresses standalone
image outputs; an enabled video still contains image frames and may use the image
decoder internally. Video requires an explicit ordered latent trajectory. Debug
features and attention remain observable. Low-level `decode`, `generate_text` and
`decode_video` serve reconstruction/inspection; user-facing emission uses `emit`.

## A concrete caller path

After constructing `model` and acquiring a clean observed `state` as in the
[world-model guide](multimodal.md#operations):

```python
from pathwm.models.tasks import Actor, OutputControl, OutputRequest, TaskRequest, TaskSession

user = Actor("user", "alex")
agent = Actor("agent", "model-1")
task = TaskSession(TaskRequest(
    "explain-1", "Explain this scene with an image.", user,
    (OutputControl("image", "required", user),
     OutputControl("audio", "disabled", user)),
))
trace = {}
proposal = model.decide(state, task, agent=agent, trace=trace)
print(proposal.raw_operation, proposal.operation, proposal.requests)

# Alternatively, execute the user's explicit request immediately.
result = model.emit(
    state, task,
    (OutputRequest("image-1", "image", user, "explain-1"),),
    produced_by=agent, trace=trace,
)
if result.errors:
    print(result.errors, result.session.remaining)
else:
    output = result.outputs[0]
    reflection = model.reflect(
        state, {"image": output.loopback(state.time)}, trace=trace,
    )
```

Use `step_task(state, task, agent=agent, ...)` for one learned decision and its
consumer. A video emission needs `video_states`; textual generation has an explicit
`max_text_tokens` bound. Inspect returned errors and action/clarification proposals.

## Generated feedback stays generated

`output.loopback(time)` constructs an `Observation` with exact provenance. The time
is when the feedback is available for reflection. `observation.derive(values, ...)`
retains ancestry and marks a derived origin. `observe` rejects both classes before
any encoder runs. Generated bytes cannot implicitly become a new user instruction.

`reflect` processes feedback through its shared modality encoder, pools valid
features and adds a learned metadata token. Metadata uses byte embeddings and a GRU
over a canonical JSON record, pooling every valid position so early actor fields
remain represented. Exact strings/IDs are retained independently of this learned
compression. Roles and IDs have no guaranteed learned semantics yet.

Only working/reasoning change during reflection. World time, observed time,
observation count, uncertainty and imagined flag remain unchanged. The state retains
the union of generated ancestry. Thinking, imagination and subsequent real
observations do not erase it. `remember` explicitly rejects the entire dependent
state. Keep the original clean observed branch for observational memory; there is
no operation that removes ancestry from a reflected branch. This conservative rule
does not yet provide a separate long-term store for generated hypotheses.

The new latent schema is `pathwm-latent-v2`; older schemas and snapshots without
ancestry are rejected. Task/state `to_dict` and `from_dict` preserve exact tensor
values, dtype and metadata with `torch.load(weights_only=True)`. Task snapshots
include partial fulfillment and failures. Loading trusts the snapshot's producer;
this is an API provenance contract, not authentication against callers stripping
tags or constructing false records. External parent IDs are retained audit links,
never fetched as evidence or treated as authority.

## Training and inspection

```bash
python experiments/multimodal.py --dataset instructions --check
python experiments/multimodal.py --dataset instructions --width 16 --steps 80 --batch-size 8 --train-windows 112 --validation-windows 112 --evaluate-every 20 --improve-every 0 --learning-rate 0.001 --output runs/my_instruction_trial
```

`InstructionEpisodes` adds disclosed synthetic language labels to the existing
moving-ball episodes. The same optimizer learns operation cross-entropy, modality
binary cross-entropy and completion binary cross-entropy alongside world-model
losses. Labels never enter inference. Train/validation/test template families are
disjoint, though vocabulary and simple concepts overlap. Instructions explicitly
describe operations and formats; this is a small language-label exercise. Completion
labels are synthetic declarations, not measured success of generated answers.

Evaluation reports raw operation/completion errors, per-modality errors (also
restricted to emit examples), raw constraint violations and post-enforcement
violations. A lexical nearest-example baseline and a deterministic instruction
mismatch control preserve state/control context. Zero enforced violations measures
the gate; it does not mean the model learned the restriction. Emit examples are a
minority, so the per-emit errors matter more than an all-example modality average.

Instruction runs save every held-out decision in `task_decisions.json`, plus
`task_demo.json`, `task_inspection.pt` and `task_output.png`. The demo explicitly
labels whether emission came from the learned selection or a separate user request.
The existing standalone report includes task metrics and demo metadata. Traces
include task tokens, attention, metadata embeddings, raw logits, exact output
provenance and reflection ancestry. They measure computation, not semantic intent.

The [task flow diagram](diagrams/task_flow.svg) regenerates with `--diagram` and
shows actual calls. Its explicit user-emission branch is separate from the learned
proposal. Swap `TaskInterpreter`, `TaskPolicy`, `MetadataEncoder`, encoders or
decoders directly in `build_model`. A new decoder works with explicit `emit`
requests; a learned policy for that modality requires a matching output head and
training targets. Older checkpoints remain with their source snapshots; start a
fresh run for the new parameter layout.
