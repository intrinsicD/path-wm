# Interactive model explorer

Generate an identified snapshot of the **current complete categorical agent**:

```bash
.venv/bin/python -m experiments.multimodal --explore
```

Open `docs/model-explorer.html` in a current browser. It is self-contained, works
offline, and needs no Graphviz, JavaScript installation or network service.
The visual style uses colored encoder/decoder shapes and flow arrows. Colors
indicate functional roles, not discussion coverage or scientific validation.
The existing architecture atlas keeps those separate records.

## Explore

- The initial **System overview** links the main learner, its training teacher,
  and the separate World State foundation recipe. **Knowledge graph** opens the
  actual synthetic `WorldStore`: entities, components, relations, evidence and
  event operations. Click records to inspect full values, ownership, provenance,
  parent links and current dataclass schemas. The complete store exports as JSON.
- `agent` is the online inference network, optimized during training. `target`
  is a frozen copy of the same architecture used as an EMA teacher. It produces
  complete-observation reference beliefs for the online network's partial views.
  After an optimizer step, `target = decay * target + (1-decay) * agent`; buffers
  are copied. The main training CLI defaults to decay 0.99. Target parameters
  receive no gradients and the teacher is not needed for deployed inference.
- **WorldSession · runtime calls** shows observed Python component calls, with
  source and call counts on arrows. These arrows are explicitly different from
  tensor dependencies or gradient paths. Select a neural component to enter its
  own actual layers and weights; select WorldStore to enter its stored records.
- **World State · neural layers** captures `experiments/world_state.py`'s actual
  `FoundationModel` and functional objective, independently of the main learner.
  Its configuration, weights, freeze rules and gradients remain separate. The
  record/interface inventory includes optional interfaces without claiming that
  they ran in this diagnostic.
- Click a module to descend. Use the breadcrumbs or Back to return. Scroll to
  zoom, drag the background to pan, and use Fit to see the complete current scope.
- Forward data flow shows actual tensor dependencies aggregated across recorded
  calls. Inputs occupy the left boundary and outputs the right. BeliefAgent uses
  functional columns; other scopes use a cycle-aware dependency layout. Arrows
  follow rounded right-angle routes through gaps between blocks. Dashed return
  routes indicate a direction back across the layout, not proof of a recurrent
  layer. Pale gold connections attach internal tensor operations. Hover or focus
  a block for blue incoming and teal outgoing paths; select an arrow to see
  operator/tensor evidence. Details opens the module inspector and captured
  local class source. Containers and shared modules remain explicit. Layout hints
  position existing modules only; every captured connection remains present, and
  newly added modules are included automatically.
- At a leaf layer, click a yellow parameter block to open its weights. The
  Weights & buffers view also lists every tensor beneath any enclosing module.
- The tensor microscope shows exact values and gradients, not downsampled
  averages. Select leading tensor indices and row/column windows, click a cell,
  or enter a flat index. Export exact tensor JSON preserves the entire tensor.
- Backward gradients and Autograd operations show PyTorch's actual backward graph
  and which functions executed. Frozen/disconnected/unused parameters have no
  gradient; this is distinct from an observed zero gradient. No forward arrow is
  simply reversed to invent a backward connection.
- Tensor operations shows direct ATen calls in a scope, with explicit page
  boundaries (72 operations per page) and name filtering (press Enter to apply). All operations remain
  embedded in the file. Objective & caller is a runtime scope, not a neural module.
- Capture shows source identity, initialization/checkpoint, exact loss terms,
  input identity and coverage limits.

## Keep it current

A standalone file is a snapshot. For automatic refresh while editing:

```bash
.venv/bin/python -m experiments.multimodal --explore --explore-serve 8765
```

Open `http://127.0.0.1:8765`. While that browser page and Python process are open,
the viewer checks source and checkpoint hashes every four seconds. Changes run a
new capture in a fresh Python process; the page reloads when it succeeds. The
header distinguishes current, rebuilding and failed updates. A failed export
retains the previous HTML and cannot report it as current. Stop with Ctrl+C.
The server binds only to loopback and serves only its HTML and status endpoint.

Supply a different destination or explicitly chosen weights:

```bash
.venv/bin/python -m experiments.multimodal --explore /tmp/my-model.html
.venv/bin/python -m experiments.multimodal --explore --explore-checkpoint /path/to/last.pt
```

Checkpoints must match the requested architecture exactly. Use `--width`,
`--image-size`, `--audio-samples` and memory sizing options to match the original
main recipe. A complete learner retains its saved teacher and bookkeeping; a
plain deployed-agent state is loaded into both agent and teacher, explicitly
labelled. The saved replay population is retained even though capture uses two
examples. Partial/other-variant checkpoints fail; they never silently leave
random parameters behind. Without a checkpoint, weights are **freshly initialized**
with the displayed seed, not weights selected from a recent experiment.

## What is exact, and what is scoped

Module containment and every parameter value come from the instantiated model.
Tensor edges come from executed ATen operations, including dependencies through
in-place writes to views. Backward edges come from `grad_fn.next_functions` and
execution hooks; newly created autograd functions are attributed to their observed
torch-call/module scope. Parameters shared by aliases are stored once. Parameter
values are stored as full-precision JSON numbers, with explicit int64/nonfinite
representations, then losslessly compressed into the HTML.

The supplied callback executes on a deep copy. CPU PyTorch, NumPy and Python RNG
states are restored, caller parameters/buffers/gradients are preserved, and no
optimizer runs. Gradients target copied model parameters, not external input
leaves. The generic callback must not mutate its input objects or perform IO.

The main recipe uses two synthetic instruction episodes, history 2, horizon 1,
and its existing belief/task losses. It records four modality paths, prediction,
correction, thinking and task heads. It does **not** exercise every input-dependent
branch or long-horizon memory eviction. World State is not attached to this main
agent: its separate foundation capture uses four synthetic descriptor/property
pairs and the existing objective. The runtime capture executes its existing
foundation exercise, then reads its saved session through `WorldSession.think`.
That read checks that retrieval/context/reasoning leave the store unchanged. All
stored records are retained, including inactive relations. The fresh seed42 store
contains one entity and no relations; the viewer reports this rather than creating
illustrative relationships. The existing exercise's conditional paths may remain
unexecuted. A main-model checkpoint does not load World State weights or data.
Unobserved
modules stay visible. An aggregate module graph can contain cycles because a
module is reused across different times; inspect individual operations for the
actual execution DAG. Parameter counts are unique within each module and include
descendants; summing parent and child counts double-counts weights.

This capture is an architecture/debugging inspection, not model training or a
capability evaluation. Separate spatial/video codecs and experimental readout
recipes are not silently combined into this agent. Their own recipes can use
`pathwm.evaluation.explorer.capture(model, callback, metadata=...)` and
`write_explorer(snapshot, path)` with explicit construction and objectives.

The recorder targets ordinary CPU PyTorch models and retains intermediate tensors
while recording, so large real batches can be expensive. The current small system
export produces approximately 21 MB of standalone HTML. The live rebuild has a
120-second timeout, including a 90-second limit on the separate foundation
capture. Both recipes and the embedded UI are watched. There is no silent graph or
weight truncation.
