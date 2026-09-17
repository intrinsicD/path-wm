# Executable model explorer

17 September 2026. User-requested engineering slice; no model or training change.

## Scope and acceptance

Generate a standalone HTML explorer from the actual instantiated module tree,
parameters, and a bounded CPU execution of the recipe's existing objective.
Start at the complete categorical agent, retain its frozen training teacher as a
separate visible branch, and descend through modules, observed tensor operations,
parameter tensors and individual values. Breadcrumbs return to enclosing scopes.
Forward edges must come from tensor provenance, never module declaration/call
order. Backward edges must come from autograd, not reversed forward arrows.
Unexecuted modules, detached paths, shared objects, fresh initialization and absent
gradients remain explicit. All parameter values are retained without sampling.

The saved file is an identified snapshot. An optional loopback-only preview checks
source/checkpoint fingerprints and regenerates it in a fresh process when they
change; stale/rebuilding/error states must remain visible. Offline snapshots cannot
claim automatic freshness. Compatible checkpoints load strictly, with no fallback
to randomly initialized partial weights. Other experimental variants use the same
library exporter with their own recipe construction/objective; do not silently
label them as the main agent.

No new trainer, configuration registry, external service, weight download or
scientific comparison. Preserve the existing atlas and its discussion/validation
colors. A recorded execution is evidence only for the declared input/loss/mode,
not all conditional branches or model quality.

## Work sequence

1. Essential failing tests: true forks/detaches, shared parameters, method scopes,
   complete weights, real autograd, and preservation of the caller's state/RNG.
2. Implement the library recorder and standalone viewer; expose it in the existing
   multimodal recipe with explicit snapshot/checkpoint/live options.
3. Run a tiny default-model capture and inspect hierarchy, graph navigation,
   weights, gradients, back navigation and live freshness in the browser.
4. Run focused and full CPU software tests; record evidence and limitations here
   and in project-state, then commit the completed implementation.

Budget: one representative synthetic batch, CPU only, no optimizer updates;
capture target <=120 seconds and <100 MiB HTML. Full software suite <=1200 seconds.
Stop rather than silently truncate the operation graph or tensor values.

## Progress

Plan and essential checks prepared. Implementation pending.
