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

Completed. `pathwm.evaluation.explorer` records the copied model, ATen tensor
provenance (including in-place view writes), real autograd execution and complete
weights/gradients. The recipe exposes snapshot, strict checkpoint and loopback
live-rebuild options. The standalone viewer supports hierarchy navigation,
pastel encoder/decoder blocks, exact operation pages and a scalar tensor microscope.
See [usage and limits](model-explorer.md).

Verification: all 632 software tests passed in an isolated source snapshot;
16 focused explorer/diagram checks passed on the final Python implementation.
The later focused checks cover strict checkpoint replay-buffer sizing and live
source identity. Another concurrent task restored the previous metadata encoder
implementation and replaced its tests after the full-suite snapshot; those source
changes were not part of this explorer change. A separate default-model audit
against current source checks every registered module, all 1,031,106 learner
parameter scalars, the exact uninstrumented loss, every gradient (including absent
gradients), forward provenance causality and backward endpoints.

Browser checks covered module → layer → direct weight navigation, exact scalar
selection, gradient heatmaps, Back/breadcrumbs, real backward operations, paging,
operation filtering and capture provenance. The live-server test covers successful
rebuild, failed rebuild retaining the last snapshot and recovery. Python lint and
JavaScript syntax checks passed. A complete capture takes about 7 seconds and
produces about 19 MB HTML, within the declared budget.

The initial view contains 601 deployed modules and 515,553 parameters; the frozen
teacher is a separate branch. Weights are fresh initialization, seed 42. The short
synthetic execution does not establish coverage of every branch or scientific
capability. No optimizer, training, model-default or validation-color changes.

## Layout refinement, 17 September

User reports dense BeliefAgent layout and confusing connections. Arrange explicit
input/output boundaries at the left/right extremes, spread internal modules into
readable columns, and route arrows through free space around nodes. Repeated-call
return edges must remain present and clearly distinguishable. Model-specific stage
hints may position existing nodes but must not create, omit or relabel connections;
unrecognized modules still receive positions. Preserve all drilling/weight views.

Bounded checks: browser geometry of boundary placement, nonoverlapping blocks,
routes avoiding unrelated blocks, preservation of all graph endpoints, and smoke
navigation in forward/backward/nested views. Reuse the recorded payload for layout
iteration; regenerate the final source-identified snapshot once. No model/training
or Python recorder changes. Budget: frontend-only edits, focused renderer checks,
and browser review; no new training or full numerical test rerun required.

Completed: fixed boundary columns, functional stage hints for BeliefAgent and a
cycle-aware dependency layout elsewhere; rounded orthogonal routes avoid blocks.
Return paths remain dashed and internal tensor-operation links use a quieter gold.
Hover/keyboard focus separates blue incoming from teal outgoing connections.
Stage hints change positions only; new/unrecognized modules remain included.

Browser geometry verified all 17 BeliefAgent overview blocks and all 41 original
connections are preserved. No node overlaps or routes through unrelated blocks
in that overview, backward view (16 nodes / 31 arrows), nested encoders and Conv2d
leaf (6 / 8), or the first ATen page (74 / 139). Weight drill-down and Back still
work. The final overview received another geometry/visual check; exact source,
weight, loss and gradient audit passed after regeneration. Twelve exporter checks
pass in 7.47s; JavaScript syntax and diff checks pass. The live preview regenerated
during layout iteration. Numerical model code and training were unchanged.

## System scope and teacher roles, 17 September

User accepts a system-level view including World State and asks why agent/target
are separate. Preserve the selected main learner snapshot and add the existing
World State foundation as a separately labelled recipe configuration. Use its own
constructor/objective and bounded synthetic runtime exercise; do not silently
attach invented World State dimensions to the main agent. Show captured runtime
component calls separately from neural tensor/autograd edges. Expose WorldStore
entities, components, relations, evidence, events, exact stored values and current
record schemas; source links/docstrings explain optional interfaces. Label the
synthetic diagnostic store and fresh weights explicitly.

Expose inference-agent versus frozen training-only EMA teacher roles, including
source-backed update rule and stop-gradient behavior. Never imply an EMA update
occurred in the no-optimizer capture. Module IDs and checkpoint formats stay exact.
Keep recipe construction in recipes; companion export crosses a CLI boundary,
not a shared-library import of experiment code. Include both recipe files in live
source tracking. Fail visibly if the companion capture cannot be generated.

Checks: meaningful failing coverage for real runtime component calls, full store
records and sources, separate neural context/roles and source tracking; focused
export/World State tests, isolated full software suite, browser system/store/neural
navigation and offline payload verification. CPU only, no optimizer, each export
under120s and100MiB; full tests capped1200s. No model or scientific capability change.

Implemented: the initial system overview links the main learner/EMA teacher,
WorldSession calls, WorldStore records, current schemas and optional interfaces.
The companion recipe exports its actual FoundationModel through a subprocess CLI:
452 modules, 111,929 weight scalars, no teacher. Its existing functional objective
supplies neural gradients; a separate diagnostic session supplies persistent data
and observed Python call edges. An additional read through the saved session
exercises retrieval/context/think even when fresh binding does not create two
owners, and asserts the store is unchanged. The fresh seed42 store has one entity,
four components, zero relations, four evidence records and two events. Full record
values, endpoint/evidence/parent links, events and schema source remain inspectable.

Final focused checks: 27 pass in12.12s. Independent uninstrumented audits match
every module, weight scalar, loss and gradient for both actual recipes; main target
has no gradients. Full store records/events roundtrip without truncation. Browser
QA covers system navigation, teacher labels/source, records→evidence→event, Back,
schemas/source and WorldSession→context→individual weight/gradient. Geometry
retains all13 runtime nodes/12 observed edges and17 main-agent nodes/41 edges,
without node overlaps or paths through unrelated nodes. Runtime layout hints only
position recorded nodes; new components receive additional slots.

The final standalone HTML is20,723,908 bytes; live source rebuild succeeds within
its configured120s timeout and watches both recipes and embedded UI. Python lint,
JavaScript syntax and whitespace checks pass. Browser automation blocked file-URL
navigation; UI QA used the loopback server serving the same standalone bytes.
Offline packaging and lossless payload extraction pass structural checks.

The isolated full suite passes643 tests in718.73s. That frozen copy precedes the
additional saved-session read and final UI refinements; the final27 focused tests
cover the changed Python path, full relation/provenance records and profiler
cleanup. The exact numerical audits and browser checks above use the final source.
