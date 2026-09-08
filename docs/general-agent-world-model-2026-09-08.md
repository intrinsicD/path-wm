# Cross-scale fusion and a world model beyond PushT

8 September 2026. Architectural clarification and research proposal; no new
training, checkpoint intervention or model implementation. This extends the
[perception proposal](versatile-perception-architecture-2026-09-08.md). The older
multimodal proposal in `world_model_design_notes.md` remains historical, not an
adopted 224-token configuration.

**Current answer:** we have not measured whether cross-scale attention improves
this model. The broader agent goal requires perception, temporal belief,
action-conditioned prediction, action generation and outcome evaluation. Spatial
cross-scale fusion is one replaceable operation within that system.

## What the current exchange actually does

`world_model/paddle/models.py:40` defines E, reused by PushT. It makes 256 fine
tokens from a 16×16 grid and 64 coarse tokens from an 8×8 grid, all width 64. Both
derive from the same RGB frame. With learned position/scale information already
added, one four-head exchange computes:

```text
F1 = F0 + attention(query=LN_f(F0), key=LN_c(C0), value=LN_c(C0))
C1 = C0 + attention(query=LN_c(C0), key=LN_f(F0), value=LN_f(F0))
```

Each branch then gets its own residual per-token MLP. The two directions use the
incoming branches; the coarse update does not read the just-updated fine branch.
Learned query/key comparisons assign weights over the other scale's positions,
then a weighted combination of projected values is added to each query token.

The intended benefit is context exchange: fine locations can use broader context,
and coarse locations can retrieve discriminative detail. These are intended
functions, not roles guaranteed by the architecture or our losses. An 8×8 grid is
a lower spatial resolution, not automatically an abstract task representation.
This block has no recurrence and sees no actions, goals or previous frames.

The existing structural test checks simultaneous updates from incoming branches.
The [curriculum report](training-curriculum-results-2026-09-07.md) shows attention
weights and descriptive feature variation. Neither establishes benefit. The
checkpoint inspection marks one fine query using its known pusher location for
visualization; this does not turn the plot into learned causal attribution.

[CrossViT](https://arxiv.org/abs/2103.14899) is a precedent for attention between
scales, but its classification architecture and summary-token exchange differ
from ours. [Feature Pyramid Networks](https://arxiv.org/abs/1612.03144) demonstrate
that useful multiscale fusion also has simpler top-down/lateral implementations.
These sources motivate alternatives, not a verdict on our implementation.

A suitable prospective comparison has three trained arms: current attention,
no exchange with both branches retained, and resize/projection fusion. Keep input
data, branch processing, output layout, selection rules and exposure matched;
report parameters and elapsed compute, including a capacity control where needed.
Train each arm under its declared curriculum. Turning attention off only at
evaluation measures dependence on an altered network, not whether training with
attention is better. If comparing control, train compatible downstream modules;
an old U/P paired with changed E is a compatibility confound. No such experiment
ran here, and success thresholds/budgets remain to be declared prospectively.

## A reusable agent loop

Represent an estimated situation using available observations and history. It may
include learned features, uncertain hidden state, exact symbolic values and
retrievable records. Keep observed facts, reported information and imagined
branches distinguishable. A common token width alone does not give them a shared
meaning.

The proposed division of responsibilities is:

| Component | Responsibility | Extension beyond the current visual prototype |
|---|---|---|
| Observation encoders | Encode images, text, structured state, sound or sensors | Modality-specific front ends with source, coordinate and time metadata; variable resolution/length |
| State updater and memory | Combine evidence over time into a current belief | Working state plus retrieval of relevant past observations, explicit resets and branch isolation |
| Action model / policy | Propose executable actions given state, goal and available interfaces | Discrete choices, continuous controls, text sequences, tool calls or generated artifacts with explicit action semantics |
| World model | Predict consequences of a candidate action and elapsed time | Future events, features, state changes, duration, failure modes and uncertainty where supported by data |
| Planner / evaluator | Compare candidates against a goal and constraints | Short imagined rollouts, longer subgoals, learned value or externally verifiable goal predicates |
| Observation decoder / readouts | Expose or reconstruct aspects of actual or predicted state | Images, masks, text/state changes, geometry and diagnostics appropriate to the domain |

One iteration is: observe → update belief → propose actions → predict/evaluate
outcomes → execute the selected action → observe its result. Replan from that
result. The loop can be implemented with separate modules, shared Transformer
blocks or some combination. The functional distinctions and data contracts still
matter when weights are shared.

An action has a type, arguments or generated payload, execution semantics and
duration. PushT target XY, a mouse click, a tool call, typed code and steering are
different action interfaces. Rendering a predicted screenshot is a readout of the
world model. Generating text to type or an image to save is an action-producing
policy/generator; predicting what happens after that output is executed is the
world model's responsibility. Reusing decoder components is possible, but these
targets and roles should not be conflated.

## What varies across the user's examples

| Application | Evidence to retain | Action and useful prediction |
|---|---|---|
| Software/computer use | Screenshot detail, visible text, accessible UI structure when available, focus, document contents, prior tool results | Click/type/call an API; predict dialog changes, saved state, execution errors or a completed operation |
| Output generation | Request, source material, exact symbols, output constraints and intermediate artifact | Emit text/code/image or modify a document; evaluate the produced artifact and predict execution/consumer outcomes when relevant |
| Driving | Multi-view spatial evidence, ego motion, road geometry, other agents and uncertain occluded state | Controls or trajectories; predict occupancy, interactions and goal progress across time |
| Internal computation | Current problem, evidence, candidate plans, retrieved records and intermediate results | Search, retrieve, simulate, calculate or revise a plan; evaluate whether additional computation improves the eventual action |

A computer task cannot preserve readable text by shrinking every screenshot to
64×64. It needs suitable image resolution and/or genuinely available structured
observations. A driving state needs metric and temporal structure. Exact source
code and filenames need faithful symbolic access. The reusable part is the
evidence/action/prediction loop; equivalent-sized image grids do not solve these
requirements.

There are also different meanings of scale: spatial resolution, temporal horizon,
task abstraction and the number of internal computation steps. Our current
cross-scale block implements only the first. A longer task can use a subgoal such
as “save the document” above a sequence of focus/type/click actions, with explicit
completion and failure checks. Such an abstraction must be trained or constructed
and evaluated; it does not follow from downsampling an image.

For internal computation, distinguish estimating the external world from changing
the agent's own workspace. Simulated action branches can support deliberation
without a model of every neural operation. A policy that chooses when to search,
retrieve or simulate is a further extension, judged by task accuracy and compute.
Computation consumes time; in a moving environment its elapsed time belongs in
the prediction horizon. A text narration of reasoning is not, by itself, an
accurate simulator or evidence that extra computation helps.

## Literature supporting different parts of the loop

| Primary source | What it demonstrates | Limit for our proposal |
|---|---|---|
| [MuZero](https://arxiv.org/abs/1911.08265) | Learned latent transitions supporting search by predicting reward, value and policy in games | Task-relevant prediction need not reconstruct pixels; reward-focused state need not preserve every property for future applications |
| [DreamerV3](https://www.nature.com/articles/s41586-025-08744-2) | World-model imagination trains an actor and critic across diverse control tasks using a common configuration | A broadly usable learning algorithm is different from one trained checkpoint already knowing every domain; imagination can train a policy without explicit search at every execution step |
| [Gato](https://arxiv.org/abs/2205.06175) | One set of policy weights supports several observation/action modalities and tasks | Shared action generation is feasible; this is not itself an explicit model of action consequences |
| [WebDreamer](https://arxiv.org/abs/2411.06559) | Uses LLM-predicted website outcomes to evaluate candidate actions and improves over reactive baselines in its web benchmarks | Direct precedent for semantic software-state prediction; imagined outcomes are fallible and must be checked against interaction |
| [UniAD](https://arxiv.org/abs/2212.10156) | Integrates perception, prediction and planning through query interfaces for driving | Domain-specific geometry and objectives remain significant; its benchmark is not proof of a universal agent or general deployment readiness |
| [Coconut](https://arxiv.org/abs/2412.06769) | Reuses LLM hidden states as intermediate reasoning inputs and studies logical reasoning | Supports trained internal latent computation; does not establish that spatial cross-attention develops reasoning or transfers to driving/software automatically |

These papers address different questions. Combining their mechanisms is a
proposal, and none establishes that a PushT-trained model becomes a general agent
by increasing capacity. Robust shared weights require suitably diverse training
and explicit transfer/retention tests. Existing pretrained language/vision models
are practical starting points for complex observations and action proposals, but
their domain priors still need grounding in the actual environment.

## Training and next evidence

Static images can help perception. Action/outcome trajectories are needed to learn
environment transitions; variable-duration workflows need event and completion
targets. Demonstrations can bootstrap action generation. Outcome feedback and
interaction can improve action choice, while held-out action combinations expose
where the learned transition model fails. Diverse task, object, layout and goal
splits test transfer; mixed-domain replay and regression checks test retention.
Longer imagined rollouts should be introduced only with measurements of compounding
error and actual decision quality. Match runtime and action-search budgets when
claiming a planning improvement.

For the stated general-agent goal, the mask/extent experiment remains a perception
test. Add a second, deliberately different environment early: a small local
software task that requires editing a value, navigating away, returning and
verifying persistence. Vary layout/text and hold out workflow combinations.
Screenshots and exposed UI state are observations; hidden simulator state serves
as labels/evaluation, not an undeclared input. Compare a reactive agent, an agent
using learned outcome predictions and a matched planner with the environment's
exact transition model as a positive control. Check final persisted contents,
not only a success-looking screenshot.

The proposed next sequence is a bounded fusion comparison, then that minimal
software action/outcome loop alongside the retained physical reference. It tests
whether interfaces and learned state support a different kind of task before a
large architecture is built. It is not authorization to launch these studies;
no previous deferred run was started or old readiness gate relaxed in this review.
