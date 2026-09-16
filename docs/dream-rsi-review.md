# Dream-RSI: relevance to PATH-WM

16 September 2026. Assessment requested by Alex; no implementation, training or
default changes. Sources: [paper](https://arxiv.org/pdf/2609.14858),
[authors' explanation](https://dream-rsi.com/) and
[release status](https://github.com/zhengkid/Dream-RSI).
The public PDF, extracted text and source hashes are saved under
`runs/literature/dream_rsi_v1/`; public-only actual Claude reviews are under
`runs/reviews/dream_rsi_v1/`.

## What the work establishes

Dream-RSI improves an executable exploration policy while keeping its coding agent
and evaluator fixed. Online search records branches and outcomes. Offline replay
reveals only outcomes reachable through the recorded continuations; policies vary
branch allocation, batching and stopping. Selected policies return to online search.
This is recorded-history simulation, not learned latent dynamics. Incumbent
inclusion guarantees nondecreasing selection score on fixed history, not future
success. See paper §3.

The same-agent Pro Lasso comparison uses 317 versus 550 discovery calls and lowers
mean solver runtime from 3587.1 to 2931.0 ms; individual datasets regress. The 162×
headline uses a different-agent comparator. Calls exclude a complete accounting of
policy development and execution costs (§4.1). A single-task guidance ablation
does not establish that semantic memory is harmful (§5.1).

Before reproducing, resolve the objective description: Equation 1 balances best
score, attempt count and parallelism, whereas Appendix B.2 describes Pareto AUC
minus a parallel penalty. Full framework/reproduction code is still pending in the
linked repository; appendix program listings are available.

## Proposed application — our inference, not a paper result

| Existing part | Useful extension | Boundary |
| --- | --- | --- |
| Experiment recipes, run manifests and understanding suite | Use comparable histories to test whether simpler allocation/early stopping reaches the same acceptance gates with less total work. | Our runs often change weights, objectives or data. They cannot simply be stitched into one counterfactual simulator. |
| Episodic World State | Expose action/outcome histories as a read-only replay view, retaining source, time, neural version and measured cost. | Entity relations and latent beliefs remain in their existing stores. A saved store revision alone is not a full neural/environment state. |
| Bounded planning and task operations | Later compare small controllers choosing which branch to extend, how much reasoning to spend and when to stop. | Extending this to general recall/think/tool/physical actions requires its own training task and evidence. |
| Shared latent thinker and transition predictor | Combine retrieved past outcomes with uncertain predictions about unseen actions. | Recorded outcomes are evidence; model rollouts are hypotheses. Never promote a predicted outcome to observed history. |
| Image/video/audio/text encoders and decoders | No change justified by this paper. | It does not establish a better codec, multimodal understanding, smaller decoder, or local-GPU training recipe. |

Concrete code anchors: `pathwm/io.py` already saves run provenance;
`pathwm/world_state/store.py` and `session.py` supply transactional history and
session snapshots. `pathwm/evaluation/planning.py::choose_sequence` scores supplied
fixed-horizon action sequences; it does not yet learn branch allocation.
`pathwm/models/tasks.py` already names think/recall/imagine/act/emit/ask/finish.
Those interfaces are possible attachment points, not evidence that the proposed
controller already exists or works.

A small trace controller could run on CPU without expanding the encoders. That is
an implementation possibility, not a measured memory budget. The paper's strong
external coding agents do not validate our tiny model's language or perception.

## Smallest useful next comparison

First audit whether one existing homogeneous task family contains complete
prefix-visible parent/child histories under fixed interfaces, model and evaluator.
Do not manufacture missing branch outcomes. If the logs are insufficient, collect
a small bounded task with explicit state/action/outcome records; do not build a
general orchestration framework to compensate.

Compare fixed-budget scheduling, a simple fixed early-stop rule and a small
adaptive rule. Separate development from held-out whole tasks/runs, including
prompt/protocol tuning; replay controllers see no future scores or task IDs that
encode outcomes. Report support coverage, quality versus actual cost, and every
required task/regression gate. Count policy-development calls/tokens, CPU time,
GPU time and wall time. Parallel width is not intrinsically valuable on one GPU.
Treat identical replay results as paired comparisons on the same histories, not
as independent new experiments.

Any apparent advantage needs fresh real executions before adoption. If the simple
rule matches the adaptive one, retain the simpler rule; that would not by itself
invalidate the published method. Unsupported actions remain unscored by replay.
Numerical acceptance gates and a compute budget must be fixed before execution.

The current [request interpretation diagnosis](request-readout-plan.md) remains
the active model task: familiar answer formats improved, unseen wording still
fails and old capabilities are not fully preserved. Replay scheduling could help
us investigate that efficiently; it does not repair the underlying semantics.

## Review status

Claude independently criticized the public-paper interpretation and emphasized
the unsupported jump from branch scheduling to general action selection. We
retain that boundary. Reconciliation corrects two overstatements: the paper does
include fresh online deployment, and matching a simple baseline would establish
local sufficiency rather than prove a replay-selection artifact. Review agreement
is methodological input, not validation of a PATH-WM improvement.

Architecture discussion follow-ups reference this proposal. Existing validation
scopes and colors remain unchanged. No training or software tests are warranted
for this documentation-only assessment.
