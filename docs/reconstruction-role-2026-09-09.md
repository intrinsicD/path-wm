# Reconstruction as a diagnostic and as a training objective

The user proposes making RGB reconstruction a separate decoder, primarily to
visualize latent state. This is a sensible design preference for the current
inspection use case. It does not by itself decide whether a reconstruction loss
should train the representation. No model, optimizer or experiment is changed
by this clarification.

| Role | Inputs and gradient contract | What it establishes |
|---|---|---|
| Diagnostic state renderer | Exactly the declared observation, memory or predicted state; detached inputs and decoder-owned optimization | How a fitted decoder can render that state; not a literal or exhaustive picture of its information |
| Auxiliary reconstruction objective | Declared state/features, with intentional gradients into specified representation modules | A learning signal whose benefit must be judged on retention, prediction and downstream tasks |
| Functional image output | Task-relevant state/context and an image-generation target | Output quality for a task that actually requires producing images |

Separate decoder weights do not isolate gradients into a shared encoder. For a
pure diagnostic renderer, stop the gradient at its inputs (or fit on a frozen
snapshot) and keep its optimizer independent. This allows improving the viewer
without using its pixel loss to alter the state being inspected. A moving encoder
can make a previously fitted renderer stale, so bind diagnostic results to the
state/decoder checkpoint pair.

Current PushT and Paddle planners score predicted position/pose readouts rather
than decoded RGB: `world_model/pusht/planner.py:CEMPlanner._evaluate` and
`world_model/paddle/planner.py:ExhaustivePlanner.plan`. Their perception trainers
currently add RGB and position/pose losses, so reconstruction also trains the
encoder. The original RGB decoder reads fine/coarse observation features; it
does not directly read recurrent memory. The overnight frozen-feature heads
already permit training RGB separately from the fixed representation.

The diagnostic input boundary is essential. Early/fine tokens are valid if they
are the state being inspected. Extra image skips cannot establish what a separate
recurrent-memory state retained. The overnight local-input reconstruction gains
show decodability from the supplied observation features, not retention by the
temporal model or accuracy of imagined futures. To inspect an imagined state,
use its predicted features and causally retained information, not real unseen
future pixels. A decoder trained on observed states may need calibration/refitting
for predicted-state errors; an observation decoder's missing-input deployment
cannot be assumed valid.

Successful images may partly reflect decoder priors; poor images may reflect a
weak or stale decoder. Keep quantitative state/task readouts beside the renderer.
Selecting the main representation on task, prediction and retention quality is
consistent with treating diagnostic RGB as optional. Do not select a worse task
head merely because its companion produces better debug pictures.

Removing pixel reconstruction from representation learning is a separate choice.
[TD-MPC2](https://arxiv.org/abs/2310.16828) and
[MuDreamer](https://arxiv.org/abs/2405.15083) are public precedents for world-model
learning without input reconstruction. They use alternative learning objectives;
their results do not show that deleting our RGB term leaves adequate learning or
generic feature retention. No such loss-removal comparison ran this turn.

Claude provided a public-only conceptual critique and a correction round through
the installed CLI. The implementing agent checked local planner/trainer paths and
the primary sources. The correction distinguishes deterministic feature states
from RSSM posterior/prior terminology, and qualifies the claim that raw-skip
decoders cannot render imagined states at all: compatible causal features or a
trained missing-detail mode can be possible, but require validation. It also
clarifies that early tokens are admissible when they are the declared tested
state. Exact briefs, responses and execution receipts are preserved in
`runs/reconstruction_role_2026-09-09/`. Peer agreement is not a new empirical result.

## Follow-up: does reconstruction help train the encoder?

The user clarifies that the main question is the effect of auxiliary RGB loss on
a **trainable encoder**, and consequently on downstream tasks and temporal
modeling. It is not a proposal to detach every reconstruction objective. A
renderer can be optional at inference while its training loss still contributes
to the representation. Keep that empirical question open.

Reconstruction puts pressure on the supplied representation to retain visible
information, including details omitted by sparse task labels. This can support
geometry, learning efficiency and later tasks. It does not guarantee that this
information is organized into accessible physical variables or a state that is
easy to predict. Pixel error can emphasize texture/background; small critical
objects contribute comparatively few pixels. A single sharp image is compatible
with different velocities and action responses, so same-frame reconstruction
alone cannot identify temporal state. These are mechanisms and hypotheses, not
measured effects of removing our reconstruction term.

The actual pathways are explicit in `world_model/pusht/training.py:perception_batch`
and the perception branch of `world_model/paddle/training.py:_stage_impl`: RGB and
position/pose losses both update E. In the later staged memory/prediction fits,
E is frozen. `world_model/paddle/rollout.py:rollout_step` feeds predicted features
and memory into subsequent updates, without decoding RGB. Thus current pixel
supervision constrains perception directly and may affect U/P indirectly; it
does not directly train recurrent memory to retain or predict pixels.

The completed results do not isolate this effect:

- Frozen-feature decoder refits and the overnight P1/D2/D3 comparisons cannot
  measure reconstruction gradients changing E.
- The nine encoder continuations all retain RGB, mask and pose objectives;
  `world_model/curriculum/perception_continuation.py:step_update` applies
  `0.5 * image_loss + task_loss` in each domain. Their domain gradient comparison
  includes RGB plus mask versus RGB plus pose, not RGB versus other objectives.
- The earlier warmup/adaptation studies do not provide a matched joint-training
  RGB-gradient ablation. Recovering RGB with a refitted decoder establishes
  recoverability, not a causal downstream benefit of RGB supervision.

Public evidence supports testing rather than assuming either answer:
[SAC+AE](https://ojs.aaai.org/index.php/AAAI/article/view/17276) demonstrates useful
auxiliary reconstruction for visual control and discusses optimization stability.
[MAE](https://arxiv.org/abs/2111.06377) demonstrates transferable representations
from masked reconstruction; its pretraining result is not evidence for any
particular same-frame joint RGB loss.
[Masked World Models](https://proceedings.mlr.press/v205/seo23a.html) combines
masked visual learning with an auxiliary reward objective and decoupled latent
dynamics. [MuDreamer](https://arxiv.org/abs/2405.15083) provides a counterpoint:
alternative value/action targets improve distracting-background robustness in
its reported settings. These systems change more than a loss coefficient and
do not determine the appropriate coefficient for this project.

### Proposed discriminating comparison; not an executed protocol

Use multiple paired seeds. Within each seed, start each arm from the same E
checkpoint and decoder/readout initialization, with identical ordered data,
other objectives, updates and evaluation cases.
E remains trainable through the other objectives in **every** arm.

| Arm | RGB gradient into E | RGB decoder training |
|---|---|---|
| Diagnostic-only RGB | Blocked | Same active objective and optimizer recipe |
| Weak auxiliary RGB | Small, predeclared multiplier | Same active objective and optimizer recipe |
| Reference auxiliary RGB | Reference multiplier | Same active objective and optimizer recipe |

Vary the gradient entering E, not the decoder's own gradient multiplier. Keep
decoder parameters, optimizer and clipping independent of task modules; audit
the actual encoder gradients and clipping. Decoder training inputs will diverge
as the encoders change, which is an intended consequence, not a violation of
the paired recipe. Hold feature routes fixed: an added image bypass would change
how much reconstruction constrains the tested representation. A reference
coefficient and the meaning of weak must be declared for normalized losses and
the chosen architecture; no numeric value is established here.

Measure held-out task accuracy, geometry tails, category/mask transfer and
retention before assessing temporal utility. Include predeclared clean and
label-preserving distractor cases to expose robustness differences. Fit fresh
compatible U/P/readouts on each frozen resulting E with matched recipes, data
and budgets. Use the same declared feature-normalization procedure, downstream
capacity and any tuning allowance. Evaluate physical multi-step errors, action
distinctions and actual control. Do not rank different learned feature spaces by
absolute latent MSE or attach an unchanged old predictor to a moved representation.
Predeclare checkpoint selection and report fixed-budget learning curves so
convergence and selection are visible. These tests assess predictive usability
under the chosen downstream recipe, not universally across every possible head.

This design tests auxiliary reconstruction during continuation from a common
initialization. It cannot establish whether earlier reconstruction pretraining
was necessary, or whether the same answer holds at another budget. Joint
encoder-dynamics optimization is also a different question from the staged
comparison. A crossed nuisance-training comparison could investigate a distractor
mechanism; a non-pixel auxiliary arm could investigate whether a measured RGB
benefit is replaceable. Neither is required to identify the initial incremental
effect under a fixed recipe, and neither is answered by that effect. Masking,
future-frame losses, latent consistency and additional architectural routes are
separate interventions after this comparison. No model, loss, experiment queue
or dataset changes are authorized by this conceptual clarification alone.

Two further public-only Claude exchanges reviewed this comparison; exact prompts,
responses and execution receipts are under
`runs/reconstruction_constraint_2026-09-09/`. The implementing agent checked the
local gradient paths and primary literature independently. The reconciliation
withdraws the claims that the three-arm comparison cannot test its scoped
hypothesis, that fitting and testing dynamics on frozen features only tests
static recoverability, and that a non-pixel control is required for the initial
incremental-effect question. It also corrects SAC+AE's stochastic-latent issue,
which the initial reply mislabeled as decoder stochasticity. Paired seeds,
distractor evaluation, declared feature routes, normalization and fitting curves
were retained as useful controls. No substantive disagreement remains about
this scoped proposal; no new empirical result follows from that agreement.
