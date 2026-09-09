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
