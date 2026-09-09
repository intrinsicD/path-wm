# Why strong models work, and where this project stands

The user asks whether architecture, data, training or scale explains the gap
between our results and high-performing modern models. All contribute and interact.
There is no evidence-based universal percentage breakdown or causal ranking for
this project. The current custom pipeline is a small research prototype; the
overnight work was a set of bounded diagnostics, not training a general world
model to its best achievable performance. This discussion authorizes no new run,
hardware purchase, download or architecture change.

## The contributions are different

| Ingredient | Contribution and relevant limitation |
|---|---|
| Architecture and interfaces | Make useful computations possible and affordable: spatial locality, context, memory, action conditioning and output resolution. Extra layers cannot recover information removed from inputs. |
| Data and coverage | Supply examples of the distinctions and transitions that matter. Diverse positions, interactions, failures and actions differ from many near-duplicate frames. Passive images alone do not specify an agent's action consequences. |
| Objectives | Reward the information and behavior the system learns: reconstruction, masked/latent prediction, task readouts, imitation or reward. A low proxy loss need not mean useful control. |
| Optimization and compute | Learning rates, normalization, loss balance, targets, sampling and enough updates determine whether fitting is stable and adequate. Size, training duration and data quantity are separate variables. |
| Pretraining and distillation | Reuse information learned at a much larger upstream budget; a small deployed model or short head fit can inherit expensive training. |
| Acting and evaluation | A learned policy, planning, feedback and appropriate success metrics turn representations into behavior and expose failures. Image quality and action success are different endpoints. |

These are interacting constraints, not a numerical quality formula. More compute
can improve a suitable recipe; it does not supply absent observations, action
labels or an objective that distinguishes task success. Conversely, a failed
small task does not rule out insufficient capacity. Controlled evidence and
learning curves should determine whether data, optimization or modest scaling
is the next useful investment.

## Concrete public examples

- [DINOv2's model card](https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md)
  identifies LVD-142M training data and small/medium models distilled from a large
  teacher. The DINOv2-S package used here inherits that visual training; our local
  decoder fitting is not the cost of learning those features from scratch.
- [DINOv3's release](https://ai.meta.com/blog/dinov3-self-supervised-vision-model/)
  reports 1.7 billion images and a 7-billion-parameter teacher, with smaller ViT
  and ConvNeXt models distilled for deployment. The
  [technical paper](https://arxiv.org/abs/2508.10104) also introduces Gram anchoring
  to address dense feature degradation during long training. This is an example
  of scale and objective/optimization design working together, not proof that
  larger models automatically preserve useful spatial features.
- [V-JEPA 2](https://arxiv.org/abs/2506.09985) combines over one million hours of
  internet video with an action-conditioned stage using less than 62 hours of
  robot data. The reported robot result is image-goal pick-and-place in two labs;
  it does not establish arbitrary software, driving or general planning capability.
  Actionless video learning and learning an agent's executable actions are distinct
  training problems, even when they share representations.
- [DreamerV3](https://www.nature.com/articles/s41586-025-08744-2) provides a
  complementary route: a world model and behavior learned from environment
  interaction, with robustness techniques and imagination training. The Nature
  version reports size/replay scaling and task-specific runs on one A100 per
  agent, not foundation-scale internet pretraining. That is not a runtime or
  training-feasibility claim for this project's RTX 3050. Its reconstruction
  gradient ablation also supplies direct positive evidence for that recipe, not
  an answer to our unresolved auxiliary-loss comparison.

These are documented examples of strong systems, not a September 2026 leaderboard
or a claim that one family is universally best. Demonstrations and benchmark
statistics have different evidential scope. Do not infer undisclosed selection
or tuning practices from the existence of a polished demonstration.

## Local evidence and practical direction

The [runtime ledger](../runs/perception_overnight_2026-09-08/runtime/evaluation.json)
counts 390,240 parameters in the custom CNN encoder and 22,056,576 in the loaded
pretrained ViT encoder. These are encoder counts, not whole-agent comparisons.
The [overnight protocol](perception-overnight-protocol-2026-09-08.md) uses RGB64
source images and 4,000 updates per main fit (category probes use 2,000). Resizing
RGB64 to the ViT's 224 input cannot restore original details already discarded.
That resolution is useful for controlled toy scenes but is not a tested basis
for reading general desktop text or fine real-world details.

In the [paired package comparison](perception-overnight-results-2026-09-09.md),
foreground IoU is 0.3579 for CNN versus 0.6574 for pretrained ViT; category AP is
0.0773 versus 0.6501. This supports using the pretrained package for these outputs.
Architecture, prior training, width and preprocessing differ, so neither the
transformer nor pretraining alone is causally isolated. Short extra-depth
continuations did not consistently improve the tested tradeoff; they do not
establish that more capacity or adequate longer training can never help.

The new perception packages have not yet been integrated with newly trained
compatible memory, prediction and control. Existing released-weight LeWM control
references remain useful evidence and should be preserved; there is no need to
pretend the project lacks all successful references or to repeat completed checks.

Recommended direction, still proposed: define one concrete end-to-end capability,
use an applicable successful reference recipe and suitable pretrained components,
then close a measured bottleneck with adequate training and coverage. Training
curves, physical multi-step errors, rare failures and control success should guide
the next investment. Keep bounded component experiments tied to that capability;
do not infer universal-agent progress from a better PCA image or RGB decoder.
The auxiliary reconstruction-gradient question remains open alongside this
broader framing. Neither a new architecture nor a compute purchase is selected.

Two actual Claude exchanges provided public-only conceptual criticism; briefs,
replies and receipts are in `runs/model_quality_2026-09-09/`. Local evidence was
checked independently and was not exported. The review added emphasis on explicit
evaluation protocols and training stability. Reconciliation withdrew unsupported
claims about undisclosed demo/hyperparameter selection and a supposed most-common
cause of simple-task failure; it also narrowed home-compute limitations to the
cited frontier pretraining recipes. There is no substantive remaining disagreement.
Peer agreement is not evidence that a particular intervention will improve this
project. No model tests or dashboard rebuild were needed for this prose-only work.
