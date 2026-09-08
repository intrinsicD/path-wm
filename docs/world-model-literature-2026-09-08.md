# Literature review and next questions — 8 September 2026

Claude completed the requested public-literature search while the decoder recovery
experiment ran. Codex checked the primary sources supporting the shortlist below
and narrowed several interpretations. These papers provide methods and relevant
precedents, not an established fix for our particular model.

[Original Claude memo](../runs/decoder_recovery_2026-09-08/literature/claude_memo.md)
and [execution receipt](../runs/decoder_recovery_2026-09-08/literature/receipt.json)
are preserved unchanged. The installed Claude CLI completed successfully in
390.2 seconds, 27 turns, with no permission denials. Its reported $1.624796 is
API-equivalent usage, not a subscription billing statement. The agent interface
had no configured agents; the CLI used an isolated public brief, only WebSearch
and WebFetch, disabled MCP servers and no local project context.

## Verified, relevant starting points

| Question | Primary source | What it supports | Transfer limit |
|---|---|---|---|
| Can a new decoder expose retained visual information? | [Inverting Visual Representations with Convolutional Networks](https://arxiv.org/abs/1506.02753) | Learned inversion reveals color and rough image structure in supervised features. | An inversion score depends on the decoder, training data and objective. It is not a complete information measure. |
| Did feature coordinates change? | [Revisiting Model Stitching](https://arxiv.org/abs/2106.07682) | A trainable connector can make separately trained frozen network halves work together, despite dissimilarity under other metrics. | Successful stitching shows compatibility for the tested outputs, not that only a coordinate change occurred. |
| Should the head learn before unfreezing E? | [Fine-Tuning Can Distort Pretrained Features](https://arxiv.org/abs/2202.10054) | Linear-probe-then-fine-tune mitigates feature distortion in the studied distribution-shift tasks. | Classification evidence; our pose regressor and small encoder need their own matched test. |
| Are reconstruction features useful beyond a linear head? | [Masked Autoencoders](https://arxiv.org/abs/2111.06377) | Masked reconstruction can produce transferable visual features. | A large masked ViT differs from our small ordinary autoencoder; nonlinear accessibility must be measured locally. |
| How should image and dynamics learning interact? | [Masked World Models](https://arxiv.org/abs/2206.14244) | Separates representation and dynamics updates; its reward auxiliary improves control versus reconstruction alone. | Online robotic learning with a different architecture. Our existing pose/velocity supervision must be accounted for before adding another objective. |
| Can operation conditioning help? | [Image World Models](https://arxiv.org/abs/2403.00504) | Conditions latent prediction on photometric transformations; conditioning, difficulty and capacity affect learned invariance/equivariance. | Photometric transformations are not physical actions or a reconstruct/predict switch. It is an analogue, not direct validation of that switch. |
| How can old behavior be retained? | [Dark Experience Replay](https://arxiv.org/abs/2004.07211) | Rehearsal plus matching past logits is a strong continual-learning baseline in its settings. | For RGB outputs, define and test a reconstruction/feature retention objective instead of copying a classification loss blindly. |
| How can applications share a stable base? | [Parameter-Efficient Transfer Learning](https://proceedings.mlr.press/v97/houlsby19a.html) | Freezes shared pretrained weights while adding task-specific adapters. | NLP evidence; freezing E alone does not preserve an updated D. Preserving the entire original E/D route preserves that function. |
| Do skips help video prediction? | [High Fidelity Video Prediction](https://arxiv.org/abs/1911.01655) | Appendix A.3 shows benefits from skips, varying by architecture, motion and whether unseen pixels enter view. | Helpful copying of static appearance is legitimate; it must be distinguished from accurate moving-object dynamics. Test-time removal alone also creates distribution shift. |
| What do register tokens do? | [Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588) | Extra tokens handle internal computation associated with high-norm background artifacts and improve studied dense-vision tasks. | Our coarse-map variation is not proof of those artifacts. These registers do not automatically persist across frames or prevent weight forgetting. |
| What is persistent token memory? | [Recurrent Memory Transformer](https://arxiv.org/abs/2207.06881) | Recurrent memory tokens pass information across segments. | Language-sequence experiments; a world model needs explicit episode resets, causal state transitions and independent batch members. |
| Must planning reconstruct pixels? | [DINO-WM](https://arxiv.org/abs/2411.04983) | Predicts pretrained spatial patch features and plans toward goal features without reconstructing the visual world. | Different representations, data and planning setup. It motivates separate control and rendering evaluation, not dropping our readiness checks. |

For the strongest task-relevance statement, Codex checked MWM's full paper:
Figure 6(c), Section 5.4 and Appendix L report reward-ablation and frozen-feature
regression results. For skips, Appendix A.3 of Villegas et al. explicitly compares
with/without skips and notes reduced distant-frame benefit when unseen pixels
enter the view. Other table entries use verified primary abstracts or the already
read main text; precise unpublished local effect sizes are not inferred from them.

## Interpretations tightened after Claude's memo

- A failed linear probe does not establish missing information; a successful probe
  is evidence of accessibility for that probe task. Claude's phrase that a linear
  probe's success cannot imply information is present reverses this distinction.
- A persistent error gap after decoder fitting does not prove information loss.
  Capacity, optimization, data and regularization remain alternatives. There is
  no justified conversion of a reconstruction-error gap into an information bound.
- A learned connector restoring output accuracy does not establish a purely
  invertible coordinate change across all inputs, or accurate physical dynamics.
- Successful skip connections are not automatically metric gaming. Compare
  matched trained architectures and moving-object/rollout metrics. A skip lesion
  alone confounds reliance with an out-of-distribution intervention.
- The original memo's recent 2026 preprints and broad statement about what happens
  in most forgetting cases are not used as settled premises. Their claims would
  need separate verification. No broad register-placement sweep is launched.

## The larger world-model problem, tied to our actual results

Paddle control improved from 36.2% to 69.0% on ordinary starts and from 20.5% to
57.5% on paired opposite histories. Resetting current memory gives 14% paired
success. This is useful learned history, but five-step position MAEs are about
4–6 pixels and speed is underestimated in paired probes. PushT remains blocked
at perception; its best test angle MAE is 27.31 degrees against 5.51 on training.
These are the [completed curriculum results](training-curriculum-results-2026-09-07.md),
not new control evaluations in the decoder experiment.

Our current best guesses are objectives/generalization first, specific spatial
readout or dynamics limitations second, and total capacity as an unresolved
possibility. This is a ranking for experiments, not established causal attribution.
The following proposals retain all the user's points:

1. **Geometry/readout.** Compare the existing flattened linear H with a compact
   spatial head, such as location heatmaps and orientation/keypoint supervision.
   Use label-consistent transforms and configuration-disjoint validation. Probe
   frozen features first to separate accessibility from changes to E. Do not
   confuse successful training-set fitting with orientation generalization.
2. **Memory.** U/R already receives position/velocity supervision. Compare frozen-U
   readouts before altering memory capacity; inspect short histories, motion
   magnitude and collisions, and test displacement/velocity consistency. Any
   rebalance must retain evaluation on the original population. Repeating the
   existing supervision unchanged is not a new memory mechanism.
3. **Prediction.** The current Paddle predictor's optimization loss is normalized
   latent MSE; H position errors are measured under no-gradient evaluation.
   Compare explicit multi-step physical losses through verified readouts while
   preserving the latent objective and actual source labels. R is imperfect, so
   a loss through it needs a validity check. Inspect collision/terminal slices and
   action ranking; lower average latent MSE does not guarantee better decisions.
4. **Retention and application routing.** Keep a separate working task decoder
   while testing generic repair. If both renderings matter, compare mixed-domain
   D training, then operation/domain conditioning or separate heads under matched
   data and budgets. A reconstruction/prediction tag does not distinguish COCO
   reconstruction from PushT reconstruction.
5. **Registers and skips.** For within-frame scratch capacity, inspect encoder
   token norms first. The current decoder is convolutional, so giving it attention
   registers changes its mechanism. For temporal memory, begin with the existing
   U instead of introducing independent recurrent state into E and D. Future
   decoding may use only predicted or causally observed appearance; actual future
   features are unavailable.

Later work should include unseen situations, action-ranking calibration and
uncertainty. [DreamerV3](https://www.nature.com/articles/s41586-025-08744-2) provides
a precedent for balancing world-model learning signals and evaluating behavior;
[PETS](https://arxiv.org/abs/1805.12114) provides a precedent for uncertainty-aware
dynamics and planning. Neither is adopted as a replacement system in this turn.
The only newly authorized implementation here is the bounded decoder-recovery
diagnostic; all wider proposals remain separate future experiments.
