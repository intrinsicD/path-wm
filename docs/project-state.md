# Current work

Read the plan for the requested task. This is a current-work index, not a session
log. Earlier decisions, negative results and run/report receipts remain in the
[complete historical record](project-history.md) and the linked owning documents;
read the particular record when needed, not the whole history at startup.

## Current priorities

- **Integrated latent agent:** the [research goal](integrated-latent-agent-goal.md)
  remains the priority: compatible latent perception/thinking/action, shared-depth
  computation and useful concept/instance memory that acquires, retrieves and
  corrects knowledge at runtime without weight retraining. Partial components do
  not establish the complete capability.
- **Integrated architecture R1/R2:** the [adopted plan](integrated-architecture-plan.md)
  implements the pixel reference and a unified session (one store/clock, separate
  instance identity and concept membership, shared source encoding, bounded actions,
  external verification, restart recovery). All 752 CPU tests pass, including independent
  action/correction/restart boundary tests. Learning remains open: texture
  variation fixes lamp recognition but identity transfer is weak; a fresh symbolic
  core now learns relational information with supplied rule codes, below the gates.
  Paired latent-identity training and exact core continuation are the next trials.
  No integrated learned capability is established.
- **Latent concept learning:** the [abstract research review](latent-concept-learning-review.md)
  compares concept induction, latent memory and prerequisites with current primary
  sources and actual Claude Opus 5.5 at max effort. Full modal reconstruction is
  not a universal prerequisite; joint learning, staged learning and pretrained
  features remain alternatives. Concept-code search is distinct from weight
  retraining. No mechanism, experiment budget or new capability is adopted.
  The [next discussion](latent-concept-learning-review.md#nächste-besprechung-fünf-latente-operationen)
  describes comparing, binding, generalizing, applying and correcting as functional
  requirements; separate modules and a concrete learning path remain unselected.
  A [proposed transfer objective](latent-concept-learning-review.md#bekannte-konzepte-als-transfersignal)
  uses known relations to supervise new-case outcomes; new-rule generalization and
  informative target representations remain to be established.
  The [concept/understanding discussion](latent-concept-learning-review.md#konzept-objektverständnis-und-latente-aktionen)
  now relates object identity/state, latent actions and conditional effects; raw
  action differences alone do not establish semantics or causal understanding.
  Alex clarifies that “concept” initially means something shared above instances.
  The [direction review](latent-concept-learning-review.md#richtungsprüfung-geteilte-struktur-über-instanzen)
  broadens the target to reusable structure; actions and analogies remain subcases,
  with examples/prototypes and inferred codes still open alternatives.
  The unchanged [protocol](shared-abstraction-spec.md) defines equations, loss, arms
  and frozen-weight updates; no trained-model result.
- **Attention backend review:** [current kernels and GPU eligibility](encoder-token-budget-plan.md#current-attention-kernels-pre-integration-review-22-september-2026). Current fused efficient attention is the baseline; native Flash/cuDNN, Flex and the new FA4 Ampere source path require matched local comparisons. Half precision and exact mask support are explicit constraints. No integration or speed claim yet.
- **Encoder/model efficiency:** use the [local encoder plan](encoder-token-budget-plan.md)
  and preceding [token-budget plan](token-budget-plan.md) for the current implemented
  baseline, reproduction commands and remaining work. Keep fine-window restrictions
  optional and the observation resampler disabled by default. Before promoting
  either, establish trained task-quality/detail-retention evidence; current short
  fits do not do so. Persistent detail retrieval, bounded routing and delta updates
  remain follow-ups, not permission to replace a working path without a scoped plan.
- **Input-architecture specification:** the [multiscale design](multiscale-modality-design.md)
  remains active: information-preserving operations, learnable filter bank,
  post-processing, then compression at each scale, with exports before compression
  and consumer-specific processing. Invertibility is preferred when compatible
  with learning; that learning condition is unresolved. The
  [randomized-attention discussion](encoder-token-budget-plan.md#randomized-factorized-attention-discussion-not-implementation)
  records options and approximation limits, without a new adopted mechanism.

## Current evidence and limits

- Packed encoder merges and bounded geometry caches are implemented; fine windows
  retain exported detail positions while restricting interactions. The
  [encoder plan](encoder-token-budget-plan.md#measured-result) owns the image256
  resource comparison, small-input overhead, 114 scoped checks and quality limits.
  Reports: `runs/encoder_token_budget_cached_v1/report.html` and
  `runs/encoder_token_quality_v1/`. Existing renderer; structural report validation.
- Complete-event preparation is reused, and actual attention allocations are
  profiled. The [token-budget record](token-budget-plan.md#results-and-adopted-change)
  owns output/gradient/RNG checks, scoped timings and the optional resampler's
  added cost. Short future-image fits still fail the copy-frame comparison;
  neither resource savings nor successful execution establishes learned utility.
  Reports: `runs/token_budget_final_v1/report.html` and `runs/token_budget_quality_v1/`.
- The [belief model](belief-model.md) and [model guide](models.md) describe the
  implemented categorical state, event transactions, causal evidence and bounded
  memory. Long-horizon learning, calibrated uncertainty and broad autonomous
  behavior need their own evidence. Preserve failed current/recent-recall results
  when selecting a bounded input/binding/readout diagnostic.

## Open design and discussion owners

| Topic | Owner and next decision |
| --- | --- |
| Architecture walkthrough and Local/Global Context | [Walkthrough](architecture-walkthrough.md): module mapping, TaskPolicy extensions and learned control remain proposed. |
| Decision and outcome verification | [Decision design](decision-design.md): distinguish declared completion from verified success; preserve scoped recall/calibration evidence. |
| Actions and instruction semantics | [Action design](action-semantics-design.md): typed execution, uncertain inferred actions and outcome checks; general execution is not established. |
| General latent processing and token roles | [Latent core](latent-core.md): shared multimodal processing and learning/readout contracts, without a new adopted layout. |
| Speech, human perception and video understanding | [Voice](agent-voice-design.md), [human perception](human-perception-discussion.md), [video test map](video-understanding-test-map.md): model selection, integration and natural-data quality remain open. |
| Demand-driven inference | [Neural Engine](neural-engine-inference.md): bounded residency, feature/state ownership and loading policy remain proposals. |

Use the [architecture discussion checklist](architecture-discussion.md) and atlas
for discussion coverage separately from evidence-backed validation. Green applies
only to its named scope; these summaries do not change colors or promote results.
The [experiment workflow](experiment-workflow.md) retains the standing design
principles, public-only external-review boundary and run/report obligations.
The [coding-agent efficiency plan](agent-token-efficiency-plan.md) is complete;
its measurements and limitations remain in that record.

Keep this page at most 8 KiB. Put new detailed results and discussion in their
existing owners; update a short status/link here when the current priority changes.
