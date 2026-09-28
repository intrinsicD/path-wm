# Current work

Task plans own current work; older decisions remain in the
[historical record](project-history.md).

## Current priorities

- **Image-code work:** [Plan](image-code-contract-plan.md): replay/editing interfaces and exact resume pass; slow reconstruction passes512images. Fast fidelity/edit quality fail. P5 code feedback lowers feature error but worsens pixel MSE5.5–6.0% in two seeds; keep default0.990CPU tests pass. KV reuse saves9–23% producer time.

- **Earlier decoder result:** [Pyramid access](native-pyramid-decoder-plan.md) gains1.09–1.71% MSE, missing20%; encoder frozen.

- **Completed slice:** [Real visual-memory connection](real-visual-memory-plan.md), with Claude Opus5.5 medium.
- [Actual-model reruns](actual-model-rerun-plan.md):36 reference checks pass (105 with regression); resource/R2 rerun. Remaining gaps open.
- **Native memory evidence:** Joint6000 passes six fixed development populations: relocation62,63,61,63,64,64/64; zero novel merges and all memory contracts pass. Retention12/12. [Plan/report](real-visual-memory-plan.md) preserve failures and scope. After the optional decoder change, fixed-policy calibration and R2 life exactly reproduce earlier outputs under a fresh source-bound manifest; six-population qualification remains historical.

- **Earlier screens:** [Affine transfer](representation-transfer-plan.md) passes surrogate gates but pixel ridge is stronger; [indexed lookup](scan-discovery-plan.md) passes without training. Actual-model counterparts remain subject to the rerun audit.

- **Integrated latent agent:** the [research goal](integrated-latent-agent-goal.md)
  remains the priority: compatible latent computation, shared depth and runtime
  concept/instance acquisition and correction without retraining. Component checks
  do not establish the complete capability.
- **Integrated architecture R1/R2:** the [plan](integrated-architecture-plan.md)
  connects one session/store/clock, shared image features, concept memory, typed
  execution and verification. J passes the primary identity screen
  (100% versus54.6% cross-lamp), but misses one fresh-scene attribute guard:
  provisional development input. All five oracle application families pass
  (nu≥0.863). Code search and eight induction paths failed; copied-example recall was learned
  but degraded during the failed transfer curriculum. CI1 (§22): color reaches core; frozen property query passes. Encoder adaptation loses shape/size retention; full prior curriculum remains open.
  §23: a frequent entry rule opens L4 induction in all 5 families (2 seeds); L16/L44/transfer open.
  Rollout, learned full R1/R2 and natural data remain unproven.
  Plan §0,10,11,16,18–19,23; evidence: `runs/latent_agent_r1/`.
- **Concept-learning discussion:** the [research review](latent-concept-learning-review.md)
  records current literature and actual Claude Opus5.5 max review. Alex's initial
  target is structure shared above instances; actions and analogies are subcases.
  Joint/staged learning, latent codes and exemplars remain alternatives. The
  [shared-abstraction protocol](shared-abstraction-spec.md) defines proposed
  frozen-weight updates; no new mechanism or trained-model result is adopted.
  28 September: [self-discovered concepts](latent-concept-learning-review.md#selbst-entdeckte-konzepte-über-die-zeit)
  (prediction-driven propose/confirm/correct loop) recorded as proposal only.
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

- **Internal core design:** [core design](core-design.md) decided 28 September with
  Alex, Claude, Fable and Codex: one shared tied core for thinking and prediction,
  object slots, width 128 with two blocks, prediction first, 8 GB local limit.
  Surprise handling (E11) and learning paths (E12: memory at runtime, weights only in
  offline versioned consolidation) also decided. Not implemented; next is a planned
  contract slice.

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
- The [belief model](belief-model.md) and [model guide](models.md) describe
  categorical state, events, causal evidence and bounded memory. Long-horizon learning, calibrated uncertainty and broad autonomous
  behavior need their own evidence. Preserve failed current/recent-recall results
  when selecting a bounded input/binding/readout diagnostic.

## Open design and discussion owners

| Topic | Owner and next decision |
| --- | --- |
| Architecture walkthrough and Local/Global Context | [Walkthrough](architecture-walkthrough.md): module mapping, TaskPolicy extensions and learned control remain proposed. |
| Decision and outcome verification | [Decision design](decision-design.md): distinguish declared completion from verified success; preserve scoped recall/calibration evidence. |
| Actions and instruction semantics | [Action design](action-semantics-design.md): typed execution, uncertain inferred actions and outcome checks; general execution is not established. |
| External ideas: World Labs RTFM/Atlas | [Review](worldlabs-review.md): keyed detail retrieval, pose-like metadata inputs, realism vs fidelity; [compact instance codes](compact-instance-memory-ideas.md) (Arc2Face etc.); nothing adopted. |
| General latent processing and token roles | [Latent core](latent-core.md): shared multimodal processing and learning/readout contracts, without a new adopted layout. |
| Speech, human perception and video understanding | [Speech plan](grounded-speech-plan.md); [learnable KG voice profiles](agent-voice-design.md); no speech runs. [Human perception](human-perception-discussion.md) and [video](video-understanding-test-map.md): integration/quality open. |
| Demand-driven inference | [Neural Engine](neural-engine-inference.md): bounded residency, feature/state ownership and loading policy remain proposals. |

Use the [discussion checklist](architecture-discussion.md) and atlas. Discussion
coverage is separate from validation; green applies only to its named scope.
The [experiment workflow](experiment-workflow.md) retains the standing design
principles and run/report obligations. For this native-memory task, Alex explicitly
authorized Claude as a same-permission implementation/review partner; the older
public-only review boundary does not block this delegated work.
The [coding-agent efficiency plan](agent-token-efficiency-plan.md) is complete;
its measurements and limitations remain in that record.

Keep this page at most 8 KiB. Put new detailed results and discussion in their
existing owners; update a short status/link here when the current priority changes.
