# First generated visual-memory training slice

**Steered before implementation:** the user now explicitly requests direct numeric
weight construction and testing first. The optimizer run below is deferred; the
immediate experiment is [direct weight construction](direct-weights-plan.md).

User authorized the assistant-authored curriculum on 12 September. Separately, the
user asks about writing useful numeric weights directly without optimization. Binary
serialization is possible, but no validated direct weight synthesis mechanism exists
for this architecture. This slice produces weights through ordinary training and must
not describe them as directly invented learned parameters.

## Scope and contract

Use the existing multimodal recipe, image encoder, categorical belief event path,
hybrid memory and working-token reader. Add an ordinary visual-memory data adapter
and readout wrapper; no separate trainer, registry or runtime teacher. Observation
images are rendered by deterministic code, with explicit event-derived labels. This
is controlled 2D visual history, not realistic webcam footage or general entity learning.

One target mug and two containers. Four RGB32 observations: target at center, target
visibly at the selected left/right container, then two identical covered views. The
question is fixed: which container was the mug last visibly associated with? Answer
left/right. The final image cannot reveal the answer. A pair shares every appearance
and final frame but differs in the earlier visible target location and answer. Model
forward receives only images and their ordinal times; labels, object coordinates,
scene IDs, renderer trace and answer text stay outside the forward boundary. The task
query is supplied as one learned constant; no language understanding claim.

Generate 256 training scene pairs, 64 development pairs and 64 evaluation pairs from
disjoint seeded scene populations (seeds 3101/3201/3301). Appearance palettes, background,
container sizes and target shape details vary; all variants of a scene stay in its
partition. Retain per-scene hashes and verify disjoint complete image histories and
final images across partitions. Record renderer version and all scene parameters.
Known rendering/visibility checks plus sampled visual inspection verify intended
labels. Only location differs within a pair. No future observation enters a query.

## Learning and measurements

Fresh own model, width16, one-scale image encoder, RGB32, recent memory2, block2,
one compressed block. All components on the active gradient path train jointly;
unused heads/modalities receive no supervision. Read final working tokens after two
thinking rounds. Final-answer cross entropy only; no positional-label auxiliary loss.
AdamW lr0.003, weight decay0.01, gradient norm cap5, batch16 (eight complete pairs),
512 updates, seed3001, FP32. Full history remains differentiable within each episode;
episodes start with fresh state. No cross-episode memory.

Evaluate final checkpoint, with development NLL/accuracy every64 updates. An untouched
evaluation split is scored once after training. Use matched categorical RNG for the
two members of each evaluation pair and fixed evaluation seeds, preserving training
RNG. Report raw logits, accuracy, NLL and fraction of pairs where both answers are
correct. Evaluate the same model with all earlier frames replaced by the final image;
matched RNG makes its logits identical within each pair, so balanced accuracy is50%.
This intervention and its information limit are the current-view control; do not call
it a separately trained baseline.

Development screen: final held-out accuracy >=90%, paired-both accuracy >=80%, and
history accuracy exceeds the current-view control by >=30 percentage points. All gates
must pass. Failure is retained; no post-hoc threshold change or test-driven tuning.
This establishes only short controlled visual recall if successful. Real-footage
training/transfer, multiple-entity binding, unknown current state and longer retention
are subsequent experiments with their own contracts.

## Budget and verification

Run on the local RTX3050 after a tiny forward/backward/inference check. GPU access is
available outside the filesystem sandbox. Limit this process's PyTorch allocator to
1GiB, log allocated/reserved training and inference peaks, and stop above a measured
1GiB reserved peak; allow up to900 active seconds for training and evaluation. These
are conservative pilot defaults, not a user-approved whole-agent budget. Other local
GPU users are untouched. Checkpoint state includes cumulative active time, weights,
optimizer, sampler/RNG and progress; CPU exact pause/resume is tested. CUDA resume is
supported but bitwise CUDA reproducibility is not claimed without direct verification.

Before implementation commit the plan and meaningful failing tests: exact paired
final pixels/opposite labels, rendered evidence visibility/location and split isolation;
past-image gradients and history-erasure invariance through the real agent; tiny-run
CPU resume and output-weight load equivalence. Commit implementation before training.
Use existing Run/checkpoint/report machinery. Preserve checkpoint, data identity, raw
metrics, prediction cache and inspected scene strips in the run report. Browser QA is
attempted through the browser skill; disclose structural-only status if unavailable.

One brief public-only Claude review checks causal/label/control pitfalls; local tests
remain authoritative. Review the results before proposing another scientific run.
