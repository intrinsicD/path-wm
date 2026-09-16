# Video output and temporal access

16 September: user requests video before returning to direction. Preserve the
direction source-rotation proposal. Existing video results concern small observed
clips, not natural-video generation or forecasting. Use the current PyTorch models,
modality_readout recipe, Run/checkpoint and standalone reports.

## Diagnosis and first comparison, registered before execution

The1024-step oracle decoder fits known combinations, but withheld color/location
combinations fail foreground quality. Its video branch adds requested time to every
source token before the native image read. The proposed first change moves that
same learned time projection to the patch queries. No extra parameters, different
RGB head, changed spatial queries, target-frame inputs or hand-coded motion.
Keep context conditioning as the default; query conditioning is opt-in.

1. Add an optional validated per-sample query offset to ImageDecoder; extend the
   TemporalImageDecoder with context/query conditioning. Check legacy output
   equivalence, context immutability, masks/gradients, time selection, identical
   parameter initialization, saved-option restoration and exact resume.
2. Add video diagnostics: soft foreground centroids (mass above background0.04),
   framewise centroid error, displacement error, direction including a minimum
   motion threshold, per-frame RGB error. Report foreground/background separately.
   Static-first-frame and reversed-output controls must expose timing failures;
   no metric computed from target facts is passed to a learned decoder.
3. Eight fixed CPU fits: seeds7201/7202 × context/query × oracle/frozen-video input.
   Original same-seed core checkpoints and cached sampled states;1024 updates,
   Adam0.003, batch24, existing normalized MSE, clip5. Train only the video decoder.
   Oracle supplies complete true factors and is explicitly not agent performance.
   Frozen runs train on video-only cached states, and evaluate all input modes.
   Source encoders/core and all other outputs remain frozen. Same initialization,
   batches, populations and parameter count within each paired comparison.
4. Primary query-conditioning benefit: held-out oracle foreground MSE improves by
   >=20% in BOTH seeds, known foreground MSE worsens by no more than0.005. Report
   failed cases. Capability is separate: existing video gates (all-factor template
   correctness>=80%, total MSE<=0.02, foreground MSE<=0.05) AND direction>=90%,
   centroid mean Euclidean error<=1 pixel, horizontal displacement MAE<=1 pixel,
   on both known and held-out cases in both seeds. Template direction alone is
   insufficient because arrow orientation already reveals direction in one frame.
   For frozen runs the primary cell is video-only; other inputs diagnose transfer.
5. Independent frozen temporal-access challenge: symmetric moving objects, reversed
   pairs containing identical frame sets and the same center frame at the same
   timestamp. Check raw order-sensitive positive control and identical middle-frame
   negative control, then existing encoder and final core access with validation-
   selected linear readers. New noise/groups across splits; no model training or
   future forecasting claim. Do not require reversed clips to have identical
   encoded states, since that would erase the signal under test.

Budget: eight1024-update fits plus tiny smoke/resume checks and frozen diagnostics.
At most four additional1024-update fits may follow one separately registered
evidence-motivated repair. No default adoption or larger encoder/decoder from one
test. Record CPU time and parameters; no equal-compute claim against future changed
architectures. Known/withheld combinations reuse the existing study and remain a
diagnostic population, not a fresh real-world confirmation.

Claude reviews use generic public conceptual briefs only. Corrections to the first
review: interpolation is informative; requesting later times is not forecasting
if the source saw the target frames. Forecasting requires a preceding observation
cutoff. Oracle output success and frozen linear accessibility are scoped evidence,
not proof of general temporal understanding. Browser QA remains unavailable under
the earlier local-file restriction; use existing renderer, structural/media checks
and static visual inspection, without alternate transport.

## Iteration 2, registered after the first eight fits

Query-side timing fails the paired foreground benefit: seed7201 held-out MSE is
0.10895 versus0.10885; seed7202 improves0.18017→0.12376. Oracle motion direction
reaches100% with query timing, but quality/composition gates remain failed. Actual
video-state decoding remains poor. Preserve all results and the context default.

One output-side follow-up: an optional learned four-entry RGB palette from the
untimed latent context; the existing patch-query attention predicts spatial/time
mixture weights. No explicit color labels, masks, position formulas or true factors
are read by this module. Training remains ordinary normalized pixel MSE. This biases
appearance sharing across positions/times; it is not a universal natural-video
representation. It adds parameters and changes factorization, so improvement cannot
uniquely establish a mechanism or an equal-compute advantage.

Four additional1024-update fits: both seeds, oracle and frozen video-only context,
query timing plus palette4, compared to the saved query/direct-RGB controls. Same
initial attention trunk, optimizer, batches and populations; new output projections
necessarily differ. Require the same >=20% held-out oracle foreground benefit and
<=0.005 known foreground regression in both seeds; original quality/motion capability
gates remain. Record palette/mixture collapse and context controls. Stop model
training after these four fits. No automatic default adoption.
