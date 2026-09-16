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

## Result, 16 September

[Complete sequences and metrics](../runs/video_readout_v1/report.html) ·
[frozen temporal-access challenge](../runs/video_readout_v1/temporal_access/report.html).
All12 registered1024-update fits finished,65.75 seconds measured CPU training,
excluding checks/reporting. Native/context and query/direct-RGB decoders both have
7104 parameters; palette4 has7804. No core/encoder/non-video weights changed.
The fresh native oracle runs reproduce the earlier1024-step weights and complete
training rows exactly. Neither experimental branch becomes the default.

Held-out oracle results (correct factors supplied; not agent performance):

| Decoder | Foreground MSE, seed7201 /7202 | Motion direction, seed7201 /7202 |
|---|---|---|
| Time in source context |0.10885 /0.18017 |83.3% /100% |
| Time in query |0.10895 /0.12376 |100% /100% |
| Query + learned palette |0.22998 /0.22553 |0% /0% |

Query timing improves foreground error31.3% in one seed and worsens it0.1% in the
other, failing the paired20% benefit screen. Known oracle quality passes for both
direct-RGB variants, but withheld quality/composition still fails. All six combined
quality/motion capability screens (three variants × oracle/frozen) fail. With actual
video-only states, context/query achieve only0% all-factor correctness on withheld
combinations and18.75–33.33% motion accuracy. Moving time cannot repair that upstream
state limitation on its own.

The palette branch collapses its spatial/time mixture fields. Palette entries are
still distinct (mean pair distance about0.39–0.43), but most pixels in a clip select
essentially one entry; known-frame spatial RGB standard deviation is at most1.1e-6.
Thus a check of palette diversity alone would miss the failure. Raw colors, mixture
fields, entropy, spatial/time variation and usage are retained in palette_inspection.*.
Low entropy is not independently proof of a cause, especially with sparse foreground.
The failure is consistent with a near-constant rendering solution under pixel loss;
no alternative loss/initialization was tested after the registered budget ended.

The new temporal challenge uses symmetric five-frame objects with exact reverse
pairs, identical center frames and identical unordered frame sets. Train/validation
use centers5/7/9; test uses6/8/10. Frozen readers are selected using validation only:

| Test direction access | Seed7201 | Seed7202 |
|---|---:|---:|
| Geometric centroid displacement reference |100% |100% |
| Ordered raw pixels, linear reader |83.3% |88.9% |
| Single middle frame or mean frame, linear reader |50% |50% |
| Video encoder features, linear reader |77.8% |69.4% |
| Final thought tokens, linear reader |38.9% |27.8% |

This separates temporal evidence from the arrow-orientation shortcut. It does not
prove irreversible loss: even raw linear readers fail some new positions, reader
dimensions differ, and the source model was not trained on this new challenge.
Each test seed has18 base trajectories paired with reversals, not36 independent
trajectories. The original oracle test has12 known and6 withheld target patterns
with repeated views, not48 independent scenes.

User architecture clarification: the general agent's video branch uses its own
MultiScaleImageEncoder(video=True). A shared framewise ImageEncoder patch stem is
followed by position/time features, causal attention and adjacent-frame pooling.
Image and video branches share this architecture, not parameter objects. The
experimental spatial image VAE is a separate path. No persistent recurrence across
arbitrary video windows or natural-video codec integration was added here.

Verification:58 scoped tests;2163 exact query-resume and2175 exact palette-resume
checks;28502 independent metric/model checks;210 NumPy temporal-reader checks.
The independent ridge solver records234 prediction differences only at numerical
ties (maximum4.32e-11), all in the intentionally uninformative middle/mean-frame controls; saved score
arithmetic and validation selection match. Raw observations, probe coefficients,
predictions and original failed candidates are preserved. Static sequence panels
inspected; report HTML/PNG/GIF checked. Browser interaction remains unavailable.
Three actual-Claude public-methodology reviews; no private code/data/results sent.
Claims that interpolation is trivial, that reverse pairs need identical encodings,
or that entropy alone identifies a mechanism were not adopted.

Next: return to shared-state direction/temporal learning using the symmetric
sequence challenge, with direct-encoder versus state-mediated reads kept separate.
Output composition/color remains an independent unresolved question. A spatial-VAE
video codec should be a separately controlled integration experiment, not silently
substituted into this categorical agent. No further training in this completed slice.

Reproduction examples (fresh output directories):

```bash
.venv/bin/python experiments/modality_readout.py --stage oracle --core runs/modality_readout_v1/formal/seed7201/core --modality video --video-conditioning query --steps 1024 --seed 7201 --device cpu --output runs/new_video_oracle
.venv/bin/python experiments/modality_readout.py --stage frozen --core runs/modality_readout_v1/formal/seed7201/core --modality video --input-mode video --video-conditioning query --video-palette 4 --steps 1024 --seed 7201 --device cpu --output runs/new_video_palette
```

The second command reproduces the failed experimental branch, not a recommended
default. Omitting both video flags retains the original context-timed RGB decoder.

Final artifact audit:170 structural/media/atlas checks across18 reports, six Claude
receipt/hash checks. All embedded PNGs and GIF frames decode; existing discussion
colors and validation scopes remain unchanged. The full-sequence summary panel was
rearranged into two rows of three cases for legibility, with all cases retained.
