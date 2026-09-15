# Current work

**Video comparison in progress, 16 September:** user requests video before the
remaining direction work. [Active protocol](video-readout-plan.md): distinguish
output conditioning from upstream temporal access; compare the same decoder with
requested time in context versus queries, oracle and frozen video-only states.
No additional model parameters; direction rotation/dropout remains pending.

**Direction input isolation and curriculum completed, 16 September:**
[Protocol/results](modality-readout-plan.md#input-isolation-and-curriculum-result-16-september),
[comparison](../runs/direction_inputs_v1/report.html),
[source-removal control](../runs/direction_inputs_v1/robustness_curriculum/report.html).
Six fresh fits plus four checkpoint continuations; shared latent architecture and
source encoders unchanged. Text-only known direction reaches100% in both seeds,
stable over10 sampled draws; audio46%/100%, simultaneous-all48%/60% in the original
evaluation draw. Text384→all768 beats matched all384→all768 by52/17 percentage
points, but complete-input capability still fails in one seed. Across10 draws,
text-warm-start complete inputs score100%/74%; without text50%/51%. This is a narrow
learning improvement relying on text, not reliable cross-modal transfer. Frozen
audio grids and their means retain100% known-direction linear access in both seeds;
learned first queries give40%/98%. No unique causal loss layer established.
Optional input selection, explicit task scopes and exact checkpoint-file starts
reuse the existing recipe; no additional model parameters or default change.
61 scoped tests, exact fresh/continuation resume, raw artifact and repeated-draw
audits pass. Five brief actual-Claude methodology exchanges; no private data exported.
Next bounded comparison: gradual source rotation/dropout after a fixed warm-up,
requiring direction from each informative modality, with matched restart/exposure
controls. Not yet run. Video remains deferred at the user's request; no new output,
memory or general modality claim. Static/HTML structural QA; browser QA unavailable.

**Fresh direction learning completed, 16 September:**
[Protocol/results](modality-readout-plan.md#result-16-september),
[comparison](../runs/direction_learning_v1/report.html),
[localization](../runs/direction_learning_v1/localization/report.html).
Eight matched fits: frozen verified encoders, fresh updater/readout, direction-only
versus all-factor training, sampled versus continuous posterior working access.
All eight task screens fail; direction-only averages52–62% and continuous access
50–63% across six known input modes. No default replacement. Frozen linear readers
recover all factors at100% from the encoder, but direction only60–75% from updater
queries. This narrows the learning/access problem without proving a unique loss layer.
Next: isolate a single informative modality before mixing modes and altering query
pooling.59 scoped tests, exact hard/continuous resume and independent artifact audits
pass. Four actual-Claude public-methodology exchanges reconciled;450s CPU training.
No new decoder, persistent-memory or general modality capability claim. Static/HTML
structural QA completed; browser interaction unavailable.

**Direction literature review, 15 September:**
[Primary-source review and proposed comparisons](direction-literature-review.md).
Categorical estimators, multi-task interference, paired-factor supervision and
attention pooling provide relevant candidates. A May2026 VQ warm-up preprint is an
analogy, not a theorem for our sampler. Current core classification has no KL loss;
standard KL-collapse fixes do not directly apply. Proposed: isolate direction learning,
then compare continuous/hard training and targeted gradient/query changes with matched
controls. Two public-only Claude methodology exchanges reconciled. No new model runs,
architecture adoption or green capability promotion.

**Multimodal diagnosis and bounded repairs completed, 15 September:**
[Protocol/results](modality-readout-plan.md#follow-up-result-15-september),
[repair report](../runs/modality_repair_v1/report.html). Frozen stage probes show
that combined-input encoder features support held-out factors much better than the
posterior/sampled state. All8 longer native oracle fits pass known-output screens;
held-out text/image and video quality still fail. Raw-logit training supervision
improves known position from71–73% to97–100%, but direction stays near50% and held-out
color worsens. All8 complete core screens fail; no default replacement. Probability
auxiliary and temperature curriculum fail paired benefit.77 scoped tests, independent
raw audits and exact auxiliary/curriculum resume pass. A read-only
[direction follow-up](../runs/modality_repair_v1/direction_localization/report.html)
finds69–90% known-direction access in posterior probabilities but50–54% in sampled
codes, with measured code collisions. Trained pre-sampling auxiliary heads also
fail direction; this is not exclusively a sampling explanation. Next: paired
direction-only updater learning from initialization with a frozen verified encoder,
then joint retention/composition and decoder recombination; not yet run.


**Multimodal readout comparison completed, 15 September:**
[protocol/results](modality-readout-plan.md),
[combined report](../runs/modality_readout_v1/formal/report.html). Two seeds,
native versus adapter1/2/4, separate frozen-state outputs then joint core/output
training:42 runs, plus8 explicit-factor output controls and2 linear probes.
All192 simultaneous-output screens fail; held-out all-four-correct is0% throughout.
Core color is more readable than location/direction. With oracle facts, audio
passes symbolic-tone recombination; text fits known strings but fails recombination,
and native image/video remain weak. Recurrent adapters are optional, not adopted as
a repair. Source/mask/gradient checks and exact GPU resume pass; raw reports preserve
negative results. Next: upstream location/direction diagnostics and output-side
recombination/spatial-conditioning controls. No general language or media capability.

**Latent-core visualization, 15 September:** [source-grounded walkthrough](latent-core.md)
and atlas §14–15 distinguish world belief, recurrent workspace, session memory and
optional graph context. Thinker already repeats shared attention/MLP; only workspace
changes. Native decoders read all state tokens through modality-specific attention.
Alex clarifies that shared thinking must remain multimodal; output branches should
learn to extract what they need, optionally with their own adapter. No mandatory
shared text/answer-plan bottleneck. Compare existing decoder reads against a small
adapter before adding one; preserve evidence and evaluate output/grounding plus
total resources. No model change, training or new capability validation;
discussion/green scopes are preserved. Nested shared-weight Transformer loops are
an allowed adapter variant: existing outer Thinker, optional inner modality readout
refinement with fixed budgets first. No nested loop or adaptive stopping added.

**Non-image modality interfaces repaired and measured, 15 September:**
[protocol/results](modality-foundation-plan.md),
[four-modality report](../runs/modality_foundation_v1/direct/report.html),
[video comparison](../runs/modality_foundation_v1/video_weighted/report.html).
Foundation now connects image/video/audio/text with masked candidate pooling,
source attribution and the actual memory/thinker. Fixed a four-code/eight-code
memory mismatch, decoder masked-NaN handling, invalid video-state acceptance and
multiscale tracing that changed the numerical attention path. Shared diagnostics
now preserve outputs, gradients and RNG exactly in all four encoders.
112 scoped tests and 6075 independent artifact checks pass; full checkpoint matches
64+64 resume exactly. Four-example direct fitting learns four words and four tones.
Video ordinary continuation gets 3/4 motion directions; object weighting gets 4/4
but worsens total RGB/background and fails the registered repair screen. Weight1
remains default. Real local audiovisual/UTF-8 inputs pass the storage/retrieval
path after documented resizing/resampling; no real-media capability training.
State-to-output adapters remain untrained in this audit; language understanding,
speech, video forecasting, cross-modal identity and live hardware remain open.
Static figures and embedded report media checked; browser QA unavailable under the
existing policy. Next: train/test state-conditioned outputs separately, then held-out
real-modality tasks. Earlier VAE quality/rate and entity-calibration work stays open.

**World State foundation implemented, 15 September:** [guide/API](world-state.md),
[protocol/results](world-state-foundation-plan.md),
[debug report](../runs/world_state_foundation_v1/final/report.html).
Versioned entity/component/relation/evidence store; atomic commits, revocable aliases,
dependent-state invalidation/replay, bounded retrieval, interchangeable neural modules
and actual BeliefAgent connection. Optional concept/self/feedback/prediction/selection
clients and bounded training/inference diagnostics are usable; general learned behavior
remains experimental. 81 focused tests and 3293 independent checks pass. A CPU 96-update
exercise learns paired histories; full checkpoint matches 48+48 resume exactly.
Attention tracing no longer changes native outputs/gradients; internal commits keep
the store and core clock aligned. Structural report QA only under existing browser
policy. Next: real candidate/identity/state tests and calibration; VAE work stays open.

**Persistent multimodal World State review, 15 September:**
[assessment and proposed milestones](world-state-proposal-review.md). Two actual
Claude review rounds support the hybrid entity/component/relation/evidence design.
Source review finds existing bounded entity/state/relation paths, but no unified
persistent multimodal store. Recommended first milestone: confusable instances,
state/history queries, bounded retrieval and restart/correction with provenance
from the start. Concept induction is a separate second milestone. Learning signals,
candidate granularity and quantitative gates still need an experiment contract.
No universal existence decay from non-observation; merges must retain attribution
for correction. Discussion only: no model changes, new training or capability
validation. Existing VAE quality/efficiency work below remains open.

**Efficient sampled-color repair, 15 September:** [protocol/results](spatial-vae-repair-plan.md),
[report](../runs/spatial_vae_repair_v1/report.html). Training-only color supervision
on the same model reduces sampled global color error81–83% and raw RGB error26–27%
in two seeds. No added inference work or parameters; training8–10% slower. KL rate
rises37–41% and grids remain, so full repair/rate screens fail. Phase residual loss
adds no consistent benefit. Color-only weights are a diagnostic candidate; original
weights retained. KL evaluated once instead of four times; shared report/color code
and cached extra draws reduce duplication. CPU loss26% faster; AOT capture/compile
microbenchmark10% shorter, not a whole-model speedup.50 tests and independent audit
pass. Next: decoder phase repair and matched achieved-rate tests; high-level codec
quality and agent integration remain open.

**VAE color/grid diagnosis, 15 September:** [results and controls](spatial-vae-color-plan.md),
[report](../runs/spatial_vae_color_v1/report.html). Constant latent fields generate
a 2/4-pixel grid at decoder shuffles, even without noise; learned phase channels
are unequal. Color accessibility drops strongly at24->4 posterior projection and
again at sampling. Extra512-update encoder-only/decoder-only/joint sampled training
changes global color error by only0.5–3.5%. Mean-only decoder training improves it
37.6% with encoder fixed, but sampled RGB error worsens to0.09944; diagnostic only,
not adopted. Grid remains.45 tests and independent data/weight/metric/replay audit
pass. Default-vs-IEEE evaluation mismatch repaired without changing four trained
weights; use reference_fp32/follow-up evaluation_fp32. Next: phase-consistent decoder
initialization and color/rate/noise interventions with real sampled-output checks.

**R/P/M/C VAE v2 implemented and tested, 15 September:**
[Protocol, commands and results](spatial-vae-v2-plan.md),
[comparison report](../runs/spatial_vae_v2/formal/report.html). Overlapping stem,
separate shuffle/processing/compression, coarsest self-attention with shared loops,
spatial posterior, latent-only decoder, detached stage probes and strict old/new
exports work. Actual Claude reviews were reconciled; private code/data/results
stayed local.42 focused tests and exact GPU resume pass;2203 artifact checks pass.

Ten512-update fits on512 new real training photos,64 validation and96 test photos
pass the learning sanity gates but all fail the photo-quality screen. C/beta0.1:
mean MSE0.017132, sampled0.018344. Lower KL pressure helps this short run more than
attention/loop changes; processing before and after compression is essentially tied
at beta1. Unequal rate/compute/parameters and one seed prevent a superiority claim.
Common RGB probes show reduced access after each channel compression, even when
feature-variance reconstruction looks good; this is not proof of irreversible loss.
58 reports verified structurally, example/error/rate figures inspected; browser QA
unavailable due to local-file policy. Formal training totals70.75s, GPU reserved
peak162MiB (allocator cache included), artifacts~388MiB. Codec stays separate from
the agent. Next: controlled duration/latent-capacity and matched-resource comparisons;
fine detail, semantic utility and state-to-latent generation remain open.

**Spatial image VAE implemented and compared, 15 September:** the
[explicit-scale codec](spatial-vae-design.md) now has base, cross-scale attention,
and attention-plus-reversible-local-mixing variants. It separates PixelUnshuffle,
processing and channel projection, keeps a spatial Gaussian posterior, and decodes
without encoder skips. Current agent components remain unchanged.
[Report](../runs/spatial_vae_v1/report.html), [protocol/results](spatial-vae-plan.md).

Matched 512-update fits on1024 real COCO photos,128 validation and192 test photos:
RGB64 mean MSE base0.019200, attention0.019419, reversible0.018487. Attention alone
worsens error1.14%; reversible improves4.79% versus attention, below the predeclared
5% minimum despite a positive paired interval. All three fail the combined photo
quality screen. Native128/odd reconstruction and mild instance-retrieval screens
pass; raw pixels also retrieve100%, so this does not validate semantic recognition.
Patterns and first-eight photo examples show severe fine-detail loss. Larger native
crops up to192×256 retain more structure but stay blurred; input area, latent size
and crop content change together. Prior samples are color blobs, not validated
general images. The validation-only collapse trigger was false; no tuning on test.

25 focused CPU tests, exact GPU pause/resume and880 independent source/weight/output/
numeric checks pass. Three fits total17.93s training-loop time; evaluation21.36s;
maximum formal GPU reserve164MiB. Small deterministic overfit control passes.
Two actual Claude method-review rounds reconciled technical errors; private
code/data/results stayed local. Report inspection found custom image outputs were
hidden by the standard gallery; explicit labelled panels now repair all38 reports
without changing weights or metrics. Structural checks and scientific-figure QA
pass; browser interaction QA is blocked by local-file URL policy.

Next proposed comparison: training duration and KL pressure separately at fixed
geometry, then latent capacity. Keep the base as a simple reference; reversible
mixing remains a candidate, not an adopted default. One short seed cannot establish
sample efficiency or architecture limits. State-to-latent generation, photo state/
recall repair and other modalities remain open.

**Architecture atlas, 14 September:** [thirteen source-grounded drawings](architecture-atlas.html)
now cover the overall loop, encoders, attention, belief state, memory, tasks,
decoders, photo generation, entities, planning and learning signals. General,
photo-specific and proposed graph/DAG paths are explicitly distinguished.
[Editable diagrams and notes](architecture-atlas.md). Documentation only; no model
or weights changed. Graph structure and local links checked; SVGs inspected;
browser interaction QA blocked by local-file URL policy.
The overview now tracks discussion coverage: red = a dedicated walkthrough remains,
blue = discussed; green = validated within a labelled test scope.
Green now marks event mechanics, bounded memory storage/causal reads, bounded search
mechanics and reflection routing, with saved evidence and limits. This does not
validate learned world prediction, general recall or output quality. Event handling
retains its pending discussion label. [Checklist and evidence](architecture-discussion.md)
keep the status revisable; update it as each topic is discussed. Initial red areas:
observation adapters, categorical belief updating, task contracts and action execution.

**Photo-detail path localized, 14 September:** the largest measured drop in spatial
accessibility is the first observation-to-state update. Matched linear and RBF
readers use 1,024 training/128 validation photos and a fresh 256-photo test suffix.
They reconstruct a 16×16 RGB layout, separating spatial structure from finer texture.
[Report](../runs/photo_detail_v1/report.html), [protocol/results](photo-detail-plan.md).

Linear grid MSE: encoder 0.00000706 → first observed state 0.033545 → second
state 0.034603 → recalled workspace 0.043975. Nonlinear readers corroborate the
main drop. Stored snapshots are bitwise copies. Both readers also flag a further
drop between the complete stored state and the reset/recalled state. The image
encoder supplies 336×32 features; observation updates compress to 30×32 state,
and recall updates only the 8×32 working/reasoning tokens after reset.

Fine detail has a separate constraint: the actual 4×4 RGB patch projection maps
48 values to 32 outputs, has 16 null directions, and its top three singular
directions hold 99.99999926% of squared weight magnitude. Tiny nonzero directions
are poorly conditioned, not declared absent. The good direct codec additionally
gets raw residual detail that never enters the agent. On fresh photos, full-image
PSNR is 31.18 dB with that channel, 19.74 dB without it, and 12.84 dB for native recall.

No agent weights changed. Closed-form readers remove probe SGD duration as a
confound, but family/sample/dimension limits prevent proving all lost information
or ruling out larger/longer-trained decoders. Six focused tests, 17 exact reader
reloads, 52 exact GPU activation/control tensors over 32 photos, 157 independent
numeric checks and 63 source snapshots verified. Extraction18.81s/86MiB, fitting
10.61s CPU. Three reports structurally checked; figure inspected, browser QA unavailable.
Actual generic Claude method review completed; private code/results stayed local.

Next proposal: train the observation/state update and recall pathways on real-photo
spatial targets with intermediate readouts. The previous generator-only training
left these modules frozen. Test finer input-detail transport separately; size and
duration comparisons remain open. No repair is claimed by this diagnostic.

**Real-photo training completed, 14 September:** continued the existing own image
generator on 1,024 local COCO photographs, with 128 validation and 256 test photos
held out from this continuation. Encoder, state/memory and codec stayed frozen.
[Report](../runs/real_photo_v1/report.html), [protocol and results](real-photo-plan.md),
[weights](../runs/real_photo_v1/training/weights.pt). Original runs are preserved.

After 2,048 updates, test reset MSE falls 0.09674→0.05867 (39.4%), PSNR
10.34→12.73 dB. Ordinary image error falls 41.4%. Reset error is 15.8% below the
training-mean baseline and 26.8–49.8% below wrong-memory, blind-history and erased
controls; swapped outputs track the swapped target. The predeclared learning and
context screens pass. However, the first eight fixed examples show mostly broad
color fields, not recognizable photographs. Direct codec reconstruction reaches
30.92 dB: it receives the image and carries residual detail directly, so this is
an adequacy control, not a compact-memory result. General generation remains open.

Same architecture, 314,576 trainable generator parameters, 370.03 training seconds,
700 MiB peak GPU reserve. Fifteen focused tests, exact GPU pause/resume and four
GPU reload comparisons, 160 independent metric values, all 520 frozen tensors and
60 source/snapshot files verified. Seven reports structurally checked; scientific
figures inspected; browser QA unavailable. Actual Claude method review used the
user-approved brief; local code/results review and remaining disagreements recorded.

Next proposed diagnostic: compare recoverable real-photo detail at encoder features,
stored state and final workspace using matched readers, then train the implicated
path. The current result does not identify a unique loss location or prove that a
larger generator is needed. The earlier synthetic placement and factual-readout
proposals remain separate, unadopted next steps.

**Decoded-image supervision implemented and iterated, 14 September:** an optional
image loss now differentiates through the frozen decoder, either from a one-step
endpoint estimate or through the actual eight-step sampler. Your agent and codec
stay unchanged. [Report](../runs/decoded_image_v1/report.html),
[protocol/results](image-output-plan.md#decoded-image-supervision-results).

At1024 updates each, adding endpoint image loss lowers fresh weighted pixel MSE58.0%
and increases familiar reset accuracy13.28%→55.47%. One withheld cell regresses;
primary repair, nonregression benefit and full capability fail. A validation diagnostic
finds familiar first-endpoint accuracy87.5%, versus62.5% after eight sampling steps.

A separate matched512-update comparison trains through all eight steps: pixel error
falls44.4% versus endpoint supervision, with no joint-category regression in any
ordinary/reset seen/withheld cell. That benefit passes, but familiar accuracy45.31%
and withheld reset14.84% remain below capability requirements. The original renderer
remains default. Post-hoc attribute checks reveal a tradeoff: reset color73.44%→90.23%
and shape46.09%→64.84%, while side64.45%→54.69%; about half the generated contrast is
on the wrong side. Lower image error does not mean reliable placement.

Four fits3072 updates215.95s, maximum GPU reserve566MiB.12 focused tests,16 exact
64-history GPU confirmation replays plus4 validation reloads,912 independently
reproduced metrics; all513 frozen source tensors and all source snapshots verified.
24 reports structurally checked and scientific figures inspected; browser QA unavailable.
Four actual public-method Claude reviews, concrete misreadings corrected in receipts;
private code/results reviewed locally. Sourcea37b994/46a392c; original runs preserved.

Next proposed repair: add a training signal for spatial placement/occupancy to the
state-conditioned generator, without target-derived input at inference. Keep the
same data and hard tests for a matched comparison. Perfect color/shape coupling in
this generator's training set is a separate generalization risk; don't silently
relax the withheld test to obtain a pass. General imagery, arbitrary prompts and
other modalities remain open; no population-optimum or unique-cause claim.

**Conditional image generator implemented and compared, 14 September:** optional
residual transformers per scale and cross-scale attention produce all image-decoder
features from the existing workspace. Agent and own codec stay frozen; no pretrained
download or default replacement. [Report](../runs/conditional_image_v1/report.html),
[implementation/results](image-output-plan.md#conditional-generator-results).

Both new methods fail the declared capability gates. Direct regression reaches100%
on combinations seen by the new generator and0% on withheld combinations. Uniform
flow sampling is poor even on seen cases. A separate pure-noise-weighting fit lowers
fresh weighted pixel error25.7% but still fails faithful generation. Original native
outputs remain100% in these neutral reference cells; its upstream/renderer already
saw all categories, so it is descriptive rather than the matched training control.
All factual answers and the frozen state/memory/codec remain unchanged.

Three1024-update fits,314,576 new trainable parameters each,84.73s formal training,
peak1116MiB.60 distinct scoped tests,1026 independently reproduced metrics,18 exact
GPU confirmation replays and3 validation reloads.26 reports structurally checked;
figures inspected, browser QA unavailable. Four actual public-method Claude reviews;
one terminology disagreement retained. No private code/data/results exported.

Next proposed comparison: decoded-image supervision through the frozen codec,
holding the generator and progress policy fixed; retain pure-noise and withheld
binding checks. Latent loss alone was insufficient at this budget; this is not an
intrinsic architecture or capacity limit. General photographic generation, arbitrary
text requests, compact memory and other modalities remain open. The separate factual
readout optimization proposal below is still pending.

**Factual readout comparison completed, 13 September:** the proposed joint continuation
fails its declared benefit. Warm/texture factual errors increase10→14/512 correlated
responses. Both fits and unchanged source pass12/12 task gates on fresh64-history cells,
but full repair remains false. Retain the prior source; no jointly trained checkpoint
adoption. [Report](../runs/factual_readout_v1/report.html),
[protocol/results](recall-repair-plan.md#factual-readout-repair-results).

Joint factual retention10/12 versus12/12 image-only. Warm recall facts92.1875–93.75%
versus95.3125–96.875% control; joint cool facts100%. Images are bitwise identical across
fitted arms: warm recall90.625–93.75%, image retention10/12. Saved training warm/texture
cross-entropy worsens too, while cool improves. A more conservative optimization test
is warranted; neither a unique cause nor a successful lower rate is established.

Added opt-in separate gradient clipping and enabled box-weighted joint cached training.
The branches are parameter-disjoint; factual learning does not directly change image
semantics. Image weights/AdamW states and all1536-step image loss/norm records match
exactly. Image clipping0 steps; joint factual clipping307 steps. No new architecture
or parameters;67,032 trainable joint versus58,128 image-only.

61 tests,10,188 independent metrics,40 exact64-history GPU reloads. Paired16-step GPU
development additionally verifies gradients and optimizer updates at every step. The
initial raw-logit preflight mismatch was corrected to use the existing evaluator's
trainability convention before any training; receipts preserved. Training119.15s,
peak646MiB. Three actual Claude reviews reconciled;45 reports structurally checked,
figures inspected. Prior browser local-file restriction remains a QA limitation.

Next: compare a lower readout learning rate with the current rate from the preserved
source, with equal data/updates and separate clips; add difficult-scene loss diagnostics,
fresh confirmation and unchanged retention/causal gates. Prior [refinement](../runs/producer_refinement_v1/report.html)
and [loss correction](../runs/recall_shape_v1/report.html) remain preserved. General scene
robustness, real imagery, learned reliability, upstream replication, CPU portability,
streaming and general generative backends remain open.

**Scene/lighting challenge and reporting repair completed, 13 September:** centered
input passes12/40 attempted cells (neutral, temporal additive and RGB-channel
additive); raw passes8/40 (neutral and textured background). All those successes
reach100% ordinary/reset factual/image categories. Centering hurts texture:
raw100% becomes76.562–86.719%. Clutter, larger objects, gain, tint and shadows remain
unreliable. Bright backgrounds fail centering coverage; accepted histories only
15.625–21.875% on screen seeds, with no accepted-subset accuracy reported.
[Report](../runs/centering_challenge_v1/report.html),
[protocol/results](recall-repair-plan.md#centering-challenge-results).

The ordinary evaluation recipe now writes input_coverage.json and completes an
explicit coverage-failed report before any model query when centering rejects a
history. No fabricated predictions, clipping, fallback or learned confidence.
Two valid full GPU exports remain bitwise exact after this repair; two fresh
rejection exports correctly report12.5% history coverage. Centering stays opt-in.

92 distinct relevant tests pass;45 focused reruns after repair.76 scored GPU replays
exact;18,012 metrics independently verified;12 legacy data cases byte-exact.80-cell
screen571.93s/298MiB, four repair checks16.37s; zero optimizer updates. Two actual
Claude public conceptual reviews with corrections reconciled; private code/results
reviewed locally.86 reports structurally checked, figures inspected, browser QA
unavailable. Separate screen311c3ca and repair716d661 source identities retained.

Next: controlled state/readout learning with scene variation and clean retention;
centered tint has100% direct-reader accuracy yet weak native answers, so no unique
encoder-loss conclusion follows. Independent upstream replication, confidence,
real recordings, CPU portability and deployed runtime tests remain open.


**Brightness repair comparison completed, 13 September:** opt-in frame centering
passes all 12 confirmation conditions with 100% factual/image-category accuracy,
including full causal task gates, neutral retention and stress benefit. It uses
unchanged source weights and a fixed RGB reference from verified neutral training
observations. Available as `--evaluate-only --center-input`; it is a task-specific
preprocessing diagnostic, not learned invariance or a default deployment policy.
[Report](../runs/brightness_repair_v1/report.html),
[protocol/results](recall-repair-plan.md#brightness-repair-results).

Matched output-head augmentation gives partial improvement but fails repair:
4/4 neutral task cells pass, 0/8 stress task cells pass; neutral retention passes
only 1/4 cells and stress benefit 6/8. Neutral-only continuation retains 100%
neutral answers but fails every stress gate. Augmented heads also underfit seen
training extremes, so failure does not uniquely diagnose an encoder problem.
Both explored source checkpoints share one upstream initialization/codec; fresh
seeds23073/23074 and RGB offsets−12/0/+12 test familiar synthetic scenes and
perturbation interpolation. Earlier negative screens remain preserved.

Four matched 1,536-update fits train only 67,032 factual/image-feature parameters;
all state/encoder/backend tensors stay fixed. Training139.52s, peak712 MiB; 48-cell
evaluation/report loop365.40s. All78 relevant tests pass;13 focused checks rerun
after fixing CUDA median incompatibility with deterministic sorting. All48 GPU
replays and original default exports are exact;12,536 metrics independently
verified plus80 descriptive training-offset accuracies. CPU numeric1e-4 fails
all48 cells; native categories differ in11, none centered. Two public-only Claude
reviews, private implementation/results reviewed locally. All53 individual reports
and overview structurally checked, figures inspected; browser QA unavailable.

Next proposed: challenge the centering reference with varied backgrounds, object
coverage and nonuniform illumination before making it a default. For learned
robustness, diagnose the head/state optimization shortfall with a fresh protocol;
no encoder replacement follows from this fixed-budget result. No further fit or
deployment launched.

**Fixed-checkpoint nuisance screen completed, 13 September:** both continuation
policies pass all six neutral cells, but none of their twelve brightness-stress
cells or five-point retention checks. Joint continuation retains 100% facts/images
on fresh background seeds 20073–20075. Image-only neutral ordinary facts/images are
96.094–100% / 98.438–100%; reset 95.313–100% / 96.875–100%. With observed RGB offsets
−16/+16, ordinary facts fall to 31.250–65.625% and images to 31.250–57.031% across
trained checkpoints. Targets/labels stay unchanged, no pixels clip, foreground
color identity and history ambiguity are preserved. These are familiar synthetic
objects from one upstream initialization/codec, not independent model replication.
[Report](../runs/output_robustness_v1/report.html),
[protocol/results](recall-repair-plan.md#fixed-checkpoint-nuisance-robustness-results).

Added evaluation-only `--input-offset` with clipping/type guards and recorded
transform provenance; default data identities and inference/training code remain
unchanged. All 67 relevant tests pass; 12,798 metric values and all 54 task gates
independently verified. All 36 trained-model GPU replays are exact; source/factual
freeze and sampled memory/workspace invariants pass. CPU numerical tolerance fails
in all 36 cells and native categories differ in 15 stressed cells; portability is
still open. All checkpoints stayed fixed: zero training, 449.78s evaluation/report
loop, 296 MiB peak GPU reservation. Two public-only Claude reviews; implementation
reviewed locally. Fifty-five individual reports and overview structurally verified,
representative figures inspected; browser QA unavailable.

Next proposed: separately declare a brightness augmentation comparison and input
normalization diagnostic, with neutral retention and fresh confirmation data. Facts
and the frozen direct reader also fail under shifts, so a decoder-only explanation
is insufficient; a failed direct reader does not prove encoder information is gone.
Do not reuse these diagnostic populations as untouched confirmation. No repair
training or normalization change has been launched.

**Image-output continuation completed, 13 September:** image-only training on both
frozen final joint states reaches100% ordinary/reset image answers on fresh test19073.
Joint continuation reaches100% facts and images for both. Image-only preserves
source8502 facts at97.656/99.219%; source8501 facts remain100%. All full task and
separate sufficiency/retention gates pass. Unchanged sources already pass the minimum
gate here, with images91.406/97.656% and92.188/90.625% ordinary/reset. This demonstrates
conditional output recoverability and reduced residual error; both trajectories
share one explored upstream initialization/codec. Earlier7793 failures remain valid.
[Report](../runs/image_continuation_v1/report.html),
[protocol/results](recall-repair-plan.md#image-output-continuation-results).

Added opt-in `--image-only` and found/fixed a GPU inference consistency issue:
parameter trainability flags affected numeric outputs under no_grad. Evaluation
now freezes flags temporarily and restores them even on failure. Original invariant
failure preserved; six unchanged exports rescored with zero extra optimization.
All categorical answers/gates unchanged. GPU exports, frozen states, factual logits
and memory provenance now exact under the corrected inference contract. CPU1e-4
numeric tolerance still fails (logits0.185367/pixels0.002075), but native categories
agree on this sample. Direct model calls require matched inference flags.

All58 relevant tests and3,056 independently recomputed metric values verified.
Four fits507.67s training,404MiB peak GPU reservation. Two bounded public-only Claude
reviews; implementation and repair reviewed locally. Fourteen individual reports
plus overview structurally verified, figures inspected; browser QA unavailable.
Next proposed: hold both policies fixed for a fresh declared robustness evaluation,
then expand history/observations. No further training or benchmark launched.

**Existing-source robustness comparison completed, 13 September:** the overall
joint-task robustness gate fails on fresh test 7793. All four new source7801 fits
(writer-only/joint × two optimizer seeds) reach 100% facts/images in ordinary and
reset modes and pass every task gate. Both unchanged source7802 joint checkpoints
fail this fresh screen: ordinary facts/images 99.219/86.719% and 92.188/88.281%;
reset 100/93.750% and 92.969/89.063%. Source-retention passes all joint cells.
Earlier test7783 passes remain scoped to that sample. Both sources were already
explored and share one codec; this is not blinded replication or new codec training.

Shape rendering is the main remaining error. In ordinary mode, the weaker joint
checkpoints have 16 and 8 cases with fully correct factual answers but incorrect
images. Color and side are 100% correct in both; shape is 86.719/88.281%. This
supports testing the image-output path, without uniquely locating a decoder fault.
[Comparison and error examples](../runs/cross_source_v1/report.html),
[protocol/results](recall-repair-plan.md#existing-source-robustness-results).

Added `--evaluate-only` to the existing recipe with immutable checkpoint provenance,
separate evaluation identity and report-failure preservation. Fixed nested JSON
settings comparison and misleading empty training charts. All 54 relevant tests
pass; focused regressions rerun after fixes. 2,794 metrics independently checked;
all eight GPU exports, frozen tensors, paired initialization/sampling and memory
provenance pass. CPU numerical gate fails (native logits0.258659/pixels0.009212);
one ordinary factual-shape answer changes, all image/reset categories agree here.
Four new fits: 515.28s training, 404MiB reserved. Four checkpoints reused; two unchanged
sources freshly scored. Three public-only Claude reviews; 12 individual reports
plus overview verified structurally, figures inspected, browser QA unavailable.
Next proposed: compare targeted image-output training on frozen final joint states
against matched continued joint training. No additional fit launched.

**Joint writer and workspace-reader learning completed, 13 September:** both new
fits pass the full controlled-task gate. Ordinary facts/images are 99.219/90.625%
and 93.750/92.188%; reset 100/95.313% and 94.531/91.406%. The matched writer-only
checkpoints fail the task gate on the same fresh test. The separate five-point
benefit gate passes only seed 8502: seed 8501 ordinary image accuracy is unchanged.
Thus replicated task success passes; the stricter combined repair-and-benefit claim
fails. These are two optimization seeds from one upstream model, familiar synthetic
objects and supplied snapshots; independent source replication remains open.

Opt-in `--train-thinker` adds workspace-reader learning to `--writer-learning trainable`
and live mixed ordinary/reset output training. Encoders, reconstruction backend and
other frozen components stay fixed; inference banks stay detached. All 52 relevant
tests pass, including joint exact resume. Default-off old exports reproduce exactly.
New GPU outputs and bank/replay checks are exact; 1,826 metrics independently checked.
CPU numerical equivalence fails: one joint factual-shape answer and one reused-baseline
image-side answer change in ordinary mode. Images still have residual artifacts;
template correctness is not general image quality. Browser QA unavailable; structural
reports and inspected figures retained. [Report](../runs/writer_reader_v1/report.html),
[protocol/results](recall-repair-plan.md#joint-observer-and-workspace-reader-results).

The preceding writer-only iteration produced partial gains but failed all full gates:
ordinary facts/images 83.59/89.84% and 86.72/82.81%; both reset 81.25/75% on its own
test population. Four fits took 481.75s; two joint fits took 369.15s. Total new formal
training this turn: six fits, 850.90s, 404MiB peak reserved. Two unchanged baselines
were freshly rescored for the joint comparison with original provenance preserved.
Four bounded public-only Claude design reviews; implementation reviewed locally.
[Earlier comparison](../runs/writer_learning_v1/report.html). Next proposed validation:
replicate the joint recipe across independently initialized upstream models before
expanding history length or distractors. No further training launched.

**Mixed ordinary/reset training completed, 13 September:** training only native
output heads on both workspaces repairs the stronger source on the controlled task.
Writer7801 ordinary facts/images96.875/96.875% versus82.8125/58.59375% with matched
reset-only training; reset97.65625/94.53125% versus97.65625/96.09375%. Full task and
policy-benefit gates pass for this source. Writer7802 remains75% in both modes,
even on training histories; replicated repair fails. No full-agent reliability claim.
Encoder, writer, thinker, reconstruction head, probes and raw banks remain frozen.
Four1536-update fits,140.83s training plus9.21s cache preparation including resume,
458MiB reserved.46 relevant tests and1,826 independently verified metrics pass;
GPU ordinary/reset exports and both live caches replay exactly,776-row restart
prefix retained. CPU numeric tolerance still fails (max logits1.104984/pixels0.026627),
but all native factual/image labels agree in this slice. Two public-only Claude
reviews, structural reports and inspected figures. [Protocol/results](recall-repair-plan.md#mixed-ordinaryreset-results),
[report](../runs/mixed_context_v1/report.html). Next proposed: test the weaker writer
and workspace formation while retaining mixed output training; no further run launched.


**Direct output-readout diagnostic completed, 13 September:** with writer AND thinker
frozen, head-only training on native reset tokens reaches96.88/98.44% facts/images
for writer7801 (raw),98.44/96.88% standardized. Both pass the predeclared reset-only
accuracy/pair/causal screen. Ordinary outputs regress to81.25/59.38% and68.75/50.78%;
all eight full gates fail. Direct stored-token routing scores78.91/95.31% raw and
75/94.53% standardized for7801;7802 stays near75–77%. No replicated route advantage,
normalization benefit or reset sufficiency. This identifies conditional native
readout success in one source, not absent state information or a complete repair.
Eight1536-update cached fits,296.76s training plus14.33s first cache preparation,
460MiB reserved.59 distinct tests and2,930 independent metrics pass. Frozen writer/thinker/
codecs/probes/banks unchanged; live shuffled-batch caches and all GPU exports exact;
774-row resume prefix retained. CPU numeric tolerance fails with five factual-side
differences across three7801 runs; image labels agree. Two public-only Claude reviews,
structural reports and all figures inspected. [Protocol/results](recall-repair-plan.md#direct-native-output-readout-results),
[report](../runs/direct_readout_v1/report.html). Next proposed: freeze upstream and
train native heads on both ordinary and reset workspaces; retain the weaker writer
as a separate accessibility problem. A stale-scaling reconfiguration bug was fixed;
all eight exports replay exactly afterward. No next training run launched.

**Frozen-reader supervision completed, 13 September:** adding a frozen stored-state
reader loss does not improve native recall over equal training. Writer7801 falls
from86.72/89.06% facts/images to84.38/86.72%; writer7802 remains74.22% for both.
All four full reliability gates and the replicated10-point benefit gate fail.
The optimized reader improves strongly on recalled tokens, and the second reader
also improves joint accuracy, but neither establishes better native output.
Stored-state accessibility remains stronger in writer7801 and reader-dependent.
Four1536-update fits,708.51s total training,424MiB reserved;49 relevant tests and
1,572 independently checked metrics pass. Frozen writer/codecs/reader and raw banks
unchanged; identical within-pair initialization/data/sampler, exact GPU exports and
bank-order workspace checks. CPU numeric tolerance still fails: writer7801 supervised
changes two factual-side labels and one image-side label. Two public-only Claude
reviews; standalone reports structurally verified and exported figures inspected.
[Protocol/results](recall-repair-plan.md#frozen-reader-supervision-results),
[report](../runs/reader_supervision_v1/report.html). The extra loss stays optional.
Next proposed: a controlled direct stored-working-token route into native output
heads, to separate workspace formation from output learning. No further run launched.

**Snapshot timing comparison completed, 13 September:** explicit snapshot ages
raise writer7801 reset facts72.66→77.34% and images75→75.78% versus equal untimed
continuation; writer7802 stays74.22% for both outputs. All reliability, replicated
10-point benefit and replicated time-sensitivity screens fail. Timestamp erasure
reduces the first temporal model to57.81/55.47%, but leaves the second unchanged;
misalignment affects both, weakly in the second. Keep the cue opt-in: response to
metadata is not general temporal understanding or a reliable repair.
Four1536-update fits,642.65s training,424MiB reserved.76 relevant tests and1,328
independent metric checks pass. GPU reload and bank-order workspace checks exact;
774-row resume prefix preserved. Source weights, writer and codecs unchanged.
CPU numeric tolerance still fails; factual labels agree, two image-side labels differ
for writer7801 temporal. Two public-only Claude reviews; structural reports and
inspected figures. [Protocol/results](recall-repair-plan.md#snapshot-timing-results),
[report](../runs/memory_time_v1/report.html). Next proposed: use the successful
direct stored-state readout as a reference for the native workspace path, and check
the weaker writer separately, before adding more memory cues. No next repair run.

**Frozen-writer recall repair completed, 13 September:** focused retraining improves
reset factual accuracy from48.44/50% to66.41/75%, and generated-image accuracy from
47.66/51.56% to63.28/75%. Fixed memory calibration scores58.59/73.44% facts and
57.03/73.44% images, worse in both matched pairs. All four runs fail the full90%
reliability/pair/intervention screen; neither condition is promoted as a complete
repair. Original checkpoints, writer, raw bank and codecs remain unchanged.
Four1536-update fits,761.23s total,424MiB reserved;41 targeted tests and1,076 independent
metric checks pass. GPU reloads exact with matched precision;768+768 resume preserves
774 ledger rows. CPU answers agree, but numeric tolerance still fails (max logits
0.224921, pixels0.019403). Fixed stored-value probes on these histories score98.44/
85.94% for writer7801 and76.56/75% for7802: incomplete access/learning remains.
Two public-only Claude reviews; reports structural-only, all comparison PNGs
inspected. [Protocol/results](recall-repair-plan.md#results),
[report](../runs/recall_repair_v1/report.html). Next proposed diagnostic: test temporal
and snapshot distinctions at the memory reader, while retaining frozen-state access
controls. Timestamps are stored but not consumed by its attention context; this is
a concrete missing cue, not yet a demonstrated cause or guaranteed repair.

**Frozen-state probes completed, 13 September:** four1536-update fits, two frozen
agents × two reader seeds. Every direct encoder probe scores100%; before-storage
location70.31–96.09%, after-recall49.22–51.56%. Initial future-side50% as expected;
all four diagnostic control screens pass. One individual pre-storage working-token
probe passes, but no state stage passes in both probe seeds. This demonstrates
partial recoverability before storage and poor recovery after recall for this
probe family; it does not prove erased information or a unique causal mechanism.
The source bank stores the pre-storage tensor exactly. Both agent checkpoints
remain unchanged.38 targeted tests,672 independently verified train/test metrics,
exact768+768 GPU resume ledger and exact GPU probe reloads pass. CPU probe replay
on the SAME GPU cache stays within1.55e-5, all categorical predictions agree; this
does not fix the older raw-encoder cross-device discrepancy.217.14s total probe
training,90MiB reserved; caches take9.28s. Two public-only Claude exchanges,
structural reports and inspected stage plot. [Protocol/results](memory-probes-plan.md#results),
[report](../runs/memory_probes_v1/report.html). Next proposed repair: train memory
reading/working-state formation against frozen stored states; keep pre-storage
probes and consider a probe sensitivity control before interpreting failures.
No native agent repair or larger encoder/decoder training performed.

**Balanced relocation comparison completed, 13 September:** both new seeds fail
the declared screen. On128 fresh-background histories with all tuples/motions
represented in training, independently trained direct encoder readers score100%.
Reset recall facts48.44%/50%, images47.66%/50%; color100%, shape96.875%/100%,
location50% in both factual outputs. Complete relocation-pair accuracy0% for both
outputs/seeds. Removing the later view leaves factual accuracy unchanged; removing
the cue or memory reduces it to6.25%. The old normalized checkpoint scores50% on
this same population. Thus balancing removes the data shortcut but does not repair
learning of location updates through the agent state/memory path. It does not prove
location information is absent from those states. No unseen-combination claim.
Frozen visual weights and both calibrations match the reference; seed7801 initial
model hash is identical.34 targeted tests pass;522 metrics and rendered-history
answers independently verify. All three GPU reloads exact. CPU/GPU pixel tolerance
still fails (max0.004826); factual labels agree, one seed7801 image label changes.
Two1536-update runs,534.83s total,402MiB peak reserved; two public-only Claude
exchanges. Reports structural-only; all quartet PNGs inspected. Old benchmark and
weights unchanged. [Protocol/results](memory-output-plan.md#relocation-results),
[report](../runs/memory_relocation_v1/report.html).
Next: frozen-state probes before changing the updater or adding training; separate
recoverable-but-unused location from inadequate state learning. No probe run yet.

**Memory-output experiment and normalization repair completed, 13 September:**
two raw-feature seeds fail (held-out recall facts25%/12.5%, images0%). Fixed input
channel calibration enables100% factual/image validation accuracy on familiar
combinations in one matched seed, but held-out joint accuracy remains0%. Crucially,
reset-recall color and shape are each128/128 correct; location is0/128. Every factual
prediction belongs to the training combinations. The parity split lets location
be inferred from appearance; results strongly suggest this correlation shortcut,
not a demonstrated loss of all entity information. Teacher decoding remains100%.
30 targeted tests, exact CPU replay/no-op gradients and unchanged frozen encoder/
decoder checks pass.180 saved metrics independently verify. GPU standalone reload
is exact on128 episodes; normalized CPU/GPU max pixel difference0.003259 fails the
tight1e-4 check, although factual and rendered-image labels agree on all128.
Three1536-update runs total810.81s; peak402MiB reserved. Four brief public-only
Claude exchanges reconciled. HTML structural-only; comparison panels inspected.
[Protocol/results](memory-output-plan.md#completed-results),
[report](../runs/memory_output_v1/report.html). Next: break appearance/location
correlations with counterfactual relocation training and a separately specified
generalization test. No further training launched; broad baseline remains intact.

**Shared modality-output boundary adopted, 12 September:** state/request/memory
conditioning feeds replaceable modality-specific generators and matching codecs.
Separate training is allowed; state alignment and cross-modal timing/content need
explicit tests. [Design](multimodal.md#adopted-output-design). Documentation only;
no backend selected, model changed or new capability result.

**Image detail and request-only output interface pass scoped screens, 12 September:**
the opt-in detail channel reduces real COCO reconstruction MSE0.012395→0.000879
(92.9%,128 reused images, zero updates). This transports source pixels; it is not
learned compression or evidence of state-mediated photo reconstruction. A learned
state-to-spatial adapter supplies every feature to the same frozen RGB head with
no image encoder in the agent. Four training stripe requests fit to MSE0.000110
after512 updates; erased/shuffled requests0.138158/0.266283. This is memorization,
not novel-prompt generation. All212 donor tensors are unchanged.35 targeted checks,
exact CPU resume, actual GPU256+256 resume, standalone reload and8-metric audit pass.
Training12.12s; peak106MiB reserved. Public-concepts-only Claude review completed;
HTML QA structural-only, comparison PNGs visually inspected.
[Protocol/results/load instructions](image-output-plan.md#completed-screens),
[report](../runs/image_output_v1/report.html). Next: compact learned visual codec and
held-out paired image/request training; audio/video generation training still open.

**Handwritten initialization trained and diagnosed, 12 September:** the exact source
binary now supports strict initialization, component freezing and resumable RGB
training. The original decoder rate collapsed both seeds; validation-only selection
of0.000003 (encoder0.0003) stabilizes training. Joint RGB MSE0.012395/0.012447 improves
10.52%/10.15% from the handwritten baseline. Decoder-only improves9–10%, encoder-only
5%; joint fails the5% advantage screen against decoder-only. Initial patch features
retain only three color averages and collapse an equal-mean checkerboard pair.
Both information loss and decoder optimization matter; this is not an additive error
attribution. Ordinary initialization at its original rate remains better0.007319/0.007859.
[Protocol/results/load instructions](hierarchy-training-plan.md#completed-diagnosis),
[report and trained binaries](../runs/hierarchy_training_v1/decoder_rate_repair/report.html).
203 full-suite tests plus15 focused rate/freeze/resume checks pass; original90 and
repaired81 raw scores independently verify. Source/frozen/export hashes and paired
sampling pass. Seven repair runs complete; the last ordinary control stops at103/384
updates under the600-second cap, so its same-rate paired comparison is incomplete.
190MiB peak reserved for completed repair runs; original521s plus repair600s.
Mask head frozen, image-only scope, reused test population, HTML QA structural-only.
Claude retry was rejected by automatic approval review; no external review occurred.

**Direct hierarchy weights completed, 12 September:** wrote the same1.80M-parameter
image hierarchy as explicit numbers, plus a separately labeled132-coefficient fitted
readout. [Plan/results/load commands](hierarchy-weights-plan.md#completed-comparison),
[report](../runs/hierarchy_weights_v1/report.html). Four GPU evaluations,25.91s,
zero optimizer updates; handwritten RGB MSE0.013853/IoU0.204784, fitted0.013693/0.072054.
Both fail perception gates and trail trained references; color transport works,
foreground perception remains weak.197 tests,36-metric audit, strict binary reload
and cached GPU resume pass;120–136MiB reserved. Previous checkpoints unchanged.
HTML QA structural-only. Claude send rejected by automatic approval review; no
external peer review was performed for this comparison.

**Hierarchy/fusion comparison completed, 12 September:** implemented the requested
per-scale transformer depth plus optional final all-scale stack.194 CPU tests pass,
including causal video/audio/text, invalid gradients and exact resume; disabled
fusion matches pre-change outputs exactly. Eight real COCO RGB/foreground runs,
two seeds,589s: fusion improves IoU0.184→0.258 at7401 but worsens0.304→0.238 at7402
with18.6% worse RGB. Extra depth helps neither seed. Both proposed improvements fail
the prespecified screen; longer shallow training improves RGB, mask effects mixed.
All mask IoUs remain below the full-foreground reference.170–184MiB reserved GPU;
independent raw metrics/hash audit and completed GPU resume pass. [Results/plan](hierarchy-fusion-plan.md#completed-comparison),
[report](../runs/hierarchy_fusion_v1/report.html) (structural-only). Keep fusion
default0 and the option available. Encoder recurrence remains a separately specified
cross-window test; no new speech, video-memory or integrated-agent claim.

**Encoder–decoder review completed, 12 September:**
[Pair-by-pair review and proposed comparisons](encoder-decoder-review.md) distinguish
short-pilot undertraining, decoder mismatch, state-path failures and missing speech
context. Every multimodal feature scale already has attention/MLP processing.
Historical extra CNN depth helped geometry but missed its gate; frozen-encoder
decoder repair recovered COCO reconstruction with a PushT retention tradeoff.
Proposed next: reuse useful vision donors, compare direct versus state-mediated
readouts, and freeze/probe the successful text donor before adding capacity.
Audio needs a meaningful temporal contract before speech training. Claude reviewed
public concepts and accepted corrections; no model changes or new training/tests.
The capability baseline remains the frozen comparison point. Numerical gates and
budgets for the next formal experiment still require a predeclared specification.

**Capability baseline completed, 12 September:**34 behavioral/mechanism checks across
13 separately identified checkpoint roles,183 passing software tests,10 explicit
coverage gaps. [Report](../runs/capabilities_v1/reference/report.html) and
[protocol, results and comparison commands](capability-baseline-plan.md#completed-baseline).
Direct visual weights retain100% accuracy on original/mirrored/dim/noisy scenes but
fall to48.44% with red/blue swapped and50% grayscale. Their other outputs are gray,
silence and empty text, with zero response to the tested audio/text/video input contrasts or opposing actions.
Structured entity/state/relation and supplied-mechanics planning screens pass;
through-agent fact identity0%, historical recall answers0/15, instructions12.5%,
and learned prediction loses to copy-last. Known planner reward/utility mismatch
persists. No weight changes. CPU evaluation211s/804MiB peak process RSS. Independent
raw checks across all34 cases, checkpoint hashes, cached resume and comparison guards
pass. Browser QA remains structural-only. This is a baseline of separate components,
not a complete jointly trained agent or evidence of general webcam capability.

**Direct weight test completed, 12 September:** at the user's explicit request,
constructed binary model weights and adjusted them without backpropagation. First
candidate family fails (39/64 final answers). A bounded numeric timing/routing revision
with34 label-fitted readout parameters passes on a fresh set:64/64 answers,32/32 complete
pairs and32/32 reversals; ordinary/erased controls50%. Zero optimizer updates,40MiB
inference allocator peak;180 CPU tests and independent artifact/metric checks pass.
[Weights and scope](direct-weights-plan.md#completed-revision),
[report](../runs/direct_weights_v1/timing/report.html) (structural-only).
This is a hand-built fixed-task circuit in the small existing model configuration,
not general pretrained-weight generation. Real webcam transfer remains open. The
previous optimizer-based visual curriculum is deferred by this user-requested test.

**Assistant-authored curriculum discussed, 12 September:** the user proposes having
the assistant generate teaching modalities and train the learner. [Curriculum proposal](webcam-memory-data-plan.md#assistant-authored-teaching-data-12-september-discussion)
combines controlled histories and checked questions/images with separate real-footage
evaluation. One brief Claude review emphasized pixel/label checks; the local
reconciliation avoids assuming a perfect automated verifier. Text/image generation
is available; realistic video/speech generation is not yet connected. No new corpus
or training run. Visual-memory interfaces, objective, gates and GPU budget still need
a concrete experiment specification; prior real-footage preference remains intact.

**Real-footage development pack prepared, 12 September:** the user prefers real
webcam footage for observation memory. Selected 12 COCO train photographs and 6
Charades official-train indoor clips, preserving source hashes, annotations and
split/subject groups. Contact sheets inspected; six imported video episodes have
timestamps and pending annotation templates. [Data and recording guide](webcam-memory-data-plan.md)
defines the first 28-take pilot and independent answer-key checks following Claude's
review. Three capture/import tests pass. No camera activation or training. Own
recordings, reviewed entity/event/query labels and the experiment gates remain open;
this selection is development material, not a trained or validated memory capability.

**S01 sequence adopted, 12 September:** first remember observations, then predict
action consequences, then plan toward a goal using those predictions. Observation
memory is the next capability target. First environment/input and numerical gates
remain open. [Recorded answers](agent-specification-questions.md#recorded-answers).
Both training and inference must fit the current local GPU; scale later.

**S03/R06 preference recorded, 12 September:** comfortable fit on the user's GPU
is required; prefer own components, with pretrained model plus adapter if quality
is inadequate. Local hardware query reports RTX3050/8192MiB. User confirmed both
training and inference must fit now; scaling comes later. Numerical headroom and
quality gates remain open. [Recorded answers](agent-specification-questions.md#recorded-answers).

**S02 long-term scope recorded, 12 September:** eventual webcam vision, speech and
writing, image/video creation and software-tool use. [Questionnaire answer](agent-specification-questions.md#recorded-answers)
is partial: first experiment and learned-versus-supplied boundaries remain open.

**Specification questionnaire prepared, 11 September:**
[85 numbered questions](agent-specification-questions.md) cover scope, goals,
evidence, perception, identity, graph learning, memory, focus, dynamics, planning,
training, execution, inspection and experiment gates. Begin with S01–S05 and then
the task/evidence/dynamics contracts. Explicit deferrals define what the first
experiment does not claim. Answers remain pending; no architecture change or
training was launched. Claude reviewed integration-level coverage.

**Whole-model audit updated, 11 September:** controlled integration is working,
but learned dynamics, uncertainty semantics and general task execution remain
separate or supplied. Concrete planner/metric objective mismatch confirmed: under
85% absence belief, current one-step planner retrieves although reported expected
utility favors stopping0.85 over retrieval0.10. Prior measurements remain valid;
they do not establish optimization of that utility. [Current readiness review](model-readiness-review.md)
replaces the stale pre-integration assessment. Next: resolve the task/cost/evidence
contract, then connect action-conditioned prediction to the integrated path.
Review only; no model changes or new training/full-suite result.

**Independent key-box replication and correction screen pass, 11 September:**
new training seed2302, fresh evaluation2431: ordinary and longer-history relocation
cases both96/96 reachable and32/32 absent; correction readout192/192. Frozen prior
model also passes (correction191/192). Ordinary memory utility advantage remains
0.015625; under relocation memory costs more than no-history, so no universal
utility benefit.31 tests, lint, cached resume and independent time-varying execution
and correction audit pass. [Report](../runs/key_box_v1/replica/report.html)
structural-only; [active plan](key-box-integration-plan.md). Next: varied change
and observation timing. Still supplied descriptors, corrections and action mechanics.

**Interleaved key-box training passes controlled screen, 11 September:**
four read pairs separated by observation events preserve working state during
training. Fresh seed2421: known content100%, reachable96/96, absent32/32;
utility0.9203125 beats no-history0.9046875 by0.015625 (required0.01).
Frozen previous model fails the same screen.30 tests, lint, cached resume and
independent execution replay pass. [Report](../runs/key_box_v1/history/report.html)
structural-only; [active plan](key-box-integration-plan.md). Next: replication and
stronger state-change/history checks. Supplied descriptors and action mechanics;
this does not validate learned dynamics or general action planning.

**Integrated key-box loop implemented, 11 September:** entity state now reaches the
actual belief-agent workspace; supplied expectimax action mechanics execute and
replan from real feedback. First training screen failed. Query-switch repair raises
known content82.81%→100% and reachable success76.04%→95.83% on matched fresh cases,
but utility0.891016<no-history0.904688, so full screen still fails. Four false stops
follow later readout confidence loss.29 distinct relevant tests, resumes and independent
execution replay pass. [Report](../runs/key_box_v1/switched/report.html) structural-only.
[Active plan](key-box-integration-plan.md). Next: full action-history readout stability.
Source-selection tuning is paused. This is controlled descriptor/explicit-dynamics
integration, not visual discovery or learned world-model planning.

**Matched-budget coverage fails utility gain, 11 September:** eligible drift
source checks29/32 versus21/32, but late utility gain0.000977<0.01. Static cost
and reset guards pass.11 source tests, cached resume and independent allocation/
feedback replay pass; costs exactly matched. [Report](../runs/entity_source_coverage_v1/reference/report.html)
structural-only. No promotion; result conditional on forced acquisition/epsilon0.5.
Next proposal: source-selection headroom diagnostic before further tuning.

**Source-local diagnosis complete, 11 September:** of32 drift source records,14
never reached an eligible all-new-feedback block,10 reached checks but stayed below
threshold,8 reset. This separates check availability from threshold outcomes; it
does not establish a repair.9 source tests, resume and independent full-feedback
reconstruction pass; original artifacts unchanged. [Report](../runs/entity_source_diagnosis_v1/reference/report.html)
structural-only. Next proposal: matched-budget feedback coverage with detector fixed.

**Variance-aware forgetting fails adaptation margin, 11 September:** development
selects z2; held-out static resets4/16 meet25% (matched fixed-trigger8/16). Late
drift utility0.731096 beats cumulative0.722993 by0.008103, below required0.01.
Other utility guards pass.29 distinct relevant tests, cached resume and independent
development/held-out replay pass. [Report](../runs/entity_source_uncertainty_v1/reference/report.html)
structural-only. No promotion or tuning; next diagnose missed/late changes from
existing traces before choosing another detector. Original gate unchanged.

**Triggered forgetting fails false-reset guard, 11 September:** late drift utility
0.742107 versus frozen0.713721 and cumulative0.723066; static utility loss0.011832
is within0.02. But10/16 unchanged worlds reset (62.5%>25%), so full screen fails.
26 relevant tests, cached resume and independent action/feedback/reset replay pass.
[Report](../runs/entity_source_change_v1/reference/report.html) structural-only.
Original gate unchanged. Next candidate: uncertainty-aware change checks with
separate stationary calibration and held-out false-alarm controls; no tuning here.

**Online recency adapts but fails stable-source guardrail, 11 September:** drift late
utility window0.760835 versus frozen0.723169 and cumulative0.733477. Static
utility falls0.780920→0.754734 (loss0.026187>0.02), so full screen fails.25 relevant
tests, resume and independent action/feedback replay pass.
[Report](../runs/entity_source_drift_v1/reference/report.html) structural-only.
Reference unchanged; fixed window is not learned drift detection. Next candidate:
evidence-triggered forgetting rather than unconditional recency, with stable-source
controls and feedback costs preserved.

**Outcome-trained source choice passes, 11 September:** a per-world action-value
table selects the useful opaque source in16/16 worlds from calibration feedback.
Evaluation accuracy79.91% versus best fixed77.00%; utility0.777112 versus0.748059.
Combined calibration/evaluation utility0.752104 exceeds stop0.728353.23 relevant
tests, resume and independent selected-feedback audit pass.
[Report](../runs/entity_source_choice_v1/reference/report.html) structural-only.
This is static-source adaptation with supplied outcome feedback, not a neural
selector or general reliability estimator. Next candidate: source-quality changes
and online updating, with explicit feedback availability and stale-value controls.

**Alternate evidence is worth its declared cost, 11 September:** selective high-noise
accuracy80.86% alternate versus73.44% same-source, both reread42.58%. Utility
0.787305 versus0.725859 despite alternate cost0.05 versus0.02. Paired descriptive
95% interval for utility gain[0.024129,0.101331].21 relevant tests, resume and
independent audit pass. [Report](../runs/entity_evidence_sources_v1/reference/report.html)
structural-only. Fixed sensor properties/policies; no learned source selection.
Next candidate: learn acquisition choice from outcome feedback without giving the
policy hidden noise/correlation labels. Reliability-estimation proposal remains separate.

**Correlated rereads expose the independence limit, 11 September:** selective
high-noise gains9.77/5.08/1.17 points at rho0/0.5/0.9; rho0.9 fails the2-point
gate. Rho1 adds no information and loses sensing cost.19 relevant tests, four
resumes and independent shared-observation/decision/cost audits pass.
[Report](../runs/entity_gate_correlation_v1/rho0.9/report.html) structural-only.
Reference unchanged; no memory integration. Next candidate: compare acquiring a
different evidence source with repeating the same source, without assuming the
agent already knows error correlation.

**Reobservation diagnostic passes, 11 September:** high-noise accuracy72.27%→82.81%
with selective rereads42.58%; ignore recall93.75%→98.44%. Duplicate replay gives
no gain. Always-two reaches87.11% and higher utility at the declared cost0.02.
18 relevant tests, cached resume and independent decision/cost audit pass.
[Report](../runs/entity_gate_reobserve_v1/reference/report.html) structural-only.
This is a supplied static-context sensing policy, not learned deferral or a new
memory operation. Next candidate: test correlated second-observation noise before
integrating sensing; independent reread benefit may not survive shared errors.

**Clean retention does not repair the tradeoff, 11 September:** weight-one teacher
KL leaves clean-runtime NLL unchanged0.0237295 and low-noise ignore90.625% at
sigma0.15; high-noise accuracy81.25%→80.08%. Full criteria fail in both arms.
16 relevant tests, two resumes and independent objective/metric audits pass.
[Report](../runs/entity_gate_retain_v1/retained/report.html) is structurally verified;
visual QA remains unavailable. Reference unchanged. Next candidate: diagnose
whether the binary write decision needs an explicit defer/reobserve option under
ambiguous cues, before another loss-weight experiment. This remains a proposal.

**Two conditional replications retain the augmentation tradeoff, 11 September:**
high-noise gains+8.98/+11.72 points over controls, but both full acceptance gates
fail. Rep1 clean-runtime NLL0.152775 exceeds0.15; rep2 NLL0.269396, development
and low-noise ignore checks fail.15 relevant tests, four cached resumes and paired
audits pass. Reference unchanged. Reports: [rep1](../runs/entity_gate_replicate_v1/rep1_augmented/report.html),
[rep2](../runs/entity_gate_replicate_v1/rep2_augmented/report.html), structural QA only.
Next candidate: explicit clean-behavior retention during noisy continuation, tested
against these preserved results on fresh contexts. No broad robustness claim.

**Noise augmentation improves recall but fails acceptance, 11 September:** on fresh
contexts high-noise accuracy80.08% versus matched control71.48% (frozen75.78%).
Low-noise0.15 ignore recall94.53% misses95% criterion; reference stays unchanged.
Both arms preserve100% clean runtime state accuracy.14 relevant tests, both cached
resumes and independent paired audit pass. [Treatment report](../runs/entity_gate_augment_v1/augmented/report.html)
and [control](../runs/entity_gate_augment_v1/control/report.html) are structurally
verified; visual QA remains unavailable. Next proposed: independent-seed replication
of the recall/false-write tradeoff before another repair or threshold choice.

**Frozen gate noise shift finds a limit, 11 September:** the primary per-class
robustness gate fails at sigma0.30 and0.60. Accept recall drops92.97% then55.47%;
ignore recall remains≥95%. Sigma0.03/0.15 pass at100%. No retraining or threshold
selection.12 relevant tests, cached resume and independent metric audit pass.
[Report](../runs/entity_gate_shift_v1/reference/report.html) is structurally verified;
visual QA remains unavailable under the prior browser policy denial. Next proposed:
noise-augmented training with fresh evaluation contexts; high-noise ambiguity
means this failure does not identify a unique architectural defect.

**Context write gate passes the bounded screen, 11 September:** the gate learns
from source-selection loss with matcher, key and interaction weights frozen.
All96 held-out cases pass source/state prediction, including allocation permutation
and repeated irrelevant cues; always/never-write controls score50%. Soft/hard
source choices agree100%. Full145-test suite and cached resume pass.
[Report](../runs/entity_gate_v1/reference/report.html),
[verification](../runs/entity_gate_v1/verification.json). Report structural checks
pass; browser policy blocked visual QA. This tests separated context regimes,
not ambiguous semantic relevance. Next proposed: predeclare a context-noise shift
comparison before extending the claim or adding graph operations.

**Remembered relation keys pass, 11 September:**140 tests pass. A learned key supports
destination-only recall after the source cue disappears:100% source/state accuracy
through replacement, gaps and allocation changes. Erasing keys yields0% source
accuracy. Only key addressing trained; persistence and write policy remain explicit.
[Report](../runs/entity_relations_v1/reference/report.html) is browser verified. Next
proposed: learn whether a new cue should overwrite or preserve a relation.

**Frozen source retrieval passes, 11 September:**136 tests pass. Descriptor queries
select the source among three records at100% accuracy, matching oracle outcomes.
Allocation permutation passes; all unknown queries roll back. No model fitting.
[Report](../runs/entity_source_v1/reference/report.html) is browser verified. The
source query is still supplied. Next proposed: remember a relation from an earlier
cue and retrieve it for a later destination-only action.

**Directed state interaction passes, 11 September:**132 tests pass. A learned
interaction uses another entity’s latent state:100% across all tested conditions,
including a second copy. The matched source-zero control scores50% reference and
29.3% composition. Frozen dynamics, transactions and browser QA pass.
[Report](../runs/entity_interaction_v1/full/report.html). Endpoints and copy type are
still supplied. Next proposed: selecting a source among distractors before claiming
learned relational retrieval or graph structure.

**Explicit idle preservation passes the bounded screen, 11 September:**128 tests
pass. With31 extra idle events, accuracy improves25%→100%; all seven frozen
conditions pass. Latents remain exactly stable across idle stretches; resets and
actions still work. This is an explicit deterministic rule, not learned belief
persistence. [Report](../runs/entity_noinfo_v1/adapted/report.html) is browser verified.
Next proposed: an update depending on another entity’s remembered state, before
claiming learned interaction edges or graph structure.

**Mixed-history adaptation improves ordering but still fails idle stability,
11 September:**123 tests pass. Reset-order accuracy rises62.5%→100% and unseen
composition38.7%→100%; long no-information remains75%, with worse NLL. Recognition
is frozen; adapted runtime/transaction checks pass. [Report](../runs/entity_state_varied_v1/adapted/report.html)
is browser verified. Next: a task-specific state-preserving no-information update,
keeping this failed baseline and fresh idle-length tests.

**Frozen temporal screen exposes state-update limits, 11 September:**121 tests pass.
Reference and repeated-toggle histories score100%; reset-order histories62.5% and
extra no-information events75%. All routing, retry/restore and latent-agreement checks
pass; model weights are unchanged. [Report](../runs/entity_temporal_v1/reference/report.html)
is browser verified. Next: broaden state-update training with fresh held-out histories;
retain this failed frozen baseline. No general temporal or graph-learning claim.

**Learned persistent state succeeds on the bounded task, 11 September:**119 tests
pass. A frozen recognizer routes observations into learned per-entity16-float states;
256/256 development pairs and persistent-runtime outputs are correct. Retry/restore,
rollback and reversed allocation-order checks pass. [Report](../runs/entity_state_v1/reference/report.html)
is browser verified. Splits share16 temporal templates and differ in descriptors;
this does not establish temporal generalization. Next: frozen held-out composition
and length tests before broader attributes or graph learning.

**Variable-count adaptation succeeds, 11 September:**115 tests pass. One256-update
run gives100% development matching/coverage and passes every fresh lifecycle gate
at capacities1/2/4/8. The matched original model still fails capacity8. Claude reviewed
the design twice. [Adapted report](../runs/entity_variable_v1/adapted_pinned/report.html)
is browser verified; weights, descriptor isolation and resume are checked. Geometry
and candidate counts changed together. Next: learned changing state bound to stable
IDs; visual discovery and learned graph structure remain open.

**Frozen growth screen completed, 11 September:** capacities1/2 pass; capacities4/8
fail the declared lifecycle gate. Capacity8 allocation/revisit accuracy is98.44%,
with no wrong-ID matches; uncertain novelty leaves some stores underfilled. All
retry/restore checks pass. Fixed checkpoint loading and evaluator ID-offset errors;
113 distinct tests pass. [Corrected report](../runs/entity_growth_v1/corrected/report.html)
is browser verified. Next: predeclare variable-cardinality training; preserve this
failed frozen reference. No new model training or graph learning yet.

**Entity lifecycle implemented, 11 September:** a bounded Python `EntityMemory`
allocates stable IDs, reuses them on confident matches, defers at capacity or low
confidence, and supports idempotent retries and JSON snapshot restoration. Claude
reviewed the transaction contract twice. Recognition prototypes remain frozen;
variable-cardinality accuracy and learned belief/graph updates remain unvalidated.
See [usage and scope](entity-learning-task.md).

**Known-versus-new matching succeeds, 11 September:** `--dataset entity-matching`
is implemented and reviewed with Claude. All 105 CPU tests pass; the 256-update run
gets 128/128 known identities and 128/128 novel rejections correct, with zero false
merges/splits on development. Novel selection coverage is 96.1%; memory swaps preserve
probabilities and metrics. [Report](../runs/entity_novelty_v1/reference/report.html)
passed browser QA. The distance-separated task is a sanity check, not general novelty
or calibration. No memory records are allocated yet; the next proposed slice is
transactional allocation/revisit behavior. See [task record](entity-learning-task.md).

**Bounded descriptor variation succeeds, 11 September:** the learned/shared reader
trained at `--entity-noise 0.2` passes every declared development gate after 256 updates;
matching and identifiable task accuracy are 100%. All 102 CPU tests pass. Oracle,
geometry, zero-noise compatibility, raw scores, resume and reorderings are verified.
[Report](../runs/entity_variation_v1/reference/report.html). Claude completed both
review and reconciliation after explicit approval of the follow-up; no conceptual
objection remains.
See [task record](entity-learning-task.md). New-entity allocation remains next,
not implemented. This is bounded synthetic drift, not general visual robustness.

**Learned association succeeds on stable descriptors, 11 September:** the
Claude-reviewed `--entity-reader shared --entity-association learned` mode passes
all fixed development gates after 256 updates. Matching improves from 37.5% to 100%
on eight held-out descriptor groups; task accuracy is 100% on 128 identifiable cases.
Task-loss gradients reach the matcher; no exact lookup is used by its forward path.
All per-frame reorderings preserve probabilities. [Report](../runs/entity_learned_v1/reference/report.html)
and [task record](entity-learning-task.md) retain the scope: stable synthetic features,
fixed two-object slots and enumerated hypotheses; graph structure is not learned.
Next proposed: feature variation and unmatched/new entities.

**Shared entity reader succeeds, 11 September:** the Claude-reviewed
`--entity-reader shared --entity-association observed` path passes all declared
development gates after 256 updates: 100% identity/state/effect accuracy on 128
identifiable cases and correct bounded ambiguity handling. All 96 CPU tests pass.
Predicted probabilities are unchanged under all eight per-frame reorderings after
undoing output order. It uses 30,021 parameters and 128 persistent state floats.
[Report](../runs/entity_shared_v1/reference/report.html) and
[task evidence](entity-learning-task.md) retain limitations: association, two object
slots and the finite hypothesis set are supplied; graph learning remains unimplemented.
Next proposed: learn association while preserving this working reference.

**Supplied association complete, 11 September:** the Claude-reviewed
`--entity-association observed` diagnostic is implemented; 94 CPU tests pass.
Matched 256-update development identity accuracy improves to 88.3%, but state/effect
remain 64.8%/57.8% and combined gates fail. The trained recurrent reader changes its
probabilities under candidate reordering. This is a useful diagnostic gain, not
reliable binding or learned graph structure. See the
[updated task record](entity-learning-task.md) and
[report](../runs/entity_alignment_v1/observed/report.html). Next proposed diagnostic:
shared per-entity updates and permutation-consistent readout.

**Entity baseline built and evaluated, 11 September:** `--dataset entities` now runs
the [controlled two-object task](entity-learning-task.md) through the existing recipe.
All 92 CPU tests pass. The fixed 256-update baseline reaches development identity/state/
effect accuracy of 60.2%/62.5%/54.7%; every combined development gate fails. Training
state accuracy reaches 100%, but identity is only 66.8%. The graph comparison is
deferred under the declared stop rule. Source, cached resume, oracle, raw scores and
browser report checks are recorded in `runs/entity_learning_v1/verification.json`.
[Open the report](../runs/entity_learning_v1/reference/report.html). Next diagnostic:
separate descriptor association from state updating; no extra run is authorized by
this result, and the earlier entity-reader bottleneck remains unresolved.

**Task definition complete, 11 September:** [the first entity-learning task](entity-learning-task.md)
defines two-object identity persistence, state updates and action-effect prediction.
Both graph structure and latent values are intended to be learned; readable labels
are inspection aids, not imposed semantics. Two actual Claude reviews are reconciled.
A finite 512-case specification check confirms an exact history oracle and final-view-only
bounds of 50% identity and 25% state-pair accuracy. These are contract checks, not model
results. Next implementation step is the task generator/reference path and existing-reader
diagnosis, followed by a bounded recurrent baseline before any graph comparison.
No new neural training or graph implementation was started in this definition slice.

**Entity-design discussion, 11 September:** the user accepted diagnosing accessible
identity first, then testing controlled two-object binding. Subsequent discussion
proposes per-entity learned beliefs and external retrieval. Two actual Claude
exchanges are reconciled in [the design note](entity-memory-design.md): candidate
extraction, uncertain association and persistent keys are separate mechanisms.
Explicit entity storage remains a proposal; no new architecture or run was started.
The follow-up runtime/payload review adds four reconciled Claude exchanges: small
read/propose/commit interfaces, explicit gradient boundaries, and optional raw,
latent or readable payloads. Recalled media may reuse modality encoders but cannot
silently enter as new observations. Source IDs, historical time and typed revisions
remain distinct; a memory-origin label alone is insufficient.

**Current slice complete, 11 September:** the
[trainable encoder initialization comparison](warm-encoder-plan.md) is implemented
as `--fact-encoder-weights` for the event fact reader. Two brief actual Claude
reviews are reconciled; all 88 CPU tests pass. Exact transfer, unchanged remaining
initialization/RNG, encoder updates and donor-bound resume checks pass. Both new
512-update recipients finished in 103.5041 active CPU seconds total. At lr0.0003,
held-out location accuracy improves from the saved cold run's 8/32 to 31/32; at
lr0.001 it improves from 24/32 to 32/32. Held-out entity and joint accuracy remain
0/32 in both warm runs. Training entity accuracy is only 3/96 in each. Both
extraction gates fail; binding is skipped and the declared two-run slice is complete.

The [reference report](../runs/warm_encoder_v1/reference/report.html) and
[learning-rate comparison](../runs/warm_encoder_v1/lr_control/report.html) passed
structural and 1280x720 browser QA. `runs/warm_encoder_v1/verification.json` binds
raw-score, source, donor, matching cold settings/sampler, cached-resume, test and
browser checks. Development combinations are reused; the donor adds 512 upstream
updates / 8192 presentations. The shared encoder also reads the fixed instruction.
These results leave entity learning unresolved without isolating its cause.
Next proposed: freeze the donor encoder in one otherwise matched diagnostic to
test whether preserving its features changes entity learning.

**Prior single-event slice, 11 September:** the
[single-event agent-reader control](event-fact-plan.md) is implemented as
`--dataset facts --fact-reader event`. Two brief actual Claude reviews are
reconciled; all 86 CPU tests pass. A strengthened gradient check confirms the loss
reaches the observed fact, and exact resume preserves cached outputs. Two matched
512-update runs took 109.4034 active CPU seconds total. At lr0.0003, held-out entity/
location accuracy was 3.125% / 25%; at lr0.001 it was 0% / 75%. Joint accuracy was
0/32 in both runs. Both extraction gates fail and binding evaluation is skipped.
No third training run, new objective or memory change was started.

The [reference report](../runs/event_fact_v1/reference/report.html) and
[learning-rate comparison](../runs/event_fact_v1/lr_control/report.html) passed
structural and 1280x720 browser QA. `runs/event_fact_v1/verification.json` binds
source snapshots, settings/data/init/sampler matching, raw-score checks, exact
resume, tests and screenshots. The existing whole path does not learn the task
under this budget; the failing component is not isolated. Recent records are
detached on storage by the existing memory policy, while the live categorical
path still carries gradients. The trainable encoder initialization follow-up is
now complete above.

**Prior direct-control slice, 11 September:** the
[direct fact extraction and binding controls](fact-learning-plan.md) run in the
same recipe (`--dataset facts`). Two short actual Claude reviews are reconciled.
All 83 CPU tests passed; 12 focused fact/run checks passed after the report revision.
The first 512-update reference passed every declared gate in 10.3502 active CPU
seconds: entity, location and joint accuracy are 100% on 96 training and 32 held-out
combinations. Mean held-out NLL is 0.211649. The fixed selector answers every
enumerated two-record query correctly, including 768 queries / 384 pairs whose
constituent facts are both held out. Coherent location swaps also pass. These
reused fact combinations are not independent samples or natural-language evidence.
No second learning-rate run was needed.

The [standalone fact report](../runs/fact_grounding_v1/reference/report.html) has
verified tables, curves and expandable examples at 1280x720, with no broken images
or horizontal overflow. `runs/fact_grounding_v1/verification.json` binds the intact
training snapshot, separate final renderer, checkpoint/results, tests and browser
receipts. CLI resume reused cached predictions and unchanged result/metric files;
an independent probability-space calculation matches every saved binding score.
Next proposed: test the same factual task through the existing agent event/reader
path. Success of this freshly trained encoder and explicit selector does not prove
the world model's learned binding, recurrent retention or memory compression.

**Prior diagnostic slice, 11 September:** the
[current/recent factual recall diagnostic](recall-learning-plan.md) is implemented
in the same recipe (`--dataset recall --recall-mode current-recent`). Actual Claude
completed one review and one reconciliation after Alex approved the export. All 79
CPU tests pass; the actual diagnostic forward/backward check has finite losses and
the expected gradients. The 256-update CPU pilot finished in 144.58 active seconds.
The final checkpoint gets 71.875% seen-location accuracy on its 40 training episodes,
but only 17.5% on the 80 seen cases among 100 fresh development episodes: current
12.5%, recent 22.5%. Both declared gates fail. Overall development task loss is 0.31
at 12% coverage, worse than always abstaining (0.25). No calibration/test data was
loaded and no new memory loss or budget extension was started.

The [standalone report](../runs/recall_diagnostic_v1/pilot/report.html) passed
structural and 1280x720 browser checks. `runs/recall_diagnostic_v1/verification.json`
binds source/results/checkpoint/report hashes, tests and screenshots. Next focus is
basic entity/location binding and generalization; this pilot does not isolate the
input encoder, query binding, readout or insufficient optimization budget.

**Prior recall slice:** historical recall is implemented in the same recipe. Read the
[usable guide](recall-task.md) and [implementation record](recall-implementation-plan.md).
All 76 CPU tests pass, including exact pause/resume, held-out split isolation and
report-failure recovery. A 256-event default-memory forward/backward check completed.
The eight-update CPU development run selected update 7. On its 15 test episodes,
factual accuracy is 20%, every decision abstains, and task loss is 0.25. The model
does not yet demonstrate useful recall. Calibration selected its upper bound T=20;
this small sample establishes no calibration guarantee.

**Overnight work complete:** all seven topic reviews and their combined reconciliation
with actual Claude are complete. The [decision agenda](remaining-decisions-2026-09-10.md)
starts with five choices and the full design, followed by the detailed alternatives,
interfaces, evidence and remaining empirical questions. Recommended next: grounded
local memory learning with current/recent recall controls, then learned marking if
its marginal signal is useful. Preserve fixed capacity, exact historical records and
separate live, hypothetical and offline replay state. Freeze the whole selected
procedure before final calibration; each later task configuration needs its own evaluation.

The final consistency check completed on 10 September at 08:31 Berlin, before the
09:00 deadline. All 21 overnight Claude responses succeeded; saved evidence and source
identity are unchanged. The heartbeat `overnight-agent-design-proposals` is paused
with its persisted status verified. These proposals remain unadopted. Receipts are
`overnight-integration-local-checks.json` and `morning-handoff-check.json` under
`runs/reviews/state_memory_design_2026-09-09/`. No model run or test suite was repeated
for the morning handoff.

**Implemented:** the categorical belief and bounded session-memory design authorized
by Alex on 9 September 2026. Read [the model guide](belief-model.md) and the
[implementation plan/record](belief-implementation-plan.md). One editable
[recipe](../experiments/multimodal.py) still owns construction, targets, training,
evaluation, checkpoint/resume and reporting. No second trainer or runtime LLM was added.

The CLI defaults to `--state-model belief`: recurrent context, grouped categorical
prior/posterior, source-only evidence and a separate task workspace. Ordered event
transactions advance executed action/time once, merge partial packets against a
fixed prior and memory snapshot, and commit once. Shared dynamics power imagination.
Thinking/reflection leave the physical belief and observational history unchanged.
Both evidence age and inferred-state age/ordinal constrain causal memory reads.

Memory contains exact recent latent envelopes, chronological compression staging,
compressed history, protected user/agent marks and gated consolidation. Separate
perception/prediction/thinking readers have learned scale gates and a null choice.
Full categorical probabilities reach readers and compression. Source-only features
remain independently encoded and compressed. The default tensor payload is bounded
at 318,040 bytes per FP32 stream, plus bounded metadata and temporary computation.
Fresh `initial_state()` starts empty memory and workspace with the same model weights;
callers discard prior task progress and plans. Individual memory resets remain withdrawn.

Learning now includes observable reconstruction/prediction likelihoods, split
categorical KL, isolated full/partial teacher targets, delayed recall, frozen-reader
compression distillation and delayed marginal mark utility. These mechanisms are
implemented; effective long-horizon memory and calibrated uncertainty are unproven.
The ordinary two-step default history is too short to train delayed recall across
the default 32-record recent store. The guide gives explicit small-memory settings
that exercise all memory scales within an eight-step development history.

**Earlier belief-slice verification:** 69 CPU tests passed at that stage. Exact
pause/resume reproduces model, optimizer, sampler, RNG and training rows, including the optional extra-update gate. New tests
cover event retry/order, masked inputs, source/belief separation, mixed batches,
full-distribution reads, memory bounds/consolidation, provenance, snapshot loading,
common planning samples/RNG restoration and future memory at equal timestamps.
Final real PushT forward/backward check is saved with the run receipts.

Three short CPU development runs completed: eight synthetic updates, eight
instruction updates and two real PushT updates; no extra proposals or GPU runs.
Every run has raw metrics, checkpoint, source snapshot, media and an offline report.
The current report renderer has now passed browser visual checks at a 1280×720
viewport, with no broken images or horizontal overflow on the three reports.
See [verification](../runs/belief_v1/verification.json),
[synthetic report](../runs/belief_v1/synthetic/report.html),
[instruction report](../runs/belief_v1/instructions/report.html), and
[real report](../runs/belief_v1/pusht/report.html).

**Learning remains weak:** synthetic held-out image MSE is 0.237670 versus 0.007451
for copying the last image. The tiny real check gives 0.236499 versus 0.000050895.
The four-example instruction evaluation has 75% operation error. These development
populations are too small and training too short for capability conclusions. They
confirm execution and expose poor current predictions, not successful world learning.
The earlier negative instruction result is unchanged and preserved in the historical
[task implementation record](multimodal-plan.md); its old run paths are unavailable
in this local checkout and were not reverified or reconstructed.

**Claude collaboration:** two actual isolated CLI exchanges reviewed abstract
implementation invariants. No private source, dimensions or results were exported.
Claude withdrew overbroad demands for parameter-disjoint encoders, hard gates,
mask tokens and a particular RNG mechanism. The adopted properties are forward
information separation, frozen event inputs, bounded replay graphs and side-effect
free planning. Remaining fixed/variable-rollout and sealing concerns were resolved
by the concrete fixed-horizon interface, open-event type checks and local tests.
Receipts: `runs/reviews/state_memory_design_2026-09-09/implementation*`.

The Gaussian reference, existing task controls/output attribution and multiscale
adapters remain usable. Python `build_model()` retains its Gaussian default; use
`state_model="belief"` explicitly in code. Its SVG diagram exporter still describes
the Gaussian reference; the new guide has the categorical flow. Checkpoint schemas
are distinct, with no implicit Gaussian conversion. Completed runs retain their
own source snapshots and must be resumed with compatible source/settings.

**Active design proposal:** [decisions from belief and memory](decision-design.md)
connects the task workspace and planner through an exact objective/cost contract,
a learned observable-outcome head, and verification of actual results. This is the
Claude-reviewed discussion requested on 9 September, completed on 10 September;
it is not implemented or a new capability claim. Historical last-observed recall
with a separate factual not-observed answer and operational abstention is the
recommended first slice. Active current-location inspection is a subsequent task
requiring observation-conditioned continuations and isolated hypothetical updates.
Existing memory bounds, reset decisions and negative results are unchanged.

**Implemented first-task contract:** [selective historical recall](recall-task-design.md)
specifies four locations plus factual not-observed, separate abstention, loss 0/1/0.25,
two retrieval rounds, complete text-observation episodes and independent calibration.
Two further abstract Claude reviews reconcile cost/calibration and data-split claims.
Report seen/old-history performance separately: recognizing only unseen entities can
beat all-abstain without remembering any locations. The subsequent implementation
and development evidence are recorded in the guide linked above.

**Next step:** declare one bounded input/binding/readout diagnostic before any new
memory-learning objective; preserve the failed current/recent pilot. Numerical
costs are explicit research defaults, not inferred application preferences. Deployment-length replay,
general mark selection, calibrated probabilities and closed-loop behavior remain open.
