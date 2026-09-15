# Non-image modality integration and diagnosis

User request: all existing modalities should work with the modular World State,
not only images. Inspect and repair concrete interfaces, then separately measure
small-task learning and real-input transport. No general language/speech/video
capability inferred from tensor shapes or toy fitting.

## Plan before implementation

1. Reuse current PyTorch multiscale encoders, native text/audio/image decoders,
   video trajectory decoding, BeliefAgent, WorldSession, Run and portable reports.
   Add the missing audio/video inputs and audio/text outputs to the foundation
   constructor. Keep modules directly replaceable; no second framework.
2. Provide a small masked feature-to-candidate adapter shared by text spans,
   audio chunks and video/image tokens. Candidate extent is supplied; whole-window
   pooling is a diagnostic baseline, not object/word discovery or cross-modal
   identity alignment. Version spaces separately until alignment is trained.
3. Test every input alone and mixed, output generation, masks, causal ordering,
   gradients, source attribution, unchanged writes during recall, and exact restore.
   Correct concrete failures before calling the interfaces complete.
4. Run isolated encoder/decoder learning checks for text, audio and video; measure
   direct feature decoding separately from retrieval/context decoding so missing
   training in the latter cannot be hidden by a direct path. Retain image control.
5. Exercise local real video/audio and UTF-8 text using explicit preprocessing and
   source identities. No synthetic speech labels or invented person identity.
   Save raw metrics, output examples, traces, checkpoint and standalone report.

## Fixed development budget and gates

CPU first, one seed 61301; small batch of four examples per modality, 128 updates
per isolated branch, Adam 0.003. Up to one additional 128-update repair run after
a documented failure. Keep native small interfaces: RGB16, audio256 samples,
short byte strings and ordered four-frame clips. These are fitting diagnostics,
not held-out perceptual quality or sample-efficiency tests. Final resource use and
per-modality loss reductions are reported; no architecture superiority claim.

Mechanical checks require exact persistence/retry behavior, finite gradients and
mask exclusion. Development learning requires at least 20% relative loss reduction
per branch; text autoregressive exact match, audio error versus silence, frame
error versus mean/copy and temporal-order sensitivity remain explicit diagnostics,
not substitutes for each other. Real media use the existing local Charades source;
report the resize/resampling/chunk durations and lack of general task labels.

## Initial findings

The current FoundationModel only registers image/text input and image output.
The general agent already contains all four input modalities, but its default
audio output is only 32 samples and text is a tiny byte model. Video output is an
ordered sequence of image decodes from states, not a pretrained temporal generator.
Current joint tests demonstrate gradients/shapes, not isolated modality learning.

## Progress and results

Initial direct fit (128 updates each) passes loss-reduction checks. Text freely
reproduces four words; audio MSE is 0.00001265 versus silence 0.1800. Video MSE is
0.014726, worse than the mean-output control (0.011801), with almost no order
sensitivity. Image control also remains weak. These are tiny fitting diagnostics.

One bounded video repair comparison is now specified before execution: from the
same direct checkpoint, 128 further video-only updates with ordinary MSE versus
normalized foreground-weighted MSE (weight12 on the known synthetic object mask).
Same data, optimizer, parameters and CPU budget. Evaluate unweighted RGB, foreground
and background MSE, wrong-context sensitivity, centroid motion and reversed input.
Repair screen: RGB beats mean output; foreground error improves at least20% against
equal-duration continuation; motion direction correct in at least3/4 development
clips with displacement above1 pixel. This is not a real-video quality standard.
One split-run replica checks exact resume; other modalities' weights must stay fixed.

Actual Claude two-round conceptual review accepts isolation/provenance/null-context
controls. Its fixed shuffled-target-training suggestion was corrected: four fixed
permuted pairs can also be memorized. Evaluation uses wrong-example/zero context and
a unigram comparator instead. Sample-size limits remain explicit; no discovery or
alignment claim. No private source/data/results exported.

## Completed implementation and failures repaired

- The foundation registers audio/video input and native text/audio output alongside
  image/text. Its four categorical codes now agree with HybridMemory, which had
  silently defaulted to eight. The old combination crashed only when a real packet
  populated memory and thinking tried to read it. BeliefAgent now rejects that
  mismatch at construction for the built-in linear memory distribution.
- CandidateEncoder pools supplied boolean regions/spans from TokenBatch or
  FeaturePyramid, preserving gradients while excluding invalid/unselected values.
  There is no learned candidate discovery or cross-modal identity alignment.
  Candidate provenance retains the originating Observation; marked generated or
  recalled material cannot be committed as a new source observation.
- Native decoders accept context validity. Attend removes invalid values before
  normalization/projection, preventing masked NaNs and their gradients from leaking.
  Text context validity and causal prefix validity remain distinct. Video decoding
  validates each state, including a singleton, before checking chronological order.
- Independent replay exposed another attention tracing path in multiscale
  ConditionedBlock. Requesting weights changed its kernel. It now shares a detached
  probability diagnostic with Attend and always uses the native output path.
  Eight tests cover all four encoders in train/eval: outputs, parameter gradients
  and RNG are exactly unchanged by tracing. Probabilities are pre-dropout diagnostics.
- The first preflight failed because the session's freshly constructed scorer was
  still in training mode; explicitly setting runtime eval mode fixes this.
  A first video-control report failed after successful training because two curve
  inclusion predicates disagreed. One shared predicate now controls generation and
  embedding. Raw results/checkpoints and the original failure receipt remain intact;
  the report was repaired without training again.

## Measured outcomes

One seed, four fixed development examples per modality, CPU, 128 updates each.
Text/audio/video use separate input branches; image cannot supply their answers.
The native image control here is separate from the spatial image VAE experiments.

| Direct codec | Measured result | Diagnostic controls / scope |
|---|---|---|
| Text | CE 0.00353067; free generation 4/4 words | Zero context CE 0.39839, wrong-example 2.09283, unigram 2.30259. Memorized short words, not language understanding. |
| Audio | Waveform MSE 0.0000126474 | Silence 0.1800, training-mean 0.1350, wrong-example 0.35951. Four tones, 256 samples each at 8 kHz (32 ms), not speech/music. |
| Video, initial | RGB MSE 0.0147261 | Mean frame 0.0118009; reversed-input output change 0.000000700. Predominantly background fitting. |
| Image control | RGB MSE 0.0130804 | Mean image 0.0119434; this tiny native decoder remains weak. |

All four exceed the registered 20% loss-reduction sanity threshold. That threshold
does not require beating the mean control and therefore does not establish useful
visual reconstruction. [Direct report](../runs/modality_foundation_v1/direct/report.html).

The bounded video follow-up includes the ordinary continuation needed to distinguish
the proposed objective repair from simply adding training. Both branches receive
128 additional video-only updates from the same direct checkpoint; non-video
weights stay exact. This matched control adds one run to the initial repair budget.

| Video objective | Total RGB MSE | Foreground MSE | Background MSE | Motion direction |
|---|---:|---:|---:|---:|
| Ordinary MSE continuation | 0.00701749 | 0.0816304 | 0.00204329 | 3/4 |
| Foreground weight 12 | 0.0170099 | 0.00702448 | 0.0176755 | 4/4 |

Foreground weighting reduces object error 91.4% against equal-duration continuation
but worsens total RGB error 142.4%. It fails the registered composite repair screen
because total error does not beat the mean control. **Not adopted; weight 1 remains
the default.** The static comparison shows colored background trails/bleeding in
the weighted branch, consistent with the numerical tradeoff. These are observed
frame reconstructions with causal feature access, not future-frame predictions.
[Comparison report](../runs/modality_foundation_v1/video_weighted/report.html),
[raw screen](../runs/modality_foundation_v1/comparison.json).

### Actual World State and real inputs

Each modality separately supplies a candidate and a real packet to WorldSession;
bounded retrieval then reaches the actual BeliefAgent thinker. Source attribution,
recall without new evidence and exact session restore pass. The report's
state-path loss is **only the first example**, using an untrained state-to-output
path. It is not a matched bottleneck comparison against the four-example codec
average and does not locate irreversible information loss. A separate trained
adapter/readout comparison is still needed.

One existing Charades source, `data/memory_media_v1/episodes/0LDP7/video.mp4`, tests
real transport. Original video is 480×270, 24 fps, 27 seconds with mono 44.1 kHz
audio. The first second is resampled to four RGB16 frames and mono 8 kHz audio;
31 complete 256-sample chunks cover 0.992 seconds and the final partial chunk is
omitted explicitly. Times describe this resampled timeline. German UTF-8 strings
exercise byte handling. Separate and combined AV/text packets, one combined event,
memory access and finite outputs pass. No real media were used to update these
codec weights; there is no speech target, live webcam or identity label here.

The custom VectorEncoder input path and bounded action/task interfaces
also pass existing focused tests. Hardware drivers, software tools, physical skill
learning and speech recognition/synthesis are outside those checks.

## Verification, artifacts and use

112 focused tests pass across modalities, multiscale processing, World State,
belief, training, tasks and Run/report infrastructure. This is not the entire
repository suite. The independent audit passes 6075 individual tensor/metadata/
numeric/artifact checks, including exact complete-checkpoint equality for 128 versus
64+64 video updates, unchanged non-video tensors and source hashes. These checks
are not 6075 independent capability trials. The corrected foundation's original
96-update descriptor exercise also passes its learning/replay gate.

Historical result arrays were captured with the old traced encoder kernel.
The audit exactly reproduces those using a scoped legacy-kernel override and
separately verifies that current native/traced outputs agree exactly. Separate
`native_after_trace_fix.npz`/`.json` files retain the current outputs and measured
small deltas; original arrays, metrics and weights were not overwritten. For the
direct run the largest delta is 0.00000190735 in text logits. Training used the
native path throughout, so this diagnosis did not require retraining.

Run-owned artifacts include raw metrics, checkpoints/source snapshots, exact
output/target arrays, autoregressive strings, WAV/GIF examples, per-modality curves,
World State/attention traces and standalone reports. Structural HTML, report hashes
and embedded-media bytes pass; curves and the comparison figure were inspected.
Browser interaction QA remains unavailable under the established local-file policy.
The direct command took 11.38 seconds including evaluation/reporting; the weighted
follow-up took 6.55 seconds. This CPU check makes no scaling or compilation claim.

```bash
.venv/bin/python -m experiments.modality_audit --check --output runs/my_modality_check
.venv/bin/python -m experiments.modality_audit --output runs/my_modality_fit
.venv/bin/python -m experiments.modality_audit --modalities video --initialize runs/my_modality_fit/last.pt --output runs/my_video_continuation
```

Use fresh output directories. `--video-foreground-weight 12` selects the failed
diagnostic candidate; it is optional, not recommended as a repaired default.
`--stop-after N` pauses; `--resume RUN` resumes the same source/settings contract.
Historical runs retain executable source snapshots; the diagnostic code change
intentionally prevents resuming them under a different fingerprint.

Evidence: [112-test receipt](../runs/modality_foundation_v1/final-tests.xml),
[independent audit](../runs/modality_foundation_v1/verification.json),
[audit script](../runs/modality_foundation_v1/audit.py),
[Claude method review](../runs/reviews/modality_foundation_v1/method-response.json),
[reconciliation](../runs/reviews/modality_foundation_v1/reconcile-response.json).

Next experiments should train state-conditioned text/audio/video outputs with
matched direct-feature and memory controls, then use held-out real-modality tasks.
Long audio streams, useful speech/language, video forecasting, general cross-modal
identity and synchronized output remain capability work. Earlier VAE artifact/rate
repairs and real-entity calibration are not superseded by this modality audit.
