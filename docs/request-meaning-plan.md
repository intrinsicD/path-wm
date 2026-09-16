# Balanced request meaning: isolate before fitting the core again

16 September2026. Suite-directed next question follows the saved request-readout
failure: source7202 often has correct content but both new request formats cannot
be followed together. Source7201 also lacks reliable content. The replay scheduler
is now available; this question needs a fixed small comparison, not another sweep.

## Preregistered diagnostic

Use the two saved `runs/request_readout_v1/seed720{1,2}/instruction` checkpoints,
unchanged. Author German requests for first color versus both colors in temporal
order. Two exact UTF-8-length-matched payload pairs (24/28 bytes), crossed with
six calibration, two validation and four entirely held-out prefix templates.
Each prefix and length occurs with both labels. These share semantic vocabulary:
this tests controlled composition, not natural language or unseen-word mastery.
A separate requested/excluded phrase-order stress pair has identical byte
histograms; report separately and never substitute for the primary direct tasks.
No model inputs receive labels, choices, IDs, format metadata or probe predictions.

Freeze each model and cache actual request-path stages on the same two opposite
calibration video contexts, observed question `.`. Read the instruction encoder,
first instruction-attention output, final task tokens and thinker working tokens.
Use fixed16-bin encoder summaries; pooling/probe capacity can hide information.
Fit existing ridge readers with calibration-only scaling and validation-only
choice from0.1/1/10. Report every regularizer, train/validation/test accuracy and
both-correct request pairs per context, raw features/predictions and saved readout
weights. No stochastic repeats of a request count as independent examples.

Controls: byte length, prefix one-hot, byte histogram; paired family-label flips
with fixed seeds9711/9712/9713 (diagnostic nulls, not a significance test). Save
actual decoder answers/EOS, content and requested-format exactness separately.
Exact frozen model/RNG preservation and causal physical-state equality are required.
Probe success is accessibility under that readout, not downstream causal use;
probe failure does not establish information loss.

## Conditional bounded repair

Only if encoder diagnostic test accuracy AND opposite-request pair accuracy are
>=80% in BOTH sources on direct requests, try interpreter-only adaptation. This
is a feasibility gate, not evidence the current interpreter uses those features.
Use original two instruction checkpoints; same new optimizer/seed and512 updates,
batch8, Adam0.001, clip5, byte CE/log259 including EOS. Compare narrow original
four requests against balanced calibration requests on the same16 calibration
videos. Train ONLY existing TaskInterpreter weights. No new neural module, intent
labels at inference, whole-core fit or request-free replay. All other parameters
and buffers must remain bit-identical; this topology bypasses task modules for
request-free calls, but also verify actual legacy outputs exactly.

Each source/arm uses the same clip-index stream, with an independent request-choice
stream; no final-checkpoint selection or hyperparameter sweep. Evaluate unseen
prefixes on all16 reserved clips and three fixed paired draws. Primary adoption:
>=80% exact answers per format AND joint pairs each draw; >=10pp joint benefit
against narrow control in both sources. Video omission/last-frame <=60% first
and sequence accuracy; request-constant control cannot pass both formats.
Keep old request-free outputs exact; no broad25-task quick pass lost or metric
(accuracy/pair/source gain) decrease>10pp. Existing failures stay visible; do not
claim general semantics or natural video understanding. If encoder feasibility
fails, stop neural fitting and retain the diagnostic result instead.

Budget: two diagnostic runs <=180s each; conditional four fits/evaluations<=600s
each plus at most six quick suites<=600s each; total formal process time<=2400s,
peak torch allocation<6GiB, artifacts<600MiB, free disk>=500MiB. CPU software checks
and review costs separate. No downloads. Development fixtures have prior reuse;
new wording templates are authored controls, not a locked external benchmark.

## Implementation and review

Reuse existing recipe, task path, ridge reader, Run/checkpoints and standalone
report renderer. Tests first: prefix/length balance and split independence, probe
label isolation, stage/physical-state/frozen preservation, and exact request-free
output preservation if fitting proceeds. Commit working code before formal runs.
Actual public-only Claude methodology critique/reconciliation is saved under
`runs/reviews/request_meaning_v1`. Code, data and measurements stay local. No
validation-color or default promotion without all declared gates.

Claude withdrew its causal-attribution objection after the scope clarification:
the probe only gates an attempted repair, and does not establish feature use.
Remaining caveats are retained: this gate has no proven search-efficiency benefit;
equal budget means equal512 updates/batch/clip stream, with actual time recorded;
ridge selection is validation-only and every regularizer is reported. No test-based
choice or mechanism claim. Initial checks fail on missing diagnostic modules.

## Diagnostic outcome and repair execution boundary

Both fixed encoder gates pass:7201 accuracy93.75%/pairs87.5%;7202 both100%.
Instruction-attention accuracy68.75%/75%, task-token81.25%/75%, working68.75%/
71.875%. Unequal feature/probe dimensions prevent a unique loss attribution.
Length/prefix controls50%/0% pairs; byte histograms100% confirm the deliberately
shared-vocabulary scope. Phrase-order stress remains poor. Actual fixed-context
answers18.75%/43.75% exact, first-word50%/100%. These two contexts are diagnostic,
not the16-clip repair endpoint. Family-flip results and all regularizers are saved.

Conditional repair now proceeds exactly as above.40 software checks pass; four
uninterrupted versus two+two smoke updates give848 exact saved-state tensor checks.
The request decoder now batches64 cached working states while preserving each
individually seeded core call. All208 saved diagnostic answers/EOS match scalar
decoding. This inference batching is fixed before formal fits; no model math,
threshold or sample population changed. Training smoke precedes this decode-only
optimization. Fresh reference suite runs will verify historical raw arrays as well.

Post-comparison diagnostic declaration: both balanced fits improve joint answers
but remain below80% (35.9375%/46.875%). Re-run the SAME frozen request diagnostic on
the two balanced checkpoints to describe how stage accessibility changed. These
are two additional short, exploratory evaluations after seeing the primary result;
they do not select weights, alter gates, or constitute independent confirmation.
The fixed initial gate and four-fit comparison remain unchanged. Charge the added
cost separately and retain calibration-overlap/shared-vocabulary limitations.

## Result — 16 September

[Report](../runs/request_meaning_v1/report.html),
[raw comparison](../runs/request_meaning_v1/comparison.json),
[verification](../runs/request_meaning_v1/verification.json).
Four512-update fits completed. No default or broad capability promotion.
Only9,936 existing interpreter parameters train out of316,765 total; encoders,
metadata encoder, world-state core, thinker and every decoder remain unchanged.

| Source / fit | First-color exact | Sequence exact | Both requests exact | Broad quick passes |
| --- | ---: | ---: | ---: | ---: |
|7201 / narrow|14.06%|25.00%|0%|0/25|
|7201 / balanced|46.88%|42.19%|35.94%|0/25|
|7202 / narrow|34.38%|37.50%|0%|1/25|
|7202 / balanced|84.38%|46.88%|46.88%|1/25|

These are worst-of-three-draw values on16 reserved clips, eight wording pairs
per clip. Wordings/draws are repeated measurements, not independent clips. The
unchanged sources also have0% joint exact. Both balanced fits exceed the narrow
control by10pp, but neither reaches the80% primary gate. Source7201 loses33.33pp
on REAL.scene.image in both fits;7202 preserves the declared broader gates.
Video omission/last-frame controls meet their limits; constant requests yield0%
joint accuracy. Phrase-order stress yields0% joint for both balanced fits.

Every full-condition generated first word is unchanged relative to its source:
3,072 exact first-word comparisons across the four fits. Mean first-color content
accuracy remains54.69%/85.42% in7201/7202. Balanced fitting raises requested word-count
accuracy35.94→85.03% and43.75→74.87%, respectively. This is partial repair of response
form, not improved video content or general instruction semantics. Word-count
correctness alone is not a successful answer and does not replace exact/EOS gates.

The exploratory post-fit probes leave encoder features bit-identical. Working-token
intent accuracy rises68.75→81.25% and71.875→75%; earlier-stage probes vary
non-monotonically. Different pooled feature dimensions and a limited linear reader
prevent a unique information-loss attribution. The input byte-histogram control is
already perfect on direct requests, so this corpus cannot establish word-order
understanding. Do not promote a probe gain to a deployed capability.

All2,208 legacy request-free output arrays are exactly unchanged across four fits;
all non-interpreter tensors and both source checkpoint files are unchanged. The
fresh suite references reproduce600 historical arrays. Clip sampler states match
across paired arms. Every run owns its checkpoint, source snapshot, raw metrics and
standalone report.19 reports and403 embedded media checks pass;43,635 raw checks
pass. The chart is visually inspected; interactive browser QA remains unavailable.

Training takes174.22s total, peak PyTorch training allocation87.4MiB (excludes CUDA
runtime/reservation). The12 formal fit/evaluation processes total1,209.86s; initial
stage diagnostics14.26s and exploratory post-fit diagnostics12.70s are separate.
Software verification579 tests passes in approximately520.15s, overlapping GPU
runs; durations are not summed as overall elapsed time. Review and smoke costs
remain separately recorded. No resource or artifact budget was exceeded.

Formal results/full579-test check use commit `0bdbbee`; saved source snapshots
retain that implementation. Final review then rejects unsupported continuous
working-belief sources in the sampled-state diagnostic, before any output is
written. Actual model execution still supports that variant. The measured sampled
sources are unaffected;41 focused checks pass after the guard, including its
negative case. The full suite was not repeated for this guard-only change.

## Next bounded question

Stop further repetitions of the four familiar requests. On the preserved balanced
models, next compare the same reserved video evidence with requests entering only
the task path versus both the observation and task paths. Hold physical evidence
and random draws fixed, separate first-word content, complete response form and
EOS, and make no new fit initially. This can test request/evidence interference
before another architecture change. The weaker-source content failure, compositional
request semantics and image-question preservation remain separate open issues.
