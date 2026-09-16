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
