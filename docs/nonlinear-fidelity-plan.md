# Nonlinear output fidelity: frozen decoder range

24 September 2026. Continuation of [the handoff](architecture-continuation-handoff.md).
This bounded diagnostic precedes any new learned inducer or codec repair. It does
not repeat the affine transfer screen. Earlier target encode/decode error is a
reference, **not a lower bound** on all possible latent predictions. R1's failed
support readers do not localize representation failure; CI1's encoder adaptation
lost old behavior. Keep both codec weights frozen and separate output range,
encoding and support inference here.

## Slices and decision

1. Register protocol and meaningful red checks: affine-range projection against
   independent least squares; support-only threshold estimation and invalid inputs;
   target poisoning; frozen output/restart integrity; report failure preservation.
2. Implement one readable recipe using the existing codec, generator, ridge helper
   and renderer. No new library module, trainer, storage owner or neural update.
   Compare stored examples with a support-inferred shared threshold structure.
3. Freeze source, evaluate two untouched populations with both existing codecs,
   independently audit arrays and run the CPU suite. Stop this selected path if a
   final decoder-range reference exceeds the fidelity gate. Record the blocker;
   do not tune thresholds, train a repair or claim the architecture is complete.

## Registered populations, comparisons and gates

CPU FP32 frozen codecs from `runs/evidence_loop_final_s{17,29}/last.pt`; SVD and
projection in float64. Only pose zero is used for binary output prediction.
Development seed926101, threshold vector(.4,.4,.4). Final seeds926117/926129,
two relations with channel thresholds(.3,.45,.6) and(.55,.35,.5). These are fresh
operators in the **same threshold family**, not novel-family generalization.
Each relation gets64 independently generated support images and64 query images;
64 other smooth images check all four familiar poses. Supplied part identity,
channel alignment and support relation grouping are oracle conveniences.
One tiny smoke uses development seed, four support/query images; one full dev
invocation on codec17; four final invocations. No fitting on query targets.

Support-only arms: raw nearest whole-image support output (pixel-distance);
latent affine ridge using existing penalty.001; channel-wise monotone threshold
estimated as the midpoint between largest negative and smallest positive support
values; the same threshold applied after input codec reconstruction. Threshold
estimation explicitly rejects missing classes or contradictory/nonmonotone labels.
The hand-specified threshold family is a strong task-specific prior, not learned
induction. Whole-image shuffled support tests label dependence by fit rejection
or observed prediction degradation; rejection is not a capability result.

Evaluator-only references: target encode/decode and orthogonal projection of true
target pixels onto the **unclipped affine decoder range**. The latter is a numerical
lower bound over unrestricted latent codes for this fixed decoder, not an inference
arm or a claim about clipped/nonlinear/changed decoders. Save singular values,
rank, residual orthogonality, optimal codes and both reconstruction arrays. Use
float64 SVD with tolerance max(shape)*eps*smax; compare against a full-column
least-squares solve and flag rank truncation/poor numerical agreement. Report
all errors unclipped; bound output clipping to visualization only.

Per final relation/cell: novel-output fidelity MSE<=.002; raw support-threshold
MSE<=.001 (task identifiability control); old familiar-pose MSE<=.001 at every pose;
weights/buffers and old outputs bit-exact; optimal projection residual normal
condition <=1e-8 (max absolute residual@W); least-squares MSE difference<=1e-9;
projection error<=target encode/decode error+1e-9. Decoder-range MSE>.002 is a
concrete blocker for **any** latent inducer constrained to this decoder on these
outputs. If projection passes but encoded reconstruction fails, encoding is an
additional repair candidate; if both pass but inference fails, investigate the
inference contract. Multiple limitations may coexist. No unregistered success
claim for binary accuracy, concept acquisition or latent advantage.

## Scope, integrity and budgets

No new runtime/session API is needed for a decoder diagnostic. Population inference
uses tensors; exact saved threshold/map reload is checked. Existing correction,
stale-source and session restart contracts remain regression tests, not new claims
of a live nonlinear agent. Inference accepts support and query input only; target
poisoning must leave fitted state and predictions bit-exact. Neural model hash
and all old reconstructions are checked before/after. Retained raw instances and
fitted state have separate storage accounting; pretraining remains sunk cost.
Prepare each decoder basis once and reuse across outputs. Preserve authoritative
source arrays; evaluator-only target use is structurally separate from inference.

Each invocation <=120s, <=2GiB RSS, two CPU threads. No GPU/training. One software
repair after development, no scientific hyperparameter repair or final tuning.
Full suite <=20min; at most four public-only actual Claude Opus5.5 medium rounds,
<=10min each. Save briefs/replies/model receipts in
`runs/reviews/nonlinear_fidelity_20260924/`. Record invocation wall/CPU/RSS including
imports and reporting. Every completed run saves raw arrays, frozen checkpoint,
result before report, source snapshot and standalone report using unchanged renderer.
Report status is independent of numeric gate outcome. Browser QA is not newly
claimed. Failed experiments remain visible and immutable.

## Pre-execution review reconciliation

Actual Claude Opus5.5 medium round1 confirmed the affine-range interpretation;
round2 supplied generic SVD/threshold code and explicitly withdrew its erroneous
requirement for disjoint support/query thresholds, categorical shuffle-degradation
rule and unregistered inconsistent-support fallback. Local implementation follows
the reviewed SVD/minimum-norm equations and strict bounded threshold contract,
with additional device/finite validation and defined zero-rank helper behavior.
Within each episode support/query share a relation; only dev/final operators differ.
Adopt constant and identity diagnostics, native-decoder discrepancy, per-image
MSE/p95 and balanced pixel error (undefined for missing classes). Edge-band scores
are not needed to decide the registered mean-MSE gate and are not a new gate.
Saved exact briefs and model receipts stay in the review directory.

## Development and sole software repair

Smoke and development v1 completed with raw data/reports; all integrity checks pass.
Development decoder optimum MSE0.0371386 versus gate0.002; target encode/decode
0.0519878; raw threshold0; reconstruction-then-threshold0.0201823. This remains
development evidence, not final. No thresholds, populations or fitting settings change.

Claude round3 correctly noted that a numerically truncated SVD range cannot in
general establish a lower bound for unrestricted real codes. The recipe already
rejects rank-deficient checkpoints before completing a result. The sole allowed
software repair makes the conclusion explicitly conditional on full retained column
rank, sigma_min>100*rank_tolerance, independent Householder QR residual gap<=1e-12,
QR/SVD output max difference<=1e-8, and existing gels MSE agreement<=1e-9. Subtract
a conservative1e-8 numerical margin before comparing the minimum with the unchanged
0.002 blocker gate. This is a cross-checked numerical result, not an interval-arithmetic
proof. Record code norms and native-dtype discrepancy. Repeat development once in
v2, preserve v1, then freeze. These additional numerical checks precede final exposure;
no scientific repair, training or gate relaxation is authorized.

## Final result: concrete decoder-output blocker

Implementation `04650b8`; registration/red checks `f751a69`. Two codecs17/29 ×
fresh populations926117/926129 × two threshold operators completed, with no
scientific changes after development. All eight cells fail the0.002 decoder-output
fidelity gate. Results, raw arrays, frozen checkpoints and structurally verified
standalone reports remain in `runs/nonlinear_fidelity_final_s{17,29}_{926117,926129}_v1/`.

| Route/reference | Final pixel MSE range | Interpretation |
| --- | --- | --- |
| Optimal affine-range projection (target oracle) | 0.03078835–0.03444457 | 15.4–17.2× above output gate |
| Target encode/decode | 0.04250526–0.04746876 | Encoder adds error; not a lower bound |
| Support-only latent affine ridge | 0.08366437–0.08939971 | Inference also remains inadequate |
| Whole-image nearest remembered raw output | 0.32615153–0.35709635 | Copying does not solve fresh instances |
| Support-fitted raw pixel threshold | 0.00002035–0.00008138 | All identifiability controls pass |
| Threshold after input reconstruction | 0.01556396–0.01989746 | Outside decoder range, still misses fidelity |

Full column rank256 in both768-output decoders; no singular direction was dropped.
All numerical cross-checks pass. Native FP32 optimal-code outputs differ from
float64 projections by at most1.80e-6 per pixel. Old outputs and neural weights
remain bit-exact; every familiar-pose MSE is below0.001; fitted map/threshold reload
is exact. These establish non-mutation and replay, not learned retention.

Independent NumPy least-squares/metric audit verifies3,584 prediction images over
eight relation cells. Four separate-process target-poisoning replays preserve
fitted state/predictions exactly, reject evaluator access and pass query permutation
equivariance. The auditor's first attempt loaded an inherited GPU checkpoint without
a CPU map; it failed before producing an audit result. `audit.log` preserves that
software-audit failure; the corrected auditor uses explicit CPU loading and passes
in `audit2.log`. No model, experiment or final array changed for this repair.

Four complete final invocations total5.118s, including imports/reporting, with peak
RSS678–697MiB. Three development/smoke invocation receipts also remain. Four actual
Claude Opus5.5 medium public-only rounds took133.16s; cumulative CLI API-equivalent
estimate$0.3733 is not subscription billing. Existing codec pretraining is a sunk
cost (3000 mean+600 variance updates per seed); new neural updates0. Review receipts
and source freeze live in `runs/reviews/nonlinear_fidelity_20260924/`. Reports use
the unchanged renderer; the saved contact sheet was visually inspected, showing
smooth optimal decoder outputs against sharp target boundaries. No browser-QA claim.

Claude's final reconciliation accepts the corrected scope: a cross-checked numerical
minimum, not a rigorous interval-arithmetic bound. Pixel scale[0,1] and RGB16 layout
fix numeric tolerances. Native/extracted affine equality is required by integrity
checks and passed in every cell. The mean-MSE margin does not apply to balanced
classification error, which is descriptive and pooled across each relation population.

**Stop decision:** no inducer restricted to this fixed affine output decoder can
plausibly meet the declared mean-MSE gate on these populations. Do not spend a new
induction budget on this blocked output contract. Multiple limits coexist: encoding
and affine inference add error too. This does not block all nonlinear concept tasks,
other output heads, pixel routes, changed decoders or natural-data methods.

**User correction after completion:** this study used the separate `DetailCodec`,
not the main multiscale representation. Its scoped numerical result remains valid,
but it does not establish a blocker in the intended architecture. The suggestion
to choose a decoder-only or residual repair of this surrogate is withdrawn as the
default next step. Work must return to the existing implementation: trace the
actual multiscale/memory/consumer path and identify a demonstrated defect or missing
connection before adding anything. Keep these experiments as historical scoped
evidence, without growing the surrogate further. Final seeds926117/926129 remain
consumed. No new training or architecture change is selected.

## Reproduction and completion

Inspect existing final artifacts rather than rerunning consumed populations. For a
software-only development check in a fresh directory:

```bash
.venv/bin/python -m experiments.nonlinear_fidelity --checkpoint runs/evidence_loop_final_s17/last.pt --output runs/my_fidelity_check --count 4 --support 4
.venv/bin/python -m pytest tests/test_nonlinear_fidelity.py tests/test_representation_transfer.py
```

This is evaluation only, so there is no training/resume CLI. `last.pt` owns the frozen
codec and fitted support state; it is not a training checkpoint. The full CPU suite
completed: **866 tests passed in 647.72 seconds, exit0; source unchanged.** Log and exit/source-freeze receipts are in the review
directory. Four focused new checks and six existing transfer checks also pass.
Portable evidence: [four-cell results and audits](../ara/evidence/tables/nonlinear_fidelity_2026-09-24.json).
