# Explicit correspondence support at matched readout capacity

16 September. Follow-up to [pair centering](video-pair-center-plan.md).
Hypothesis: missing correspondence range contributes to larger-shift errors. Compare
exposed matching support while holding representation, readout shape and crop fixed.

## Protocol before fits

Reuse the existing frozen image VAE (4x12x12 posterior means for48x48 RGB; stride4),
CausalLatentMixer and local_correlation. Three arms share radius3 computation, valid
interior margin3, and seven reserved cosine-match channels at offsets[-3..3]:
NONE (mode=train) exposes zeros; LOCAL (mode=correlation,active_radius2) exposes[-2..2]
and zeros two outer slots; WIDE (mode=correlation,active_radius3) exposes all seven.
Current/temporal features are cropped to the same6x6 interior in every arm. No wrap,
padding, extra learned matching weights or inferred ground-truth displacement in inputs.
All arms execute the same radius3 correlation computation before masking. Equal nominal
parameter count and forward compute; useful input channels/active parameter gradients
intentionally differ. This is not equal information access. Two extra reserved columns
and common margin3 change the reference vs previous margin2/five-channel experiments;
compare WITHIN this study, retain old results, do not claim exact historical fit replay.

Same16 training clips x4frames, exhaustive48 cyclic phases, opposite-prefix partners,
train magnitudes2/4/6/8px. Evaluation2/4,6/8,3/5/7,10/11;fixed dt, first jump2d<=22<24.
Four crossed cells temporal7401/7402 x head7501/7502; sampler7600,512updates x8pairs,
matched phase sampling,AdamW0.003,wd0.0001,clip1,CPU FP32. Loss fixed in ALL arms to
CE+0.1 per-pair offset SmoothL1. This is an experimental fixed objective, not default
adoption of the failed prior repair. Twelve fits total; same initialization and sampled
pairs per cell. Each inference gets only its own three-frame clip; no partner, labels,
metadata, test-time calibration or artificial-reversal constraint. Matching uses only
previous/current feature grids; this does not give a temporal prediction model.

All13 prior evaluation/confirmation clips are development (historical1KKYX plus12
inspected clips). Historical validation remains monitoring only with shared subject.
Select4 new confirmation clips from existing local Charades metadata by SHA256 of
`pathwm-motion-matching-v1:`+ID, length>=8s,file exists, unique new subjects, excluding
ALL subjects of30 previously used clips. Save selection/source hashes before training.
No score selection/replacement. Subject/file/crop checks are task-level leakage guards,
not universal pretraining/near-duplicate guarantees. Raw RGB and encoded per-frame
class-marginal equality checked; ambiguous raw examples retained and reported.

Full capability: ALL four cells and all four fresh groups meet source-macro accuracy
>=90%,pair>=80%,flip>=90%,each-source accuracy>=80%; current/previous/unordered50%,
current/unordered pair0 and swapped-prefix outputs exactly pair-swapped. Benefit gates
are separate WIDE-vs-LOCAL and WIDE-vs-NONE: mean intermediate/extrapolation gain>=3pp,
no fresh cell/group loss>2pp and no inspected-source cell/group loss>2pp (all13 sources,
not a masking macro-average). Adoption requires WIDE full capability AND both benefits.
These are finite engineering screens, not significance tests or seed-population claims.

Diagnostic, fixed before results: frozen feature cosine volumes on common margin3,
support2 and3. For each clip, sign(max negative-offset mean cosine - max positive-offset
mean cosine) predicts direction; exact zero resolves to class0 and ties within1e-6
are reported, never excluded. Also report zero-offset-best fraction including ties.
These are hand-defined alignment readouts, not learned dynamics. Unit tests verify
coordinate sign with known nonwrapped shifts. Store scores/labels/source attribution,
per-source/per-magnitude accuracy and frozen-grid versus raw RGB matching comparisons.
No oracle-derived feature/threshold injected into training. Compare single-sequence
classification, common offset/relative bias, ordering and margins for learned arms.

Budget:12 formal fits,45s each,<=540s training. Only extra fit8 versus4+4 for mechanics.
<=120MiB run artifacts,>=300MiB disk reserve; no source copying/downloads. Cache frozen
features once; time preparation/evaluation/reporting separately. Do not inspect fresh
model scores until all12 complete. No conditional fits, tuning, early stopping or
checkpoint selection. Preserve unsuccessful candidates. Same underlying image codec
and original checkpoints remain intact. Controlled horizontal periodic pans only:
no natural motion, dense optical flow, streaming, forecasting or agent capability claim.

## Implementation

Extend OrderReadout in experiments/video_order.py with configurable correlation_radius
and active_radius, defaults2/present radius. Keep existing local_correlation primitive
in pathwm/models/video_vae.py unchanged. Add CLI/resume identity fields. No new trainer,
model hierarchy or report renderer. Tests first: cosine sign/reference/gradients/bounds,
common crop/parameter/masking contracts, input isolation, invalid radius combinations.
Run existing video/image/modality regressions and real exact restart; independent raw
metric/gate/exposure/source audits. Claude reviews public hypothetical methods only;
review receipts under runs/reviews/video_matching_v1, no private exports.

## Independent methodology review

Actual Claude CLI review and two concise reconciliation replies are retained under
runs/reviews/video_matching_v1 (public hypothetical methodology only). Accepted:
nominal parameter count is not equal effective capacity; disabled head-column data
gradients must be zero, although weight decay can still act. Auxiliary centering reads
only final logits after masking and gives no hidden wide-correlation route. Claude
withdrew the incorrect claim that10/11px exceeds radius3 at stride4. We retain the
important fractional-grid/nonlinear-encoder caveat and mandatory frozen cosine diagnostic.
We do not adopt its proposed conditional exclusion gate: a failed simple cosine reader
cannot prove unrecoverability, and excluding difficult cases would weaken this evaluation.
Means, per-source/magnitude breakdowns, thresholds and diagnostic aggregation were fixed
above. Remaining concern about overinterpreting aggregates is addressed through scope:
a trained classifier with a frozen image encoder on finite constructed pans, not a
frozen classifier, universal equivariance or natural video understanding.

## Execution notes before formal fits

85 scoped video/image/modality tests and Ruff pass. Real8 versus4+4 restart passes
8661 exact checks at the specified0.1 objective, including model/optimizer/RNG, sampler,
raw predictions and exposure. The ignored orchestration helper initially duplicated
mode kwargs and failed after preparation, before ANY formal fit. Inspection also found
the first smoke had inadvertently used default CE; it is preserved separately under
check-pre-objective-fix. Fixed the helper and repeated the same8 versus4+4 mechanics
with0.1. This adds16 smoke updates beyond the original mechanics budget, not another
scientific fit or coefficient search. Both logs remain. The12 formal fits are unchanged.
Per-run automatic evaluation writes fresh scores to artifacts; they are not inspected
until all12 fixed fits finish (no score-dependent stopping/selection). Preparation and
scoring/reporting are timed separately from the per-fit45s training cap.

## Result (16 September)

Twelve512-update fits completed:22.144s CPU training,21.786s preparation,
243.564s through preparation/training/evaluation/reports; mandatory alignment diagnostic
2.066s separately. Same1630 nominal parameters and4096 sampled pairs per fit.
Frozen-cache, initialization, sample/exposure and raw metric checks pass.

Four fresh sources, source-macro accuracy averaged over the four temporal/head cells:

| Displacement | NONE | LOCAL +/-2 | WIDE +/-3 | Fixed cosine +/-3 |
|---|---:|---:|---:|---:|
| trained2/4px | 98.00% | 95.22% | 96.62% | 96.71% |
| trained6/8px | 94.57% | 97.40% | 95.78% | 99.25% |
| untrained3/5/7px | 97.44% | 97.57% | 97.64% | 99.93% |
| untrained10/11px | 78.75% | 86.77% | 90.07% | 99.93% |

The fixed cosine column is an untrained, explicitly designed direction rule on the SAME
frozen grids; it is not another fit, learned dynamics, or universal video capability.
Radius2 fixed cosine reaches94.69% on10/11px versus99.93% at radius3. Raw pixel reference
is100% on all fresh groups, with no ambiguous examples removed. Fresh radius3 large-shift
accuracy per source is at least99.74%; the thirteen inspected sources average99.82%, with
minimum97.66%. The old1KKYX source is100% for this rule at10/11px. At least this task's
useful directional evidence remains accessible in the frozen encoder output. A failed
learned readout is therefore not evidence that the encoder destroyed all motion cues.
This does not localize every individual mistake or establish arbitrary feature equivariance.

ALL trained-arm full capability gates fail. WIDE minus NONE mean intermediate/large
transfer gain is5.758pp, but worst fresh cell/group-2.73pp and worst inspected source
-28.26pp fail preservation. WIDE minus LOCAL transfer gain1.685pp, worst fresh-3.91pp
and inspected-26.82pp also fails. A bigger explicit search window helps large shifts
on average here, but concatenation into the generic learned head is not a reliable fix.
No default promotion, decoder changes, coefficient search or extra scientific fits.

85 tests,8661 exact restart checks at the specified objective,92229 independent raw
checks,5256 frame-marginal comparisons and34 source-file hashes pass.17 reports are
structurally verified (includes preserved pre-fix mechanics); comparison panel inspected.
Browser interactions were not validated; renderer unchanged. Artifacts ~72.6MB within
120MiB cap. Reports and weights remain under runs/video_matching_v1, including failed
setup and pre-objective smoke. Main outputs: report.html,result.json,comparison.json,
alignment.npz/alignment.json,verification.json and the twelve per-fit reports/checkpoints.

Next proposed experiment: a structured, learnable direction readout that explicitly
compares opposite-offset matching evidence before combining appearance/context. Compare
against this generic head AND the fixed cosine rule; test small displacements as well
as large ones. Keep single-sequence inference, no forced partner at inference, fresh
confirmation and source-level preservation. Do not simply enlarge convolution kernels
or retrain the image codec from these results. Natural motion, object tracking, temporal
forecasting, streaming and integration into the agent core remain separate open tests.

## Reproduce one arm

```bash
.venv/bin/python -m experiments.video_order \
  --output runs/my_matching_wide \
  --source-manifest data/motion_matching_v1/sources.json \
  --displacement-spec data/motion_matching_v1/expanded.json \
  --temporal-source runs/video_context_v1/seed7401/k3/history/last.pt \
  --balanced-training --matched-phase-sampling \
  --mode correlation --correlation-radius 3 --active-radius 3 \
  --pair-center-weight 0.1 --head-seed 7501 --seed 7600 --steps 512
```

Use active-radius2 for LOCAL, or mode train for NONE while keeping radius3.
The ignored execute.py/analyze.py record this exact bounded orchestration and audit;
the supported user entry point remains the existing recipe. Defaults remain radius2
and no paired auxiliary loss. Model loading uses saved radius/active radius; old
checkpoints default to radius2. Resume validates objective and radius settings.
