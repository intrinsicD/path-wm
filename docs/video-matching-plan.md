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
