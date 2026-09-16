# Source diversity versus correlated frame count

16 September; follow-up to [initialization comparison](video-stability-plan.md).
Keep architecture, image codec, temporal/head initialization and optimization fixed.
Test source-content coverage before increasing model size or update duration.

## Protocol fixed before fitting

Three populations, each crossed with temporal7401/7402 and head7501/7502:

| Arm | Source clips | Frames per clip | Images | Paired phase examples |
|---|---:|---:|---:|---:|
|small|4 historical train|4|16|1536|
|dense|same4|16|64|6144|
|broad|same4 plus12|4|64|6144|

All frames use fps2, resize80x64, top-left crop origin(4,12) to48x48. Small/broad
decode the first2s, dense first8s. Each image supplies all48 circular phases and
displacements2/4px. No reflection or correlation augmentation. Controlled periodic
pans of real content; no natural motion, forecasting or agent-integration claim.
BROAD versus DENSE is the matched-image-count diversity contrast; SMALL is an anchor.
More time coverage within dense is part of the intervention and not scene diversity.
Same512 updates,8 pairs, sampling seed7600, AdamW0.003/wd0.0001, clip1. Match integer
pair indices for dense/broad and all initializations. Uniform sampling means equal
expected per-image exposure in dense/broad, but4x per-clip exposure in dense.
Record realized source/image/pair exposure. Epoch counts differ for small.

Selection uses existing local Charades official-train metadata, raw video files;
no download/copy or selection by model score. Exclude all six historical subject
groups, require metadata duration>=8s and an existing file. Sort IDs by SHA256 of
`pathwm-motion-diversity-v1:` + ID; take one clip per previously unseen subject.
First12 expand training: TIGIP,TYHA8,AXJUB,DPCGS,SG8ZR,R979C,CAND1,QO7SM,S3FY2,
CF92N,A2771,FNM0L. Next4 confirmation:0XP8L,7YV59,L8HMR,DPKMU;4 frames each.
Persist source metadata/hashes, frame/image provenance and manifests before fits.
Reject cross-split exact file/crop duplicates or confirmation-subject overlap;
do not silently replace a problematic example after evaluation. Within-source
repeated/ambiguous images remain, with oracle ambiguity reported.

Historical validation12VVC shares subjectKFGP with train0EJAG. Keep it only for
development curves; no stopping/selection. Historical1KKYX remains known/wide
development monitoring. Four new subject-disjoint confirmation clips are reported
separately (`confirm_known`, `confirm_wide`), never used for updates/selection.
Fresh means unused by this motion-task experiment, not guaranteed absent from every
upstream dataset or previous unrelated audit. Metadata/file hashes do not certify
independent households/scenes. Every source result and macro-average is visible.

Twelve formal fits,45s training cap each,<=540s total. No conditional extra fits,
hyperparameter changes or checkpoint selection after results. Eight versus4+4
restart is the only extra fitting, for mechanics. <=50MiB output artifacts,
>=300MiB free disk, no preserved runs removed. Precompute frozen features once per
population. Do not inspect confirmation model scores until all12 fits complete.

Capability gate per arm: every initialization cell, both confirmation displacement
groups: source-macro accuracy>=90%, pair accuracy>=80%, prefix-flip>=90%, every
individual source accuracy>=80%; current/previous/unordered exactly50%, current and
unordered pair accuracy0. Diversity-benefit gate: broad minus dense mean>=3pp across
the8 cell/group comparisons, no cell/group regression>2pp. Record BOTH gates.
Four cells at one sampler and four confirmation clips support descriptive contrasts,
not population uncertainty/significance. Keep source/cell ranges, errors and margins.

## Implementation and review

Reuse `experiments.video_order`, Run and unchanged reports. Add an optional small
source manifest, source/frame validation and attribution, per-source metrics, and
actual sampled-exposure logging. No new training framework or model module.
Test duplicate file/subject guards, frame-count requirements, confirmation gating
independent of training labels, exact source macro arithmetic and sampling exposure.
Run paired-direction, video-codec and image-codec regressions plus exact restart.

Claude reviewed a public-only hypothetical protocol; no code, photos or measured
results exported. Its mistaken per-frame4x repetition and four-seeds-per-cell claims
are being corrected: dense/broad have equal64 image indices; there are four crossed
initializations total per arm. Static RGB/latent marginal checks already guard
single-frame artifacts; low-level temporal matching is valid for this narrow task.
Review receipts live in runs/reviews/video_diversity_v1/.
