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
no download/copy or selection by model score. Exclude subjects of all six historical clips (five distinct groups), require metadata duration>=8s and an existing file. Sort IDs by SHA256 of
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
were corrected and acknowledged: dense/broad have equal64 image indices; there are four crossed
initializations total per arm. Static RGB/latent marginal checks already guard
single-frame artifacts; low-level temporal matching is valid for this narrow task.
Review receipts live in runs/reviews/video_diversity_v1/.


Pre-fit local lineage audit resolves Claude's remaining unknown-codec concern:
`spatial_vae_repair_v1` uses the exact hashed `C_beta0.1` export; that base declares
no pretrained source and seeded initialization. Both run manifests record COCO-only
image training, not Charades. Keep the weaker fresh-motion-task scope; this is not
universal duplicate detection. See runs/video_diversity_v1/codec-provenance.json.
The reviewer confirms per-image versus per-clip exposure and four-cell limitations.


## Results

All12 fixed fits complete:19.947s CPU training; preparation, evaluation and reports
additional. No changes to optimization, architecture or source selection after results.
The confirmation metrics below average sources equally, then the four crossed cells.

| Arm | Confirm2/4px accuracy | Confirm6/8px accuracy | Confirm6/8px range across cells |
|---|---:|---:|---:|
|small4x4|98.23%|83.61%|69.37–93.98%|
|dense4x16|99.28%|83.16%|68.52–96.52%|
|broad16x4|99.93%|77.51%|63.35–94.56%|

All three full four-cell capability gates FAIL. BROAD-minus-DENSE averages-2.502pp
across both displacement groups; worst cell/group-16.439pp, so the diversity-benefit
gate FAILS. One cell passes the complete gate in SMALL and DENSE; none in BROAD.
Higher familiar-displacement accuracy is not a full repair. All train scores100%
except one DENSE99.927%; all static controls50%. Raw alignment oracle100%, no ambiguous
confirmation cases, so these measured direction targets are recoverable from RGB.

The displacement breakdown sharpens the next question: BROAD averages99.85% at2px,
100% at4px,86.30% at6px and68.72% at8px. Wider content coverage transfers well at
trained displacement magnitudes on these new sources, but does not solve untrained
magnitudes; it worsens their mean in this comparison. This is not proof that data
breadth is harmful in general, or that architecture, encoder access or optimizer
implicit bias is the unique cause. Four clips/cells are descriptive evidence only.
Historical monitored-source performance differs substantially, so preserve those
results rather than replacing them with the easier fresh familiar-displacement group.

Each fit samples4096 pairs. SMALL visits16 images/1433 distinct pairs; DENSE/BROAD
visit64 images/2966 distinct pairs. DENSE and BROAD have exactly identical sampled
integer pair sequences and per-image counts, with different content/source grouping.
Their per-clip counts differ intentionally. All four initialization cells share these
sampling patterns; model initial states match across arms within each cell.

Validation:74 unique scoped tests;3571 exact8 versus4+4 restart comparisons including
source attribution and exposure;14771 independent metric/artifact checks;3312 exact
RGB/feature marginal identities across prepared populations.22 source-file hashes
and72 cached tensors unchanged. All16 baseline normal-logit arrays (four cells x four
legacy populations) exactly reproduce the previous comparison.15 reports structurally
verified; comparison panel inspected, browser interaction not checked. About29.1MB
artifacts; no images/videos copied and no new model downloads.

Post-run code review caught a guard gap: confirmation could be accepted when another
split's subject was missing, or a source ID was reused for a different file. Added a
failing regression case, fixed both guards and reran23 affected tests.210 field checks
show the stricter reader returns the same valid experimental records. It changes no
fitted data/model or outcomes. Exact replay is recorded for the experiment snapshot
before this input-only guard fix; original snapshots remain available.

Two actual-Claude public reviews were reconciled. The remaining unknown-codec concern
was resolved locally through the recorded COCO-only initialization/continuation chain.
No private code, images or measurements were sent; metadata/hash grouping still does
not guarantee universal scene/near-duplicate independence.

[Comparison report](../runs/video_diversity_v1/report.html),
[verification](../runs/video_diversity_v1/verification.json),
[per-displacement/source diagnostics](../runs/video_diversity_v1/displacement-diagnostics.json),
[exposure counts](../runs/video_diversity_v1/exposure.json).

Next proposed bounded comparison: vary displacement coverage during training while
holding source content and model size fixed. Distinguish trained magnitudes, unseen
intermediate magnitudes and extrapolation; retain this now-inspected confirmation
set as development evidence and reserve new sources for a further confirmation.
Do not call displacement magnitude speed without varying time intervals. This is a
proposal, not an extra unregistered run or a guarantee of repair.

### Running a configured population

```bash
.venv/bin/python -m experiments.video_order --output runs/my_motion_diversity \
  --source-manifest data/motion_diversity_v1/broad.json --balanced-training \
  --seed 7600 --head-seed 7501 --steps 512 \
  --temporal-source runs/video_context_v1/seed7401/k3/history/last.pt
```

The manifest lists explicit paths, file hashes, subjects and frame counts under train,
validation,evaluation,confirmation. Source IDs/subject labels stay out of model inputs.
Prepared data identities, per-source metrics and `exposure.npz` make the actual images
and sampled pairs inspectable. Omitting the manifest retains the original six-source
recipe. Earlier runs require their saved code snapshots for exact resume.
