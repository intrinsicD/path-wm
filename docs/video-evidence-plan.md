# Shared evidence scoring for horizontal direction

16 September. Follow-up to [correspondence support](video-matching-plan.md). User
clarified that the structured direction reader precedes code/compilation cleanup.
Question: can a small learned opposite-offset comparison retain the useful fixed
cosine rule and improve on a generic appearance/temporal/correlation head?

## Fixed protocol before implementation/fits

Keep image encoder/decoder unchanged and frozen. Use existing radius3 horizontal
cosine volume with common margin3 (12x12 latent ->6x6 interior). Structured head:
spatial mean per offset; shared candidate inputs[cosine,abs(offset)/3,zero-offset
cosine]; Linear3->16,SiLU,bias-free Linear16->1 residual. Zero-initialize last layer,
so initial candidate scores equal cosine. Same scorer for every signed offset.
No signed offset, appearance, temporal feature or class bias supplied. Compare the
maximum negative and maximum positive candidate score; logits[-d/2,d/2] with
 d=exp(clamp(log_scale,-4,6))*(max_negative-max_positive), log_scale starts0.
Only last2 frames of the usual3-frame input used. No artificial partner at inference.
The absolute-offset/cosine residual can change decisions; it is not just a global
monotonic recalibration. Channel reversal swaps logits, but this is NOT guaranteed
video-time reversal equivariance. This is a task-specific geometry prior.

Implement one small CorrespondenceDirectionHead in pathwm/models/video_vae.py and
optional readout='evidence' in existing OrderReadout/recipe; default remains pooled.
Evidence mode requires mode=correlation and full active radius; no unused temporal
module/weights allocated. Model loading/resume records readout/radii, and source
metadata identifies temporal weights as unused for evidence. No new trainer/report
framework, no attention, larger encoder or decoder changes.

Compare POOLED (existing1630 parameters, temporal source7401) and EVIDENCE (81 learned
parameters; no temporal source). Two independent head seeds7501/7502, sampler7600,
same16 training clips x4frames,48 cyclic phases,train2/4/6/8px,512updates x8pairs,
AdamW.003/wd.0001,clip1,CE+0.1 pair-offset SmoothL1,CPU FP32. Parameters/compute and
initial weights differ between architectures; do NOT claim a matched-capacity causal
isolation. Initial EVIDENCE predictions equal the no-fit cosine reference. Report
fixed rule separately and preserve two exact POOLED replays of earlier same-seed
wide models where tensors/populations match. No duplicate temporal-seed cells for a
head that has no temporal weights. No epoch selection or coefficient sweep.

All17 earlier evaluation/confirmation sources become development. Select4 new existing
Charades clips by SHA256('pathwm-motion-evidence-v1:'+ID),duration>=8s,file exists,
unique new subjects; exclude ALL subjects of34 prior clips. Save selection/hash before
fits. Historical validation shares a training subject; monitoring only. Fresh groups
2/4,6/8,3/5/7,10/11px. Keep raw ambiguous cases and report them. Compute/check RGB and
encoded single-frame class marginals. Source identity guards do not prove universal
pretraining/near-duplicate independence. All formal fits fixed before fresh scores are
inspected; per-run automatic scoring is retained without influencing later fits.

Full capability: BOTH seeds and all4 fresh groups have source-macro accuracy>=.90,
pair>=.80,flip>=.90,each-source accuracy>=.80; current/previous/unordered accuracy.50,
current/unordered pair0, swapped-prefix outputs exactly pair-swapped.
Structured-vs-pooled benefit: mean intermediate/extrapolation gain>=3pp, no fresh
seed/group or inspected-source seed/group regression>2pp. All17 inspected sources
individually gated. Separate learning-vs-fixed benefit: mean over all4 fresh groups
>=0.5pp gain, no fresh seed/group or inspected-source seed/group regression>1pp.
Adoption requires structured capability AND both benefits. Finite engineering screens,
no statistical or broad video-capability claim. Failure of learning-vs-fixed means we
must not sell retained hand-designed competence as learned improvement.

Mandatory diagnostics: raw logits, per-source/magnitude/group metrics and matched pair
exposure; initial/final candidate scores, corrections/corruptions and unchanged decisions
vs fixed rule, residual magnitude and positive scale; zero/nonzero residual gradients,
channel-reversal error before/after learning, single-sequence batch independence.
Static confidence is descriptive (static motion is outside the two-class training task),
not proof of calibration; report max probability and >=.95 fraction for controls.

Budget:4 formal fits<=45s each (<=180s training), one8 versus4+4 mechanics check.
No additional score-dependent fits. Cache frozen features once for formal comparison;
fixed cosine evaluation is zero fits. <=80MiB artifacts,>=300MiB free disk; no media
copies/downloads. Run85 existing scoped checks plus new meaningful tests, real exact
resume, independent raw/exposure/gate audits, existing standalone reports and panel
inspection. No renderer change. Preserve all negative results/defaults.

## Plan/check order

1. Red tests: initial cosine equality/sign, tied candidate sharing, reversal after
   nonzero weights, equal sides, decision-changing residual, gradients after one
   optimizer update, radius/config validation, untouched pooled path, input isolation.
2. Commit plan/red checks; implement optional small head and trace/config plumbing.
3. Verify tiny real8 versus4+4 replay/reports, then commit working slice.
4. Run fixed4-fit comparison and no-fit baseline. Audit raw results, preserve regressions,
   update state/atlas and record scoped results. Claude reviews public methods only.

## Claude methodology reconciliation

Actual Claude reviewed a public hypothetical brief, then acknowledged that a shared
scorer receiving only ABSOLUTE offsets guarantees channel-set reversal symmetry.
Its original signed-input concern does not apply to this chosen interface. We accept
that residual weighting can still overfit (symmetry is not generalization), report both
seed values/ranges, and explicitly exclude temporal corruption, natural motion and
tracking capability claims. Scale is bounded by the declared log clamp; static confidence
remains diagnostic. Receipts/briefs under runs/reviews/video_evidence_v1. No private code,
media or measured results were exported. These limits do not warrant extra tuning/fits.

## Mechanics before formal fits

89 scoped tests and Ruff pass. Real8 versus4+4 evidence-readout training passes10239
exact checks (model,optimizer,RNG,sampler,training rows,raw predictions/exposure and
non-confirmation scores). Reports structurally verified; no renderer changes.
The two readout architectures differ intentionally; generic default remains intact.

## Results (16 September)

Four512-update fits complete in5.226s measured CPU training;21.068s preparation and
85.282s through formal preparation/training/evaluation/reporting. Zero-step fixed-rule
report and8.814s candidate diagnostics follow. Same4096 sampled pairs per trained fit;
POOLED1630 versus EVIDENCE81 parameters. No encoder/decoder update or default promotion.

Fresh source-macro accuracy, mean and range across the TWO head seeds:

| Group | POOLED | EVIDENCE | Fixed cosine, no fit |
|---|---:|---:|---:|
| trained2/4px |99.45% (99.32–99.58)|99.69% (99.67–99.71)|99.54%|
| trained6/8px |99.45% (99.32–99.58)|100% (both)|100%|
| unseen3/5/7px |99.99% (99.98–100)|100% (both)|100%|
| unseen10/11px |92.53% (91.99–93.07)|100% (both)|100%|

EVIDENCE and FIXED pass the full fresh capability screen; POOLED fails. Neither benefit
screen passes, so the composite adoption gate fails. EVIDENCE-vs-POOLED transfer mean
+3.741pp and fresh nonregression pass, but worst inspected source/group regresses5.21pp
(A8LZE,2/4px,head7502). EVIDENCE-vs-FIXED gains only0.0366pp averaged across4 fresh groups
and has a worst inspected regression14.714pp (6RQHT,10/11px,head7501). This is a source
preservation failure, despite excellent new-source accuracy. No trained-motion learning
claim is inferred from initial hand-designed geometry or near-ceiling fresh performance.

Learning really changes the scoring function: final residual maxima5.90/5.11 and
positive scales1.679/1.662; channel-reversal error stays exactly0. On fresh sources,
only2/4px decisions change: seed7501 corrects6 and corrupts1 of3072 clips; seed7502
corrects5 and corrupts1. Other fresh groups make exactly the fixed rule's decisions.
Static controls have50% accuracy as required, but1.69–1.82% of current/previous-only
examples receive>=95% confidence after learning, versus0 for unscaled fixed cosine.
This flags a limitation, not a calibrated probability comparison across scales.

Post-result diagnosis (no extra fit or gate change): on6RQHT at10/11px, fixed accuracy
100% becomes85.29/85.55%; the learned residual corrupts113/111 of768 decisions and
corrects none. Among those corruptions,112/110 winning candidates move to absolute
offset1, one to2, none remain at3. Original winners were mostly3 (89/88) or2 (24/23).
Thus offset-dependent refinement can override useful wider matches in these cases.
The stored centered residuals remove irrelevant common shifts; a large raw residual
alone is not the diagnosis. This does not prove a unique optimizer or generalization
cause. See failure-diagnosis.json for exact candidate vectors and indices.

89 tests,10239 exact restart checks,42682 independent raw/gate/exposure checks,
36 exact prior-POOLED model/prediction comparisons,6120 RGB/encoded marginal checks,
38 source-file hashes and40 immutable cached tensors pass.8 reports structurally
verified, panel inspected; no browser-interaction claim. Artifacts ~48.6MB (<80MiB).
Actual-Claude public review/reconciliation receipts retained; no private export.

The optional structured head is implemented and usable. For this task the fixed rule
remains a strong reference; extra learning has not earned replacement. Next proposed
bounded question: separate decision-preserving evidence calibration from unconstrained
offset-dependent corrections, and require inspected-source preservation before enabling
rank-changing refinements. Do not respond by simply enlarging the encoder or stacking
more layers. Natural motion, temporal corruption, tracking, stationary/unknown classes,
streaming and agent integration remain open, separate from this two-direction probe.

## User path

```bash
.venv/bin/python -m experiments.video_order \
  --output runs/my_direction_evidence \
  --source-manifest data/motion_evidence_v1/sources.json \
  --displacement-spec data/motion_evidence_v1/expanded.json \
  --balanced-training --matched-phase-sampling --mode correlation \
  --correlation-radius 3 --active-radius 3 --readout evidence \
  --pair-center-weight 0.1 --head-seed 7501 --seed 7600 --steps 512
```

Use --readout pooled for the existing head; supply temporal-source for its saved
initialization. Evidence does not allocate/use temporal weights. --steps0 evaluates
the initial fixed cosine decision rule without fitting; its seed initializes dormant
residual weights but does not change initial outputs. Run resume checks readout/radii,
objective, source/config/code/environment. Main artifacts: report.html,result.json,
comparison.json,decision-changes.json,static-confidence.json,failure-diagnosis.json,
verification.json and each model's evaluation.npz/evidence.npz/checkpoint/report.
