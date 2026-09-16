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
