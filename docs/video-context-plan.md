# Temporal context through frozen image features

16 September. Follow-up to the shared image/video codec and the user's spatial
neighborhood question. Test useful history for a missing current-frame region,
not general motion understanding or future generation. Keep the existing codec,
Run/checkpoint/report infrastructure and original image checkpoint.

## Plan before implementation

- Extend the optional causal mean mixer with an odd spatial kernel and optional
  repeated shared spatial-only residual block. Allow injecting a mixer into
  VideoVAE. Preserve default behavior and legacy state-dict keys.
- Essential tests: neutral initialization, future/invalid-input invariance,
  spatial gradient reach, unchanged three-frame temporal horizon, shared loop
  parameters, and restoration. Run these before training.
- Extend the existing video recipe with a separate `inpaint` task. Freeze image
  encoder AND decoder; gradients through the decoder still train the temporal
  module. Last frame has a centered16x16 RGB patch replaced by0.5 before encoding.
  Prior frames are visible. Targets never enter the model.48x48 decoded real frames,
  same six previously inspected source-disjoint development videos as before.
- Arms: spatial3x3, spatial5x5, spatial3x3 plus two iterations of one shared
  spatial-only residual convolution. All see current/two previous frames only.
  Seeds7401/7402;256 updates;batch2;AdamW lr0.001;gradient clip1. Equal exposure,
  not equal parameter count or compute. Record both. No hyperparameter selection.
- Objective: deterministic posterior-mean RGB reconstruction with4x weight on
  masked pixels and1x outside, normalized by weighted RGB pixel count. No KL or
  image fine-tuning; this is conditional repair on a fixed codec, not VAE fitting.
  Include paired clean clips in the loss to constrain regression.
- Controls: frozen occluded-frame reconstruction, clean frozen image reference,
  wrong history and zero history, plus a trained current-only control for EACH
  architecture with past RGB inputs zeroed (black frames, same timestamps/validity).
  Each control matches its architecture's parameters and compute.
- Gate for an expanded candidate: >=5% masked RGB MSE reduction versus3x3 AND
  its own trained current-only on the reserved development source, in BOTH seeds;
  <=2% relative regression on clean and outside-mask MSE versus3x3; >=5% masked
  MSE benefit of correct history over zero AND shuffled history. Also report
  these controls for3x3. Thresholds are a development screen, not a significance
  claim. Preserve all failed results; default unchanged unless separately adopted.
- Budget: twelve fits, <=60s each, <=720s total formal training; one <=8-update
  smoke/resume check per necessary path, no extra efficacy fits. <=30MiB new
  artifacts initially estimated; after the completed8-step report/checkpoint smoke,
  allow <=36MiB new artifacts including both restart-check reports, while retaining
  >=300MiB free disk. This budget refinement precedes formal runs; no efficacy
  settings change. Stop explicitly if that reserve cannot hold.
- Record protocol/source identities, raw metrics, small reconstruction/error
  examples, standalone reports and exact restart check. No report-renderer change.
  Compare source weights before/after. Actual Claude reviews a public generic
  methodology brief only; save receipts and reconcile material concerns.

Current-only is included before execution to distinguish ordinary spatial inpainting
from useful temporal context. The tiny single reserved source and fixed mask are
development limitations; broad motion, variable masks and scaling remain untested.

Claude's public review recommends matched spatial-only controls for every enlarged
architecture; added above before runs, preserving total time/disk budget. The
encoder processes masked frames independently with per-frame normalization: observed
surrounding pixels and real past frames are permitted context, not target leakage.
Wrong history means a different clip's past, not reversed order; no order-sensitivity
claim. Mask size stays fixed before results, not selected on the reserved source.

Reconciliation: measure dense Conv2d/Conv3d MACs by executed module hooks for one
batch, including each shared iteration, and wall time separately. Current-only
arms execute the same shapes/operations, not a sparse/shortened path; compare the
recorded MACs within each pair. Report excludes normalization/activation/backward,
so it is an approximate compute measure, not total FLOPs. No remaining design
disagreement. To fit disk budget, checkpoints store trainable temporal weights plus
optimizer/RNG; the frozen image checkpoint is referenced by path and SHA256.

Before formal runs:68 scoped tests pass (12 video wrapper/task checks plus56 spatial
VAE/modality regressions). The two new interface tests first failed informatively
on absent configuration arguments. An8-step versus4+4 shared-refinement run passes
1447 exact checkpoint/optimizer/RNG/train-row/evaluation checks. Timing and extra
intermediate validation rows are excluded. Only this turn's completed pytest temp
directory was removed to recover its120MiB; historical runs and data are preserved.

## Result,16 September

All twelve fixed256-update fits completed in84.1408s CPU training including periodic
validation (120.16s full orchestration). Source image weights and their loaded state
hash are unchanged. The opt-in module accepts spatial kernel and shared spatial-only
iterations; default3x3/zero iterations preserves legacy weights and behavior.

| Seed | Architecture | Masked MSE, history | Matched current-only | History benefit vs current-only | Change vs3x3 history |
|---|---|---:|---:|---:|---:|
|7401|3x3|0.041020|0.094754|56.71%|reference|
|7401|5x5|0.049223|0.083695|41.19%|20.00% worse|
|7401|3x3 + shared loop2|0.042468|0.092276|53.98%|3.53% worse|
|7402|3x3|0.049214|0.093471|47.35%|reference|
|7402|5x5|0.052253|0.074730|30.08%|6.17% worse|
|7402|3x3 + shared loop2|0.054966|0.090984|39.59%|11.69% worse|

Both expanded candidates fail the registered gate.3x3 uses1340 temporal parameters,
5x5 uses3644, shared loop2 uses1924. Measured dense convolution MACs per batch are
264.63M,267.28M,265.96M respectively including the shared image codec; each has exactly
the same MAC count as its own current-only control. Other operations/backward are
excluded. This is not a matched-resource comparison between different architectures.

Correct history gives only0.48–1.32% less masked error than different-clip history,
below the5% screen for all variants/seeds.3x3 specifically gives1.32%/0.86% benefit.
Consequently the much larger improvement versus blank-history training does not
demonstrate detailed temporal matching or motion understanding. Other clips from the
same reserved video share scene content; generic appearance reuse is a plausible,
untested explanation. Correct-history benefit over black-history evaluation is
13–38%, which still does not establish sensitivity to the correct sequence.

Current-only counterparts are matched architecture controls, not newly weakened
models. All masks are applied to RGB before encoding, with clean targets used only
in losses/metrics. Frozen image components carry no gradients and their state hashes
match the source. The clean direct image path is unchanged. Video clean-input errors
are reported separately; no blanket image/video-quality improvement is claimed.

[Standalone comparison](../runs/video_context_v1/report.html), all12 child reports
and two restart-check reports are structurally verified; the reconstruction/error
panel was inspected visually. Browser interaction was not validated.405 independent
artifact checks include NumPy error recomputation, source/checkpoint identities,
matched MACs/initialization/sampling and both actual-Claude response receipts.
68 scoped tests and1447 exact restart checks pass. No extra efficacy fits followed
the results. New artifacts fit36MiB, disk stays above300MiB reserve.

The codec extension is available for experiments, not adopted as a quality repair.
Next proposed task: same current observation and similar scene appearance, but paired
histories that require different motion-dependent answers. Include direction/speed
and source-order controls before expanding kernels further. This is not implemented
or validated by this fixed-mask reconstruction comparison.

Run a new comparison arm with the existing recipe (use a fresh output directory):

```bash
.venv/bin/python -m experiments.video_vae --task inpaint --steps 256 \
  --seed 7401 --spatial-kernel 5 --output runs/my_video_context
```

Add `--current-only` for its matched no-history training control, or choose
`--spatial-kernel 3 --spatial-iterations 2` for shared spatial refinement.
Explicit `--resume` requires the same task/configuration; the image source is checked
by SHA256 and the Run checkpoint restores temporal weights, optimizer and RNG.
