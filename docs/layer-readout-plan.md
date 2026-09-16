# Optional per-scale layer readout: first reversible architecture comparison

16 September2026. User explicitly requests separately selectable variants so an
unsuccessful change can be rolled back. Default native model and source weights
remain available. This is the first bounded slice of the reference plan, not an
implementation of all literature candidates.

## Inspection and hypothesis

`BeliefAgent._features` already concatenates every `FeaturePyramid` scale. Merely
adding all-scale access would duplicate existing behavior. Instead expose earlier
processing states at the SAME grid within each `ScaleProcessor`. The hierarchy
continues to pass its original finished representation into the next scale.

Optional readout: `final + tanh(gate_i) * (mean(previous_layers) - final)`.
Previous means depth, never future frames. Entry features after scale identity
and all non-final residual-block outputs form the previous layers. Current
depth1 therefore compares its entry and processed result. One zero-initialized
scalar per scale;12 additional scalars across four3-scale encoders. This is a small
RAEv2-inspired transfer hypothesis, not RAEv2 reproduction or extra Transformer.
Gate zero must exactly recover native outputs and RNG. Encoder propagation,
token layout, masks and availability times must stay unchanged.

## Fixed development comparison

- Reuse `experiments/modality_readout.py`, Run, capabilities and report renderer.
  Add `--encoder-readout native|layers`, persisted in initialization/resume metadata.
  Defaults retain historical checkpoint behavior. No parallel experiment framework.
- Native versus layers, seeds7201/7202/7203; same source encoders from
  `runs/modality_readout_v1/formal/seed7201/core`, same per-seed data and sampled
  batches, fresh paired core weights/Adam. Only readout gates and ordinary core
  train; all original encoder and decoder weights remain frozen.
-512 updates, existing factor CE and rotating source schedule, batch24, lr0.003,
  existing clipping5 and categorical sampling. No new loss or curriculum.
- GPU full-step smoke first; maximum6GiB process budget and180s per training arm.
  CPU2-thread diagnostics max180s each. Stop on invalid gradients, exceeded budget
  or <500MiB free disk. Total new artifacts<=500MiB. No parameter search/restarts.
- Evaluate existing15 implemented symbolic capability screens on each trained
  checkpoint. Keep14 broad unimplemented and4 decoder-not-run cases visible.
  All modalities alone, together and complementary; three categorical draws per
  checkpoint are not three independent trained seeds. Existing screens unchanged.
- Primary: per-seed mean held-out JOINT accuracy across text/image/audio/video/all,
  averaged across the suite's three fixed draws. Candidate-minus-native median
  must be>=0.05. For every matched seed/mode/split/factor, min-draw accuracy must
  not regress>0.02. All existing15 measured screens must pass in all seeds.
- Record actual time, parameter count, allocator peak, process VRAM and source
  preservation. Added arithmetic is a measured cost, not exact FLOP parity.
  Passing is a go/no-go screen for a NEW source-group replication; this inspected
  symbolic population cannot promote a general default. No new default this run.
  Failure rejects this scalar/entry-readout variant at this budget, not all layer
  aggregation. No significance claim or seed selection from three development fits.

## Plan → checks → implementation → comparison

1. Essential red tests: exact native/RNG preservation at gate0, output-only
   influence, source gradients frozen/gates live, future/invalid masks, model
   metadata reload and two spatial sizes. Commit plan/red checks.
2. Optional readout inside existing hierarchy; extend ordinary recipe loading and
   freeze controls. No changes to native checkpoint keys or default computation.
3. Focused regression tests; tiny GPU full/resumed run and strict checkpoint equality.
4. Six predeclared fits, existing suite diagnostics, raw-score audit, standalone
   reports. Fix implementation bugs; retain failed architecture results.

Before source edits, the complete existing pytest suite is being run against
revision69b42ab in `runs/architecture_test_inventory_v1/`. Frozen curriculum
checkpoint evaluations initially considered there are superseded by the paired
new fits; they have not been launched. Do not report them as executed.

## Claude critique

Actual public-only review in `runs/reviews/layer_readout_v1/` accepts this as a
nonduplicating comparison. Incorporated: symbolic results only justify replication;
depth mixing is a low-capacity hypothesis; additional parameters/compute explicitly
counted; no inference that a failure rejects general aggregation. Native is the
last-only control. Rich attention-weighted alternatives are deferred to avoid
adding several new factors. A fresh core is paired across arms, not a frozen core.

## Implementation checks before comparison

The pre-change suite passed542 tests in496.32s. Initial new checks failed as
expected without the implementation. After implementation, three new fixtures
were corrected to include the required observation time axis; legacy encoder-only
checkpoints now retain strict loading without requiring a manifest. Adapter
checkpoints still require their architecture metadata.41 initial focused tests then
passed; adding the strict checkpoint/init check and broader regressions gives
69 passing tests in18.65s.

Six GPU smoke updates exercise each input mode, with all12 gates changing and
original frozen parameters preserved. An uninterrupted run and3+3 explicit resume
match exactly across2749 checkpoint leaves, including optimizer, sampler, CUDA
RNG and training rows. Source checkpoint hash is unchanged. The initial isolated
worktree CLI import selected the installed main checkout; setting PYTHONPATH=.
fixed it before training. Two test invocations named nonexistent files and ran
no tests; corrected regression output is retained. No fits were selected or
repeated based on quality.

Old completed runs keep their saved executable source. Exact resume under edited
source remains intentionally refused; new layer/native runs resume under their
own recorded source and settings. Existing checkpoint initialization remains
supported.

## Ergebnis / Result — 16 September

[Comparison report](../runs/layer_readout_v1/report.html),
[raw comparison](../runs/layer_readout_v1/comparison.json),
[verification](../runs/layer_readout_v1/verification.json).
The6 preregistered GPU fits and6 existing capability evaluations completed. No
budgets exceeded, no fit rejected/repeated based on quality, no default changed.
The scalar variant is available independently via `--encoder-readout layers`.
It is inspired by [RAEv2](https://arxiv.org/abs/2605.18324), not a reproduction of
its multi-layer feature aggregation or published performance.

| Seed | Native held-out joint | Layer readout held-out joint | Difference |
|---|---:|---:|---:|
|7201|0%|0%|0 pp|
|7202|0%|0%|0 pp|
|7203|0.139%|0%|-0.139 pp|

Joint means all three factors correct on one example; each entry averages five
input modes and three categorical draws. Every checkpoint passes1/15 implemented
screens (image-source color dependence), fails14;14 broader cases remain
unimplemented and4 decoder cases unrun. Median gain0 pp fails the+5 pp criterion.
Worst min-draw factor regression is-25 pp for held-out text color, seed7202,
failing the2 pp limit. All three comparison gates fail. This rejects this
particular scalar, entry-versus-final, frozen-encoder variant at512 updates as a
repair; it does not reject richer layer aggregation or longer/different training.

The trained mixture gates are nonzero for all12 scales in every candidate.
Original source encoder tensors and all frozen output tensors match exactly;
sampling streams, data and first training losses match across paired arms.
The original source checkpoint is unchanged. Thus failure is not simply the
absence of training of the new parameters.

Diagnostic clue: held-out audio has100% linear access to all three factors at
the encoder in every arm/seed. Location access is0–27.08% at the posterior and
0–14.58% at the final working-state probe. These are reader-specific accessibility
measurements with different feature dimensions, not proof of destroyed information
or a uniquely localized cause. Prioritize the existing feature-to-core learning
problem; enlarging the encoder is not supported by this comparison. Do not infer
an improvement in reconstruction, natural video, speech or dialogue.

Resources:297013 versus297025 total parameters;113677 versus113689 trainable.
Training takes38.88–39.92s per arm,236.53s total. CUDA allocator peak is86.04 versus
86.92MiB; reserved110 versus112MiB. Independently sampled process VRAM peaks at
318MiB (one-second sampling can miss transient peaks). This is a measured fit for
this tiny symbolic experiment, not the complete agent at real-media resolution.
New artifacts occupy about247MiB, below500MiB; all180s invocation caps respected.

Verification:542 pre-change tests,69 focused variant/regression tests,2749 exact
restart checks and55080 artifact checks.15 standalone reports use the unchanged
renderer; structure, embedded images and the plotted comparison were inspected.
Interactive browser QA remains unavailable. Two actual public-only Claude review
receipts were verified; Claude reviewed the methodology, not private code/results.

## Run or restore the variant

Use a NEW output directory each time. Native and layers remain separate choices:

```bash
.venv/bin/python -m experiments.modality_readout --stage core \
  --encoder-readout layers --device cuda --seed 7201 --steps 512 \
  --encoder-source runs/modality_readout_v1/formal/seed7201/core \
  --output runs/my_layer_comparison/layers

.venv/bin/python -m experiments.modality_readout --stage capabilities \
  --core runs/my_layer_comparison/layers --seed 7201 --device cpu \
  --output runs/my_layer_comparison/layers_capabilities
```

Repeat the first command with `--encoder-readout native` and another output
folder for the paired reference. Checkpoint evaluation restores the architecture
from metadata; adapters cannot silently disappear. Explicit resume requires the
same training arguments (including layers), source and environment plus `--resume`.
The exact finite comparison calls and audit are retained under the run directory.
