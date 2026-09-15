# Multimodal recurrent readout comparison

User authorized testing each modality separately and together, including the
optional nested Transformer readouts. This is a bounded mechanism study, not a
language/speech/photo/video quality claim. Existing real-media transport checks
remain separate; no new dataset download or pretrained backend.

## Plan registered before implementation and results

1. Reuse the native multiscale image/video/audio/text encoders, categorical
   BeliefAgent, actual observation/memory/Thinker path, native decoders, Run and
   standalone report. Add an ordinary optional recurrent output adapter, shared
   parameters across its iterations, and a time-conditioned image decoder for
   observed video reconstruction from the final state. No persistent-store rewrite.
2. Paired controlled data: color (3), starting location (3), direction (2).
   Text descriptions, colored arrows, symbolic tone sequences and four-frame arrow
   clips encode the same factors. Audio is a learned symbolic code, not speech.
   Canonical outputs retain all three factors. Inputs have independent small
   nuisance variation. Hold out all combinations with (color+location)%3==0;
   training and held-out combinations share individual factor values.
3. Train one common core per seed using only factor-classification losses, not
   output reconstruction. Six balanced input modes: each modality alone, all four
   complete inputs, and complementary inputs (image color only, audio location
   only, video direction only, text a constant request). This auxiliary supervision
   intentionally teaches a controlled shared representation, not discovered concepts.
4. Freeze that exact core and cache its actual state tokens. For each output
   separately train native readout, adapter1, adapter2, adapter4 from identical
   native decoder initialization. Adapter residual starts at zero. Each adapter
   contains the same registered parameters at every iteration count. Context is
   read once per output, outside text's autoregressive loop; text still uses its
   prefix-conditioned native attention afterward. Preserve the native reference.
5. Separately train all four outputs and the common core together for each variant,
   starting from the same factor-trained core and decoder initialization. Never
   attribute joint-training gains to readout alone. Independently trained frozen
   branches can also run simultaneously on one state; their output equality is a
   mechanical test, not proof of absence of shared-core training interference.
6. Evaluate all input modes against all output modalities on seen-combination fresh
   inputs and held-out combinations. Save zero/wrong-context controls, freely
   generated text as well as teacher-forced CE, image/audio/video errors, decoded
   factors, cross-output agreement and simultaneous all-correct rate. Complementary
   input ablations must remove the source of one factor, not merely a redundant
   complete modality. Agreement alone can mean all branches are wrong.
7. Test masks/NaNs and gradients, no state mutation, repeated weight sharing,
   native-equivalent initialization, trace invariance and full checkpoint resume.
   Review raw arrays independently, inspect rendered examples, preserve failures.

## Budget and predeclared screens

Two seeds 7201/7202; width24, fixed two outer Thinker iterations. Per seed:
768 factor-core updates; 256 updates per separate output/variant; 384 joint updates
per variant; batch24, Adam0.003, deterministic FP32, GPU if available. Training
examples12 views for each of12 known combinations (144); validation4 views/known
combination (48); test4 new views/known combination (48) and8 views/held-out
combination (48). Source seeds and exact arrays saved. Test data never choose a
checkpoint or tuning value. Final fixed update used, not best test result.

Model/data plumbing: finite outputs, exact source/state preservation, reproducible
resume and masked-payload exclusion. Core factor diagnostic: >=90% accuracy per
factor on fresh known combinations for every complete input mode; held-out and
complementary performance reported separately even if this screen fails.

Output utility screen, per input/output cell: >=80% complete factor accuracy;
text also >=80% free exact match; image/video RGB MSE<=0.02, foreground MSE<=0.05;
audio normalized MSE<=0.20 of the training-mean predictor. These are controlled-task
screens. Joint success requires every output to pass plus >=70% all-four-correct
examples. Report seen and held-out screens separately. Adapter benefit: >=5 percentage
points improvement in held-out all-four-correct for both seeds without any output
losing >5 points. This does not establish resource-matched superiority; added
parameters, executed attention calls, measured latency and GPU peak are reported.
Untied-depth and equal-compute controls are needed before a weight-sharing advantage
claim. No automatic architecture adoption from this screen.

One bounded repair may address a diagnosed implementation or training failure;
predeclare it and retain the failed run/control. Do not extend every failed model
until it passes. Small synthetic success cannot establish general conversation,
speech, photorealism, natural-video forecasting or useful long-term retrieval.

## Independent methodology review

Actual Claude reviewed a generic public conceptual question without project data,
source or measurements. It highlighted capacity/compute, frozen versus joint
learning, modality shortcuts and free generation versus teacher forcing. Applied:
fixed-core comparisons, balanced exposure, complementary evidence, separate joint
results and generation metrics. A frozen-state gain establishes improved access
under that training setup, not a unique mechanism independent of extra capacity.
Semantic combination holdout and missing-source tests are distinct from an unseen
modality-pair benchmark; this study does not conflate them.

## Preflight before the formal comparison

62 focused tests pass, including joint gradients to all input/output branches,
template metric references, source complementarity and identical native decoder
initialization. Two-update CPU core/joint runs render complete reports. GPU8-update
full versus4+4 resume matches the complete checkpoint exactly. The tiny joint graph
costs about0.23s/update on GPU in this preflight versus0.09s/update on CPU; select CPU
for the formal study to avoid launch overhead. This changes the preferred device
before measured comparisons, not data/model/budget. GPU mechanics are tested;
formal resource reporting therefore has no GPU allocation peak.
