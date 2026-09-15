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

## Diagnostic follow-up registered during the first-seed comparison

The first core's original factor head learns color but poorly reads location and
direction. Its output variants also fail held-out complete-factor checks so far.
Before attributing this to information loss, fit a fresh linear ridge readout on
the exact frozen state tokens. Train-only per-coordinate centering/scaling;
regularization in {0.01,0.1,1,10,100} chosen only on validation factor accuracy.
Use all six input modes with the same three factor targets. Retain a permuted-label
control. Report known and held-out combinations per factor, save coefficients and
statistics. This tests linear accessibility; failure is not proof of absence.

Also allow a positive output control using explicit ground-truth factor tokens,
clearly marked oracle context. Same native decoder initialization, training data,
256 updates per output and final held-out evaluation. It is neither agent inference
nor a model improvement. It separates decoder/task difficulty from the learned
state path; success on oracle context does not guarantee it can read learned states.
No primary checkpoint or test-selected setting is changed by these diagnostics.

## Results and interpretation

[Combined report](../runs/modality_readout_v1/formal/report.html). Completed42 primary
runs (2 cores,32 separate frozen-output fits,8 joint core/output fits),8 native
oracle-output controls and2 closed-form state probes. Both pretrained core factor
screens fail. All192 registered simultaneous-output screens fail, and every variant
gets0% all-four-correct on held-out combinations. These are failures of this small
model/training setup, not a general rejection of recurrent readout or multimodal thought.

Mean complete color+location+direction accuracy, two seeds and six input modes:

| Training/readout | Known-combination all-four-correct | Held-out all-four-correct |
|---|---:|---:|
| Frozen / native | 3.0% | 0% |
| Frozen / adapter1 | 8.9% | 0% |
| Frozen / adapter2 | 6.1% | 0% |
| Frozen / adapter4 | 2.4% | 0% |
| Joint / native | 3.6% | 0% |
| Joint / adapter1 | 3.6% | 0% |
| Joint / adapter2 | 3.5% | 0% |
| Joint / adapter4 | 5.7% | 0% |

No adapter passes the two-seed benefit screen. Native joint per-output known
accuracy is30.2% text,31.4% image,28.6% audio and22.0% video; held-out is0%,0%,0%,7.1%.
The occasional nearest-template video match is not a quality pass. Wrong/empty
context and complementary missing-source rows remain visible in each child report.
The failed full-information task prevents a successful fusion claim.

**Fresh state probe:** validation selects ridge10 in both seeds. With all complete
inputs, known color is100%, location60–67%, direction44–48%; held-out location is
only2–8%. Replacing the original factor head does not repair reliable accessibility.
This does not prove that every possible reader would fail or locate irreversible loss.

**Explicit-factor oracle:** same native decoder initialization and256 updates.
This control supplies target facts, so it is not evidence of learned agent behavior.

| Output | Known complete accuracy, both seeds | Held-out complete accuracy, both seeds |
|---|---:|---:|
| Text, freely generated | 100% | 0% |
| Image | 50% | 0% |
| Audio, symbolic tone sequence | 100% | 100% |
| Video, observed sequence reconstruction | 25% | 0% |

Audio also passes waveform error gates on the oracle control. Text reproduces
known strings but fails recombination despite complete supplied facts. Image/video
fail their semantic/quality screens even there. Consequently the current problem
includes both learned-state availability/readout and output learning/generalization;
it cannot be attributed solely to the decoder or solely to lost latent information.
These controls do not distinguish training duration, architecture, objective and
capacity as the unique cause. The separate spatial image VAE is not this native
16x16 agent-output head and was not retrained here.

## Implementation and verification

`RecurrentOutputAdapter` is optional, begins as identity, preserves masks and source
state, and shares a local/cross-attention pair across iterations. It adds9841
parameters per modality;1/2/4 repeats have the same parameter count but2/4/8 extra
attention calls. Native full study model297013 parameters; with all adapters336377.
The actual outer Thinker runs twice, followed by the local output loop. No adaptive
inner-to-outer feedback or dynamic stopping was added. `TemporalImageDecoder` accepts
explicit output times; it does not execute world dynamics or ingest target frames.

GPU8-step full versus4+4 checkpoint resume is exact over5612 recursive checks.
Final72 scoped tests pass. Independent saved-array audit verifies49430 checks over
1152 metric cells and all42 formal runs. These counts measure integrity, not success
on49430 tasks. Formal training uses about538 CPU seconds including core pretraining;
evaluation/report time is additional. No GPU training-memory estimate from these CPU
fits; GPU preflight allocation is recorded separately. Empty objective charts in
closed-form diagnostic reports were found and fixed without changing weights/results.
Reports embed exact media; summary/probe/image figures inspected. Browser interaction
QA remains unavailable under the existing local-file policy.
An additional5289 checks verify the oracle/probe artifacts and final report media.

## Next bounded work

Keep the native reference; do not adopt a recurrent variant as a repair. Diagnose
encoder features, posterior/logit readout and final state separately for location
and direction. Test a training intervention with a matched longer-training control
before changing the categorical core. For text, test output-side recombination on
oracle facts first; for image/video, establish decoder quality with adequate spatial
and temporal conditioning. Audio's positive control justifies testing its learned
state interface next. Natural language/speech/photo/video require separate datasets
and quality targets after this controlled failure is repaired.

Reproduce the main comparison with:

```bash
.venv/bin/python experiments/modality_readout.py --stage suite --output runs/readout_new --device cpu
```

Individual child runs accept `--stage core|frozen|joint|oracle`, `--variant`,
`--modality`, `--core`, `--seed`, `--steps` and `--output`. Resume a specific child
with the same arguments plus `--resume`; `--stop-after` pauses without changing
its declared final step count. Oracle is explicitly a ground-truth control.

## Follow-up registered 15 September, before new results

User authorizes implementation/review/test/repair of the failed controlled task.
Preserve all v1 weights and populations. Reuse this recipe and report pipeline;
no new general trainer, decoder default, text-only thinking or soft-state bypass.

1. Frozen stage diagnosis on both original core checkpoints: per-modality encoder
   features (all scale tokens), final posterior probabilities, sampled codes,
   post-observation tokens, final Thinker tokens. Per-input-mode linear readers,
   train-only normalization/centering and validation-only ridge choice from
   .01/.1/1/10/100. Shuffled training labels and explicit factor controls. Save
   inputs, coefficients, predictions, distributions and stage dimensions.
   Unequal reader dimension and lack of nonlinear probes prevent exact loss-location
   claims. Posterior-probe accuracy is a comparison, not an upper bound.
2. Native oracle outputs: repeat each of four modalities with the same initial
   native weights and1024 updates (v1 used256), both seeds7201/7202. Identical
   splits/loss/optimizer/inputs. Final fixed update only. Reuse v1 semantic and
   signal-quality gates. This tests training duration without architecture change.
3. Bounded training-only repair: continue each original core with fresh Adam for
  768 extra updates, either ordinary factor loss or the same plus weight1 factor
   CE on posterior probabilities through a small linear auxiliary head. Both
   branches register identical extra head parameters, initialized without changing
   original core/output RNG. Same sampled minibatches and stochastic-state path.
   The extra head is unused at inference. Report both known and held-out factors
   across all six modes and source omissions; retain>=90% per-factor known screen.
   Repair benefit requires>=5pp mean location/direction accuracy gain over matched
   continuation in both seeds, with no>5pp color loss; capability still requires
   all original core gates. No unique regularization mechanism or sampling-loss
   cause claim: shuffled/norm-matched auxiliary interventions are deferred unless
   that stronger attribution is needed.
4. Verify capture leaves tensors/gradients/RNG unchanged, probes never update source
   parameters or use test normalization/selection, new head only affects training,
   initialization remains paired, and full checkpoint resume is exact. Independent
   raw-artifact checks, standalone reports and static visual QA. Browser QA remains
   unavailable under the existing policy; do not bypass it.

Budget:2 diagnostic runs,8 longer oracle fits,4 core continuation fits, one tiny
preflight/resume check. CPU chosen from prior measured small-model preflight. Keep
finite fixed budgets; additional architecture interventions require a separately
recorded, evidence-motivated amendment before execution. Do not select a default
from test outcomes. This scope tests three symbolic factors, not natural language,
speech, photos, general video or concept discovery.

Actual Claude public-only review: `runs/reviews/modality_repair_v1/`. Adopted reader
limits and separation of regularization efficacy from unique mechanism. Independent
code review/tests remain local; no code, data or measurements exported.
