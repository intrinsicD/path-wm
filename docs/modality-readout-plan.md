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

Diagnostic preflight correction: held-out text is one byte longer than training
text. Flattening variable pyramids made the diagnostic feature columns incompatible;
both failed attempts are retained under diagnosis/. Source model was unchanged.
Repair only pads each captured scale to64 token slots, then concatenates modalities
in stable sorted order. It does not pad/alter the model inputs or resize features.
A cap overflow raises an error. Nonconstant feature counts are reported. Existing
RidgeReader supplies train-only statistics (std floor1e-4); alpha conversion retains
raw-Gram ridge units. Original failed reports remain labelled failed.

Core intervention benefit is measured on fresh known combinations, averaged over
all six input modes and the location/direction factors; held-out results remain a
separate mandatory report. Both branches use fresh optimizer state. No claim of
unique mechanism or calibration follows from auxiliary-loss improvement.

## Second bounded iteration, registered after first measurements

Both probability-auxiliary continuations fail the registered benefit screen. On the
original first-seed all-input training batch, the updater-head gradient norm from
this auxiliary loss is0.00798; normalized raw logits give0.48087 using the identical
auxiliary head/batch. This is a local gradient diagnostic, not proof of repair.
Original posteriors have mean maximum probabilities around0.99 in most groups.

Add exactly two separate training interventions, two seeds each, from the SAME
original checkpoint with768 extra updates, same fresh Adam/batches, same registered
auxiliary head and factor task:

- `raw_aux`: weight1 auxiliary factor CE on the final correction head's raw logits,
  normalized per sample over32 channels with non-affine LayerNorm. Captured inside
  this forward only. No raw-logit or auxiliary output supplied at inference.
- `warmup`: ordinary downstream factor CE, no auxiliary loss; divide correction
  logits by a training temperature decreasing linearly from10 to1 over the first
  512 updates, then256 ordinary updates. Every step still samples categorical codes.
  Evaluation ALWAYS temperature1. This tests a training curriculum, not a new
  decoder, deterministic probability-state bypass or permanent temperature change.

Reuse prior benefit/core gates and report all conditions, not just the winner.
Equal updates are not exact compute/gradient-norm matching. The previously examined
held-out split is an exploratory development holdout; new independent tasks/splits
are needed before a generalization claim beyond it. No further search in this slice.
Claude's generic saturation critique motivates checking the sampled downstream path;
normalization is a deterministic transformation, not an extra information source.
Probability and raw-logit auxiliary objectives both intentionally bypass sampling
for that TRAINING loss, while deployment continues to use the native state path.

## Follow-up result, 15 September

[Combined repair report](../runs/modality_repair_v1/report.html) and each linked child
retain the source checkpoints, scalar metrics and examples. Actual Claude reviewed
and reconciled public-only methodology in four short replies; it did not inspect
private code or results. Two failed diagnostic attempts remain visible alongside
the corrected reports; no trained model was changed by that padding repair.

**Frozen diagnosis:** all-input and complementary encoder features permit100%
linear recovery of all three held-out factors in both seeds. Held-out position
from their posterior probabilities is0%. Isolated audio encoder features also give
100% held-out recovery, whereas image/text/video are less consistent. Known encoder
factors are98–100%; location/direction become harder at posterior and sampled-code
readouts. Feature dimensions differ and only linear readers were fitted, so this
localizes measured accessibility failures, not mathematically irreversible loss.

**Decoder duration control:**1024 native updates instead of256, with the entire
first256 training rows independently verified identical. All8 known-combination
oracle output/quality screens pass (4 modalities,2 seeds). Known text/audio remain
100%; image rises50→100%, video25→100%. On held-out combinations, text/image remain
0%, audio remains100% with good waveform quality. Video nearest-template correctness
rises to83%/33% but foreground MSE0.109/0.180 fails quality. This demonstrates that
training duration explained part of the known-example failure; it does not establish
compositional generation, realistic media or a need for a bigger decoder.

**Core continuation:** all8 full known-factor core screens still fail. Mean over the
six input modes, reported separately per seed:

| Intervention | Known position | Known direction | Held-out color | Held-out position |
|---|---|---|---|---|
| Ordinary continuation |72.6% /71.2% |50.3% /48.6% |97.6% /76.7% |0.0% /11.8% |
| Probability auxiliary |60.8% /72.6% |50.0% /51.7% |94.4% /83.3% |0.3% /11.1% |
| Normalized raw-logit auxiliary |100% /97.2% |49.7% /50.0% |78.8% /69.1% |14.2% /12.2% |
| Temperature warmup |77.4% /66.3% |51.4% /52.1% |76.7% /95.5% |3.5% /2.8% |

Only raw-logit supervision passes the declared **known-factor benefit** comparison:
mean position/direction gain13.4/13.7 percentage points without known-color loss.
This is almost entirely position improvement, not direction recovery. Held-out
color degrades versus ordinary continuation. Neither it nor temperature/probability
variants is adopted as the default. No inference parameters, extra thought loops,
soft-state shortcut or raw-logit decoder input are required by these training options.
The training-only head contains264 parameters; separate output weights stay frozen.

**Checks:**77 scoped tests pass. Independent NumPy audits pass2648 checks for frozen
stage coefficients/predictions and10273 checks for raw targets/metrics, output gates,
unchanged frozen weights, exact initial oracle training rows and embedded reports.
Full8-update versus4+4 checkpoints match exactly for probability auxiliary (4502
recursive checks) and temperature curriculum (4492). These counts are integrity
checks, not capability successes. The16 formal training runs take about451 CPU
seconds, excluding diagnostics/evaluation/reporting. Static score/error/learning
figures were inspected; browser interaction QA remains unavailable. Final summary
verification binds the report hash and checks all18 child links and summary means.

No additional full output sweep was launched: sampled direction remains near50%
and oracle text/image still fail new combinations. The next bounded question is
whether training retention/composition **from initialization**, before the categorical
posterior becomes strongly saturated, can learn direction and generalize; this is
proposed, not run. A separate output-side composition test remains necessary. The
reused held-out combinations are exploratory; confirm on a new split/task before a
broader claim. The user's shared multimodal latent design is preserved.

Example commands (fresh output directories required):

```bash
.venv/bin/python experiments/modality_readout.py --stage diagnose --core runs/modality_readout_v1/formal/seed7201/core --seed 7201 --output runs/stage_diagnostic_new --device cpu
.venv/bin/python experiments/modality_readout.py --stage core --initial runs/modality_readout_v1/formal/seed7201/core --posterior-aux 1 --posterior-source raw --steps 768 --output runs/raw_aux_new --device cpu
.venv/bin/python experiments/modality_readout.py --stage repair-report --reference runs/modality_readout_v1 --output runs/modality_repair_v1
```

The diagnostic head/curriculum are optional recipe choices. Ordinary interfaces and
old completed runs remain valid. Core/oracle initialization resets Adam explicitly;
`--resume` instead restores the complete compatible run including optimizer and RNG.

## Direction follow-up: what is actually failing?

Alex asks why direction is difficult and what is needed to solve it. A read-only
follow-up uses the two saved `raw_aux` checkpoints, without further model training:
[updater localization report](../runs/modality_repair_v1/direction_localization/report.html),
[independent verification](../runs/modality_repair_v1/direction_localization/verification.json).
Both directions are represented in every known color/location pair. Complete inputs
include explicit direction words and symbolic tones as well as arrows/motion; this
task does not require discovering physical motion from ambiguous real video.

The trained auxiliary head itself reaches only50% known-direction accuracy with all
inputs, before sampling, in both seeds (`direction_check.json`). Thus sampling alone
does not explain the failure of that trained readout. Fresh train/validation-only
linear readers on the same frozen models find:

| Stage, complete inputs / known combinations | Seed7201 | Seed7202 |
|---|---:|---:|
| Updater query tokens, before mean |87.5% |54.2% |
| Mean query |75.0% |56.3% |
| Raw categorical logits |77.1% |56.3% |
| Normalized raw logits |75.0% |58.3% |
| Posterior probabilities |68.8% |89.6% |
| Sampled categorical codes |54.2% |50.0% |

Color and position reach100% in all these known-combination cells. New combinations
remain weak. Linear accessibility is not monotonic: the nonlinear probability
transform can make a factor easier for a linear reader without adding information.
Different reader dimensions/regularization and48 examples per test cell prevent
attributing each percentage drop uniquely to its preceding operation.

There is also a concrete code collision: in seed7201, the48 known test examples have
six distinct argmax code tuples, and every tuple occurs equally with both directions.
For the actual saved sampled tuples, the optimistic best direction lookup on those
same samples is58.3% /50.0% in the two seeds. These are finite-population diagnostics
of the code tuple alone, not generalization estimates or bounds on the full state.
The continuous posterior can carry distinctions that do not reliably survive as
distinct sampled codes. The first-stage reads/trained heads also need better learning;
neither “only the decoder” nor “only sampling” is established as the sole cause.

The model has4 groups of8 codes, nominally4096 tuples for18 task combinations. This
does not prove adequate effective capacity, but does not suggest a simple shortage
of addressable codes. Early averaging may hinder retaining separate relationships;
the unequal seed results do not prove that replacing it will repair direction.

**Next bounded test, proposed rather than run:** freeze an encoder with verified
direction access and train the updater/readout from initialization on paired inputs
whose only change is direction. Require correct direction from the actual sampled
state, then jointly require color/position retention, held-out combinations, each
modality and complementary inputs. Compare ordinary query averaging against retaining
query-specific information before the categorical projection only if the controlled
learning test still fails. Keep the latent multimodal core; no language-only reasoning
or mandatory direction-specific permanent storage field is implied. A larger model
or more loops has not been shown necessary by these results.

The new audit independently recomputes reader fits, validation selection, predictions,
metrics, checkpoint hashes and embedded image in206 checks. Static heatmap inspected;
browser interaction remains unavailable. This follow-up changes no deployed weights.

## Literature follow-up, 15 September

[Twelve primary works and their transfer limits](direction-literature-review.md)
cover categorical gradient estimators, VQ dimensional collapse, shared-objective
optimization, paired observations, attention pooling and object dynamics. This is
literature review, not new training or proof of our failure mechanism. In particular,
the current factor objective has no KL term; VQ commitment/codebook results are not
direct categorical-sampler results; hard-temperature warmup did not test an unquantized
start. Claude's public-only critique was reconciled against the original papers.
Direction-only versus all-factor learning from fresh updater initialization remains
the first proposed test. Continuous/hard controls and measured gradient/query repairs
are subsequent alternatives, not a bundled default rewrite. Budgets/gates for a new
training study must be declared before running it.

## Fresh updater learning, 16 September — preregistered

First isolate task competition from the previous joint encoder training. Four CPU
runs: seeds7201/7202, paired direction-only and all-factor objectives,1152 updates,
batch24, Adam0.003, clipping5, rotating the existing six input modes. Load only
encoders from each original `modality_readout_v1/formal/seed*/core` checkpoint and
freeze them (parameters and buffers); initialize every other component afresh with
the paired seed. All-factor CE is the existing mean of three terms; direction-only
is direction CE/3 so its coefficient is unchanged. Fresh Adam, no auxiliary head,
no temperature change. The original encoder probes validate accessibility on these
populations; new updater runs do not establish unseen-dataset generalization.

Primary endpoint: actual hard-sampled post-think direction accuracy >=90% in EACH
of text/image/audio/video/all/complementary known-combination test cells, in BOTH
seeds. Report each factor and held-out combination separately, without selecting on
them. All-factor complete gate additionally requires every factor >=90% in all six
known cells. Save predictions/cache, hashes, raw per-step losses and standalone reports.

At updates0,127,254,... record separate unweighted factor gradients on the shared
updater: norms and pairwise cosines. Step127 rotates modes rather than repeatedly
sampling the same input type. This is observational: sign conflict alone does not
prove harmful task interference. A direction-only advantage warrants a controlled
repair comparison, not a causal conclusion or immediate PCGrad adoption. Both
failing warrants a readout/continuous-path diagnostic before adding task weighting.
A follow-up may use at most four matched1152-update fits, preregistered separately
once this first comparison chooses the question. No full output sweep yet; the
encoder/updater retention test is distinct from output decoder generation.

Implementation stays in the existing recipe: optional encoder-only initialization,
objective selection and observational gradient audit. Tests must show preserved
fresh nonencoder initialization, frozen encoders, exact unchanged native loss, and
identical gradients/RNG with and without the audit. Verify short exact resume before
formal runs. Claude reviews generic public methodology only; local code/evidence
review remains ours. Browser QA remains unavailable; inspect static report assets
and structurally verify reports without bypassing the prior local-file restriction.

Claude review reconciled before training (`runs/reviews/direction_learning_v1/`):
the direction gradient contribution is exactly identical at fixed weights/batch;
total update norms need not match. Near-zero cosine is not conflict, and sparse
negative cosines cannot establish causation. Soft evaluation of a hard-trained
reader is a distribution shift; failure would not prove upstream information loss.
Claude acknowledged these corrections. Use fresh independent probes/controls for
localization; keep the persistent hard-state contract unchanged in this slice.

### Iteration 2 — registered after the four hard-read fits

All four first-round task screens fail. Direction-only known accuracy averaged over
six modes is52.43% /62.15%; the second seed reaches100% only for complementary
inputs. Joint known direction is51.04% /59.38%. This does not establish task
competition as the sufficient explanation. Proceed with four additional CPU fits,
same seeds7201/7202,1152 updates, optimizer, data order, frozen source encoders,
initialization and two objectives. The sole change is **continuous working readout**:
project posterior probabilities through the existing readout matrix into temporary
world-context tokens before Thinker, instead of projecting the sampled one-hot.
Continue to draw and retain native categorical state/IDs; no new parameters, permanent
soft-state substitution, memory-write change or encoder training. The ordinary
sampled readout remains default. This is a separately trained continuous control,
not a soft-only evaluation of a hard-trained model.

Same90%-in-every-known-input-cell gates, scored explicitly as continuous access;
all-factor gate also requires color/position. Primary matched benefit: direction
accuracy increases by at least10 percentage points averaged across the six modes
in both seeds (report per objective). Stronger capability gate remains separate.
Held-out combinations are exploratory reused combinations. If continuous access
passes or materially improves, evaluate10 independent categorical draws per test
cell plus a clearly labelled hard-read counterfactual using the continuous-trained
weights. This is not proof of recovery from stored hard codes or universal modality
competence. A pass warrants subsequent persistence/closed-loop/output tests before
any default adoption. If it fails, stop this budget and document the failed control.

Claude's generic follow-up recommends preserving initialization/RNG, keeping the
stored hard-state contract explicit, and withholding default adoption. Interpretation
is ours: a continuous benefit does not distinguish representation detail, estimator
bias/variance and optimization without further controlled tests. Two meaningful
checks protect identical sampled state/RNG and differentiable continuous access.

Both continuous objectives also fail the registered paired benefit and capability
screens. No additional model training in this budget; the conditional ten-draw
confirmation is not triggered. Before interpreting the failures, run one read-only
localization: the four final jointly trained checkpoints (hard/continuous × both
seeds), complete inputs only, the existing fixed train/validation/test populations.
Freeze all model parameters and fit the existing train-standardized ridge readers,
validation choice from alpha0.01,0.1,1,10,100. Inspect encoder features, updater queries
before/after mean, posterior probabilities, sampled codes, observed tokens and final
thought tokens. Save features, coefficients, validation scores and predictions.
Selection does not inspect test labels. This tests linear accessibility with different
reader dimensions, not an information-theoretic proof of a unique loss location.

### Result, 16 September

[Comparison report](../runs/direction_learning_v1/report.html) ·
[frozen localization](../runs/direction_learning_v1/localization/report.html).
Eight complete fits, each1152 updates,237234 core parameters /113677 trainable
parameters,450.16 seconds total measured CPU training. All eight task gates fail.
No evidence for adopting continuous access as the repair; native sampled readout
remains default. All source encoders and frozen outputs/monitor/action weights are
unchanged. Paired runs have identical initial weights, populations, optimizer,
trainable parameter set, final sampling RNG and batch-sampler state.

Known-combination direction accuracy, averaged over six input modes:

| Objective | Readout | Seed7201 | Seed7202 |
|---|---|---:|---:|
| Direction only | Sampled code |52.43% |62.15% |
| Direction only | Posterior probabilities |50.35% |62.85% |
| All three factors | Sampled code |51.04% |59.38% |
| All three factors | Posterior probabilities |48.61% |55.56% |

The direction-only seed7202 sampled run reaches100% on complementary inputs, but
not on the other five input modes. The shared input-mode training regime remains
a possible source of interference; direction-only removes competition among target
factors, **not** competition among modalities. Negative gradient cosines occur in
10–80% of the ten audited points per run/pair, with no consistent causal pattern.
These sparse observations do not justify PCGrad or selecting a new loss weight.

Fresh linear readers on the four jointly trained checkpoints, complete inputs only:
encoder features give100% for all three factors on known AND held-out combinations.
Known-direction access is60.4–75.0% in updater query tokens,54.2–68.8% after mean,
56.3–77.1% in probabilities,50.0–52.1% in sampled codes, and54.2–75.0% in thought
tokens. Thus useful signal is demonstrably available at the encoder and becomes
harder to access through the updater/readout path. This is not proof that averaging,
sampling or a single layer is uniquely responsible: probes have different widths,
linear access is non-monotonic, and failed probes cannot establish information absence.
The continuous control still uses the native sampled initial/dynamic state; it tests
continuous **posterior working access**, not fully deterministic or fully unquantized
world-model training. No stored-state or memory persistence success follows from it.

Verification:59 distinct scoped tests passed across readout/belief/foundation tests;
2909 exact checkpoint comparisons each for8 versus4+4 hard and continuous runs;
changed-objective resume rejected without mutation;31376 raw artifact checks,
94 paired-control/Claude receipt checks,297 frozen reader/score checks. Four short
actual-Claude public-methodology exchanges, including acknowledgment of corrections.
Local code and measured results were reviewed independently without exporting them.
The first summary script had a syntax error, corrected with its failed source retained;
the probe verification initially differed at1.5e-10 under a different CPU thread count,
then passed at the original tolerance after restoring the experiment's two-thread
configuration. No training result was overwritten or discarded. Static plots inspected;
HTML structurally verified. Browser interaction QA remains unavailable.

**Next bounded question, not yet executed:** can a fresh updater learn direction on
one fully informative input modality alone, before introducing mixed input-mode
training? Use audio's explicit direction tone as a positive-control task, then text,
then combined inputs. Separately compare direct feature read, cross-attention query
read and query-preserving projection only if this locates a specific failure. Avoid
bundling query changes, gradients, extra loops and model scaling. Decoder composition,
persistent memory and real-media generation remain separate open capability gates.

Examples (fresh output directory required):

```bash
.venv/bin/python experiments/modality_readout.py --stage core --encoder-source runs/modality_readout_v1/formal/seed7201/core --factor-task direction --gradient-audit-every 127 --seed 7201 --steps 1152 --device cpu --output runs/fresh_direction_example
.venv/bin/python experiments/modality_readout.py --stage core --encoder-source runs/modality_readout_v1/formal/seed7201/core --factor-task all --belief-readout probabilities --gradient-audit-every 127 --seed 7201 --steps 1152 --device cpu --output runs/continuous_access_example
```

These controls reuse the existing recipe, modules, Run/checkpoint and report renderer.
`load_initial` and frozen output/diagnostic consumers recover the saved readout mode
from run metadata; a continuous checkpoint is not silently interpreted as sampled.

Final report/atlas audit:103 structural checks across14 completed run/analysis
reports; PNG payloads decode, referenced child reports exist, atlas IDs are unique,
and every high-level discussion color and existing validation scope is preserved.
The rendered summary is available in the workspace; no browser-interaction claim.

## Input-mode isolation, 16 September — new preregistration

User briefly considered video first, then explicitly returned to direction; no video
changes or runs were made. Continue with actual Claude methodology review and preserve
the prior failed references. Add an optional training input selector to the existing
recipe, leaving the default six-mode rotation unchanged. A single-input task screen
must name its trained input mode; scores on other modes remain transfer diagnostics.
Missing test cells must never pass a screen by vacuous truth.

First comparison: six fresh-updater CPU fits, seeds7201/7202 × audio-only, text-only,
and simultaneous-all-inputs. Same frozen encoders from the original seed-specific
core, native sampled working readout, direction CE/3, Adam0.003, clipping5, batch24,
1152 updates and gradient audit every127. Fixed raw factor populations and the same
paired initialization as the preceding study. Reference: the saved two direction-only
six-mode-rotation fits; no need to spend another duplicate training budget.

Primary feasibility gate: >=90% known-combination direction in the TRAINED input
condition in both seeds, using actual sampled states. All-factor and all-input
capability gates remain separate; passing direction-only never implies those pass.
Report held-out combinations and all six input modes individually. Confirm passing
trained conditions over10 independent categorical draws, reporting pooled and worst
draw accuracy; require the worst draw >=90% for a stable narrow pass. These repeated
draws are sampling checks on the same48 examples, not new independent test data.

Save a384-update checkpoint before resuming to1152, and evaluate it read-only. A
single audio/text mode then has384 direction-bearing source presentations per example
batch count, matching the192 standalone plus192 simultaneous-all occurrences of that
source in the1152-update rotating reference. This matches counts, not exact examples,
optimizer history, total input information or compute. The primary1152 comparison
has equal updates but greater per-source exposure, so cannot alone establish modality
interference. The all-input condition has a different source/compute budget too.

If isolated inputs succeed, a subsequent targeted schedule/combined-input test may
use at most four additional1152-update fits, with a separately recorded preregistration.
If isolated inputs fail, first inspect frozen direct/mean/query accessibility before
altering projection or gradients. No immediate model scaling, text-only thinking,
default architecture adoption or output decoder claim.

Before running, source-matched existing encoder probes confirm audio direction100%
on known/held-out in both seeds; text known100%/97.9%, held-out89.6%/75.0%. Therefore
text held-out failure cannot be assigned solely to the updater. Claude's public-only
review and reconciliation retain exposure/trajectory/two-seed caveats: a projection
change also alters optimization and possibly capacity; neither its success nor failure
uniquely localizes information loss. No separate modality subspace is mandated.

### Iteration 2: registered curriculum versus matched optimizer restart

First six fits complete. Text-only known direction reaches100% in both seeds;
audio-only46%/100%, simultaneous-all48%/60%. The384-update text checkpoints score
100%/77%, and were chosen before results, not by selecting a best checkpoint. A
frozen audio pooling diagnostic finds100% known direction from both raw/normalized
grids AND their means in both seeds, but first learned query access40%/98%. This
argues against assuming that normalization or query averaging necessarily erased
this input's direction. It does not prove a unique optimization failure.

Token-count balancing was discussed with Claude but is not implemented now. Recorded
all-input attention already gives a small image source up to63% of refinement mass,
while sources have84 audio/21 image/45 text/73 video tokens. Counts alone do not
explain this learned selection, and a uniform source prior is not a reliability
estimate. Preserve this as a candidate, not a diagnosed repair.

Run four further CPU continuations: seeds7201/7202 × text384→all768 versus
all384→all768. Load the exact saved384-update checkpoint, keep the original encoders
frozen, reset Adam identically, native sampled state, direction CE/3 and the existing
settings. Each complete trajectory has1152 updates; no additional latent parameters.
Control both initialization provenance and optimizer reset rather than comparing a
fresh optimizer against uninterrupted training. Do not add an unused auxiliary head
for these encoder-frozen continuations. Extend --initial to accept a specific saved
checkpoint file as well as a run directory; preserve its run-level readout metadata.

Primary repair benefit: known complete-input direction gains>=10 percentage points
in BOTH seeds relative to matched all→all restart control. Capability requires>=90%
for complete inputs in both seeds and worst-of10 independent sampled evaluations.
Also test complete inputs without text, and every single input modality. A text-only
cue may support complete-input success without cross-modal transfer; report that
separately and do not call it general multimodal repair. Held-out combinations remain
exploratory. Stop this bounded training budget after these four continuations.
