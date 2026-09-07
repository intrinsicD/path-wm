# Paddle observer history-start experiment — prospective draft

**Status: coordinator-reviewed prospective follow-up; implementation authorized,
training follows essential tests and a verified smoke.** The coordinator reviewed this plan after the PushT launch. The earlier optional direction question about observer starts
has not received an answer; this draft does not represent one. The working
assumption is that a bounded, separately recorded training-distribution test is
within the user's request to iterate the world model. State that assumption
before launching any reviewed implementation, with an opportunity for user steering.

## Question and preserved reference

Does exposing U to zero-memory starts throughout existing training trajectories
improve its three-observation motion/position estimates at unfamiliar initial
locations, while retaining ordinary full-history accuracy?

The [independent diagnosis](../runs/paddle/collaboration/codex_memory_diagnosis.md)
induced a large error on the **same** validation observations by replacing full
history with their last three frames: R y/vx MAE changed from 3.585/0.893 to
24.239/3.592 on 23 matched observations. Paired images were accurately encoded by
E/H, but the observer behaved as if its third frame were still near the ordinary
initial y range. This supports a training-initialization gap; it does not prove
that architecture or optimization is otherwise sufficient.

Preserve the immutable reference data, models, logs and control evidence:

- Source: `data/paddle/baseline`, fingerprint
  `c6d255dc9919b1bb9ce38182f9d180047f964754a32c616db072bc965a3be54f`.
- Frozen E/D/H: `runs/paddle/baseline/perception/checkpoints/best_00009750.pt`.
- Original U/R: `runs/paddle/baseline/memory/checkpoints/best_00010000.pt`.
- Original P1 at 10,000 updates, its selected 20,000-update continuation, and
  subsequent P5 in `runs/paddle/baseline` / `runs/paddle/continuation_v1`.

At U update10,000, the original fixed ordinary validation has normalized state
MSE0.01733964 and R MAE `[2.78267,3.04628,.80920,.54549,5.02473]` in
`[x,y,vx,vy,paddle_x]` order. The existing 50-pair validation has MAE
`[9.51550,26.89105,4.31146,.45474,3.19025]` after its third observations.
These are reference measurements, not new acceptance thresholds.

## Single training intervention

Use the same 5,000 training episodes, existing images/actions/labels, frozen E,
U attention/GRU128, R linear readout, and original masked normalized state-MSE
objective. Add no layers, H inputs, attention modifications, loss terms, labels,
new trajectories, or diagnostic pair examples. Do not alter physics or rendering.

A training batch still contains eight independent sequences:

1. Four complete episodes sampled uniformly from the 5,000 training episodes,
   with replacement. Their source start index is zero.
2. Four complete suffixes sampled uniformly, with replacement, from the global
   list of valid `(source_episode, source_start_index)` pairs. If an episode has
   L frames, eligible augmented starts are **1 through L−3 inclusive**. Each
   suffix therefore contains at least three observations and two real actions.
   Start zero belongs only to the original-episode branch.

Use a fixed alternating full/suffix slot order in each batch. This makes the
mixture exactly 50/50 **by sampled sequences**, not by frames, loss scalars, or
source episodes. The current immutable train population contains 169,808 valid
augmented starts. Sampling uniformly from this list intentionally weights an
episode in proportion to its number of eligible starts; do not implement
uniform episode selection followed by a uniform start and call it equivalent.
Record actual frame/scalar presentations because equal update counts do not
imply equal sequence lengths or wall-clock compute.

Keep a separate NumPy generator for each source stream: full-episode seed1701
and suffix seed38802 (`1701+37101`). Save both RNG states in every resume
checkpoint. Selection/validation/plotting must consume neither training stream.
No weighting or rejection based on velocity, y position, collision, diagnostic
pair resemblance, or eventual performance is allowed.

### Exact suffix recurrence and masks

For a selected source episode and start s, provide frames `s..L−1` and actions
`s..L−2`, keeping the recorded terminal/truncated frame and flags. Initialize
memory to zero before source frame s. Assimilate that frame using the paddle
initial previous-action marker `[0,0,0]`; do not expose source action `s−1`.
At relative index j≥1, U receives the actual recorded executable action
`action[s+j−1]` as the existing three-way one-hot vector.

Supervise the three R position coordinates at every valid suffix frame. Mask
vx/vy at **relative indices0 and1**, even when the corresponding absolute source
indices exceed one; supervise them from relative index2 onward. Retain the
original normalization `[64,64,6,6,64]`, as specified by brief section10B. The first two motion targets are absent
from the objective, not zero-valued targets. Later labels refer to the exact
recorded source state, including true terminal zero velocities.

Backpropagate through each entire suffix. Padding uses a valid executable action
with a mask; it neither updates memory nor contributes any coordinate loss.
The objective remains the sum of squared normalized errors over all valid
coordinates in the batch divided by their valid scalar count. Do not add separate
full/suffix loss weights or weight the first three observations specially.

## Initialization, budget, and immutable outputs

Proposed run root: `runs/paddle/history_start_v1/`. Keep the reference directories
read-only. Start fresh U/R and optimizer state, using seed1701 and the same model
construction/load order as the original memory stage. Load exactly the selected
reference E/D/H. Record an immutable update-zero U/R snapshot and initial tensor
fingerprint before training. The historical update-zero weights were not retained;
reconstructing the original seed/order is an initialization match, not an
independent byte comparison with an unavailable historical snapshot.

Train **exactly10,000 U optimizer updates**, batch8: 40,000 full-episode draws and
40,000 suffix draws. Use the same constant AdamW3e-4, weight decay1e-4 on the
existing matrix/kernel parameter groups, gradient clip1, FP32, TF32 disabled,
and CPU thread setting. Disable observer early stopping for this fixed-budget
comparison; the original observer also reached its full10,000 updates. Stop only
for a correctness/nonfinite/resource failure and preserve an incomplete result;
do not silently extend the budget or replace a failed run.

Use the existing raw RGB/state/action cache without changing source values.
Report optimizer updates, sampled sequences by type, observed frames, supervised
scalars, distinct source episodes/starts, gradient norms, peak memory and elapsed
time. A short pre-training implementation smoke can check throughput, with
explicit development provenance and no claimed target success.

Save new checkpoint/data/config/code identities, optimizer and both RNG states,
validation population manifests, counters, and exact parent perception SHA256.
Retain selected and final checkpoints separately. Resume must reproduce the next
mixed batch and update, preserve source starts/masks, and reject a changed mixture,
source list, seed, normalization, or frozen E identity.

## Validation and prospective checkpoint selection

Freeze all validation lists before the first new update and before comparing
candidate U snapshots:

- **Ordinary:** the same 64 full validation episodes listed in the original
  memory `paddle_manifest.json.validation_indices`.
- **Augmented starts:** 64 distinct valid `(episode,start)` pairs sampled without
  replacement from all 16,615 eligible starts in the 500 validation episodes,
  using NumPy seed38920 (`1701+37219`). Use the same suffix eligibility, initial
  marker, relative warm-up mask, and complete-suffix recurrence as training.
  Publish the exact pairs and their source identities.
- **Paired diagnostic:** the existing 50 generated validation pairs from
  `history_pairs(count=50, seed=7000)`, both members, with byte-identical final
  images verified. No new pair selection or pair-generated training examples.

At updates0,250,500,…,10,000, calculate valid-scalar-normalized R state MSE
separately on ordinary and augmented validation populations. Select U/R by

`selection_score = 0.5 * ordinary_state_MSE + 0.5 * augmented_start_state_MSE`.

Each component first sums its actual errors and valid scalar denominator across
its complete population. Do not pool frames first, average per-episode means,
reuse a full-frame velocity mask on suffixes, or let unequal lengths change the
predeclared 50/50 validation weighting. Break exact score ties by the earlier
checkpoint. The selected snapshot is the lowest joint score, even if another
snapshot looks better on the pair plots.

**Paired results never select checkpoints, tune mixture weights, set patience,
or choose among source-start lists.** They are a held-out validation diagnostic
of the stated cold-start hypothesis. The test pairs and test episodes do not
enter this stage, selection, or debugging.

The changed validation selector is explicit. To separate training-distribution
effects from selection effects, the primary matched-budget diagnostic compares
both arms at **exactly U update10,000**; also publish the prospectively selected
mixed U and its update. The historical reference selected update10,000 by its
ordinary criterion. Do not describe differently selected snapshots as a pure
initialization-distribution comparison without this final-update control.

## Required diagnostic readout

Primary outputs are paired final-frame R **vx, vy and y** errors and ordinary
retention, each separately in physical units. Report all five R coordinates,
mean/p95/max, pair counts, and per-pair errors for transparency. Include vx sign
accuracy and the current-frame-only reset control on the identical paired images.
Report the original velocity target of <0.5 units/interval for vx and vy; do not
invent an R-y engineering target, since the brief's position thresholds apply to H.

For ordinary and augmented validation, report:

- State MSE and each coordinate's physical MAE, with exact coordinate denominators.
- Relative-age2 errors separately from later observations, so averaging long
  suffixes cannot conceal continued failure after three frames.
- Per-episode/per-suffix records, source start position/velocity, and source index.
- Matched full-history versus exact three-frame resets on the original diagnostic
  observations, using the same stored rows rather than choosing favorable cases.

Show absolute deltas and ratios against the immutable reference at update10,000.
Define ordinary retention descriptively as those exact coordinate/MSE deltas;
there is no pre-existing retention margin, so this draft does not fabricate a
binary passing gate. Pair-level paired-bootstrap intervals (fixed seed38921,
2,000 draws) may summarize uncertainty while preserving both members together.
They describe one training seed and a finite diagnostic population, not variation
across independent trained models.

An improvement in paired R does not itself prove useful planning: R is not the
planner's scoring head, and memory coordinates can remain differently distributed
for cold and full histories even when their R readouts agree.

## Downstream predictor and control sequence

After the fixed U budget, freeze the jointly selected U/R. Build **new** observer
caches by replaying original full training and validation episodes from their true
initial frames. Cache identities include the new U fingerprint and exact frozen E.
Never reuse the reference U's cache, predictor, or dependency identity. Predictor
training receives only its ordinary source windows; augmenting P's memory-start
distribution would be a separate change and is not silently included here.

Reuse the exact E-only train variance statistics, after verifying E/data identity;
this intervention does not change the frame population used for those statistics.
Start a fresh zero-output-head P. The accepted downstream ceilings are P1 **20,000 updates at batch64**, matching the completed reference's
cumulative P1 budget, and P5 **10,000 updates at batch16**. Retain the reference
learning rate, optimizer, losses, frozen modules, fixed1024 validation windows,
objective-based predictor checkpoint selection and eight-check early stopping.
Report the new P1 at10,000 against the original10,000 baseline and the final
selected P1 against the preserved20,000 continuation, with actual work explicit.

The original K1-to-K5 gate remains mandatory: selected normalized latent loss
and **each H ball-x, ball-y and paddle-x MAE** must strictly improve over matched
copy-S on the fixed ordinary K1 validation population. Do not bypass it because
paired R improved. If it passes, initialize P5 from that selected new P1 with a
fresh optimizer and the same new-U dependency. If it fails, retain the failure
and diagnose; no budget extension is implied by this draft.

Only after those stages and structural checks should the coordinator schedule a
new, separately identified held-out evaluation against the same frozen ordinary
starts/test pairs and controller protocol. Preserve all prior outcomes. Report
ordinary success, paired first-action/first-interception success, P1..P5 errors,
latency/work, and failed gates. Reusing known test cases is a repeated development
evaluation, not a newly untouched confirmation set. No test outcome may choose U,
P, the suffix mixture, or another continuation within this experiment.

## Essential implementation checks before any launch

1. An index-coded episode verifies suffix frame/action/label slicing and the
   unavailable preceding-action marker; mutating future/source-prestart labels
   cannot affect earlier observer inputs or targets.
2. Velocity masks use relative suffix age; a start at a large source index still
   masks its first two frames. Positions stay supervised, terminal flags remain
   intact, and padding preserves memory and scalar denominators.
3. The sampler produces exactly four full/four suffix records, draws only from
   frozen training membership, supports all valid suffix starts, and reproduces
   the next batch after save/resume. A validation call cannot advance its streams.
4. A small independent scalar reference checks joint validation weighting across
   unequal lengths/masks. Changing paired diagnostic results cannot affect the
   selected checkpoint or its objective score.
5. New U cache/dependency identities reject old-U activations or predictor weights.
   Frozen E/D/H parameters remain identical; frozen U continues transmitting
   gradients during P's multi-step training, as protected by existing tests.
6. Complete the tiny CPU slice, inspect selected original/suffix images and
   age-indexed targets, preserve raw evidence, and refresh/verify canonical HTML
   before claiming the implementation workflow complete. Commit plan/tests after
   their informative failures, then implementation, before any substantive run.

This plan changes one scientific input distribution. It does not claim the
fixed architecture will meet the task targets, and it does not schedule itself.

## Implementation interface accepted2026-09-07

A separate `world_model.paddle.history_training` module supplies the changed
observer experiment, preserving the original trainer behavior.
`slice_episode(episode,start)` returns its complete source suffix without
mutating source arrays; `HistoryStartSampler(lengths,seed=1701,suffix_seed=38802)`
exposes `sample(batch_size=8)->list[(episode,start)]`, `state_dict()` and
`load_state_dict(state)`. Full/suffix slots alternate exactly; batch size must
be positive and even. `joint_validation_score(ordinary,suffix)` returns equal
weights of valid-scalar-normalized state MSE. `train_history_memory(config,data,
perception,run,resume=False)` writes compatible U/R checkpoints plus separate
history-start provenance and both RNG states. Extend the CLI with an explicit
`train-history-memory` command; new config `configs/paddle/history_starts.yaml`
uses source baseline data and new runs only. The new trainer must not silently
change generic `train-memory` behavior. Existing generic predictor training can
consume its validated frozen U checkpoint and exact new dependencies.
