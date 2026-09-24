# Frozen representation transfer screen

24 September 2026. Active continuation of the architecture handoff. This is a
bounded test of whether the existing learned detail representation supports an
analytic reusable map across unseen instances. It is not learned induction,
autonomous association, compositional reasoning, or completion of R1/R2.

## Decision and measurable slices

1. Specify and test affine inference against an independent numerical reference;
   verify support/query isolation and correction/restart contracts.
2. Reuse the frozen DetailCodec checkpoints from the completed evidence loop;
   compare stored examples with a map inferred from paired support observations.
   Use WorldStore, EpisodeClient and WorkingContext for a live correction/readback
   demonstration. No new trainer, neural module, storage protocol or CLI framework.
3. Run one development population, inspect all controls, freeze source, then run
   two fresh populations with each of the two existing codec checkpoints. Stop at
   the declared capability gates or a concrete failed gate; no architecture sweep.

R1's unsuccessful support readers do not prove the representation is unusable.
CI1's adaptive encoder lost old shape/size accuracy; keep weights frozen here and
measure reconstruction separately. Neither frozen-weight integrity nor successful
store updates establish a learned acquisition mechanism.

## Before-run protocol

Existing checkpoints: `runs/evidence_loop_final_s17/last.pt` and `s29/last.pt`.
No training or calibration updates. CPU FP32 codec, float64 centered ridge solve;
penalty 0.001 on the sum of squared residuals, intercept unpenalized. No query
normalization or fitting. All support pairs use supplied part correspondence and
relation grouping. One shared map is fit over the four parts of 64 support tiles
(256 pairs); 64 independent query tiles and 64 discrimination tiles per relation.
Smooth RGB16 images use the existing generator. Continuous random instances are
fresh; this is a procedural distribution, not natural visual recognition.

Development population 925101: channel cycle (1,2,0), and 50/50 identity/cycle mix.
Final populations 925117 and 925129: reverse channels (2,1,0), and 70/30
identity/reverse mix. The final operators are not the development operators and
were not codec training objectives. These are unseen operators within an affine
hypothesis class, NOT unseen mathematical relation families. A threshold at 0.4
is a separately reported nonlinear diagnostic with no required outcome.
Each population/relation has independent support, query and discrimination draws.
No tuning on final observations; consumed populations cannot later be called fresh.

Arms: identity, nearest support output in latent distance, mean latent displacement,
latent ridge, shuffled-pair latent ridge, and strong channel-shared pixel-space ridge (3×3 plus intercept). Record
true-output encode/decode floor and four familiar-pose reconstruction as separate
metrics. Do not claim benefit from learned features if pixel ridge solves the task.
Hard discrimination negatives are the same input under the other affine relation,
and the true relation on another instance. This is another use of the prediction,
not independent evidence of a broad semantic concept.

**Gates, separately for each final affine operator and checkpoint/population:**
latent ridge pixel MSE <=0.002; <=50% of the better nearest-copy/mean-displacement
MSE; <=50% of shuffled-pair MSE; discrimination accuracy >=90%; true-output
reconstruction floor <=0.001. Channel-shared pixel-ridge MSE <=0.00001 is an identifiability/control
check (rank and conditioning reported). Dense pixel maps would be underidentified
on the smooth-pattern support; do not use that as the oracle. The channel-shared
reference has a stronger architectural prior, not matched parameter count.
Familiar-pose reconstruction MSE <=0.001 at each pose. Exact neural-weight hash,
old reconstructions, corrected-from-scratch fit and CPU restart predictions must
be unchanged; stale source pins must be rejected. Correction is source-issued,
not learned outlier detection. Numeric gate failure is retained, not an exit error.
No gates on the nonlinear diagnostic; success there would not automatically imply
leakage or a lax metric. Shuffled predictions beating identity alone is not leakage.

**Budgets:** one tiny software smoke, one development run on checkpoint17, four
final invocations; <=120 seconds and <=2 GiB peak RSS per invocation, two CPU
threads. One implementation repair allowed after development, with fresh output;
no scientific hyperparameter repair, additional training or final-set tuning.
Full CPU suite <=20 minutes. Claude at medium: at most four public-only rounds,
<=10 minutes each. Record complete invocation wall/CPU time, peak RSS, fitting and
prediction costs, map/example storage, unchanged model parameter bytes, and source
identity. Report generation is included in total invocation timing. Existing
pretraining is a sunk cost and must not be counted as zero acquisition cost.

## Design review and standing principles

Actual Claude Opus 5.5 medium round1 called this a calibrated inductive-bias
baseline, not general concept acquisition. Adopt strong pixel-space inference,
reconstruction floor, shuffled support, hard negatives and separate integrity
checks. Decline a new autoencoder/seed sweep in this bounded reuse-first screen.
Reconcile overly categorical shuffled/nonlinear control stopping rules in round2.
Public briefs/responses/model receipts: `runs/reviews/representation_transfer_20260924/`.
Private source and measurements stay local.

Prepare source representations once; infer query-independent state and reuse it.
Preserve source evidence rather than discarding instances after fitting. Give
weights, evidence, derived maps and disposable context explicit ownership. Source
correction invalidates the derived map; recomputation is charged. Separate cheap
proposals from target-based evaluation and include all local execution costs.
No new diagram discussion coverage is earned by assistant-only work.

## Status

Plan registered before execution. Four essential checks fail collection because
the recipe does not yet exist (red-tests.log, exit2). Claude round2 withdrew both
categorical stopping rules and the five-seed minimum; adopted channel-shared
pixel reference, fixed derangement, logged rank/shrinkage and unclipped excess MSE.
Its generic centered-ridge code is adapted locally and independently checked.


## Completed screen and decision

The development run passed. One allowed implementation repair moved the shuffle
from part rows to whole images, added explicit candidate-order/finite-value checks,
and exposed the capability gate in the unchanged report renderer. Development v1
and the tiny smoke remain preserved; the smoke's quality failure is not a learning
result. Development v2 passed. Numeric thresholds and model settings did not change.
No final population was used to tune the implementation.

All four final cells passed every affine gate:

| Codec seed | Population | Reverse MSE | Mix MSE | Discrimination, both |
| --- | --- | ---: | ---: | ---: |
| 17 | 925117 | 0.000256037 | 0.000211428 | 100% |
| 17 | 925129 | 0.000251141 | 0.000215578 | 100% |
| 29 | 925117 | 0.000250782 | 0.000209716 | 100% |
| 29 | 925129 | 0.000246985 | 0.000212748 | 100% |

The latent map's error is 18.4–116.3× lower than the better copying/displacement
baseline. Most error is already present in true-output reconstruction. Channel
ridge reaches roughly 1e-13 MSE: the stronger channel-sharing prior solves this
photometric problem more accurately, so there is no learned-representation
advantage. Each source split has 64 whole images; 256 parts and 16384 pixel rows
are not independent population samples. All candidate degeneracy counts are zero.
The fixed full denominator and strict tie rule remain unchanged.

The nonlinear threshold diagnostic has latent MSE0.08308–0.08375, output-codec
floor0.05093–0.05189 and pixel-ridge MSE0.07596–0.07598. Discrimination still reaches
100%, demonstrating why easy outcome separation must not substitute for faithful
reconstruction. This is a diagnostic limitation, not a secretly failed affine gate
or evidence of arbitrary concept learning. Binary threshold outputs differ from
the smooth training distribution. These measurements do not isolate a unique
architectural cause.

All four familiar-pose reconstruction gates, frozen-weight/old-output integrity,
source correction, stale-read rejection, unrelated-state preservation and exact
float64 map/restart checks pass. Integrity is mechanically expected with frozen
weights and tested source contracts; it is not learned retention. Population
predictions use tensors; one live episode/context demonstration per invocation
uses supplied feature observations. The entire population does not run through a
learned autonomous session, and the generated predictions do not become evidence.

**Decision:** stop this affine feasibility screen at its passing gates. Do not add
a learned inducer to a task solved by the strong baseline. The next representation
priority is useful non-affine structure and novel output fidelity with preservation
of prior detail/behavior; diagnose the target-reconstruction floor before proposing
an inference repair. A new task/family and budget must be declared first. Existing
R1/CI1 failures and general learned induction remain unresolved. These final
populations are now consumed.

## Evidence, cost and reproduction

Raw arrays, maps, unchanged codec checkpoint, source snapshot, standalone report,
and a live memory/session snapshot are in
`runs/representation_transfer_final_s{17,29}_{925117,925129}_v1/`.
Run `OMP_NUM_THREADS=2 .venv/bin/python -m experiments.representation_transfer
--checkpoint runs/evidence_loop_final_s17/last.pt --output runs/my_transfer`
for the development screen. Add `--final --seed 925117` to reproduce an already
consumed final population. Tiny software smoke: `--support 2 --count 2`. There is
no training/resume CLI; `last.pt` is an audit artifact, while the checked live
restart uses `memory/session.pt` and `memory/context.json`.

Independent augmented-normal-equation fits and raw-target/prediction/distance
recomputation audited all 9,216 prediction/choice rows. A separate process replaced
query targets/candidates with NaNs: all 12 relation cells retained bitwise-identical
maps/predictions. Image derangements, split disjointness, fixed denominators, source
hashes, checkpoint equality, gates and report status were checked. The initial
standalone poison-audit launch lacked PYTHONPATH; it failed import, then passed
with the project root explicitly supplied. Both logs remain. No result was rerun
or changed for that audit repair.

Final complete process invocations took 2.62–2.97s each (10.98s total, imports and
report rendering included). Internal CPU time1.81–1.84s each; peak RSS579–582MiB.
All declared resource limits passed. A float64 map uses33,280 bytes versus131,072
bytes for paired FP32 support codes; retained source evidence and model storage
are additional, so this is not a total-memory compression claim. Per-stage timing,
parameter bytes and raw storage remain in each run. No neural weights were updated.
Inherited training cost is 3000 mean +600 variance updates per checkpoint and is
explicitly excluded from these new runtime times, not treated as free learning.

Four public-only Claude rounds took202.40s total wall time. The CLI reports a
cumulative API-equivalent estimate of$0.5413478, not subscription billing; per-round
usage and cumulative cost are distinguished in `claude-summary.json`.

Four public-only Claude rounds at requested medium effort are verified by
`modelUsage` naming `claude-opus-5-5`. Claude supplied the generic ridge reference,
challenged the design, and reviewed the public implementation contract; private
implementation and measurement review stayed local. Round4 acknowledged the
repairs and withdrew candidate exclusion. Requested extra isolation/derangement
checks passed locally. No additional population or training was authorized by
review agreement. Exact briefs, responses, model receipts, frozen source manifest,
auditors and logs are in `runs/reviews/representation_transfer_20260924/`.

Reports use the unchanged renderer and have structural verification; no new
browser-QA claim. A saved three-relation contact sheet was visually inspected:
affine outputs track their targets, while the threshold examples visibly lose
sharp binary detail even at the target-code reconstruction floor. Full frozen-source CPU suite: **862 passed**, exit0 in620.35s, source hashes
unchanged. Six focused numeric/integrity checks passed before the final runs.
Final receipts: `full-suite-exit.json`, `source-freeze.json`, `doc-checks.json` and
`raw-artifact-audit.json` in the review directory. All seven evaluation invocations
are costed in `all-run-costs.json`; complete development/failed-check process
overhead was not instrumented and is not presented as zero.

Plan/red commit: `8b24950`; tested implementation: `5925311`. Research event N578
and staged interpretation O402 bind the portable evidence; no broad claim promoted.
