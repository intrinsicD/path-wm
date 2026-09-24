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
