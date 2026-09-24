# Nonlinear output fidelity: frozen decoder range

24 September 2026. Continuation of [the handoff](architecture-continuation-handoff.md).
This bounded diagnostic precedes any new learned inducer or codec repair. It does
not repeat the affine transfer screen. Earlier target encode/decode error is a
reference, **not a lower bound** on all possible latent predictions. R1's failed
support readers do not localize representation failure; CI1's encoder adaptation
lost old behavior. Keep both codec weights frozen and separate output range,
encoding and support inference here.

## Slices and decision

1. Register protocol and meaningful red checks: affine-range projection against
   independent least squares; support-only threshold estimation and invalid inputs;
   target poisoning; frozen output/restart integrity; report failure preservation.
2. Implement one readable recipe using the existing codec, generator, ridge helper
   and renderer. No new library module, trainer, storage owner or neural update.
   Compare stored examples with a support-inferred shared threshold structure.
3. Freeze source, evaluate two untouched populations with both existing codecs,
   independently audit arrays and run the CPU suite. Stop this selected path if a
   final decoder-range reference exceeds the fidelity gate. Record the blocker;
   do not tune thresholds, train a repair or claim the architecture is complete.

## Registered populations, comparisons and gates

CPU FP32 frozen codecs from `runs/evidence_loop_final_s{17,29}/last.pt`; SVD and
projection in float64. Only pose zero is used for binary output prediction.
Development seed926101, threshold vector(.4,.4,.4). Final seeds926117/926129,
two relations with channel thresholds(.3,.45,.6) and(.55,.35,.5). These are fresh
operators in the **same threshold family**, not novel-family generalization.
Each relation gets64 independently generated support images and64 query images;
64 other smooth images check all four familiar poses. Supplied part identity,
channel alignment and support relation grouping are oracle conveniences.
One tiny smoke uses development seed, four support/query images; one full dev
invocation on codec17; four final invocations. No fitting on query targets.

Support-only arms: raw nearest whole-image support output (pixel-distance);
latent affine ridge using existing penalty.001; channel-wise monotone threshold
estimated as the midpoint between largest negative and smallest positive support
values; the same threshold applied after input codec reconstruction. Threshold
estimation explicitly rejects missing classes or contradictory/nonmonotone labels.
The hand-specified threshold family is a strong task-specific prior, not learned
induction. Whole-image shuffled support tests label dependence by fit rejection
or observed prediction degradation; rejection is not a capability result.

Evaluator-only references: target encode/decode and orthogonal projection of true
target pixels onto the **unclipped affine decoder range**. The latter is a numerical
lower bound over unrestricted latent codes for this fixed decoder, not an inference
arm or a claim about clipped/nonlinear/changed decoders. Save singular values,
rank, residual orthogonality, optimal codes and both reconstruction arrays. Use
float64 SVD with tolerance max(shape)*eps*smax; compare against a full-column
least-squares solve and flag rank truncation/poor numerical agreement. Report
all errors unclipped; bound output clipping to visualization only.

Per final relation/cell: novel-output fidelity MSE<=.002; raw support-threshold
MSE<=.001 (task identifiability control); old familiar-pose MSE<=.001 at every pose;
weights/buffers and old outputs bit-exact; optimal projection residual normal
condition <=1e-8 (max absolute residual@W); least-squares MSE difference<=1e-9;
projection error<=target encode/decode error+1e-9. Decoder-range MSE>.002 is a
concrete blocker for **any** latent inducer constrained to this decoder on these
outputs. If projection passes but encoded reconstruction fails, encoding is an
additional repair candidate; if both pass but inference fails, investigate the
inference contract. Multiple limitations may coexist. No unregistered success
claim for binary accuracy, concept acquisition or latent advantage.

## Scope, integrity and budgets

No new runtime/session API is needed for a decoder diagnostic. Population inference
uses tensors; exact saved threshold/map reload is checked. Existing correction,
stale-source and session restart contracts remain regression tests, not new claims
of a live nonlinear agent. Inference accepts support and query input only; target
poisoning must leave fitted state and predictions bit-exact. Neural model hash
and all old reconstructions are checked before/after. Retained raw instances and
fitted state have separate storage accounting; pretraining remains sunk cost.
Prepare each decoder basis once and reuse across outputs. Preserve authoritative
source arrays; evaluator-only target use is structurally separate from inference.

Each invocation <=120s, <=2GiB RSS, two CPU threads. No GPU/training. One software
repair after development, no scientific hyperparameter repair or final tuning.
Full suite <=20min; at most four public-only actual Claude Opus5.5 medium rounds,
<=10min each. Save briefs/replies/model receipts in
`runs/reviews/nonlinear_fidelity_20260924/`. Record invocation wall/CPU/RSS including
imports and reporting. Every completed run saves raw arrays, frozen checkpoint,
result before report, source snapshot and standalone report using unchanged renderer.
Report status is independent of numeric gate outcome. Browser QA is not newly
claimed. Failed experiments remain visible and immutable.
