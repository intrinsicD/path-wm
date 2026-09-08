# Encoder investigation — running results

Status: diagnostic/reference slice complete; three-seed depth-by-exchange
comparison and its audits are running. This is a partial research record, not a
final model recommendation. See the [frozen protocol](encoder-study-protocol-2026-09-08.md)
and [verified dashboard](../runs/experiment_dashboard.html).

## Completed frozen-representation diagnostics

| Validation-selected readout | Test angle MAE | Test object x/y MAE | Test q |
|---|---:|---:|---:|
| Coupled spatial, 6,000-update budget | 7.98° | 13.36 / 18.59 | 2.32 |
| Independent spatial, selected update1,200 | 20.98° | 12.14 / 12.85 | 2.10 |
| Frozen custom A + fresh D/H | 23.30° | 13.04 / 13.32 | 2.33 |
| Frozen DINO + trained adapter + fresh D/H | 15.57° | 18.09 / 13.91 | 2.29 |
| Frozen native DINO + scaled linear pose head | 16.45° | 10.68 / 7.10 | 1.65 |

q=max(each position MAE/8 world units, angle MAE/10°); q≤1 is required. None passes.
All2,506 test frames belong to reused, configuration-disjoint holdouts. One seed
per diagnostic does not estimate training-seed variance. Head packages and some
objectives differ; these are not pure architecture or information-loss tests.

The coupled head's within-run2,000→6,000 extension improves validation q4.339→2.403
and angle17.86°→7.77°. The independent head's best validation q was already reached
at1,200; extra training did not improve that selected score. The historical coupled
run is not an exact continuation: tiny GPU gradient differences grow despite
matching initialization and draws. The new within-run budget window is the valid
budget comparison.

The unscaled native head was severely unstable (selected test q12.02; final
validation q644.26). The prospectively amended fixed input scaling preserves
linear function capacity and makes this control useful, yielding test q1.65.
Neither its initial failure nor its improvement proves an encoder information
limit. Native width, readout capacity, scaling and pose-only objective differ
from the adapter+RGB comparison.

## Completed common RGB/foreground audit

The task-only A encoder and COCO-warmup encoder have different training lineages;
this audit compares output recoverability and does not measure pre/post forgetting.
The earlier decoder-recovery experiment remains the matched forgetting evidence.
All representations are frozen; fresh identical RGB and mask decoders receive the
same initialization,2,000updates and frame draws. Prepared COCO subsets contain
4,096/512/512 train/validation/test images; crowd pixels are ignored.

| Frozen source | Test RGB MSE ↓ | Foreground IoU ↑ | Dice ↑ |
|---|---:|---:|---:|
| Task-only custom A | 0.007349 | 0.3183 | 0.4399 |
| COCO-warmup custom E | 0.005035 | 0.3090 | 0.4320 |
| Task-fitted DINO adapter | 0.037109 | 0.5832 | 0.6959 |

Always-foreground IoU is0.3235; the training mean-mask baseline is0.1236. Thus the
two custom readouts do not beat the stronger trivial IoU control at this budget,
while DINO does. Pixel recoverability and useful foreground readout quality differ
substantially. This supports measuring multiple outputs rather than choosing an
encoder from reconstruction alone. It does not isolate pretraining, architecture,
adapter compression or decoder optimization, nor prove irrecoverable information
loss. COCO overlap with DINO's pretraining is not excluded. The union mask measures
annotated foreground coverage, not instance separation, extent or all possible
objects.

## Verification and remaining work

The default repository suite passed369 tests; all3 opt-in installed-browser checks
also passed. The subsequent native-scaling invariant test passed with the affected
suite. Every completed run refreshed and browser-verified the dashboard. Raw
selection, sample streams and per-frame metrics are reconciled by
`scripts/report_encoder_study.py`. Pinned DINO source/official weights, failed
development artifacts and all corrections remain recorded.

The recurring Claude workflow is [adopted](claude-collaboration-workflow.md).
Two protocol exchanges completed successfully, including correction/retraction
of unsupported diagnostics. A subsequent implementation-source review was rejected
before execution by automatic approval review; no payload was sent and local
review continued. See the protocol for the specific scope and receipt.

Remaining: complete all12 paired custom-E runs and their common audits, inspect
observed latent states, calculate depth/exchange interactions and conditional
group uncertainty, apply existing downstream gates, and replace this partial
record with a final interpretation. Additional scales/registers/software tasks
remain conditional follow-ups, not silently scheduled expansions.

## First completed paired seed (7107; confirmation pending)

| Arm | Selected validation q | Test q | Test angle MAE |
|---|---:|---:|---:|
| Shallow, exchange on | 2.8742 | 2.4747 | 24.75° |
| Deeper, exchange on | 1.3961 | 1.0678 | 2.21° |
| Shallow, exchange off | 2.2508 | 2.6718 | 26.72° |
| Deeper, exchange off | 2.0722 | 1.7347 | 9.92° |

Depth with exchange produces a large geometry gain in this seed. Deeper/on test
pusher x/y MAE is8.50/8.54 world units, object x/y5.80/5.42 and angle2.21°; pusher
position keeps q above1. Validation pusher x/y9.07/11.17 likewise fails the gate.
No downstream training is triggered for these four arms. This is one of three
paired seeds; do not promote it to a final cross-scale or depth conclusion.

A read-only data-support audit found no pusher centers outside the observed image
in train/validation/test; only4of2,506 test frames have centers within15world units
of a border. This does not rule out occlusion, visual ambiguity or other limits.
The audit also checks all207 prepared PushT files and the82,783-image COCO cache
against their original manifests; all hashes match.

A separate analytic objective audit shows a10° angular error between unit vectors
contributes124.45times the pose loss of one8-world-unit coordinate error (31.11times
four such coordinate errors). This follows from the current normalized-coordinate
and sin/cos MSE, and makes loss calibration a plausible follow-up. It does not prove
that loss scaling caused the measured pusher-position errors. The current factorial
keeps that objective unchanged.

## Second completed paired seed (7108; final seed pending)

| Arm | Selected validation q | Test q | Test angle MAE |
|---|---:|---:|---:|
| Shallow, exchange on | 2.5410 | 3.0759 | 30.76° |
| Deeper, exchange on | 1.4123 | 1.4099 | 2.90° |
| Shallow, exchange off | 2.2266 | 2.7429 | 27.43° |
| Deeper, exchange off | 1.8636 | 1.5985 | 12.50° |

The second paired seed also favors depth with exchange. Deeper/on pusher x/y
test MAE is 11.28/9.87 world units, object x/y 5.44/6.65. All four validation
gates fail, so these models also stop before downstream training. Final inference
waits for the third paired seed and completed output audits.

## Observed perception inspection

The first declared seed's four selected encoders are inspected on the same six
fixed test frames. Panels show ground-truth/predicted pusher and body-origin pose,
reconstructions, fine/coarse PCA features and normalized attention entropy.
PCA is fitted on 256 training frames separately for each encoder; colors are not
aligned across models and the top three components omit other feature dimensions.
The panels are observed states, not imagined rollouts or evidence of planning.

For these six frames, deeper/on mean fine attention entropy is 0.9992 and coarse
entropy is 0.7981 on a 0–1 scale. Nearly uniform fine attention is consistent with
aggregate context, but does not establish its function or redundancy. A trained
fusion comparison would be needed to test that explanation. Raw states, PCA bases,
checkpoint hashes and all four visually inspected panels are preserved under
`runs/encoder_study_2026-09-08/evaluation/internals/` and in the verified dashboard.
