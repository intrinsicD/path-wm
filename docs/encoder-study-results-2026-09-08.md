# Encoder investigation — completed results

8 September 2026. **The residual-block encoder with cross-scale attention is the
strongest geometry candidate in this study. It still fails the position readiness
gate, and its generic output audits do not establish a versatile world model.**
All 33 formal runs completed: 110,000 optimizer updates and 12,160,000 frame
presentations. No training jobs remain active.

The user-adopted [critical Claude workflow](claude-collaboration-workflow.md) is
now linked from the standing experiment workflow. The [frozen protocol](encoder-study-protocol-2026-09-08.md),
[verified dashboard](../runs/experiment_dashboard.html) and
[raw reconciliation](../runs/encoder_study_2026-09-08/evaluation/integrity.json)
preserve the comparison, failures and exact values.

## What the trained depth/exchange comparison establishes

With exchange enabled, adding two residual blocks per branch reduces selected
validation q by **51.43%, 44.42% and 52.10%** across seeds 7107/7108/7109. Enabling
exchange in the deeper encoder reduces validation q by **32.63%, 24.22% and
48.33%**. Both meet the prospectively declared candidate signal: at least 10%
improvement in all three paired seeds.

The effect depends on the architecture. In the shallow encoder, exchange worsens
validation q by 27.70%, 14.12% and 9.33%. Adding depth without exchange changes it
by −7.93%, −16.30% and +1.35%, failing that candidate rule. Its test results improve,
but validation and training-seed behavior are less consistent. Cross-scale
attention is therefore useful in the tested deeper stack; this is not evidence
that it universally helps, or that it beats simpler trained fusion alternatives.

| Seed | Encoder / exchange | Selected validation q ↓ | Test q ↓ | Test angle MAE ↓ |
|---|---|---:|---:|---:|
| 7107 | Shallow / on | 2.874 | 2.475 | 24.75° |
| 7107 | Deeper / on | 1.396 | 1.068 | 2.21° |
| 7107 | Shallow / off | 2.251 | 2.672 | 26.72° |
| 7107 | Deeper / off | 2.072 | 1.735 | 9.92° |
| 7108 | Shallow / on | 2.541 | 3.076 | 30.76° |
| 7108 | Deeper / on | 1.412 | 1.410 | 2.90° |
| 7108 | Shallow / off | 2.227 | 2.743 | 27.43° |
| 7108 | Deeper / off | 1.864 | 1.599 | 12.50° |
| 7109 | Shallow / on | 2.968 | 2.990 | 29.90° |
| 7109 | Deeper / on | 1.422 | 1.247 | 2.18° |
| 7109 | Shallow / off | 2.715 | 2.863 | 28.63° |
| 7109 | Deeper / off | 2.751 | 2.405 | 24.05° |

q is the maximum of each of four position MAEs divided by 8 world units and angle
MAE divided by 10°. The unchanged readiness requirement is q≤1. **All 12 arms
fail on validation**, so none progresses to compatible memory/predictor training.
This does not establish that control is impossible; it means the agreed
progression gate remains unmet and control was not evaluated for these encoders.

For the deeper/on candidate, remaining test errors are:

| Seed | Pusher x/y MAE, world units | Body-origin x/y MAE, world units | Orientation MAE |
|---|---:|---:|---:|
| 7107 | 8.50 / 8.54 | 5.80 / 5.42 | 2.21° |
| 7108 | 11.28 / 9.87 | 5.44 / 6.65 | 2.90° |
| 7109 | 9.61 / 9.98 | 5.35 / 5.52 | 2.18° |

The body origin is the dataset's rigid-body reference point, not a redefined
visual center. Pusher error keeps q above 1 despite much better orientation and
body-position accuracy. All deeper/on runs select update 4,000, the final allowed
update; convergence has not been established.

![Depth and exchange results](../runs/encoder_study_2026-09-08/evaluation/factorial_q.png)

This tests a residual-block package: added spatial computation, normalization and
295,936 parameters, with the same original D/H and 320×64 exported latent layout.
It does not isolate depth from capacity or optimization. Shared initial tensors,
head initialization, frame draws and validation indices match within each seed.
Each run used 4,000 updates and batch 128 under the original RGB+pose objective.

Whole-configuration bootstrap intervals use 2,000 paired resamples of 22 test
groups, conditional on the trained models. The depth/exchange test interaction
is negative in each seed: −0.470, −0.522 and −1.285 q; conditional 95% intervals
are [−0.992, −0.137], [−0.995, −0.167] and [−2.002, −0.593]. These intervals do not
estimate training-seed uncertainty. One deeper exchange contrast (seed 7108)
has an interval spanning zero despite its favorable point estimate. Three seeds
and reused, previously inspected holdouts do not establish population-wide
significance or equivalence. Exact contrasts and draws are in
[factorial_summary.json](../runs/encoder_study_2026-09-08/evaluation/factorial_summary.json).

## Cost and training behavior

On the RTX 3050, deeper/on training takes 309–336 seconds versus 234–259 seconds
for shallow/on, including validation/checkpoint work but excluding standalone
evaluation and reporting. Peak allocated training memory is 1.52 versus 1.39 GB
(decimal). Active encoder parameters increase from 94,304 to 390,240.

The development-only, synchronized batch-one encoder latency is approximately
0.63 ms for shallow/on, 1.13 ms for deeper/on, 0.38 ms for shallow/off and 0.90 ms
for deeper/off. These are encoder measurements, not planning latency. DINO's
separate batch-32 profile is not a directly comparable latency measurement.

At the last deeper validation checkpoint within the paired shallow run's elapsed
budget, deeper/on q is 1.695 / 1.753 / 1.564 versus shallow final q
2.874 / 2.541 / 2.968. Its advantage remains in this limited wall-time comparison;
there is no separate test evaluation of those earlier checkpoints. The
[update curves](../runs/encoder_study_2026-09-08/evaluation/factorial_learning.png)
and [elapsed-time curves](../runs/encoder_study_2026-09-08/evaluation/factorial_walltime.png)
retain all validation spikes. Gradient-clipping frequencies are retained in the
[compute audit](../runs/encoder_study_2026-09-08/evaluation/compute_and_optimization.json);
different model gradient norms do not by themselves identify a causal mechanism.

## Frozen-representation diagnostics

| Validation-selected readout | Test angle MAE | Test body-origin x/y MAE | Test q |
|---|---:|---:|---:|
| Coupled spatial, 6,000-update budget | 7.98° | 13.36 / 18.59 | 2.32 |
| Independent spatial, selected update 1,200 | 20.98° | 12.14 / 12.85 | 2.10 |
| Frozen task-only custom A + fresh D/H | 23.30° | 13.04 / 13.32 | 2.33 |
| Frozen DINO + trained adapter + fresh D/H | 15.57° | 18.09 / 13.91 | 2.29 |
| Frozen native DINO + scaled linear pose head | 16.45° | 10.68 / 7.10 | 1.65 |

None passes readiness. These diagnostics use one seed each; their head packages,
capacity and some objectives differ. The independent head removes orientation
calculation through the estimated body-origin point, but still shares its feature
trunk. It is not a pure test of gradient coupling.

The coupled head's new within-run 2,000→6,000 window improves validation q
4.339→2.403 and angle 17.86°→7.77°. The independent head reaches its best q at
1,200. Its extra updates do not improve the selected score. The historical coupled
run is not an exact continuation: tiny GPU gradient differences grow despite
identical initialization and draws; deterministic GPU algorithms were not enabled.
The new within-run window is the relevant budget comparison.

The unscaled native DINO head is severely unstable (selected test q 12.02; final
validation q 644.26). The prospectively declared input-scaling correction preserves
linear function capacity and yields test q 1.65. Both runs are preserved. Neither
result proves an encoder information limit. Native width, head capacity, scaling,
pose-only objective and budget differ from the adapter+RGB comparator.

DINO uses the official pinned source and verified official weights. RGB64 is
resized to 224 without an additional crop, preserving the field of view. Upsampling
does not add observed detail. The 384→64 adapter is trained; its coarse map pools
the same fine features and adds no independent backbone information. Pretraining,
architecture and adapter capacity are not isolated in this comparator.

## Common RGB and foreground output audit

All sources are frozen, with fresh identical RGB/mask decoders, initialization,
frame draws and 2,000-update budgets. There are 4,096 / 512 / 512 COCO train /
validation / test views. The two decoder modules are separate but share a joint
gradient-clipping operation. Selection uses validation RGB MSE + mask BCE.

| Frozen source | Test RGB MSE ↓ | Foreground IoU ↑ | Dice ↑ |
|---|---:|---:|---:|
| 7107_deeper | 0.009314 | 0.3156 | 0.4395 |
| 7107_deeper_no_exchange | 0.009543 | 0.3144 | 0.4386 |
| 7107_no_exchange | 0.006987 | 0.3500 | 0.4815 |
| 7107_reference | 0.009259 | 0.2960 | 0.4094 |
| 7108_deeper | 0.007797 | 0.3354 | 0.4637 |
| 7108_deeper_no_exchange | 0.008135 | 0.3159 | 0.4408 |
| 7108_no_exchange | 0.005912 | 0.3177 | 0.4467 |
| 7108_reference | 0.007117 | 0.2745 | 0.3898 |
| 7109_deeper | 0.008795 | 0.3077 | 0.4290 |
| 7109_deeper_no_exchange | 0.007827 | 0.3328 | 0.4613 |
| 7109_no_exchange | 0.007150 | 0.3100 | 0.4385 |
| 7109_reference | 0.006914 | 0.3016 | 0.4198 |
| custom | 0.007349 | 0.3183 | 0.4399 |
| dino | 0.037109 | 0.5832 | 0.6959 |
| warmup | 0.005035 | 0.3090 | 0.4320 |

Always-foreground test IoU is **0.3235**, training mean-mask IoU is 0.1236, and
always-empty IoU is 0.0020. Deeper/on improves IoU over shallow/on in all three
seeds, but beats the stronger trivial foreground baseline in only one seed.
Its RGB reconstruction error is higher in all three paired comparisons.
The DINO adapter reaches IoU 0.5832 despite substantially worse RGB reconstruction.

This supports measuring multiple useful outputs rather than selecting E from
reconstruction or pose alone. It does not prove incompatible representations,
irrecoverable information loss, or a generally superior backbone. Readout
optimization and pretraining differ; COCO overlap with DINO pretraining is not
excluded. Task-only A and the COCO-warmup E have different training lineages, so
their comparison is output recoverability, not a matched pre/post forgetting
measurement. The earlier decoder-recovery experiment remains the matched
forgetting evidence.

Masks are the union of annotated non-crowd instances; crowd pixels are ignored.
IoU/Dice use a fixed 0.5 threshold, value 1 when both valid masks are empty, and
exclude all-invalid images. All 512 test images contain valid pixels.
[Per-source summaries](../runs/encoder_study_2026-09-08/evaluation/probe_summary.json)
include foreground-fraction strata. This measures annotated foreground coverage,
not instance separation, per-object extent or every possible object.

Full six-view panels: [task-only custom A](../runs/encoder_study_2026-09-08/evaluation/custom_output_panels.png),
[COCO warmup](../runs/encoder_study_2026-09-08/evaluation/warmup_output_panels.png),
[DINO adapter](../runs/encoder_study_2026-09-08/evaluation/dino_output_panels.png).
The dashboard embeds a compact, matched comparison of the first two fixed views
to stay within its payload limit; the full panels and all raw arrays are retained.

## Internal perception states

[All four first-seed panels](../runs/encoder_study_2026-09-08/evaluation/internals/curriculum_analysis.json)
and the dashboard show the same six fixed test frames, pose predictions,
reconstructions, fine/coarse PCA maps and normalized attention entropy. Raw
latents, training-only PCA bases and checkpoint hashes are preserved. PCA uses
256 training frames separately per E, so colors are not aligned across encoders
and the first three components omit other information.

The [deeper/on panel](../runs/encoder_study_2026-09-08/evaluation/internals/deeper_perception.png)
has mean fine attention entropy 0.9992 and coarse entropy 0.7981 on a 0–1 scale.
Nearly uniform fine attention is consistent with broad aggregate context. It does
not establish redundancy or explain the measured benefit. A trained simple-fusion
comparison would test that hypothesis. These are observed states, not imagined
futures, and attention pictures are not causal evidence.

## Interpretation and next decision

Use deeper/on as the next experimental reference, preserving the original model.
Do not expand the number of exported scales yet. The immediate missing capability
is accurate pusher position under the existing image view and objective.

A bounded next comparison should distinguish a longer unchanged training budget
from a prospectively calibrated position/angle objective. At readiness-scale
errors, the current pose MSE assigns a 10° error between unit orientation vectors
124.45 times the loss of one 8-unit position-coordinate error (31.11 times four
such position errors). This is an analytic scaling fact, not proof of the observed
error's cause or a validated remedy. Reweighting must also control total loss and
gradient scale and retain angle, RGB and mask checks. This follow-up is proposed,
not already trained.

A data-support audit found no pusher centers outside the RGB view; only 4 of
2,506 test frames have centers within 15 world units of an image border. It does
not rule out occlusion, ambiguity or readout limitations. Extra levels, registers,
higher observed resolution, mixed-domain/conditioned decoders, simpler fusion and
a software-persistence task remain separate staged studies. No new evidence here
establishes temporal memory, action consequences, computer use or driving ability.
Compatible U/P training is still required before comparing complete world models.

## Verification and reproducibility

All 33 formal results reconcile with their exact selected validation checkpoints,
update counts, presentations and raw per-frame pose/RGB/mask metrics. Paired
factorial streams/initialization and all probe populations/baselines match.
The prepared PushT files and COCO frame cache match their source manifests.
Failed development artifacts, unstable native-head results and prospective
corrections remain visible.

The full repository suite passes **379 tests**, including installed-browser
checks, with no skipped tests. The final HTML passes package, validation and
browser verification at widths 1440 and 390, including source-dialog interaction.
Exported scientific figures and all four internal-state panels were visually
inspected. The [loader repair](encoder-checkpoint-loading-note.md) preserves
encoder depth/exchange metadata in ordinary loading and inference export; all
12 trained E/H/D outputs match the experiment factory exactly.

The exact frozen training source is preserved at commit
b95906cabdad0c3df7d2ff0aa70196729b5e4c33. Loader integration happened only after the
27-unit remaining schedule finished and all frozen source hashes matched.
Code and reports are tracked; datasets, checkpoints, caches, exact replies and
large HTML/image artifacts remain local under data/ and runs/.

Claude completed two execution-protocol exchanges and accepted explicit
corrections; the earlier design review had three exchanges. Automatic approval
review rejected an additional implementation-source/local-results export as
beyond that payload's authorization. Nothing was sent for that rejected review,
and implementation review continued locally. Peer agreement is not empirical
validation. Receipts and replies remain under the experiment's collaboration
directory.
