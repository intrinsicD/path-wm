# E1 common-base overnight development evidence — 2026-09-05

Product: ABI-v2 video/audio evidence for the modality-neutral belief and a replaceable H1 interface.
Research: matched E1_common_base R0/R1 development; no formal E0/E1 freeze or H1 result.

Provenance: ai-executed measurements; ai-suggested interpretations and next experiment. The user
authorized autonomous overnight continuation; silence was not treated as endorsement of a result.

## Completed comparison

24 source archives (107,607,337,075 bytes before extraction) are verified. The normalized manifest
contains 8,646 train / 3,645 held-out 10-second clips from 305 / 126 disjoint recording groups.
Windows span 0.5 seconds; future offset 0.5 seconds and same-recording A/V shift 2 seconds.
Video is 8 Hz, 64×64 RGB; audio mono 16 kHz. No scene label enters representation objectives.

Four full R0 runs use 10,000 updates, batch 64, seeds 0/1, and the same clean runtime `0bb78d5`.
Only covariance changes (0 versus 0.002); the initial panels match exactly within each seed.
Both covariance sources pass all ten R0 conditions; both controls fail the two rank gates.
Four 10,000-update runs on the 20-clip examples still fail both future gates. That duration
diagnostic does not select the coefficient; the matched full-corpus comparison retains 0.002.

The two R1 continuations each complete 10,000 further updates on clean runtime `e49c451`,
initialized from their own SHA-bound passing R0 source with fresh optimizer/RNG and analytic
clock transport. Seed 0 passes all seven conditions; seed 1 fails only synchrony (-0.977 pp).
Both retain positive independent teacher-copy future intervals; neither paired balanced-time
interval excludes zero. B0 is deferred.

## R0 Gate Table

| seed | covariance | gate | video_future | audio_future |
|---|---|---|---|---|
| 0 | 0 | Fail: both ranks | +0.027073 | +0.001024 |
| 0 | 0.002 | Pass | +0.045454 | +0.003842 |
| 1 | 0 | Fail: both ranks | +0.027473 | +0.001434 |
| 1 | 0.002 | Pass | +0.049802 | +0.003246 |

## R1 Gate Table

| seed | gate | video_to_audio | audio_to_video | sync_pp | video_rank | audio_rank | failures |
|---|---|---|---|---|---|---|---|
| 0 | Pass | +0.1447 | +0.1463 | +1.76 | 0.3353 | 0.3187 | None |
| 1 | Fail | +0.1534 | +0.1572 | -0.98 | 0.3332 | 0.2982 | synchrony_accuracy_above_chance=-0.00976562 does not satisfy greater 0 |

## Timing Intervals

| seed | stage | accuracy_pp | interval_pp |
|---|---|---|---|
| 0 | R0 source | +0.45 | -1.00 to +1.88 |
| 0 | R1 trained | +1.41 | -0.22 to +3.08 |
| 0 | Paired R1 − R0 | +0.96 | -1.40 to +3.34 |
| 1 | R0 source | +0.26 | -1.33 to +1.88 |
| 1 | R1 trained | +0.89 | -0.76 to +2.55 |
| 1 | Paired R1 − R0 | +0.63 | -1.59 to +2.85 |

## Future Intervals

| stage | seed | modality | advantage | interval |
|---|---|---|---|---|
| R0 source | 0 | Video | +0.043840 | +0.037857 to +0.050212 |
| R0 source | 0 | Audio | +0.000820 | +0.000424 to +0.001244 |
| R0 source | 1 | Video | +0.044328 | +0.038259 to +0.050760 |
| R0 source | 1 | Audio | +0.000981 | +0.000788 to +0.001194 |
| R1 trained | 0 | Video | +0.037311 | +0.030620 to +0.044334 |
| R1 trained | 0 | Audio | +0.004693 | +0.003850 to +0.005580 |
| R1 trained | 1 | Video | +0.036159 | +0.029033 to +0.043558 |
| R1 trained | 1 | Audio | +0.002631 | +0.001735 to +0.003570 |

## Retrieval Context Table

| seed | control | r0_margin | r1_margin | r1_interval | paired_interval |
|---|---|---|---|---|---|
| 0 | Other clip | +0.0008 | +0.1518 | +0.1282 to +0.1757 | +0.1276 to +0.1741 |
| 0 | Same scene, other recording | -0.0012 | +0.0265 | +0.0085 to +0.0443 | +0.0095 to +0.0457 |
| 0 | Same recording, other clip | +0.0006 | +0.0038 | -0.0033 to +0.0106 | -0.0045 to +0.0106 |
| 0 | Same clip, +2 seconds | +0.0003 | +0.0000 | -0.0028 to +0.0029 | -0.0032 to +0.0026 |
| 1 | Other clip | -0.0105 | +0.1552 | +0.1297 to +0.1806 | +0.1390 to +0.1923 |
| 1 | Same scene, other recording | +0.0002 | +0.0288 | +0.0083 to +0.0490 | +0.0071 to +0.0496 |
| 1 | Same recording, other clip | -0.0004 | +0.0025 | -0.0051 to +0.0099 | -0.0050 to +0.0107 |
| 1 | Same clip, +2 seconds | +0.0005 | +0.0008 | -0.0020 to +0.0037 | -0.0026 to +0.0034 |

## Scene Points

| seed | model | video | audio |
|---|---|---|---|
| 0 | Untrained | 33.28% | 41.63% |
| 0 | R0: covariance 0 | 38.32% | 46.62% |
| 0 | R0: covariance .002 | 36.48% | 46.65% |
| 1 | Untrained | 32.83% | 41.19% |
| 1 | R0: covariance 0 | 35.87% | 46.53% |
| 1 | R0: covariance .002 | 37.09% | 46.25% |

## Scene Differences

| seed | modality | delta_pp | interval_pp |
|---|---|---|---|
| 0 | Video | -1.84 | -4.65 to +1.08 |
| 0 | Audio | +0.02 | -1.36 to +1.37 |
| 1 | Video | +1.23 | -1.70 to +4.21 |
| 1 | Audio | -0.28 | -1.51 to +0.97 |

## Methods and implementation proofs

- Official panel: 32 batches ×64 held-out windows, separate evaluation RNG. Raw checkpoint,
  metric, summary, threshold, copied-spec and final-panel values reconcile for every full run.
- Independent timing/future: one SHA-selected start per all 3,645 held-out clips, fixed before
  R1 outcomes; 10,000 recording-group bootstrap resamples, seed 950003. Intervals condition on
  fitted encoders, fixed split and readouts; they do not estimate training-seed variability.
- Context controls are exploratory after seed 0: every eligible negative per category, both
  directions averaged; query groups resampled with the negative pool fixed, seed 950017.
- Scene readout: fixed ridge coefficient 0.01, one window at 3 seconds, train-only fitting and
  standardization; paired group resampling within ten scene strata. Covariance-minus-control
  intervals include zero in both modalities/seeds. Rank is not a demonstrated scene advantage.
- Constant-input and within-position controls show substantial audio position structure. Eighty
  native-clock checks agree on first track timestamps, but 8 Hz frame selection lags by up to
  32.9 ms; normalized grid time is not exact native exposure time.
- Correct shared-window-end timestamps with unconverted R0 time weights caused video rank
  0.1825 and future advantage -0.5017 on a frozen preview. Analytic Fourier coefficient rotation
  and bias translation restore 0.34235/+0.04813 on equivalent physical windows; the largest
  panel difference is 2.15245e-7. Non-time weights and RNG are exact. ABI bf16 rounding limits
  output bit equivalence, separately from the tested pre-rounding time functions.
- One-batch prefetch reduced mean steady update time from 0.256668 to 0.183243 seconds (28.6%)
  in a warm-cache compatibility probe with virtual IDs over unchanged example bytes. All four
  100-update trajectories and actual CUDA recovery were exact; this is a runtime probe, not data.
- The final runtime exactly replays the earlier 250-step R0 CUDA trajectory. Actual R1 CUDA
  interruption/recovery reproduces weights, EMA, optimizer, random streams and panels exactly.
- 196 fast tests pass; two opt-in tests are deselected. Every completed seed refreshed the
  dashboard. The canonical report has two native charts and eight tables, structural-only QA;
  browser layout, source-dialog interaction and chart rendering were not verified.

## Metric counterexample and next question

Current video/audio both equal an orthogonal clip axis e. Shifted video is 0.5e+sqrt(3)/2 f;
shifted audio is 0.5e-sqrt(3)/2 f. The three audiovisual metrics pass (retrieval 1, synchrony
+0.5) while balanced assignment is always wrong and change alignment is -0.25. This isolates
metric semantics; it does not construct a trained checkpoint or test its separate collapse guards.
The next controlled source/objective experiment is in `docs/common-base-next-experiment.md`.
It is planned, not implemented or run. A unique cause for the measured timing weakness remains open.

## Source bindings

- Machine-readable snapshot: `ara/evidence/tables/e1_common_base_overnight_2026-09-05.json`, SHA-256 `6ba8bf4d6705165910332fe430ad70762062e2361f80de2bf2d0ca46bdcd7892`.
- Manifest SHA-256: `fa284fe55b9e0bc35e970f592a28b703f5fbaabb46efe9669e72f880c8c905d9`.
- Report: `runs/overnight/common_base_20260905/report.html`; exact source producer and input identities accompany it.
- Implementation commits include 5cbe0a6, 0320c0a, 83beccf, 74f87f7, cad31df, 0bb78d5,
  f54976c, fde5729, 5d58418 and 4ac3d90. Source-bound R1 spec: 1d5711a. Decisions: e49c451, ca65ce0.
- `runs/overnight/common_base_20260905/full_comparison_validation.json` — SHA-256 `2e50a443af541f1d3bba865afe8403d303e3402ee16cf5c601b4a731645fc91a`.
- `runs/overnight/common_base_20260905/r1_comparison_validation.json` — SHA-256 `c206072b7fe4b672010f4a75b18630bd5e1919364deacddda1d794ec4df48001`.
- `runs/overnight/common_base_20260905/time_transport_compatibility.json` — SHA-256 `1f0df846dadf0d2483e75d6408b56893cc4d7879842869299af42045b7b9f95f`.
- `runs/overnight/common_base_20260905/time_transport_r0_compatibility.json` — SHA-256 `1ca65c8ca025cbf1962767177535b69789af1cb004733a56e4dd17ff5b937126`.
- `runs/overnight/common_base_20260905/r1_cuda_recovery_compatibility.json` — SHA-256 `02df5b5beccdd77f90c849d615b524f9e41f9528a0bd32f36793a9d0866b5a4c`.
- `runs/overnight/common_base_20260905/report_qa.json` — SHA-256 `6d2c2fcc64585cd0815311a2b4e8e5ebfce3f0fd7ced690ca94b45c7dca761d8`.
- R0 balanced seed 0: checkpoint SHA-256 `d005f709145bf9d280bda7047e3d233b40b9a5d05b1f8468815aaa040332c535`.
- R0 control seed 0: checkpoint SHA-256 `d7146ece43fe09c56764954a8a473b98ac4569a359173d3ab180b623aca3cb5f`.
- R0 balanced seed 1: checkpoint SHA-256 `d4c3df7a6a5409c01a1e4c0584d6fac6560a3aef5dc1c91a9d0707d573eab57f`.
- R0 control seed 1: checkpoint SHA-256 `2b0af61d375ce1e55ec849f1ff90079d2d23f4b524feb4b910a943d8f0db14f1`.
- R1 av seed 0: checkpoint SHA-256 `fde285e6abe87844d7d8806bdc6e916d80604ee4ea0c4391f223ea79287b04f8`.
- R1 av seed 1: checkpoint SHA-256 `8fc6f86f16867896de3b87d7f78001d99b6e657a47ba5b26af8872bc732d0a46`.
