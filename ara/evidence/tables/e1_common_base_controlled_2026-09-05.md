# E1_common_base controlled physical A/V evidence — 5 September 2026

Product: ABI-v2 evidence for the modality-neutral belief and replaceable H1 interface.
Research: controlled source / frozen readouts in E1_common_base development; no H1 or freeze result.
Provenance: experiments ai-executed; interpretation ai-suggested.

- Dynamic source: 192 clips, 128 train / 64 eval, 64/32 disjoint recording groups. Raw matching
  1.0, swapped 0.0, coordinate MAE 0.002079; constant source exactly chance with all ties.
- Both predeclared 5,000-update R0 seeds fail video/audio rank and video temporal retrieval.
  Video ranks 0.155671/0.108127, audio 0.214671/0.231884. Independent teacher-copy future
  intervals remain positive in both modalities/seeds. Original gates are preserved.
- Frozen physical readouts introduced after seed 0's failure retain more information through
  token layouts. Video current/future readouts improve from initialization; audio current
  coordinate readouts decline. These are conditional probe results, not intrinsic information bounds.
- Equal-width follow-up: both readout types output 192 features and fit 3,072 weights plus
  target means. Two fixed projections give current-coordinate R² video 0.760–0.821 vs
  pooled 0.257–0.321, audio 0.941–0.951 vs 0.639–0.744. All tested paired current/future
  MSE-gain group intervals favor projections. Constant features give the exact train mean;
  all shuffled-target current R² values are negative. No encoder is updated by labels.
- Exact eight-coordinate code: std 1, coordinate error <=1.15e-8, rank fraction 0.039775
  at either 32 or 120 tokens / width 192. This is not a full dynamical state or gate exemption.
- The R1 balanced assignment loss is preserved on isolated branch dev/controlled-balanced-time,
  commit 57876df. It passes 205 tests and exact default CPU loss/gradient/RNG replay, but no
  candidate R1 panel exists. Main passes 202 tests; both suites deselect two opt-in tests.

The companion JSON preserves original receipts, hashes and executable audit code. The patch
records the isolated branch without implying it was merged or marked GREEN. Original run
ledgers, tensors and report remain under runs/controlled_av_20260905/. The report has two
charts/five tables and structural verification only; browser layout/chart rendering are unverified.

The next step is physical-diagnostic calibration and a compact readout across variable token
layouts, with encoders fixed and a declared budget. Keep audio-objective changes separate.
Do not relax the failed gates or widen R1 to TAU from these sources.
