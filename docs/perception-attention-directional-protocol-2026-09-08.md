# Direction-specific attention reliance

Adaptive read-only follow-up after the completed bilateral intervention. Source
results show fine-from-coarse head entropies0.9988–0.9996 and about1.04% relative
output change under uniform weights; coarse-from-fine includes one head at0.328
entropy and about47.7% output change. Uniformizing both directions removes all
fresh-case tolerance passes. Those facts motivate separating the directions;
they do not by themselves identify which direction caused the failure.

Reuse the same frozen CNN, P1 heads9107/9108/9109,512fresh cases, FP16 feature
round trip and baseline CPU/GPU tolerance. Compare normal, fine-from-coarse only
uniform, coarse-from-fine only uniform, fine-from-coarse only zero, and
coarse-from-fine only zero. No training or checkpoint selection. Report mean q,
per-case tolerance fraction and RGB MSE, preserving per-case raw predictions.
Interpret as reliance under perturbation, not the achievable score of a model
trained with pooling or no exchange. These are explicitly adaptive diagnostics.

Add an essential check that suppressing one direction leaves the independently
computed other output grid exactly unchanged. Reuse existing reference/entropy
and hook-cleanup checks. First run16development cases/one seed; verify dashboard,
commit, then seal and evaluate all512cases/three heads. CPU4threads,5minute cap
by expectation from the preceding14-second bilateral evaluation, no GPU queue
overlap. Keep original bilateral snapshots/results intact.
