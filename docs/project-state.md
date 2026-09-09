# Current work

**Completed:** the modular restart and its verification.
[Migration record](migration.md) describes retained components and evidence.

The active tree now has `pathwm/`, two editable Python recipes, focused tests and
short user/workflow guides. Retired model packages, dated scripts/configs, old
notebooks/docs, third-party vendoring and the plugin dashboard builder are removed.
They remain recoverable from `archive/pre-modular-2026-09-09`. Data and old runs
remain on disk; the new package has no dependency on that historical source.
Residual cache-only folders from the retired code and tests have also been removed.

Sixteen CPU tests pass, including from a clean Git snapshot without the old source.
Reference checks found zero difference in 505 scoped comparisons. Short CPU
perception-to-dynamics and GPU COCO/ViT runs train, pause and resume. Their reports
passed desktop/mobile browser checks; saved source snapshots match the active code.
Packaging and imports work independently.

Local examples: [perception](../runs/start_here/perception/report.html),
[dynamics](../runs/start_here/dynamics/report.html),
[ViT + COCO](../runs/start_here/coco_vit/report.html).

Daily starting points: [perception recipe](../experiments/perception.py),
[experiment guide](experiments.md), [model guide](models.md). New experiments should
edit recipes and modules directly, following [the standing workflow](experiment-workflow.md).
No further training is queued. These short checks establish the development path,
not model quality or closed-loop control. The next scientific experiment should be
declared in a small recipe and plan; do not revive the historical overnight queue.
