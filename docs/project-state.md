# Current work

**Active:** final verification of the completed modular restart.
[Migration record](migration.md) describes retained components and evidence.

The active tree now has `pathwm/`, two editable Python recipes, focused tests and
short user/workflow guides. Retired model packages, dated scripts/configs, old
notebooks/docs, third-party vendoring and the plugin dashboard builder are removed.
They remain recoverable from `archive/pre-modular-2026-09-09`. Data and old runs
remain on disk; the new package has no dependency on that historical source.

Sixteen CPU tests pass. Reference computation checks found zero difference in 505
scoped comparisons. Real CPU perception-to-dynamics and GPU COCO/ViT development
runs train, pause and resume; final example reports are being verified. Packaging
and imports work independently. A final clean-Git snapshot check is next.

Daily starting points: [perception recipe](../experiments/perception.py),
[experiment guide](experiments.md), [model guide](models.md). New experiments should
edit recipes and modules directly, following [the standing workflow](experiment-workflow.md).
These checks do not establish model quality or closed-loop control.
