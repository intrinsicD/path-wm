# Current work

**Active:** finish the user-authorized modular restart.
[Migration plan and evidence](migration.md) define the keep/remove choices.

The new package has independent CNN/DINO encoders, output heads, causal memory and
prediction, data adapters, two editable recipes and per-run offline reporting.
Sixteen CPU tests pass, including exact full/resumed training equality in both
loops. The retained CNN/head/memory computations pass 500 reference comparisons;
DINO/local decoding passes another five, all with zero observed difference.

The perception recipe has completed a short real-data pause/resume run and desktop/
mobile report QA. Final sequence and GPU COCO examples are being checked. Next:
remove retired active source and verify packaging, standalone imports and final
reports. These are development checks, not new model-quality/control claims.

Reference source: `archive/pre-modular-2026-09-09` at
`e95b6a6252cae72402de0dd93f419e8e93d25d12`. Data and completed runs remain on disk.
