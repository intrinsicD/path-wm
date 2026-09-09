# Current work

**Implemented:** one small multimodal world model, built in
[the editable recipe](../experiments/multimodal.py). Read
[the concrete architecture](multimodal.md) and [the implementation record](multimodal-plan.md).

The model has image/video/audio/text adapters, five latent token groups, explicit
observation and thinking operations, bounded episodic memory, probabilistic imagined
futures, action proposals, candidate planning, self-error diagnostics, interventions
and a measured update gate with rollback. It has 113,560 parameters by default.
Every component is directly constructed as an ordinary PyTorch module.

Twenty-eight CPU tests pass, including all 16 retained checks. Short synthetic and
real PushT runs complete, pause and resume. Synthetic full/resumed model, teacher,
replay, optimizer and RNG state match exactly. Reports contain outputs, attention,
activation magnitude, memory provenance and group-zeroing effects; raw tensors are
saved for deeper inspection.

Local results: [synthetic](../runs/multimodal_v1/synthetic/report.html),
[real PushT](../runs/multimodal_v1/pusht/report.html), and
[verification receipt](../runs/multimodal_v1/verification.json).
Reports pass structural/media/provenance checks. **Browser visual QA is blocked**:
the browser URL policy rejected local HTML navigation. No alternate route was used.

These are development weights. Image predictions remain worse than copying the
last observation; synthetic audio remains worse than silence. Token roles, physical
understanding, useful planning, calibrated uncertainty, scalable memory and broad
self-improvement remain scientific questions. No large run is queued.

The next discussion should review the actual state and component interfaces before
declaring a scientific experiment. Earlier perception/dynamics recipes remain
focused references. Historical source is preserved in
`archive/pre-modular-2026-09-09`; data and completed results remain intact.
The real-data recipe uses `data/pusht_world_model/cchi_v1` directly because the older
`data/pusht64` shortcut is absent. [Migration record](migration.md).
