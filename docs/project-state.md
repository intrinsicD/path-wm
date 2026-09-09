# Current work

**Implemented:** conditioned multiscale input features for image, video, audio and
text, following an actual Claude design review and reconciliation. See the slice in
[the implementation record](multimodal-plan.md).

**Available:** reproducible [architecture](diagrams/architecture.svg) and
[data-flow](diagrams/data_flow.svg) diagrams. Regenerate with
`python experiments/multimodal.py --diagram`; use `--diagram-depth 3` for more layers.
The hierarchy comes from instantiated modules; the flow records real values passed
between example calls. The four modalities also have individual scale-flow diagrams.
All 25 exports reproduce byte for byte on this runtime.

**Implemented:** one small multimodal world model, built in
[the editable recipe](../experiments/multimodal.py). Read
[the concrete architecture](multimodal.md) and [the implementation record](multimodal-plan.md).

The model has three processed feature scales for each input modality, five latent token groups, explicit
observation and thinking operations, bounded episodic memory, probabilistic imagined
futures, action proposals, candidate planning, self-error diagnostics, interventions
and a measured update gate with rollback. It has 325,704 parameters by default.
Every component is directly constructed as an ordinary PyTorch module.

Every scale finishes conditioned residual transformer processing before a coarser
stage or consumer reads it. Masked pooling and optional local cross-scale attention
carry fine features upward. A shared 16-value code comes from pre-observation
working/reasoning state or an explicit user override. Code zero stays neutral after
training. Availability includes code time, while sensor timing is retained separately.

Forty-two CPU tests pass, including ten multiscale and four diagram checks.
Two-update synthetic and real PushT development runs complete. Synthetic
full/resumed model (including controller/teacher/replay), optimizer, all saved RNG
and training rows match exactly. Reports show PCA for all available input scales;
raw features, masks, attention and code provenance are saved for inspection.

Local results: [synthetic](../runs/multiscale_v1/synthetic_full/report.html),
[real PushT](../runs/multiscale_v1/pusht/report.html), and
[verification receipt](../runs/multiscale_v1/verification.json).
Reports pass structural/media/provenance checks. **Browser visual QA is blocked**:
the browser URL policy rejected local HTML navigation. No alternate route was used.

These are development weights. Image predictions remain worse than copying the
last observation; synthetic audio remains worse than silence. Token roles, physical
understanding, useful planning, calibrated uncertainty, scalable memory and broad
self-improvement remain scientific questions. No large run is queued.

Earlier single-scale results remain intact under `runs/multimodal_v1/`. Their
checkpoints require their source snapshots; no silent migration to the new layout.

The next discussion should review the actual state and component interfaces before
declaring a scientific experiment. Earlier perception/dynamics recipes remain
focused references. Historical source is preserved in
`archive/pre-modular-2026-09-09`; data and completed results remain intact.
The real-data recipe uses `data/pusht_world_model/cchi_v1` directly because the older
`data/pusht64` shortcut is absent. [Migration record](migration.md).
