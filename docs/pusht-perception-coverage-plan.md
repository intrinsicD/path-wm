# Prospective PushT perception coverage comparison

This experiment tests whether broader, independently varied object poses improve the existing E/D/H observer's held-out localization and orientation. The architecture, RGB-plus-pose loss, optimizer and training budget remain fixed. The negative source-only reference is preserved. No memory or predictor training is part of this experiment.

The local diagnosis is recorded in `runs/pusht_world_model/collaboration/perception_diagnosis/analysis.md`. It found a widening source train–validation pose gap, a nearly white selected decoder, and no demonstrated source alignment or circular-angle arithmetic bug. The separate synthetic pose grid is a stress diagnostic; it is not held-out CCHI evidence.

## Arms and fixed budget

| Item | Preserved source-only reference | New coverage arm |
|---|---|---|
| Original source frames | All 20,493 CCHI training frames | One copy of those same 20,493 frames |
| Supplement | None | 20,493 simulator static observations |
| Sampling | Uniform frame draws with replacement | Uniform draws from the concatenated 40,986 frames |
| Updates / batch | 1,000 / 128 | 1,000 / 128 |
| Initialization / sampler seed | 3107 | 3107 |
| Precision / optimizer | FP32, existing AdamW settings | Identical |
| Loss | Mean RGB MSE + mean six-coordinate pose MSE | Identical |
| Validation | Existing fixed 2,048 CCHI validation indices | Exactly the same indices |
| Selection | Lowest combined validation loss, including initialization | Identical |

The new arm has a 50% probability of choosing each population on every draw, not an enforced 64/64 split in each batch. It retains the existing NumPy training sampler and checkpoint RNG semantics. No extra training-sampler draws are introduced to assign populations. Total examples remain 128,000; source-frame exposure is expected to halve while independent pose coverage increases. This exposure tradeoff is part of the intervention and will be stated in the result.

Compare both selected checkpoints and the matched final update 1,000. Report exact selected updates, held-out pusher XY, block XY, wrapped angular MAE/MSE, sin/cos norms, RGB and pose losses, region reconstruction diagnostics, and the fixed pose-grid stress results separately. The original frame-sampled validation population and the earlier group-balanced diagnostic population must remain separately labeled. The comparison will be descriptive for one seed; no new absolute quality threshold, control success, or convergence claim is inferred. Expansion to U/P needs a separate decision after these observer diagnostics.

## Supplement generation

Use a private `numpy.random.default_rng(73107)`. For each index draw five independent uniform components in this order:

- Pusher x and y: `[16, 496)` world units.
- Block x and y: `[96, 416)` world units.
- Block angle: `[0, 2π)` radians.

Generate exactly 20,493 requests. These are independently drawn **requested** poses. Contact and wall corrections during the native reset can change the resulting pose and induce physical dependence. Preserve every observation and label it with the wrapper's actual post-reset `env.pose`; do not clip labels, reject overlaps, or filter by learned predictions, validation examples, goals, reconstruction quality, or any source pose. Requested bounds do not promise that all post-reset coordinates remain in those bounds.

Each sample uses `PushTEnv.reset(pose5_world=requested, seed=73107 + index)` without source velocities or a source-derived goal. Keep the native fixed green overlay. The wrapper renders RGB96 and applies the already fixed lossless integer-to-uint8 conversion followed by `cv2.INTER_AREA` to RGB64. Model input remains float32 divided by 255. Labels are actual `[pusher_xy, block_xy]/512, sin(theta), cos(theta)` in that order. These static observations carry no transition actions, memory history, or velocity labels.

Source metadata is read only to record the CCHI dataset fingerprint, training episode/group identities and frame count. No training, validation, or test source poses/pixels are consulted to generate or filter supplement samples. No held-out pose is incorporated into the sampler. The source train count sets the declared equal-population size, not any spatial distribution.

## Data interface and immutable identity

New module: `world_model/pusht/perception_data.py`.

- `prepare_supplement(output, count=20493, seed=73107, *, source_manifest=None) -> manifest`. The default source manifest is the repository's prepared `data/pusht_world_model/cchi_v1/manifest.json`; callers may provide a different metadata path explicitly. This function creates the supplement or resumes a compatible incomplete preparation.
- `verify_supplement(output) -> report` with `passed`, `fingerprint`, and `count`. Validate the completed manifest, array schema, file hashes and exact pose-target relationship using bounded batches. Reject corruption.
- `SupplementFrames(output, expected_fingerprint)` exposes read-only `frames`, `targets`, `poses`, and `manifest`. It verifies the expected supplement identity before returning arrays.
- `MixedPerceptionSamples(source_samples, supplement)` requires `source_samples.dataset.split == 'train'`, matching source fingerprint/training identities and exactly equal frame counts. It preserves `.dataset` and `.lengths` from the source, exposes `.frames` of length `2*N`, and provides `frame_batch(indices, device)` in the original requested order, including duplicate indices. Indices `<N` route to the existing source batch path; indices `>=N` route to supplement row `index-N`. No validation adapter or episode/transition fiction is introduced.

Storage is `frames.npy` uint8 `[N,64,64,3]`, `poses.npy` float64 `[N,5]`, `pose_targets.npy` float32 `[N,6]`, plus a manifest. Memory-mapped construction is bounded; the complete RGB store is about 252 MB. The manifest records requested distributions, private RNG and reset seeds, actual-label semantics, canonical preprocessing, native simulator/wrapper/generator code hashes, dependency versions, source training provenance, array hashes and a derived fingerprint. Training saves a distinct population identity in its manifest/checkpoints/results and verifies it on resume; the original source dataset identity is retained separately.

Completed preparation is idempotent only when count, seed, source identity and generator protocol agree and all files verify. It never silently overwrites a completed dataset. Incomplete preparation keeps an atomic progress receipt with hashes for committed chunks. On resume verify that prefix before continuing; uncommitted trailing writes may be regenerated deterministically. A mismatch or corrupted committed prefix fails visibly. A completed manifest is published only after all arrays and labels verify.

## Essential tests before implementation

Use tiny temporary arrays and a fake wrapper for most tests; no GPU workload.

1. Deterministic independent pose requests match the private RNG, obey declared request bounds, preserve global RNG state, and use no source frame/pose access. A fake reset deliberately shifts the actual pose; stored labels must reflect that shift. The generated frame is the wrapper's canonical output. One tiny real-wrapper check confirms RGB96-to-AREA64 parity.
2. Exact mixture routing preserves indices, duplicates, population boundaries, input normalization and target order. Reject held-out source adapters, unequal counts and mismatched provenance.
3. Completed preparation verifies hashes/schema/target semantics and is idempotent. Changed expected identity, modified bytes, incompatible seed/count, or corrupted committed prefix are rejected. Interrupted preparation resumes to the same arrays as an uninterrupted run.

After the plan and informative red tests are committed, implement the bounded data path, run its focused checks, prepare the full supplement, and only then let the coordinator run the new perception arm in a separate directory. Refresh the canonical source-backed dashboard after each completed experiment stage. The original source data, reference checkpoints and all earlier experiment results remain immutable.
