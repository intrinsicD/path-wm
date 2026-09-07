"""Audited CCHI records with primitive absolute actions and causal labels.

Pixels are streamed into one read-only RGB64 NumPy array. Small NPZ records
retain episode boundaries and labels; no random-frame access decompresses an
entire image trajectory. The source is immutable and remains authoritative.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

import cv2
import h5py
import numpy as np


SCHEMA_VERSION = "pusht-cchi-eup-trajectories-v1"
ACTION_SCHEMA_VERSION = "pusht-absolute-xy512-start-minus1-v1"
LABEL_SCHEMA_VERSION = "pusht-pose6-backward-displacement5-train-rms-mask-before2-v2"
PREPROCESSING = {
    "schema": "pusht-rgb96-integer-cast-uint8-area64-v1",
    "source_shape": [96, 96, 3], "output_shape": [64, 64, 3],
    "source_cast": "finite integer-valued RGB in [0,255] to uint8 losslessly",
    "resize": "cv2.resize((64,64),interpolation=cv2.INTER_AREA)",
    "output_dtype": "uint8", "channel_order": "RGB",
    "model_normalization": "float32/255", "opencv_version": cv2.__version__,
}
_SPLITS = ("train", "validation", "test")
_END_REASON = "source_boundary_termination_unknown"


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _fingerprint(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _atomic_json(path, value):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_npz(path, episode):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            np.savez_compressed(stream, **episode)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def canonical_frame(rgb96):
    """The identical source/live RGB96 → uint8 RGB64 preprocessing contract."""
    rgb = np.asarray(rgb96)
    if rgb.shape != (96, 96, 3) or not np.issubdtype(rgb.dtype, np.number):
        raise ValueError("RGB input must be a numeric [96,96,3] array")
    if not np.isfinite(rgb).all():
        raise ValueError("RGB input must be finite")
    if np.any((rgb < 0) | (rgb > 255)) or np.any(rgb != np.floor(rgb)):
        raise ValueError("RGB source cast must be lossless: integer values in [0,255]")
    return cv2.resize(rgb.astype(np.uint8, copy=False), (64, 64), interpolation=cv2.INTER_AREA)


def make_targets(poses_world, motion_scales=None):
    """Return float32 pose6/motion5 labels and their [L,11] boolean mask.

    Motion is a backward displacement per 0.1-second source interval. It is
    never an invented instantaneous velocity and uses no future observations.
    Prepared episodes always pass recorded train-only RMS scales. The optional
    default preserves explicit world512/pi units for standalone helper callers.
    """
    poses = np.asarray(poses_world, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 5 or not np.isfinite(poses).all():
        raise ValueError("poses must be finite [L,5] world-coordinate labels")
    scales = np.asarray([512, 512, 512, 512, np.pi] if motion_scales is None else motion_scales, dtype=np.float64)
    if scales.shape != (5,) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("motion_scales must contain five finite positive physical scales")
    targets = np.zeros((len(poses), 11), dtype=np.float32)
    targets[:, :4] = poses[:, :4] / 512
    targets[:, 4] = np.sin(poses[:, 4])
    targets[:, 5] = np.cos(poses[:, 4])
    targets[1:, 6:10] = np.diff(poses[:, :4], axis=0) / scales[:4]
    angle_change = np.diff(poses[:, 4])
    targets[1:, 10] = np.arctan2(np.sin(angle_change), np.cos(angle_change)) / scales[4]
    mask = np.ones(targets.shape, dtype=np.bool_)
    mask[:2, 6:] = False
    return targets, mask


def previous_action_vectors(actions):
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 2 or not np.isfinite(actions).all() or np.any((actions < 0) | (actions > 1)):
        raise ValueError("executable actions must be finite [T,2] vectors in [0,1]")
    previous = np.full((len(actions) + 1, 2), -1, dtype=np.float32)
    previous[1:] = actions
    return previous


def initial_configuration_groups(initial_poses):
    """All connected components under the repository's 5px/.05rad rule.

    This preserves the circular-angle union rule in scripts/prepare_pusht_pilot.py;
    every near variant, including transitive chains, stays in one split.
    """
    poses = np.asarray(initial_poses, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 5 or not len(poses) or not np.isfinite(poses).all():
        raise ValueError("initial poses must be a nonempty finite [N,5] array")
    unique, inverse = np.unique(poses, axis=0, return_inverse=True)
    pusher = np.linalg.norm(unique[:, None, :2] - unique[None, :, :2], axis=-1)
    block = np.linalg.norm(unique[:, None, 2:4] - unique[None, :, 2:4], axis=-1)
    angle = np.abs((unique[:, None, 4] - unique[None, :, 4] + np.pi) % (2*np.pi) - np.pi)
    near = (pusher <= 5) & (block <= 5) & (angle <= .05)
    parent = np.arange(len(unique))

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for a, b in np.argwhere(np.triu(near, k=1)):
        parent[root(b)] = root(a)
    _, labels = np.unique([root(i) for i in range(len(unique))], return_inverse=True)
    return labels[inverse].astype(np.int64)


def _source_layout(h5):
    required = {"pixels", "state", "action", "ep_len", "ep_offset"}
    if not required.issubset(h5):
        raise ValueError(f"source HDF5 missing fields {sorted(required - set(h5))}")
    lengths, offsets = h5["ep_len"][:], h5["ep_offset"][:]
    if (lengths.ndim != 1 or offsets.shape != lengths.shape or not len(lengths)
            or not np.issubdtype(lengths.dtype, np.integer) or not np.issubdtype(offsets.dtype, np.integer)
            or np.any(lengths < 2)):
        raise ValueError("episode lengths/offsets must be integer vectors with at least two frames per episode")
    if not np.array_equal(offsets, np.r_[0, np.cumsum(lengths[:-1])]):
        raise ValueError("episode offsets must be contiguous, disjoint and start at zero")
    count = int(lengths.sum())
    for name, shape in {"pixels": (count, 96, 96, 3), "state": (count, 5), "action": (count, 2)}.items():
        if h5[name].shape != shape or not np.issubdtype(h5[name].dtype, np.number):
            raise ValueError(f"invalid source {name} shape/dtype; expected numeric {shape}")
    # These labels are retained exactly in float32, matching the CCHI conversion.
    for name in ("state", "action"):
        if h5[name].dtype != np.dtype("float32"):
            raise ValueError(f"source {name} must be float32 to preserve exact CCHI labels")
    return lengths.astype(np.int64), offsets.astype(np.int64)


def _fit_normalization(source, episode_ids, lengths, offsets, source_sha256):
    """Float64 streaming RMS from training-group source rows only."""
    motion_sum = np.zeros(5, dtype=np.float64)
    action_sum = np.zeros(2, dtype=np.float64)
    motion_count = action_count = 0
    with h5py.File(source, "r") as h5:
        for ep in episode_ids:
            row, length = int(offsets[ep]), int(lengths[ep])
            poses = h5["state"][row:row+length].astype(np.float64)
            actions = h5["action"][row:row+length-1].astype(np.float64)
            if not np.isfinite(poses).all():
                raise ValueError("training poses must be finite before fitting normalization")
            if not np.isfinite(actions).all() or np.any((actions < 0) | (actions > 512)):
                raise ValueError("training used actions must be finite and within world bounds [0,512]")
            angle = np.diff(poses[:, 4])
            motion = np.column_stack([np.diff(poses[:, :4], axis=0), np.arctan2(np.sin(angle), np.cos(angle))])[1:]
            motion_sum += np.square(motion).sum(axis=0, dtype=np.float64)
            motion_count += len(motion)
            action_offsets = (actions - poses[:-1, :2]) / 512
            action_sum += np.square(action_offsets).sum(axis=0, dtype=np.float64)
            action_count += len(action_offsets)
    if not motion_count or not action_count:
        raise ValueError("training groups need at least one valid t>=2 motion label and one transition")
    motion_rms = np.sqrt(motion_sum / motion_count)
    floors = np.array([1., 1., 1., 1., .01], dtype=np.float64)
    return {
        "schema": "pusht-train-rms-motion-and-action-offset-v1",
        "fit_split": "train", "source_episode_ids": list(episode_ids), "source_sha256": source_sha256,
        "arithmetic": "float64 streaming root mean square, without centering",
        "motion_scales": np.maximum(motion_rms, floors).tolist(),
        "motion_rms_unfloored": motion_rms.tolist(), "motion_scale_floors": floors.tolist(),
        "motion_count": motion_count, "motion_fit_indices": "each source episode t>=2",
        "motion_units": ["world_units_per_interval"]*4 + ["radians_per_interval"],
        "action_offset_rms": np.sqrt(action_sum / action_count).tolist(),
        "action_count": action_count, "action_offset_definition": "(action_world[t]-pusher_xy[t])/512, t<L-1",
    }


def _identity(source, seed):
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    with h5py.File(source, "r") as h5:
        lengths, offsets = _source_layout(h5)
        initial = h5["state"][offsets]
        groups = initial_configuration_groups(initial)
        source_schema = {key: {"shape": list(h5[key].shape), "dtype": str(h5[key].dtype)}
                         for key in ("pixels", "state", "action", "ep_len", "ep_offset")}
    available = np.unique(groups)
    if len(available) < 3:
        raise ValueError("at least three initial-configuration groups are required for nonempty splits")
    shuffled = np.random.default_rng(int(seed)).permutation(available)
    ntrain = min(max(1, int(.8 * len(available))), len(available) - 2)
    nval = min(max(1, int(.1 * len(available))), len(available) - ntrain - 1)
    partitions = np.split(shuffled, [ntrain, ntrain+nval])
    splits = {split: {"group_ids": sorted(ids.tolist()),
                      "source_episode_ids": np.flatnonzero(np.isin(groups, ids)).tolist()}
              for split, ids in zip(_SPLITS, partitions)}
    source_info = {"path": str(source), "sha256": _file_hash(source), "bytes": source.stat().st_size,
                   "schema": source_schema, "archive_sha256": None, "archive_hash_verified": False}
    conversion = source.parent / "conversion.json"
    if conversion.exists():
        receipt = json.loads(conversion.read_text())
        source_info["conversion_receipt_sha256"] = _file_hash(conversion)
        source_info["archive_sha256"] = receipt.get("archive_sha256")
    archive = source.parent / "pusht.zip"
    if archive.exists():
        digest = _file_hash(archive)
        if source_info["archive_sha256"] not in (None, digest):
            raise ValueError("source archive checksum differs from conversion receipt")
        source_info.update(archive_sha256=digest, archive_hash_verified=True, archive_path=str(archive))
    identity = {
        "schema_version": SCHEMA_VERSION, "action_schema_version": ACTION_SCHEMA_VERSION,
        "label_schema_version": LABEL_SCHEMA_VERSION, "source": source_info,
        "preprocessing": PREPROCESSING, "preprocessing_fingerprint": _fingerprint(PREPROCESSING),
        "split_protocol": {"seed": int(seed), "rng": "numpy-PCG64", "numpy_version": np.__version__,
                           "train_fraction": .8, "validation_fraction": .1,
                           "position_tolerance": 5., "circular_angle_tolerance": .05,
                           "grouping": "all-union-connected-components", "small_population": "reserve_one_group_per_split"},
        "splits": splits, "group_ids": groups.tolist(), "source_lengths": lengths.tolist(),
        "source_offsets": offsets.tolist(), "frame_interval_seconds": .1,
        "final_stored_action": "excluded_no_observed_successor", "end_reason": _END_REASON,
        "normalization": _fit_normalization(source, splits["train"]["source_episode_ids"], lengths, offsets, source_info["sha256"]),
    }
    return identity


def _episode(h5, source_episode, identity):
    offset = identity["source_offsets"][source_episode]
    length = identity["source_lengths"][source_episode]
    poses = h5["state"][offset:offset+length]
    targets, mask = make_targets(poses, motion_scales=identity["normalization"]["motion_scales"])
    actions_world = h5["action"][offset:offset+length-1]
    if not np.isfinite(actions_world).all() or np.any((actions_world < 0) | (actions_world > 512)):
        raise ValueError(f"source episode {source_episode} used actions must be finite and within world bounds [0,512]")
    frames = np.empty((length, 64, 64, 3), dtype=np.uint8)
    for t in range(length):
        frames[t] = canonical_frame(h5["pixels"][offset+t])
    record = {
        "actions_world": actions_world, "actions": actions_world / np.float32(512),
        "poses_world": poses, "pose_targets": targets[:, :6], "motion_targets": targets[:, 6:],
        "motion_mask": mask[:, 6:], "source_episode": np.array(source_episode, dtype=np.int64),
        "source_rows": np.arange(offset, offset+length, dtype=np.int64),
        "group_id": np.array(identity["group_ids"][source_episode], dtype=np.int64),
        "timestamps": np.arange(length, dtype=np.float64) * .1, "end_reason": np.array(_END_REASON),
        "identity_fingerprint": np.array(_fingerprint(identity)),
    }
    return frames, record


def _read_record(path):
    try:
        with np.load(path, allow_pickle=False) as saved:
            return {key: saved[key] for key in saved.files}
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid episode record {path}: {exc}") from exc


def _same_record(actual, expected, path):
    if actual.keys() != expected.keys():
        raise ValueError(f"episode record schema mismatch in {path}")
    for key in expected:
        if actual[key].dtype != expected[key].dtype or not np.array_equal(actual[key], expected[key]):
            raise ValueError(f"episode record source alignment/label mismatch for {key} in {path}")


def _manifest_identity(manifest):
    return {key: manifest[key] for key in (
        "schema_version", "action_schema_version", "label_schema_version", "source", "preprocessing",
        "preprocessing_fingerprint", "split_protocol", "splits", "group_ids", "source_lengths", "source_offsets",
        "frame_interval_seconds", "final_stored_action", "end_reason", "normalization")}


def _load_manifest(output, complete=True):
    manifest = json.loads((output / "manifest.json").read_text())
    try:
        identity = _manifest_identity(manifest)
    except KeyError as exc:
        raise ValueError(f"dataset manifest schema missing {exc}") from exc
    if (manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("action_schema_version") != ACTION_SCHEMA_VERSION
            or manifest.get("label_schema_version") != LABEL_SCHEMA_VERSION or manifest.get("preprocessing") != PREPROCESSING
            or manifest.get("identity_fingerprint") != _fingerprint(identity)):
        raise ValueError("dataset schema/preprocessing/identity fingerprint mismatch")
    if complete and not manifest.get("complete"):
        raise ValueError("dataset preparation is incomplete; resume prepare first")
    if manifest.get("complete") and manifest.get("fingerprint") != _fingerprint({k: v for k, v in manifest.items() if k != "fingerprint"}):
        raise ValueError("completed dataset manifest fingerprint mismatch")
    return manifest


def prepare(source, output, seed=3107):
    """Prepare or verify/resume an immutable source/group/preprocessing identity.

    Each episode's flat pixels are flushed before its atomic label record.
    Interrupted preparation adopts an existing record only after exact source
    comparison. Completed datasets are verified without writing any bytes.
    """
    source, output = Path(source).resolve(), Path(output).resolve()
    identity = _identity(source, seed)
    fingerprint = _fingerprint(identity)
    old = None
    if (output / "manifest.json").exists():
        old = _load_manifest(output, complete=False)
        if old["identity_fingerprint"] != fingerprint:
            raise ValueError("existing dataset source/seed/identity fingerprint differs; use a distinct output")
        if old.get("complete"):
            verify_dataset(output, source=source)
            return old
    elif output.exists() and any(output.iterdir()):
        raise ValueError("output contains unrelated files without a dataset manifest")
    output.mkdir(parents=True, exist_ok=True)
    frame_count = sum(identity["source_lengths"])
    manifest = {**identity, "identity_fingerprint": fingerprint, "complete": False, "episodes": [],
                "frame_store": {"path": "frames.npy", "shape": [frame_count, 64, 64, 3],
                                "dtype": "uint8", "order": "source_episode_then_frame", "sha256": None}}
    if old is None:
        _atomic_json(output / "manifest.json", manifest)
    old_entries = {entry["path"]: entry for entry in old["episodes"]} if old else {}
    final_frames, partial_frames = output / "frames.npy", output / ".frames.incomplete.npy"
    frame_path = final_frames if final_frames.exists() else partial_frames
    if frame_path.exists():
        frames_store = np.load(frame_path, mmap_mode="r" if final_frames.exists() else "r+")
        if frames_store.shape != tuple(manifest["frame_store"]["shape"]) or frames_store.dtype != np.uint8:
            raise ValueError("partial flat frame store shape/dtype mismatch")
    else:
        if old_entries or any(output.rglob("*.npz")):
            raise ValueError("partial frame store missing for existing episode records")
        frames_store = np.lib.format.open_memmap(partial_frames, mode="w+", dtype=np.uint8,
                                               shape=tuple(manifest["frame_store"]["shape"]))
    try:
        with h5py.File(source, "r") as h5:
            for ep, (offset, length) in enumerate(zip(identity["source_offsets"], identity["source_lengths"])):
                split = next(name for name in _SPLITS if ep in identity["splits"][name]["source_episode_ids"])
                relative = f"{split}/{ep:06d}.npz"
                path = output / relative
                frames, record = _episode(h5, ep, identity)
                if path.exists():
                    if relative in old_entries and _file_hash(path) != old_entries[relative]["sha256"]:
                        raise ValueError(f"episode checksum mismatch in {path}")
                    _same_record(_read_record(path), record, path)
                    if not np.array_equal(frames_store[offset:offset+length], frames):
                        raise ValueError(f"partial frame store source mismatch for episode {ep}")
                else:
                    if final_frames.exists():
                        raise ValueError("published flat frame store has a missing episode record")
                    frames_store[offset:offset+length] = frames
                    frames_store.flush()
                    _atomic_npz(path, record)
                manifest["episodes"].append({"source_episode": ep, "group_id": identity["group_ids"][ep],
                    "split": split, "path": relative, "sha256": _file_hash(path), "bytes": path.stat().st_size,
                    "frame_offset": offset, "frames": length, "transitions": length-1})
                _atomic_json(output / "manifest.json", manifest)
    finally:
        del frames_store
    if not final_frames.exists():
        os.replace(partial_frames, final_frames)
    manifest["frame_store"].update(sha256=_file_hash(final_frames), bytes=final_frames.stat().st_size)
    manifest.update(complete=True, episode_count=len(manifest["episodes"]), frame_count=frame_count,
                    transition_count=frame_count-len(manifest["episodes"]),
                    stored_bytes=final_frames.stat().st_size + sum(e["bytes"] for e in manifest["episodes"]))
    manifest["fingerprint"] = _fingerprint(manifest)
    _atomic_json(output / "manifest.json", manifest)
    return manifest


class EpisodeDataset:
    """One small label NPZ plus read-only flat pixel views per episode."""

    def __init__(self, path, split):
        if split not in _SPLITS:
            raise ValueError(f"unknown dataset split {split!r}")
        self.path, self.split = Path(path), split
        self.manifest = _load_manifest(self.path)
        self.fingerprint = self.manifest["fingerprint"]
        self.entries = [entry for entry in self.manifest["episodes"] if entry["split"] == split]
        self.lengths = [entry["frames"] for entry in self.entries]
        store = self.manifest["frame_store"]
        if store["path"] != "frames.npy" or store["order"] != "source_episode_then_frame":
            raise ValueError("invalid flat frame store identity/order")
        self.frames = np.load(self.path / store["path"], mmap_mode="r", allow_pickle=False)
        if self.frames.shape != tuple(store["shape"]) or self.frames.dtype != np.uint8:
            raise ValueError("flat frame store shape/dtype mismatch")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]
        record = _read_record(self.path / entry["path"])
        offset, length = entry["frame_offset"], entry["frames"]
        record["frames"] = self.frames[offset:offset+length]
        return record

    def window_indices(self, horizon, min_history=2):
        if (isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or horizon < 1
                or isinstance(min_history, bool) or not isinstance(min_history, (int, np.integer)) or min_history < 0):
            raise ValueError("horizon must be positive and min_history nonnegative integers")
        return np.asarray([(ep, t) for ep, length in enumerate(self.lengths)
                           for t in range(min_history, length-horizon)], dtype=np.int64).reshape(-1, 2)


def verify_dataset(output, source=None):
    """Check hashes, every source/label/action/frame, and cross-split reuse.

    Duplicate rendered frames are reported rather than silently equating
    initial-configuration grouping with full trajectory independence.
    """
    output = Path(output)
    manifest = _load_manifest(output)
    source = Path(source if source is not None else manifest["source"]["path"]).resolve()
    if _file_hash(source) != manifest["source"]["sha256"]:
        raise ValueError("source HDF5 checksum mismatch")
    if _file_hash(output / "frames.npy") != manifest["frame_store"]["sha256"]:
        raise ValueError("flat frame store checksum mismatch")
    identity = _manifest_identity(manifest)
    # Recompute grouping and RNG split from the source, without trusting labels.
    current_identity = _identity(source, manifest["split_protocol"]["seed"])
    current_identity["source"] = identity["source"]  # Relocation is allowed after content-hash verification.
    if current_identity != identity:
        raise ValueError("source split/group/schema identity mismatch")
    entries = manifest["episodes"]
    if [e["source_episode"] for e in entries] != list(range(len(identity["source_lengths"]))):
        raise ValueError("source episode population/order mismatch")
    frames = np.load(output / "frames.npy", mmap_mode="r", allow_pickle=False)
    if frames.shape != tuple(manifest["frame_store"]["shape"]) or frames.dtype != np.uint8:
        raise ValueError("flat frame store schema mismatch")
    seen_groups, frame_splits, trajectory_splits = {}, {}, {}
    report = {"passed": True, "episodes": 0, "frames": 0, "transitions": 0,
              "splits": {name: 0 for name in _SPLITS}, "fingerprint": manifest["fingerprint"]}
    with h5py.File(source, "r") as h5:
        for ep, entry in enumerate(entries):
            offset, length = identity["source_offsets"][ep], identity["source_lengths"][ep]
            split = next(name for name in _SPLITS if ep in identity["splits"][name]["source_episode_ids"])
            expected_entry = {"source_episode": ep, "group_id": identity["group_ids"][ep], "split": split,
                              "path": f"{split}/{ep:06d}.npz", "frame_offset": offset,
                              "frames": length, "transitions": length-1}
            if any(entry.get(k) != v for k, v in expected_entry.items()):
                raise ValueError(f"manifest episode/split/boundary mismatch at episode {ep}")
            path = output / entry["path"]
            if _file_hash(path) != entry["sha256"] or path.stat().st_size != entry["bytes"]:
                raise ValueError(f"episode checksum/size mismatch in {path}")
            expected_frames, expected_record = _episode(h5, ep, identity)
            _same_record(_read_record(path), expected_record, path)
            if not np.array_equal(frames[offset:offset+length], expected_frames):
                raise ValueError(f"flat frames/source preprocessing mismatch at episode {ep}")
            seen_groups.setdefault(entry["group_id"], set()).add(split)
            trajectory = hashlib.sha256(expected_frames.tobytes())
            for key in ("actions_world", "poses_world"):
                trajectory.update(expected_record[key].tobytes())
            trajectory_splits.setdefault(trajectory.hexdigest(), set()).add(split)
            for frame in expected_frames:
                frame_splits.setdefault(hashlib.sha256(frame.tobytes()).hexdigest(), set()).add(split)
            report["episodes"] += 1
            report["frames"] += length
            report["transitions"] += length-1
            report["splits"][split] += 1
    report.update(cross_split_groups=sum(len(s) > 1 for s in seen_groups.values()),
                  cross_split_duplicate_frame_hashes=sum(len(s) > 1 for s in frame_splits.values()),
                  cross_split_duplicate_trajectory_hashes=sum(len(s) > 1 for s in trajectory_splits.values()),
                  independence_limitation="Disjoint initial-configuration groups do not prove independence of later frames or trajectory segments.")
    for name in ("episode", "frame", "transition"):
        if manifest[f"{name}_count"] != report[f"{name}s"]:
            raise ValueError(f"manifest {name} count mismatch")
    if report["cross_split_groups"]:
        raise ValueError("initial-configuration group leaked across splits")
    return report
