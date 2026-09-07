"""Atomic, resumable trajectory recording and exact replay verification.

Source populations stay episode-disjoint. Frames, labels, and action timing are
recorded independently; metadata is never passed as an observation to a model.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from .env import ENVIRONMENT, PaddleEnv


SCHEMA_VERSION = "paddle-trajectories-v1"
SPLIT_SEED_BASES = {"train": 0, "validation": 5000, "test": 5500}
_SPLIT_MAX_COUNTS = {"train": 5000, "validation": 500, "test": 500}
_ACTION_STREAM = 0x50414444


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _fingerprint(value) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


ENVIRONMENT_FINGERPRINT = _fingerprint(ENVIRONMENT)


def _file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _atomic_npz(path: Path, episode: dict) -> None:
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


def previous_action_vectors(actions: np.ndarray) -> np.ndarray:
    """T executable actions become T+1 aligned U inputs, starting with zero."""
    actions = np.asarray(actions)
    if actions.ndim != 1 or not np.issubdtype(actions.dtype, np.integer) or np.any((actions < 0) | (actions > 2)):
        raise ValueError("actions must be a vector of integer executable IDs 0,1,2")
    result = np.zeros((len(actions) + 1, 3), dtype=np.float32)
    result[1:] = np.eye(3, dtype=np.float32)[actions]
    return result


def _generation_config(config: dict) -> dict:
    dataset = config.get("dataset", {})
    counts = {}
    for split, maximum in _SPLIT_MAX_COUNTS.items():
        count = dataset.get(split, maximum)
        if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= maximum:
            raise ValueError(f"dataset.{split} must be an integer in [0,{maximum}] to retain disjoint seed ranges")
        counts[split] = count
    return {
        "counts": counts,
        "split_seed_bases": SPLIT_SEED_BASES,
        "warmup_actions": [1, 1],
        "policy": "independent-uniform-three-actions-after-warmup",
        "action_rng": "numpy-PCG64-SeedSequence([episode_seed,0x50414444])",
    }


def _episode_metadata(seed: int, split: str, config_fingerprint: str) -> dict:
    initial = PaddleEnv(seed=seed).state.tolist()
    return {
        "schema_version": SCHEMA_VERSION,
        "environment_fingerprint": ENVIRONMENT_FINGERPRINT,
        "config_fingerprint": config_fingerprint,
        "split": split,
        "seed": seed,
        "initial_state": initial,
        "initial_rng": {"bit_generator": "PCG64", "seed": seed},
        "action_rng": {"bit_generator": "PCG64", "seed_sequence_entropy": [seed, _ACTION_STREAM]},
        "initial_previous_action": [0, 0, 0],
        "warmup_actions": [1, 1],
        "numpy_version": np.__version__,
    }


def record_episode(seed: int, split: str, config_fingerprint: str) -> dict:
    """Record through miss/time limit, including initial and terminal frames."""
    metadata = _episode_metadata(seed, split, config_fingerprint)
    env = PaddleEnv(state=metadata["initial_state"])
    action_rng = np.random.default_rng(np.random.SeedSequence([seed, _ACTION_STREAM]))
    frames = [env.render()]
    states = [env.state.copy()]
    actions, events = [], []
    terminated, truncated, hits = [False], [False], [0]
    while not (env.terminated or env.truncated):
        action = 1 if env.step_index < 2 else int(action_rng.integers(0, 3))
        frame, terminal, cutoff, info = env.step(action)
        frames.append(frame)
        states.append(env.state.copy())
        actions.append(action)
        terminated.append(terminal)
        truncated.append(cutoff)
        hits.append(env.hit_count)
        events.extend(info["events"])
    return {
        "frames": np.stack(frames),
        "actions": np.asarray(actions, dtype=np.int64),
        "states": np.stack(states),
        "timestamps": np.arange(len(frames), dtype=np.float64) * 0.05,
        "terminated": np.asarray(terminated, dtype=np.bool_),
        "truncated": np.asarray(truncated, dtype=np.bool_),
        "hit_counts": np.asarray(hits, dtype=np.int64),
        "metadata_json": np.array(_canonical(metadata)),
        "events_json": np.array(_canonical(events)),
    }


def _load_episode(path: Path) -> dict:
    try:
        with np.load(path, allow_pickle=False) as source:
            episode = {name: source[name] for name in source.files}
        episode["metadata"] = json.loads(str(episode["metadata_json"]))
        episode["events"] = json.loads(str(episode["events_json"]))
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(f"invalid episode file {path}: {exc}") from exc
    return episode


def _validate_episode(episode: dict, expected_metadata: dict, path: Path) -> None:
    metadata = episode["metadata"]
    # NumPy version is provenance, not a license to replace saved initial states.
    # Existing metadata must retain the physical population and stream identity.
    for key, expected in expected_metadata.items():
        if key in ("numpy_version", "initial_state"):
            continue
        if metadata.get(key) != expected:
            raise ValueError(f"episode metadata/fingerprint mismatch for {key} in {path}")
    actions = episode["actions"]
    steps = len(actions)
    specifications = {
        "frames": ((steps + 1, 64, 64, 3), np.uint8),
        "actions": ((steps,), np.int64),
        "states": ((steps + 1, 5), np.float64),
        "timestamps": ((steps + 1,), np.float64),
        "terminated": ((steps + 1,), np.bool_),
        "truncated": ((steps + 1,), np.bool_),
        "hit_counts": ((steps + 1,), np.int64),
    }
    for name, (shape, dtype) in specifications.items():
        array = episode.get(name)
        if array is None or array.shape != shape or array.dtype != np.dtype(dtype):
            raise ValueError(f"invalid {name} shape/dtype in {path}; expected {shape}/{dtype}")
    if steps < 2 or steps > 200 or np.any((actions < 0) | (actions > 2)) or not np.array_equal(actions[:2], [1, 1]):
        raise ValueError(f"invalid executable actions/warmup in {path}")
    term, trunc = episode["terminated"], episode["truncated"]
    if term[:-1].any() or trunc[:-1].any() or bool(term[-1]) == bool(trunc[-1]) or (trunc[-1] and steps != 200):
        raise ValueError(f"invalid termination/truncation alignment in {path}")
    if not np.array_equal(episode["timestamps"], np.arange(steps + 1, dtype=np.float64) * 0.05):
        raise ValueError(f"timestamp alignment mismatch in {path}")
    if not np.array_equal(episode["states"][0], metadata["initial_state"]):
        raise ValueError(f"initial state metadata mismatch in {path}")
    if episode["hit_counts"][0] != 0 or np.any(np.diff(episode["hit_counts"]) < 0):
        raise ValueError(f"invalid cumulative hit counts in {path}")
    if not np.isfinite(episode["states"]).all():
        raise ValueError(f"non-finite state labels in {path}")


def generate_dataset(config: dict, output: str | Path) -> dict:
    """Collect or resume the requested immutable episode population.

    A manifest is persisted at startup and every 25 episodes; after an
    interruption deterministic filenames allow recovery of the intervening
    atomically completed episodes, after their metadata has been verified.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    generation = _generation_config(config)
    config_fingerprint = _fingerprint(generation)
    manifest_path = output / "manifest.json"
    old_entries = {}
    old = None
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old.get("schema_version") != SCHEMA_VERSION or old.get("environment_fingerprint") != ENVIRONMENT_FINGERPRINT or old.get("config_fingerprint") != config_fingerprint:
            raise ValueError("existing dataset schema/environment/config fingerprint differs; use a distinct output directory")
        old_entries = {entry["path"]: entry for entry in old["episodes"]}
    elif any(output.rglob("*.npz")):
        raise ValueError("existing NPZ files have no manifest; refusing to adopt unrelated data")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "environment": ENVIRONMENT,
        "environment_fingerprint": ENVIRONMENT_FINGERPRINT,
        "generation_config": generation,
        "config_fingerprint": config_fingerprint,
        "episodes": [],
        "complete": False,
    }
    if not manifest_path.exists():
        _atomic_json(manifest_path, manifest)
    for split, count in generation["counts"].items():
        for offset in range(count):
            seed = SPLIT_SEED_BASES[split] + offset
            relative = f"{split}/{seed:06d}.npz"
            path = output / relative
            expected = _episode_metadata(seed, split, config_fingerprint)
            if path.exists():
                episode = _load_episode(path)
                _validate_episode(episode, expected, path)
                checksum = _file_hash(path)
                if relative in old_entries and checksum != old_entries[relative]["sha256"]:
                    raise ValueError(f"episode checksum mismatch in {path}")
            else:
                episode = record_episode(seed, split, config_fingerprint)
                _atomic_npz(path, episode)
                checksum = _file_hash(path)
            manifest["episodes"].append({
                "split": split, "seed": seed, "path": relative,
                "sha256": checksum, "frames": len(episode["frames"]),
                "transitions": len(episode["actions"]), "bytes": path.stat().st_size,
            })
            if len(manifest["episodes"]) == 4 and not (old and old.get("complete")):
                sample = manifest["episodes"]
                print(_canonical({
                    "event": "dataset_storage_estimate",
                    "sample_episodes": len(sample),
                    "sample_frames": sum(entry["frames"] for entry in sample),
                    "sample_bytes": sum(entry["bytes"] for entry in sample),
                    "estimated_population_bytes": sum(entry["bytes"] for entry in sample) / len(sample) * sum(generation["counts"].values()),
                }), flush=True)
            if len(manifest["episodes"]) % 25 == 0 and not (old and old.get("complete")):
                _atomic_json(manifest_path, manifest)
    entries = manifest["episodes"]
    sample = entries[:min(4, len(entries))]
    sample_bytes = sum(entry["bytes"] for entry in sample)
    sample_frames = sum(entry["frames"] for entry in sample)
    manifest.update({
        "complete": True,
        "episode_count": len(entries),
        "frame_count": sum(entry["frames"] for entry in entries),
        "transition_count": sum(entry["transitions"] for entry in entries),
        "compressed_bytes": sum(entry["bytes"] for entry in entries),
        "storage_sample": {
            "episodes": len(sample), "frames": sample_frames,
            "compressed_bytes": sample_bytes,
            "raw_rgb_bytes": sample_frames * 64 * 64 * 3,
            "estimated_population_bytes": sample_bytes / len(sample) * len(entries) if sample else 0,
            "feature_cache_bytes_float32": sum(entry["frames"] for entry in entries) * 20480 * 4,
        },
    })
    if old and old.get("complete"):
        if old != manifest:
            raise ValueError("completed dataset manifest totals/metadata are inconsistent")
    else:
        _atomic_json(manifest_path, manifest)
    return manifest


class EpisodeDataset:
    """Stream one NPZ episode at a time, with decoded metadata/events aliases.

    ``__getitem__`` retains all exact NPZ arrays from the brief and additionally
    exposes ``metadata`` (dict) and ``events`` (list) for diagnostic consumers.
    """
    def __init__(self, path: str | Path, split: str):
        self.path = Path(path)
        self.manifest = json.loads((self.path / "manifest.json").read_text())
        if split not in SPLIT_SEED_BASES:
            raise ValueError(f"unknown dataset split {split!r}")
        if self.manifest.get("schema_version") != SCHEMA_VERSION or self.manifest.get("environment_fingerprint") != ENVIRONMENT_FINGERPRINT:
            raise ValueError("dataset schema/environment fingerprint is incompatible")
        if not self.manifest.get("complete"):
            raise ValueError("dataset collection is incomplete; resume generate first")
        self.split = split
        self.entries = [entry for entry in self.manifest["episodes"] if entry["split"] == split]
        self.fingerprint = _fingerprint(self.manifest)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index: int) -> dict:
        return _load_episode(self.path / self.entries[index]["path"])


def verify_dataset(path: str | Path) -> dict:
    """Replay every saved action and compare frames, labels, flags and events."""
    path = Path(path)
    report = {"passed": True, "episodes": 0, "frames": 0, "transitions": 0, "splits": {}}
    seen = set()
    for split in SPLIT_SEED_BASES:
        dataset = EpisodeDataset(path, split)
        manifest = dataset.manifest
        if manifest["config_fingerprint"] != _fingerprint(manifest["generation_config"]):
            raise ValueError("manifest configuration fingerprint does not match generation configuration")
        report["splits"][split] = len(dataset)
        expected_count = dataset.manifest["generation_config"]["counts"][split]
        if len(dataset) != expected_count:
            raise ValueError(f"manifest split count mismatch for {split}")
        for index, entry in enumerate(dataset.entries):
            seed = SPLIT_SEED_BASES[split] + index
            if entry["seed"] != seed or seed in seen or entry["path"] != f"{split}/{seed:06d}.npz":
                raise ValueError("dataset split membership/seed sequence is invalid")
            seen.add(seed)
            episode_path = path / entry["path"]
            if _file_hash(episode_path) != entry["sha256"]:
                raise ValueError(f"episode checksum mismatch in {episode_path}")
            episode = dataset[index]
            expected = _episode_metadata(seed, split, dataset.manifest["config_fingerprint"])
            _validate_episode(episode, expected, episode_path)
            if entry["frames"] != len(episode["frames"]) or entry["transitions"] != len(episode["actions"]):
                raise ValueError(f"manifest trajectory counts mismatch in {episode_path}")
            env = PaddleEnv(state=episode["metadata"]["initial_state"])
            if not np.array_equal(env.render(), episode["frames"][0]):
                raise ValueError(f"initial frame replay mismatch in {episode_path}")
            events = []
            for i, action in enumerate(episode["actions"]):
                frame, terminated, truncated, info = env.step(int(action))
                if not np.array_equal(frame, episode["frames"][i + 1]) or not np.array_equal(env.state, episode["states"][i + 1]):
                    raise ValueError(f"frame/state replay mismatch at transition {i} in {episode_path}")
                if terminated != episode["terminated"][i + 1] or truncated != episode["truncated"][i + 1] or env.hit_count != episode["hit_counts"][i + 1]:
                    raise ValueError(f"boundary flags/hits replay mismatch at transition {i} in {episode_path}")
                events.extend(info["events"])
            if events != episode["events"]:
                raise ValueError(f"event replay mismatch in {episode_path}")
            report["episodes"] += 1
            report["frames"] += len(episode["frames"])
            report["transitions"] += len(episode["actions"])
    report["fingerprint"] = dataset.fingerprint
    for ledger_name, report_name in [("episode_count", "episodes"), ("frame_count", "frames"), ("transition_count", "transitions")]:
        if dataset.manifest[ledger_name] != report[report_name]:
            raise ValueError(f"manifest total {ledger_name} does not match recorded trajectories")
    return report


def history_pairs(count: int, seed: int) -> list[dict]:
    """Dedicated near-interception histories; pair seeds should be split-disjoint.

    Each mapping has center, pair_seed, and two members (direction -1 then +1).
    A member contains initial_state, states[3,5], frames[3,64,64,3], actions[2],
    control_state[5], and correct_action. Frames are actual simulator outputs;
    control starts after all three observations, at ball y=46.
    """
    pairs = []
    for offset in range(count):
        pair_seed = int(seed) + offset
        center = float(np.random.default_rng(pair_seed).uniform(26, 38))
        members = []
        for direction in (-1, 1):
            initial = np.array([center - 12 * direction, 40, 6 * direction, 3, center], dtype=np.float64)
            env = PaddleEnv(state=initial)
            frames, states = [env.render()], [env.state.copy()]
            for _ in range(2):
                frames.append(env.step(1)[0])
                states.append(env.state.copy())
            members.append({
                "direction": direction, "initial_state": initial,
                "states": np.stack(states), "frames": np.stack(frames),
                "actions": np.array([1, 1], dtype=np.int64),
                "control_state": env.state.copy(),
                "correct_action": 0 if direction < 0 else 2,
            })
        if not np.array_equal(members[0]["frames"][-1], members[1]["frames"][-1]):
            raise RuntimeError(f"paired final frames differ for seed {pair_seed}")
        pairs.append({"center": center, "pair_seed": pair_seed, "members": members})
    return pairs


def test_history_cases(output: str | Path) -> dict:
    """Certify all 50 validation/100 test pairs by 27-sequence enumeration."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cases = []
    for split, count, seed in [("validation", 50, 7000), ("test", 100, 8000)]:
        for pair in history_pairs(count, seed):
            for member in pair["members"]:
                first_actions, successful_sequences = set(), []
                for sequence in itertools.product(range(3), repeat=3):
                    env = PaddleEnv(state=member["control_state"])
                    for action in sequence:
                        env.step(action)
                    if env.hit_count:
                        first_actions.add(sequence[0])
                        successful_sequences.append(list(sequence))
                toward = member["correct_action"]
                expected_sequences = {(toward, toward, toward), (toward, toward, 1)}
                passed = first_actions == {toward} and {tuple(sequence) for sequence in successful_sequences} == expected_sequences
                cases.append({
                    "split": split, "pair_seed": pair["pair_seed"], "center": pair["center"],
                    "direction": member["direction"], "correct_action": member["correct_action"],
                    "successful_first_actions": sorted(first_actions),
                    "successful_three_action_sequences": successful_sequences,
                    "expected_successful_three_action_sequences": [list(sequence) for sequence in sorted(expected_sequences)],
                    "passed": passed,
                })
    report = {
        "schema_version": "paddle-history-verification-v2",
        "environment_fingerprint": ENVIRONMENT_FINGERPRINT,
        "passed": all(case["passed"] for case in cases),
        "validation_pairs": 50, "test_pairs": 100,
        "enumerated_sequences_per_member": 27,
        "cases": cases,
    }
    _atomic_json(output / "history_checks.json", report)
    return report


test_history_cases.__test__ = False
