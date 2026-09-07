"""Scientific input invariants for the separate CCHI E/U/P experiment."""

import hashlib
import json
from pathlib import Path

import cv2
import h5py
import numpy as np
import pytest

from world_model.pusht import data


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "source.h5"
    lengths = np.array([6, 7, 8, 6, 9, 7, 6, 8, 7, 6], dtype=np.int64)
    offsets = np.r_[0, np.cumsum(lengths[:-1])]
    n = int(lengths.sum())
    rows, cols = np.indices((96, 96))
    pixels = np.empty((n, 96, 96, 3), np.float32)
    actions = np.empty((n, 2), np.float32)
    poses = np.empty((n, 5), np.float32)
    for ep, (offset, length) in enumerate(zip(offsets, lengths)):
        for t in range(length):
            row = int(offset + t)
            pixels[row] = np.stack(((rows + row) % 256, (cols * 3 + row) % 256,
                                    (rows + cols + row * 2) % 256), axis=-1)
            actions[row] = [row + 10, 500 - row]
            poses[row] = [20 + ep * 30 + t, 30 + t * 2,
                          100 + ep * 20 + t * 3, 110 + t * 4, (6.25 + t * .04) % (2*np.pi)]
        # The final stored command has no observed successor and must never be used.
        actions[int(offset + length - 1)] = np.nan
    with h5py.File(path, "w") as h5:
        for name, value in {"pixels": pixels, "action": actions, "state": poses,
                            "ep_len": lengths, "ep_offset": offsets}.items():
            h5[name] = value
    archive = tmp_path / "pusht.zip"
    archive.write_bytes(b"synthetic acquisition archive identity")
    (tmp_path / "conversion.json").write_text(json.dumps({"archive_sha256": _sha(archive)}))
    return path


def test_alignment_labels_resize_and_episode_boundaries(source, tmp_path):
    output = tmp_path / "prepared"
    manifest = data.prepare(source, output, seed=3107)
    assert manifest["complete"]
    assert manifest["source"]["sha256"] == _sha(source)
    assert manifest["source"]["archive_sha256"] == _sha(source.parent / "pusht.zip")
    assert manifest["preprocessing"]["opencv_version"] == cv2.__version__
    assert manifest["preprocessing"]["output_shape"] == [64, 64, 3]
    assert manifest["frame_store"]["sha256"] == _sha(output / "frames.npy")
    assert manifest["frame_store"]["order"] == "source_episode_then_frame"
    seen, group_splits = set(), {}
    with h5py.File(source, "r") as h5:
        for split, count in [("train", 8), ("validation", 1), ("test", 1)]:
            dataset = data.EpisodeDataset(output, split)
            assert len(dataset) == count
            assert len(dataset.lengths) == count
            assert isinstance(dataset.frames, np.memmap)
            assert not dataset.frames.flags.writeable
            for index, episode in enumerate(dataset):
                ep = int(episode["source_episode"])
                offset, length = int(h5["ep_offset"][ep]), int(h5["ep_len"][ep])
                assert ep not in seen
                seen.add(ep)
                group = int(episode["group_id"])
                assert group_splits.setdefault(group, split) == split
                np.testing.assert_array_equal(episode["source_rows"], np.arange(offset, offset + length))
                np.testing.assert_array_equal(episode["poses_world"], h5["state"][offset:offset+length])
                np.testing.assert_array_equal(episode["actions_world"], h5["action"][offset:offset+length-1])
                np.testing.assert_array_equal(episode["actions"], episode["actions_world"] / 512)
                assert episode["frames"].shape == (length, 64, 64, 3)
                assert episode["frames"].dtype == np.uint8
                assert not episode["frames"].flags.writeable
                assert dataset.entries[index]["frame_offset"] == offset
                with np.load(output / dataset.entries[index]["path"], allow_pickle=False) as record:
                    assert "frames" not in record.files
                for t in range(length):
                    expected = cv2.resize(h5["pixels"][offset+t].astype(np.uint8), (64, 64), interpolation=cv2.INTER_AREA)
                    np.testing.assert_array_equal(episode["frames"][t], expected)
                    np.testing.assert_array_equal(episode["frames"][t], data.canonical_frame(h5["pixels"][offset+t]))
                np.testing.assert_array_equal(episode["timestamps"], np.arange(length) * .1)
                assert str(episode["end_reason"]) == "source_boundary_termination_unknown"
                assert not episode["motion_mask"][:2].any()
                assert episode["motion_mask"][2:].all()
                assert np.isfinite(episode["actions"]).all()
            for horizon in (1, 5):
                windows = dataset.window_indices(horizon, min_history=2)
                expected = [(i, t) for i, length in enumerate(dataset.lengths)
                            for t in range(2, length - horizon)]
                assert list(map(tuple, windows)) == expected
                for i, t in windows:
                    episode = dataset[int(i)]
                    assert len(episode["actions"][t:t+horizon]) == horizon
                    assert episode["source_rows"][t+horizon] == episode["source_rows"][t] + horizon
    assert len(seen) == 10
    report = data.verify_dataset(output, source=source)
    assert report["passed"] and report["episodes"] == 10
    assert report["frames"] == 70 and report["transitions"] == 60
    assert report["cross_split_groups"] == 0


def test_targets_are_causal_wrapped_backward_displacements():
    poses = np.array([[0, 10, 20, 30, 2*np.pi-.02],
                      [1, 12, 23, 34, .02],
                      [4, 18, 32, 46, .08],
                      [8, 26, 44, 62, .15]], dtype=np.float64)
    targets, mask = data.make_targets(poses)
    np.testing.assert_allclose(targets[:, :4], poses[:, :4] / 512)
    np.testing.assert_allclose(targets[:, 4:6], np.stack([np.sin(poses[:, 4]), np.cos(poses[:, 4])], axis=-1))
    np.testing.assert_allclose(targets[1:, 6:10], np.diff(poses[:, :4], axis=0) / 512)
    np.testing.assert_allclose(targets[1:, 10], np.array([.04, .06, .07]) / np.pi)
    assert mask[:, :6].all() and not mask[:2, 6:].any() and mask[2:, 6:].all()
    changed = poses.copy()
    changed[3] += 17
    changed_targets, _ = data.make_targets(changed)
    np.testing.assert_array_equal(targets[:3], changed_targets[:3])
    np.testing.assert_array_equal(targets[:3], data.make_targets(poses[:3])[0])


def test_previous_action_marker_does_not_shift_or_alias_origin():
    actions = np.array([[0, 0], [.5, 1]], np.float32)
    previous = data.previous_action_vectors(actions)
    np.testing.assert_array_equal(previous, [[-1, -1], [0, 0], [.5, 1]])
    for invalid in (np.array([0, 0]), [[-.01, .4]], [[1.01, .2]], [[np.nan, 0]]):
        with pytest.raises(ValueError):
            data.previous_action_vectors(invalid)


def test_prepared_motion_and_action_scales_use_only_training_groups(source, tmp_path):
    output = tmp_path / "prepared"
    manifest = data.prepare(source, output)
    norm = manifest["normalization"]
    train_ids = manifest["splits"]["train"]["source_episode_ids"]
    assert norm["fit_split"] == "train" and norm["source_episode_ids"] == train_ids
    motion, offsets = [], []
    with h5py.File(source, "r") as h5:
        for ep in train_ids:
            row, length = int(h5["ep_offset"][ep]), int(h5["ep_len"][ep])
            poses = h5["state"][row:row+length].astype(np.float64)
            angle = np.diff(poses[:, 4])
            physical = np.column_stack([np.diff(poses[:, :4], axis=0), np.arctan2(np.sin(angle), np.cos(angle))])
            motion.extend(physical[1:])  # frame indices >=2, not the first displacement.
            offsets.extend((h5["action"][row:row+length-1].astype(np.float64) - poses[:-1, :2]) / 512)
    expected_scales = np.maximum(np.sqrt(np.mean(np.square(motion), axis=0)), [1, 1, 1, 1, .01])
    np.testing.assert_allclose(norm["motion_scales"], expected_scales, rtol=1e-14)
    np.testing.assert_allclose(norm["action_offset_rms"], np.sqrt(np.mean(np.square(offsets), axis=0)), rtol=1e-14)
    assert norm["motion_count"] == len(motion) and norm["action_count"] == len(offsets)
    for split in ("train", "validation", "test"):
        for episode in data.EpisodeDataset(output, split):
            poses = episode["poses_world"].astype(np.float64)
            angle = np.diff(poses[:, 4])
            physical = np.column_stack([np.diff(poses[:, :4], axis=0), np.arctan2(np.sin(angle), np.cos(angle))])
            np.testing.assert_allclose(episode["motion_targets"][1:] * expected_scales, physical, atol=1e-6)
            targets, mask = data.make_targets(poses, motion_scales=expected_scales)
            np.testing.assert_array_equal(targets[:, 6:], episode["motion_targets"])
            np.testing.assert_array_equal(mask[:, 6:], episode["motion_mask"])
    # Perturb only held-out motion and commands while retaining initial poses/groups.
    heldout_ids = manifest["splits"]["validation"]["source_episode_ids"] + manifest["splits"]["test"]["source_episode_ids"]
    with h5py.File(source, "r+") as h5:
        for ep in heldout_ids:
            row, length = int(h5["ep_offset"][ep]), int(h5["ep_len"][ep])
            changed = h5["state"][row+1:row+length]
            changed[:, :4] += np.arange(1, length)[:, None] * 100
            changed[:, 4] += np.arange(1, length) * .4
            h5["state"][row+1:row+length] = changed
            h5["action"][row:row+length-1] = 0
    second = data.prepare(source, tmp_path / "heldout_changed")
    assert second["splits"] == manifest["splits"]
    assert second["normalization"]["motion_scales"] == norm["motion_scales"]
    assert second["normalization"]["action_offset_rms"] == norm["action_offset_rms"]


def test_initial_grouping_is_transitive_circular_and_requires_both_positions():
    starts = np.array([[0, 0, 100, 100, 2*np.pi-.01],
                       [4, 0, 100, 100, .01],
                       [8, 0, 100, 100, .03],
                       [0, 0, 106, 100, .01],
                       [0, 0, 100, 100, .11]], dtype=np.float64)
    groups = data.initial_configuration_groups(starts)
    assert groups[0] == groups[1] == groups[2]  # 0 and 2 only connect through 1.
    assert len(set(groups)) == 3
    assert groups[3] != groups[0] and groups[4] != groups[0]


def test_prepare_keeps_near_group_variants_together(source, tmp_path):
    with h5py.File(source, "r+") as h5:
        offsets = h5["ep_offset"][:]
        h5["state"][offsets[1]] = h5["state"][offsets[0]] + [4, 0, 0, 0, 0]
        h5["state"][offsets[2]] = h5["state"][offsets[0]] + [8, 0, 0, 0, 0]
    manifest = data.prepare(source, tmp_path / "groups")
    entries = {entry["source_episode"]: entry for entry in manifest["episodes"]}
    assert len({entries[i]["group_id"] for i in (0, 1, 2)}) == 1
    assert len({entries[i]["split"] for i in (0, 1, 2)}) == 1
    assert sum(len(s["group_ids"]) for s in manifest["splits"].values()) == 8


def test_completed_dataset_resume_is_byte_immutable_and_rng_private(source, tmp_path):
    output = tmp_path / "prepared"
    global_before = np.random.get_state()
    first = data.prepare(source, output)
    global_after = np.random.get_state()
    assert global_before[0] == global_after[0]
    np.testing.assert_array_equal(global_before[1], global_after[1])
    assert global_before[2:] == global_after[2:]
    before = {p.relative_to(output): (p.read_bytes(), p.stat().st_mtime_ns) for p in output.rglob("*") if p.is_file()}
    assert data.prepare(source, output) == first
    assert before == {p.relative_to(output): (p.read_bytes(), p.stat().st_mtime_ns) for p in output.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="fingerprint|identity|seed"):
        data.prepare(source, output, seed=3108)
    with h5py.File(source, "r+") as h5:
        h5["action"][0] = [99, 98]
    with pytest.raises(ValueError, match="fingerprint|identity|source"):
        data.prepare(source, output)
    assert before == {p.relative_to(output): (p.read_bytes(), p.stat().st_mtime_ns) for p in output.rglob("*") if p.is_file()}


@pytest.mark.parametrize("mutation,match", [
    ("fractional_pixel", "integer|lossless"), ("nonfinite_pixel", "finite"),
    ("out_of_bounds_action", "action|domain|bound"), ("nonfinite_action", "finite|action"),
    ("nonfinite_pose", "finite|pose"), ("overlapping_episodes", "offset|contigu|episode"),
])
def test_invalid_source_is_rejected(source, tmp_path, mutation, match):
    with h5py.File(source, "r+") as h5:
        if mutation == "fractional_pixel": h5["pixels"][0, 0, 0, 0] = .5
        elif mutation == "nonfinite_pixel": h5["pixels"][0, 0, 0, 0] = np.nan
        elif mutation == "out_of_bounds_action": h5["action"][0, 0] = 513
        elif mutation == "nonfinite_action": h5["action"][0, 0] = np.nan
        elif mutation == "nonfinite_pose": h5["state"][0, 0] = np.nan
        elif mutation == "overlapping_episodes": h5["ep_offset"][1] -= 1
    with pytest.raises(ValueError, match=match):
        data.prepare(source, tmp_path / "invalid")


def test_tampered_episode_checksum_is_rejected(source, tmp_path):
    output = tmp_path / "prepared"
    manifest = data.prepare(source, output)
    episode_path = output / manifest["episodes"][0]["path"]
    with np.load(episode_path, allow_pickle=False) as saved:
        values = dict(saved)
    values["actions"][0, 0] += .01
    np.savez_compressed(episode_path, **values)
    with pytest.raises(ValueError, match="checksum"):
        data.verify_dataset(output)
    with pytest.raises(ValueError, match="checksum"):
        data.prepare(source, output)


def test_tampered_flat_frames_are_rejected(source, tmp_path):
    output = tmp_path / "prepared"
    data.prepare(source, output)
    frames = np.load(output / "frames.npy", mmap_mode="r+")
    frames[0, 0, 0, 0] ^= np.uint8(1)
    frames.flush()
    del frames
    with pytest.raises(ValueError, match="checksum"):
        data.verify_dataset(output)


def test_interrupted_preparation_resumes_only_verified_episodes(source, tmp_path, monkeypatch):
    output = tmp_path / "prepared"
    atomic_npz = data._atomic_npz
    calls = 0
    def fail_on_second(path, episode):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated interrupted episode write")
        return atomic_npz(path, episode)
    monkeypatch.setattr(data, "_atomic_npz", fail_on_second)
    with pytest.raises(OSError, match="simulated interrupted"):
        data.prepare(source, output)
    existing = next(output.rglob("*.npz"))
    before = existing.read_bytes(), existing.stat().st_mtime_ns
    monkeypatch.setattr(data, "_atomic_npz", atomic_npz)
    data.prepare(source, output)
    assert (existing.read_bytes(), existing.stat().st_mtime_ns) == before
    assert data.verify_dataset(output)["passed"]


def test_cross_split_duplicate_frames_are_visible_without_fake_independence(source, tmp_path):
    with h5py.File(source, "r+") as h5:
        offsets = h5["ep_offset"][:]
        for offset in offsets:
            h5["pixels"][offset] = h5["pixels"][0]
    output = tmp_path / "prepared"
    data.prepare(source, output)
    report = data.verify_dataset(output)
    assert report["cross_split_groups"] == 0
    assert report["cross_split_duplicate_frame_hashes"] >= 1
    assert "initial" in report["independence_limitation"]
