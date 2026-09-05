"""Structural and sampling conformance for the training-only R0/R1 data boundary."""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
import torch
import yaml

import contracts

ROOT = Path(__file__).resolve().parents[2]


def _module():
    try:
        return importlib.import_module("training.av_data")
    except ModuleNotFoundError:
        pytest.fail("no implementation: training.av_data.build_representation_data(cfg, root)")


def _store(tmp_path: Path):
    cfg = yaml.safe_load((ROOT / "configs/dev/common_base.yaml").read_text())
    cfg["data"]["manifest"] = "manifest.jsonl"
    cfg["data"]["shard_root"] = "shards"
    cfg["data"]["video"].update(frames_per_second=4, resolution=8)
    cfg["data"]["audio"].update(sample_rate=16, channels=1)
    cfg["data"]["window"].update(
        duration_seconds=0.5,
        future_offset_seconds=0.5,
        shifted_offset_seconds=1.0,
    )
    shard_root = tmp_path / "shards"
    shard_root.mkdir()
    records = []
    definitions = [
        ("train-a", "train", "place-a", 0),
        ("train-b", "train", "place-b", 40),
        ("train-c", "train", "place-c", 80),
        ("train-d", "train", "place-d", 120),
        ("eval-a", "eval", "place-e", 20),
        ("eval-b", "eval", "place-f", 60),
    ]
    for clip_id, split, group_id, base in definitions:
        frame_index = torch.arange(12, dtype=torch.uint8)
        video = (base + frame_index)[:, None, None, None].expand(-1, 3, 8, 8).clone()
        audio_index = torch.div(torch.arange(48), 4, rounding_mode="floor")
        audio = ((base + audio_index).float() / 255.0)[None]
        torch.save(
            {
                "video": video,
                "audio": audio,
                "video_fps": 4,
                "audio_sample_rate": 16,
                "duration_seconds": 3.0,
            },
            shard_root / f"{clip_id}.pt",
        )
        records.append(
            {
                "manifest_version": 1,
                "clip_id": clip_id,
                "split": split,
                "group_id": group_id,
                "shard": f"{clip_id}.pt",
                "duration_seconds": 3.0,
                "source": {"audio": f"audio/{clip_id}.wav", "video": f"video/{clip_id}.mp4"},
            }
        )
    (tmp_path / "manifest.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    return cfg, _module().build_representation_data(cfg, tmp_path)


def _equal_batch(left: contracts.RepresentationBatch, right: contracts.RepresentationBatch) -> bool:
    for view in ("current", "future", "shifted"):
        left_view, right_view = getattr(left, view), getattr(right, view)
        if left_view.keys() != right_view.keys():
            return False
        for modality in left_view:
            for field in ("values", "timestamps", "valid_mask"):
                if not torch.equal(getattr(left_view[modality], field), getattr(right_view[modality], field)):
                    return False
    return True


def test_manifest_data_is_deterministic_and_stage_semantics_are_explicit(tmp_path):
    _, store = _store(tmp_path)
    assert isinstance(store, contracts.RepresentationData)

    first = store.sample("train", "representation_unimodal", 16, torch.Generator().manual_seed(7))
    replay = store.sample("train", "representation_unimodal", 16, torch.Generator().manual_seed(7))
    assert _equal_batch(first, replay)
    assert not first.shifted
    assert set(first.current) == set(first.future) == {"video", "audio"}
    assert first.current["video"].values.shape == (16, 2, 3, 8, 8)
    assert first.current["video"].values.dtype == torch.uint8
    assert first.current["audio"].values.shape == (16, 1, 8)
    assert first.current["audio"].values.dtype == torch.float32
    assert first.current["video"].valid_mask.dtype == torch.bool
    assert first.current["audio"].valid_mask.dtype == torch.bool
    video_ids = first.current["video"].values[:, 0, 0, 0, 0].long() // 40
    audio_ids = (first.current["audio"].values[:, 0, 0] * 255).round().long() // 40
    assert (video_ids != audio_ids).any(), "R0 must not imply audiovisual correspondence"

    synchronized = store.sample("train", "representation_av", 8, torch.Generator().manual_seed(11))
    assert set(synchronized.shifted) == {"video", "audio"}
    current_video = synchronized.current["video"].values[:, 0, 0, 0, 0].long()
    current_audio = (synchronized.current["audio"].values[:, 0, 0] * 255).round().long()
    shifted_video = synchronized.shifted["video"].values[:, 0, 0, 0, 0].long()
    shifted_audio = (synchronized.shifted["audio"].values[:, 0, 0] * 255).round().long()
    assert torch.equal(current_video // 40, current_audio // 40)
    assert torch.equal(shifted_video // 40, shifted_audio // 40)
    assert torch.equal(current_video // 40, shifted_video // 40)
    assert (shifted_video > current_video).all()


def test_manifest_rejects_recording_group_leakage(tmp_path):
    cfg, _ = _store(tmp_path)
    lines = [json.loads(line) for line in (tmp_path / "manifest.jsonl").read_text().splitlines()]
    lines[-1]["group_id"] = "place-a"
    (tmp_path / "manifest.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in lines),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="recording group"):
        _module().build_representation_data(cfg, tmp_path)


def test_r1_modalities_share_physical_time_without_revealing_the_shift(tmp_path):
    _, store = _store(tmp_path)
    batch = store.sample("train", "representation_av", 4, torch.Generator().manual_seed(17))
    for view_name in ("current", "future", "shifted"):
        view = getattr(batch, view_name)
        video, audio = view["video"], view["audio"]
        # Every fourth audio sample is captured at the same instant as a video frame in this fixture.
        assert torch.equal(video.timestamps, audio.timestamps[:, ::4]), "R1 sensors use different time origins"
        assert torch.equal(video.timestamps[0], torch.tensor([-0.5, -0.25]))
        assert torch.equal(audio.timestamps[0], torch.arange(8) / 16 - 0.5)
        for modality, observation in view.items():
            assert (observation.timestamps < 0).all(), "every sample precedes the shared update time"
            assert torch.equal(observation.timestamps, batch.current[modality].timestamps), "shifted time leaks the negative label"
    r0 = store.sample("train", "representation_unimodal", 4, torch.Generator().manual_seed(17))
    assert torch.equal(r0.current["video"].timestamps[0], torch.tensor([-0.25, 0.0]))
    assert torch.equal(r0.current["audio"].timestamps[0], torch.arange(8) / 16 - 7 / 16)
