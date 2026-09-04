"""TAU ingestion must preserve official pairing/splits before media reaches training."""
from __future__ import annotations

import csv
import importlib
import json
from pathlib import Path

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]


def _module():
    try:
        return importlib.import_module("training.av_data")
    except ModuleNotFoundError:
        pytest.fail("no implementation: training.av_data.ingest_tau_av(cfg, root)")


def _write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _fixture(tmp_path: Path, *, leak: bool = False) -> dict:
    cfg = yaml.safe_load((ROOT / "configs/dev/common_base.yaml").read_text())
    cfg["data"]["source"]["subset"] = "development"
    cfg["data"]["raw_root"] = "raw"
    cfg["data"]["metadata_root"] = "metadata"
    cfg["data"]["shard_root"] = "shards"
    cfg["data"]["manifest"] = "manifest.jsonl"
    cfg["data"]["video"].update(frames_per_second=4, resolution=8)
    cfg["data"]["audio"].update(sample_rate=16, channels=1)

    rows = []
    for clip_id, group in [
        ("airport-city-1-100", "city-1"),
        ("bus-city-2-200", "city-2"),
        ("park-city-3-300", "city-1" if leak else "city-3"),
        ("tram-city-4-400", "city-4"),
    ]:
        audio = f"audio/{clip_id}.wav"
        video = f"video/{clip_id}.mp4"
        rows.append(
            {
                "filename_audio": audio,
                "filename_video": video,
                "scene_label": clip_id.split("-")[0],
                "identifier": group,
            }
        )
        for relative in (audio, video):
            path = tmp_path / "raw" / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"source")
    fields = ["filename_audio", "filename_video", "scene_label", "identifier"]
    _write_tsv(tmp_path / "metadata" / "meta.csv", fields, rows)
    split_fields = ["filename_audio", "filename_video", "scene_label"]
    _write_tsv(
        tmp_path / "metadata" / "evaluation_setup" / "fold1_train.csv",
        split_fields,
        [{key: row[key] for key in split_fields} for row in rows[:2]],
    )
    _write_tsv(
        tmp_path / "metadata" / "evaluation_setup" / "fold1_evaluate.csv",
        split_fields,
        [{key: row[key] for key in split_fields} for row in rows[2:]],
    )
    return cfg


def test_ingestion_uses_metadata_pairing_and_writes_versioned_normalized_shards(tmp_path):
    cfg = _fixture(tmp_path)
    calls = []

    def fake_decode(video_path, audio_path, *, video_fps, resolution, audio_sample_rate, audio_channels):
        calls.append((video_path, audio_path))
        assert video_path.stem == audio_path.stem
        return {
            "video": torch.zeros(12, 3, resolution, resolution, dtype=torch.uint8),
            "audio": torch.zeros(audio_channels, 48, dtype=torch.float32),
            "video_fps": video_fps,
            "audio_sample_rate": audio_sample_rate,
            "duration_seconds": 3.0,
        }

    records = _module().ingest_tau_av(cfg, tmp_path, decoder=fake_decode)

    assert len(records) == len(calls) == 4
    manifest = [json.loads(line) for line in (tmp_path / "manifest.jsonl").read_text().splitlines()]
    assert {record["split"] for record in manifest} == {"train", "eval"}
    assert all(record["manifest_version"] == 1 for record in manifest)
    assert all((tmp_path / "shards" / record["shard"]).is_file() for record in manifest)
    assert {record["group_id"] for record in manifest if record["split"] == "train"}.isdisjoint(
        {record["group_id"] for record in manifest if record["split"] == "eval"}
    )
    shard = torch.load(tmp_path / "shards" / manifest[0]["shard"], weights_only=True)
    assert shard["video"].dtype == torch.uint8
    assert shard["audio"].dtype == torch.float32
    assert "scene_label" not in shard


def test_ingestion_fails_before_decoding_when_official_splits_leak_a_location(tmp_path):
    cfg = _fixture(tmp_path, leak=True)

    def forbidden_decoder(*args, **kwargs):
        pytest.fail("leakage must be rejected before decoding media")

    with pytest.raises(ValueError, match="recording group"):
        _module().ingest_tau_av(cfg, tmp_path, decoder=forbidden_decoder)
