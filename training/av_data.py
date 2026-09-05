"""Real synchronized audio/video ingestion and R0/R1 window sampling.

What: TAU source media and official folds become one normalized tensor shard per clip plus a versioned
JSONL manifest; training reads only those shards through ``RepresentationData``.
How: ingestion joins metadata before decoding, rejects recording-location leakage, and stores uint8
video plus float32 mono audio. R0 draws each modality independently; R1 draws aligned current/future
windows and same-recording wrong-time views from a shared frame-index clock.
Why: the first common-base result must use real A/V while corpus conventions and labels remain outside
the model/ABI. Per-clip shards bound random reads and make the expensive decode resumable (DDR §22).
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import pickle
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
import torch.nn.functional as F
import yaml

from contracts import RepresentationBatch, TemporalObservation

Decoder = Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class AVClipRecord:
    manifest_version: int
    clip_id: str
    split: str
    group_id: str
    shard: str
    duration_seconds: float
    source: Mapping[str, str]
    shard_sha256: str | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AVClipRecord":
        try:
            return cls(
                manifest_version=int(value["manifest_version"]),
                clip_id=str(value["clip_id"]),
                split=str(value["split"]),
                group_id=str(value["group_id"]),
                shard=str(value["shard"]),
                duration_seconds=float(value["duration_seconds"]),
                source=dict(value["source"]),
                shard_sha256=value.get("shard_sha256"),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid audiovisual manifest record: {value}") from error


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _integer_count(seconds: float, rate: int, name: str) -> int:
    value = seconds * rate
    rounded = round(value)
    if seconds <= 0 or rate <= 0 or not math.isclose(value, rounded, abs_tol=1e-7):
        raise ValueError(f"{name}={seconds} must produce an integer count at rate {rate}")
    return int(rounded)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing TAU metadata file: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle, delimiter="\t")]


def _split_rows(data_cfg: Mapping[str, Any], metadata_root: Path) -> tuple[dict[str, str], list[dict[str, str]]]:
    split_cfg = data_cfg["split"]
    assignments: dict[str, str] = {}
    selected: list[dict[str, str]] = []
    for split, field in (("train", "train_file"), ("eval", "eval_file")):
        for row in _read_tsv(metadata_root / str(split_cfg[field])):
            key = row.get("filename_video", "")
            if not key:
                raise ValueError(f"{field} has a row without filename_video")
            previous = assignments.setdefault(key, split)
            if previous != split:
                raise ValueError(f"clip {key!r} appears in both official splits")
            selected.append({**row, "split": split})
    return assignments, selected


def _tau_sources(data_cfg: Mapping[str, Any], root: Path) -> list[dict[str, Any]]:
    raw_root = _resolve(root, data_cfg["raw_root"])
    metadata_root = _resolve(root, data_cfg["metadata_root"])
    assignments, split_rows = _split_rows(data_cfg, metadata_root)
    meta_rows = _read_tsv(metadata_root / "meta.csv")
    by_video = {row["filename_video"]: row for row in meta_rows}
    by_stem = {Path(row["filename_video"]).stem: row for row in meta_rows}
    if len(by_video) != len(meta_rows) or len(by_stem) != len(meta_rows):
        raise ValueError("TAU metadata contains duplicate video or clip identifiers")

    subset = str(data_cfg["source"]["subset"])
    sources: list[dict[str, Any]] = []
    if subset == "examples":
        example_root = raw_root if raw_root.name == "examples" else raw_root / "examples"
        for muxed in sorted(example_root.rglob("*.mp4")):
            row = by_stem.get(muxed.stem)
            if row is None:
                raise ValueError(f"example clip {muxed.name!r} is absent from TAU meta.csv")
            source_key = row["filename_video"]
            if source_key not in assignments:
                raise ValueError(f"example clip {muxed.name!r} is absent from the official fold")
            sources.append({"row": row, "split": assignments[source_key], "video": muxed, "audio": muxed})
    elif subset == "development":
        for split_row in split_rows:
            source_key = split_row["filename_video"]
            row = by_video.get(source_key)
            if row is None:
                raise ValueError(f"official fold entry {source_key!r} is absent from meta.csv")
            sources.append(
                {
                    "row": row,
                    "split": split_row["split"],
                    "video": raw_root / row["filename_video"],
                    "audio": raw_root / row["filename_audio"],
                }
            )
    else:
        raise ValueError(f"unsupported TAU source subset {subset!r}")
    if not sources:
        raise ValueError(f"no TAU {subset} media found under {raw_root}")

    clip_ids: set[str] = set()
    groups = {"train": set(), "eval": set()}
    for source in sources:
        row = source["row"]
        clip_id = Path(row["filename_video"]).stem
        group_id = row.get(str(data_cfg["split"]["group_column"]), "")
        if not group_id:
            raise ValueError(f"TAU clip {clip_id!r} has no recording group")
        if clip_id in clip_ids:
            raise ValueError(f"duplicate TAU clip id {clip_id!r}")
        clip_ids.add(clip_id)
        source["clip_id"] = clip_id
        source["group_id"] = group_id
        groups[source["split"]].add(group_id)
    overlap = groups["train"] & groups["eval"]
    if overlap:
        raise ValueError(f"recording group leakage across train/eval: {sorted(overlap)[:5]}")
    return sources


def _validate_shard(payload: Mapping[str, Any], data_cfg: Mapping[str, Any], source: str) -> None:
    video, audio = payload.get("video"), payload.get("audio")
    resolution = int(data_cfg["video"]["resolution"])
    channels = int(data_cfg["audio"]["channels"])
    if not isinstance(video, torch.Tensor) or video.dtype != torch.uint8 or video.ndim != 4:
        raise ValueError(f"{source}: video must be uint8 (T,3,H,W)")
    if tuple(video.shape[1:]) != (3, resolution, resolution):
        raise ValueError(f"{source}: video shape {tuple(video.shape)} has wrong channels/resolution")
    if not isinstance(audio, torch.Tensor) or audio.dtype != torch.float32 or audio.ndim != 2:
        raise ValueError(f"{source}: audio must be float32 (C,S)")
    if audio.shape[0] != channels or not torch.isfinite(audio).all() or audio.abs().max() > 1.00001:
        raise ValueError(f"{source}: audio has wrong channels, non-finite values, or range")
    if int(payload.get("video_fps", -1)) != int(data_cfg["video"]["frames_per_second"]):
        raise ValueError(f"{source}: video frame rate does not match the spec")
    if int(payload.get("audio_sample_rate", -1)) != int(data_cfg["audio"]["sample_rate"]):
        raise ValueError(f"{source}: audio sample rate does not match the spec")
    duration = float(payload.get("duration_seconds", 0.0))
    if duration <= 0 or not math.isfinite(duration):
        raise ValueError(f"{source}: duration must be finite and positive")


def ingest_tau_av(
    cfg: Mapping[str, Any],
    root: Path,
    *,
    decoder: Decoder | None = None,
    workers: int = 1,
    cache_only: bool = False,
    max_clips: int | None = None,
) -> tuple[AVClipRecord, ...]:
    """Serialize shard writers; only complete ingestion may publish a manifest."""
    shard_root = _resolve(root, cfg["data"]["shard_root"])
    shard_root.mkdir(parents=True, exist_ok=True)
    with (shard_root / ".ingestion.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _ingest_tau_av_locked(cfg, root, decoder=decoder, workers=workers,
                                     cache_only=cache_only, max_clips=max_clips)


def _ingest_tau_av_locked(
    cfg: Mapping[str, Any], root: Path, *, decoder: Decoder | None,
    workers: int, cache_only: bool, max_clips: int | None,
) -> tuple[AVClipRecord, ...]:
    data_cfg = cfg["data"]
    if data_cfg.get("kind") != "synchronized_av_manifest" or data_cfg.get("dataset") != "tau_urban_audio_visual_scenes_2021":
        raise ValueError("ingest_tau_av requires the TAU synchronized manifest config")
    if workers < 1:
        raise ValueError("ingestion workers must be positive")
    if max_clips is not None and (not cache_only or max_clips < 1):
        raise ValueError("max_clips must be positive and is only allowed for cache-only ingestion")
    decoder = decode_av_media if decoder is None else decoder
    sources = _tau_sources(data_cfg, root)  # Validates all split/group invariants before expensive decode.
    shard_root = _resolve(root, data_cfg["shard_root"])
    if cache_only:
        # Opportunistic prefill cannot publish data. Complete ingestion later rehashes every
        # source, including any file whose extraction changed after this cache was written.
        sources = [source for source in sources
                   if all(source[modality].is_file() for modality in ("video", "audio"))
                   and not (shard_root / f"{source['clip_id']}.pt").exists()]
        sources = sources[:max_clips]
    else:
        for source in sources:
            for modality in ("video", "audio"):
                if not source[modality].is_file():
                    raise FileNotFoundError(f"missing TAU {modality} source: {source[modality]}")
    manifest_path = _resolve(root, data_cfg["manifest"])
    shard_root.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    # One worker owns each unique clip. Cache identity includes source bytes and normalization,
    # so separate development audio cannot inherit a muxed-example shard (DDR §24).
    def ingest_one(source: dict[str, Any]) -> AVClipRecord:
        shard_name = f"{source['clip_id']}.pt"
        shard_path = shard_root / shard_name
        checksums = {path: _sha256(path) for path in {source["video"], source["audio"]}}
        identity = {
            "decoder_version": 1,
            "video_sha256": checksums[source["video"]],
            "audio_sha256": checksums[source["audio"]],
            "video_fps": int(data_cfg["video"]["frames_per_second"]),
            "resolution": int(data_cfg["video"]["resolution"]),
            "audio_sample_rate": int(data_cfg["audio"]["sample_rate"]),
            "audio_channels": int(data_cfg["audio"]["channels"]),
        }
        payload = None
        if shard_path.exists():
            try:
                cached = torch.load(shard_path, map_location="cpu", weights_only=True)
                if isinstance(cached, Mapping) and cached.get("ingestion_identity") == identity:
                    _validate_shard(cached, data_cfg, source["clip_id"])
                    payload = cached
            except (EOFError, OSError, RuntimeError, ValueError, IndexError, pickle.UnpicklingError):
                pass  # An interrupted/legacy cache is rebuilt from the verified source pair.
        if payload is None:
            payload = decoder(
                source["video"], source["audio"],
                video_fps=identity["video_fps"], resolution=identity["resolution"],
                audio_sample_rate=identity["audio_sample_rate"],
                audio_channels=identity["audio_channels"],
            )
            _validate_shard(payload, data_cfg, source["clip_id"])
            payload = {**payload, "ingestion_identity": identity}
            temporary_shard = shard_path.with_suffix(".pt.tmp")
            torch.save(payload, temporary_shard)
            temporary_shard.replace(shard_path)
        return AVClipRecord(
            manifest_version=int(data_cfg["manifest_version"]),
            clip_id=source["clip_id"], split=source["split"], group_id=source["group_id"],
            shard=shard_name, duration_seconds=float(payload["duration_seconds"]),
            source={
                "video": str(source["video"]), "audio": str(source["audio"]),
                "video_sha256": identity["video_sha256"], "audio_sha256": identity["audio_sha256"],
                "scene_label": str(source["row"].get("scene_label", "")),
            },
            shard_sha256=_sha256(shard_path),
        )

    records: list[AVClipRecord] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        # map preserves official fold order regardless of decode completion order.
        for record in pool.map(ingest_one, sources):
            records.append(record)
            if len(records) % 100 == 0 or len(records) == len(sources):
                label = "Cache prefill" if cache_only else "Ingestion"
                print(f"{label}: {len(records)}/{len(sources)} clips", flush=True)
    if cache_only:
        return tuple(records)

    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(record.__dict__, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    temporary.replace(manifest_path)
    return tuple(records)


class ManifestAVData:
    def __init__(self, cfg: Mapping[str, Any], root: Path) -> None:
        self.data_cfg = cfg["data"]
        self.shard_root = _resolve(root, self.data_cfg["shard_root"])
        manifest_path = _resolve(root, self.data_cfg["manifest"])
        if not manifest_path.is_file():
            raise FileNotFoundError(f"missing A/V manifest {manifest_path}; run python -m training.av_data first")
        self.fingerprint = _sha256(manifest_path)
        self.video_fps = int(self.data_cfg["video"]["frames_per_second"])
        self.audio_rate = int(self.data_cfg["audio"]["sample_rate"])
        self.window_frames = _integer_count(
            float(self.data_cfg["window"]["duration_seconds"]), self.video_fps, "window.duration_seconds"
        )
        self.future_frames = _integer_count(
            float(self.data_cfg["window"]["future_offset_seconds"]), self.video_fps, "window.future_offset_seconds"
        )
        self.shifted_frames = _integer_count(
            float(self.data_cfg["window"]["shifted_offset_seconds"]), self.video_fps, "window.shifted_offset_seconds"
        )
        if self.audio_rate % self.video_fps:
            raise ValueError("audio sample rate must be divisible by normalized video frame rate")
        self.samples_per_frame = self.audio_rate // self.video_fps
        self.window_samples = self.window_frames * self.samples_per_frame
        records = [AVClipRecord.from_dict(json.loads(line)) for line in manifest_path.read_text().splitlines() if line]
        if not records:
            raise ValueError("audiovisual manifest is empty")
        expected_version = int(self.data_cfg["manifest_version"])
        if any(record.manifest_version != expected_version for record in records):
            raise ValueError(f"manifest version does not match configured version {expected_version}")
        if len({record.clip_id for record in records}) != len(records):
            raise ValueError("audiovisual manifest has duplicate clip ids")
        self.records = {split: tuple(record for record in records if record.split == split) for split in ("train", "eval")}
        if not all(self.records.values()) or any(record.split not in self.records for record in records):
            raise ValueError("audiovisual manifest needs non-empty train and eval splits only")
        train_groups = {record.group_id for record in self.records["train"]}
        eval_groups = {record.group_id for record in self.records["eval"]}
        overlap = train_groups & eval_groups
        if overlap:
            raise ValueError(f"recording group leakage across train/eval: {sorted(overlap)[:5]}")
        required = (self.window_frames + max(self.future_frames, self.shifted_frames)) / self.video_fps
        for record in records:
            if record.duration_seconds + 1e-7 < required:
                raise ValueError(f"clip {record.clip_id!r} is too short for configured R0/R1 views")
            if not (self.shard_root / record.shard).is_file():
                raise FileNotFoundError(f"manifest shard is missing: {self.shard_root / record.shard}")

    def _payload(self, record: AVClipRecord, cache: dict[str, Mapping[str, Any]]) -> Mapping[str, Any]:
        if record.clip_id not in cache:
            # Map the shard so a short window does not copy every unused video frame (DDR §25).
            payload = torch.load(self.shard_root / record.shard, map_location="cpu", weights_only=True, mmap=True)
            _validate_shard(payload, self.data_cfg, record.clip_id)
            cache[record.clip_id] = payload
        return cache[record.clip_id]

    def _draw(self, split: str, batch_size: int, generator: torch.Generator) -> tuple[list[AVClipRecord], list[int]]:
        if split not in self.records:
            raise ValueError(f"unknown data split {split!r}")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        records = self.records[split]
        indices = torch.randint(len(records), (batch_size,), generator=generator).tolist()
        selected = [records[index] for index in indices]
        starts = []
        maximum_offset = max(self.future_frames, self.shifted_frames)
        for record in selected:
            total_frames = int(math.floor(record.duration_seconds * self.video_fps + 1e-7))
            maximum_start = total_frames - self.window_frames - maximum_offset
            starts.append(int(torch.randint(maximum_start + 1, (1,), generator=generator)))
        return selected, starts

    def _observation(
        self,
        modality: str,
        records: list[AVClipRecord],
        starts: list[int],
        offset_frames: int,
        cache: dict[str, Mapping[str, Any]],
        *,
        shared_window_end: bool = False,
    ) -> TemporalObservation:
        values = []
        if modality == "video":
            for record, start in zip(records, starts, strict=True):
                payload = self._payload(record, cache)
                first = start + offset_frames
                values.append(payload["video"][first : first + self.window_frames])
            timestamps = torch.arange(self.window_frames, dtype=torch.float32) / self.video_fps
        elif modality == "audio":
            for record, start in zip(records, starts, strict=True):
                payload = self._payload(record, cache)
                first = (start + offset_frames) * self.samples_per_frame
                values.append(payload["audio"][:, first : first + self.window_samples])
            timestamps = torch.arange(self.window_samples, dtype=torch.float32) / self.audio_rate
        else:
            raise ValueError(f"unsupported normalized modality {modality!r}")
        stacked = torch.stack(values)
        # R1 sensors share a physical update time despite differing sample rates (DDR §31).
        # Each view uses its own window end, so timestamps cannot reveal a shifted negative.
        reference = self.window_frames / self.video_fps if shared_window_end else timestamps[-1]
        timestamps = timestamps - reference
        timestamps = timestamps.expand(len(records), -1).clone()
        return TemporalObservation(stacked, timestamps, torch.ones_like(timestamps, dtype=torch.bool))

    def sample(
        self,
        split: str,
        stage: str,
        batch_size: int,
        generator: torch.Generator,
    ) -> RepresentationBatch:
        if stage not in {"representation_unimodal", "representation_av"}:
            raise ValueError(f"A/V representation data does not serve stage {stage!r}")
        cache: dict[str, Mapping[str, Any]] = {}
        if stage == "representation_unimodal":
            video_records, video_starts = self._draw(split, batch_size, generator)
            audio_records, audio_starts = self._draw(split, batch_size, generator)
            current = {
                "video": self._observation("video", video_records, video_starts, 0, cache),
                "audio": self._observation("audio", audio_records, audio_starts, 0, cache),
            }
            future = {
                "video": self._observation("video", video_records, video_starts, self.future_frames, cache),
                "audio": self._observation("audio", audio_records, audio_starts, self.future_frames, cache),
            }
            return RepresentationBatch(current, future, {})

        records, starts = self._draw(split, batch_size, generator)
        current = {modality: self._observation(modality, records, starts, 0, cache, shared_window_end=True) for modality in ("video", "audio")}
        future = {
            modality: self._observation(modality, records, starts, self.future_frames, cache, shared_window_end=True)
            for modality in ("video", "audio")
        }
        shifted = {
            modality: self._observation(modality, records, starts, self.shifted_frames, cache, shared_window_end=True)
            for modality in ("video", "audio")
        }
        return RepresentationBatch(current, future, shifted)


def build_representation_data(cfg: Mapping[str, Any], root: Path) -> ManifestAVData:
    if cfg["data"].get("kind") != "synchronized_av_manifest":
        raise ValueError(f"unknown representation data kind {cfg['data'].get('kind')!r}")
    return ManifestAVData(cfg, root)


def _resize_square(frame: Any, resolution: int) -> torch.Tensor:
    image = torch.from_numpy(frame.to_ndarray(format="rgb24")).permute(2, 0, 1)
    height, width = image.shape[-2:]
    side = min(height, width)
    top, left = (height - side) // 2, (width - side) // 2
    image = image[:, top : top + side, left : left + side].float()[None]
    image = F.interpolate(image, (resolution, resolution), mode="bilinear", align_corners=False, antialias=True)
    return image[0].round().clamp(0, 255).to(torch.uint8)


def decode_av_media(
    video_path: Path,
    audio_path: Path,
    *,
    video_fps: int,
    resolution: int,
    audio_sample_rate: int,
    audio_channels: int,
) -> dict[str, Any]:
    """Decode one separate or muxed source pair using PyAV, then crop to a common duration."""
    if audio_channels != 1:
        raise ValueError("the initial common-base audio encoder ingests one normalized mono channel")
    try:
        import av
    except ImportError as error:
        raise RuntimeError("TAU ingestion needs PyAV; install path-wm[data]") from error

    frames = []
    with av.open(str(video_path)) as container:
        if not container.streams.video:
            raise ValueError(f"no video stream in {video_path}")
        stream = container.streams.video[0]
        origin = None
        next_time = 0.0
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            timestamp = float(frame.pts * frame.time_base)
            if origin is None:
                origin = timestamp
            relative = timestamp - origin
            if relative + 1e-7 >= next_time:
                frames.append(_resize_square(frame, resolution))
                next_time += 1.0 / video_fps
                while next_time <= relative + 1e-7:
                    next_time += 1.0 / video_fps
    if not frames:
        raise ValueError(f"no decodable video frames in {video_path}")

    audio_parts = []
    with av.open(str(audio_path)) as container:
        if not container.streams.audio:
            raise ValueError(f"no audio stream in {audio_path}")
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=audio_sample_rate)
        for frame in container.decode(stream):
            for converted in resampler.resample(frame):
                audio_parts.append(torch.from_numpy(converted.to_ndarray()).float())
        for converted in resampler.resample(None):
            audio_parts.append(torch.from_numpy(converted.to_ndarray()).float())
    if not audio_parts:
        raise ValueError(f"no decodable audio samples in {audio_path}")
    audio = torch.cat(audio_parts, dim=-1).clamp(-1.0, 1.0).contiguous()
    video = torch.stack(frames)
    common_frames = min(video.shape[0], audio.shape[1] * video_fps // audio_sample_rate)
    common_samples = common_frames * audio_sample_rate // video_fps
    if common_frames < 1 or common_samples < 1:
        raise ValueError(f"no overlapping normalized audio/video duration for {video_path}")
    return {
        "video": video[:common_frames].contiguous(),
        "audio": audio[:, :common_samples].contiguous(),
        "video_fps": video_fps,
        "audio_sample_rate": audio_sample_rate,
        "duration_seconds": common_frames / video_fps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest the configured TAU A/V subset into normalized shards")
    parser.add_argument("spec", type=Path)
    parser.add_argument("--workers", type=int, default=1, help="concurrent clip decoders")
    parser.add_argument("--cache-only", action="store_true", help="prefill available sources without publishing a manifest")
    parser.add_argument("--max-clips", type=int, help="bound one cache-only prefill batch")
    args = parser.parse_args()
    torch.set_num_threads(1)  # Parallelize clips, not every resize inside each worker.
    repository = Path(__file__).resolve().parents[1]
    spec_path = args.spec if args.spec.is_absolute() else repository / args.spec
    cfg = yaml.safe_load(spec_path.read_text())
    records = ingest_tau_av(cfg, repository, workers=args.workers,
                            cache_only=args.cache_only, max_clips=args.max_clips)
    counts = {split: sum(record.split == split for record in records) for split in ("train", "eval")}
    label = "Cached (no manifest published)" if args.cache_only else "Ingested"
    print(f"{label} {len(records)} synchronized clips: {counts['train']} train, {counts['eval']} eval")


if __name__ == "__main__":
    main()
