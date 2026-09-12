"""User-run local webcam capture or video import; creates an unannotated episode."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess


def prepare_episode(
    output,
    *,
    session,
    source=None,
    device="/dev/video0",
    seconds=30,
    fps=15,
    width=640,
    height=480,
):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", session):
        raise ValueError("Use a session ID with letters, numbers, dash or underscore")
    if not math.isfinite(seconds) or not 1 <= seconds <= 120 or not 1 <= fps <= 60:
        raise ValueError("Capture must be 1–120 seconds at 1–60 fps")
    if any(type(v) is not int or v < 16 or v > 1920 or v % 2 for v in (width, height)):
        raise ValueError("Use even dimensions from 16 to 1920")
    source = Path(source).resolve() if source is not None else None
    if source is not None and not source.is_file():
        raise FileNotFoundError(source)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    video = output / ("video" + (source.suffix if source else ".mp4"))
    metadata = dict(
        schema=1,
        episode_id=output.name,
        session_id=session,
        annotation_status="pending",
        status="capturing" if source is None else "importing",
        source=str(source) if source else device,
        source_kind="import" if source else "webcam",
        invoked_at_utc=datetime.now(timezone.utc).isoformat(),
        video=video.name,
    )
    record = output / "episode.json"
    record.write_text(json.dumps(metadata, indent=2) + "\n")
    try:
        if source is not None:
            shutil.copyfile(source, video)
        else:
            command = [
                "ffmpeg",
                "-n",
                "-f",
                "v4l2",
                "-framerate",
                str(fps),
                "-video_size",
                f"{width}x{height}",
                "-i",
                device,
                "-t",
                str(seconds),
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "20",
                str(video),
            ]
            metadata["capture_command"] = command
            subprocess.run(command, check=True)
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,avg_frame_rate:frame=best_effort_timestamp_time",
                "-show_frames",
                "-of",
                "json",
                str(video),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        probe = json.loads(result.stdout)
        times = [float(f["best_effort_timestamp_time"]) for f in probe["frames"]]
        if (
            not times
            or not all(math.isfinite(t) for t in times)
            or any(b <= a for a, b in zip(times, times[1:]))
        ):
            raise ValueError("Video needs finite strictly increasing frame timestamps")
        with video.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        metadata.update(
            status="ready_for_annotation",
            video_sha256=digest,
            frames=len(times),
            first_source_pts_s=times[0],
            frame_times_s=[t - times[0] for t in times],
            timestamp_basis="decoded video PTS relative to first frame; invocation UTC is not exposure time",
            stream=probe["streams"][0],
        )
        annotations = dict(
            schema=1,
            session_id=session,
            episode_id=output.name,
            status="pending",
            entities=[],
            events=[],
            queries=[],
            reviews=[],
        )
        (output / "annotations.json").write_text(
            json.dumps(annotations, indent=2) + "\n"
        )
    except BaseException as exc:
        metadata.update(status="failed", error=type(exc).__name__ + ": " + str(exc))
        raise
    finally:
        record.write_text(json.dumps(metadata, indent=2) + "\n")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--record", action="store_true", help="Activate the selected webcam and record"
    )
    mode.add_argument(
        "--input",
        type=Path,
        help="Import an existing video instead of opening a camera",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--session",
        required=True,
        help="Keep related recordings under the same session ID",
    )
    parser.add_argument("--device", default="/dev/video0")
    parser.add_argument("--seconds", type=float, default=30)
    args = parser.parse_args()
    prepare_episode(
        args.output,
        session=args.session,
        source=args.input,
        device=args.device,
        seconds=args.seconds,
    )
    print(f"Saved {args.output}; annotations are pending.")


if __name__ == "__main__":
    main()
