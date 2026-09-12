import json
import shutil
import subprocess

import pytest


def test_import_preserves_video_timestamps_and_pending_annotations(tmp_path):
    from experiments.capture_webcam import prepare_episode

    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg tools unavailable")
    source = tmp_path / "source.mp4"
    subprocess.run([
        "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=blue:s=32x32:r=5",
        "-t", "1", "-c:v", "libx264", str(source),
    ], check=True)
    before = source.read_bytes()
    output = tmp_path / "take"
    prepare_episode(output, session="session-a", source=source)
    record = json.loads((output / "episode.json").read_text())
    assert record["status"] == "ready_for_annotation"
    assert record["session_id"] == "session-a"
    assert record["annotation_status"] == "pending"
    assert record["frames"] == 5
    assert record["frame_times_s"] == pytest.approx([0, .2, .4, .6, .8])
    assert (output / "video.mp4").read_bytes() == before == source.read_bytes()
    assert json.loads((output / "annotations.json").read_text())["queries"] == []
    with pytest.raises(FileExistsError):
        prepare_episode(output, session="session-a", source=source)


def test_invalid_capture_settings_do_not_open_device(tmp_path, monkeypatch):
    from experiments.capture_webcam import prepare_episode

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("opened device"))
    with pytest.raises(ValueError):
        prepare_episode(tmp_path / "bad", session="session-a", seconds=-1)
    assert not (tmp_path / "bad").exists()
