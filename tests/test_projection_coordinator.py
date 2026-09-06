"""An attempted duplicate coordinator must preserve the active job's receipt."""
import fcntl
import subprocess
import sys
from pathlib import Path


def test_duplicate_coordinator_does_not_overwrite_active_progress(tmp_path):
    base=tmp_path/'experiment';base.mkdir()
    progress=base/'progress.json';original=b'{"kind":"start","name":"active_training"}'
    progress.write_bytes(original)
    with (base/'coordinator.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        result=subprocess.run([sys.executable,'-m','scripts.projection_experiment','--base',str(base)],
                              cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,timeout=30)
    assert result.returncode!=0
    assert progress.read_bytes()==original
    assert not (base/'stages.jsonl').exists()
    assert 'already running' in result.stderr.lower()
