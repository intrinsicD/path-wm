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


def test_failed_reporting_cannot_inherit_previous_passed_receipt(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    import pytest
    from scripts import projection_experiment as experiment
    base = tmp_path / 'experiment'
    (base / 'logs').mkdir(parents=True)
    (tmp_path / 'runs').mkdir()
    (tmp_path / 'runs/experiment_dashboard.receipt.json').write_text(
        json.dumps({'stages': {'verification': 'passed'}}))
    monkeypatch.setattr(experiment, 'ROOT', tmp_path)
    monkeypatch.setattr(experiment.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(returncode=7))
    with pytest.raises(RuntimeError, match='failed'):
        experiment.run_stage(base, 'complete_raw_failed_reporting', [], lambda: True, timeout=10)
    rows = [json.loads(line) for line in (base / 'stages.jsonl').read_text().splitlines()]
    assert rows[-1]['raw_complete'] is True
    assert rows[-1]['returncode'] == 7
    assert rows[-1]['dashboard_verified'] is False
