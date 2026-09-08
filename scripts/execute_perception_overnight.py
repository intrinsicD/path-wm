"""Execute a prospectively sealed six-fit P1 plan, with per-fit verified HTML."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from world_model.curriculum.data import file_hash
from world_model.curriculum.perception_cache import ROOT
from world_model.pusht.checkpoints import json_atomic

TRAIN_FILES = [Path('world_model/curriculum') / name for name in (
    'perception_program.py', 'perception_training.py', 'perception_heads.py',
    'perception_cache.py', 'data.py', 'encoder_masks.py', 'pose_accessibility.py')]


def seal():
    plan = [dict(encoder=encoder, seed=seed, result=str(ROOT / 'p1' / f'seed_{seed}' / encoder))
            for seed in (9107, 9108, 9109) for encoder in ('cnn', 'vit')]
    code = {str(p): file_hash(p) for p in TRAIN_FILES}
    caches = {encoder: file_hash(ROOT / 'cache' / encoder / 'manifest.json') for encoder in ('cnn', 'vit')}
    frozen = ROOT / 'p1_execution_plan.json'
    if frozen.exists():
        identity = json.loads(frozen.read_text())
        if identity['plan'] != plan or identity['code'] != code or identity['caches'] != caches:
            raise ValueError('frozen P1 code/data/plan changed; explicit amendment needed')
    else:
        snapshot = ROOT / 'p1_frozen_source'; snapshot.mkdir(parents=True, exist_ok=True)
        for p in TRAIN_FILES:
            (snapshot / p.name).write_bytes(p.read_bytes())
        protocol = Path('docs/perception-overnight-protocol-2026-09-08.md')
        (snapshot / 'protocol.md').write_bytes(protocol.read_bytes())
        identity = dict(plan=plan, code=code, caches=caches, protocol_sha256=file_hash(protocol),
            git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            frozen_at=datetime.now(timezone.utc).isoformat(), updates=4000, batch_size=64,
            max_seconds_per_fit=2400, validation_every=100, selector='minimum q; earliest exact tie')
        json_atomic(frozen, identity)
    return identity


def main():
    identity = seal()
    lock = ROOT / 'p1_coordinator.lock'
    import fcntl
    with lock.open('w') as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock_file.write(str(os.getpid())); lock_file.flush()
        for unit in identity['plan']:
            out = Path(unit['result'])
            if (out / 'curriculum_analysis.json').exists():
                print('Preserve evaluated', out, flush=True); continue
            if datetime.now(timezone.utc) >= datetime(2026, 9, 9, 4, 0, tzinfo=timezone.utc):
                print('Training launch cutoff reached', flush=True); break
            # Check the sealed identity before every process boundary.
            seal()
            args = [sys.executable, '-m', 'scripts.run_perception_program', 'train',
                    '--encoder', unit['encoder'], '--seed', str(unit['seed'])]
            if (out / 'curriculum_manifest.json').exists():
                args.append('--resume')
            log = ROOT / f'p1_{unit["seed"]}_{unit["encoder"]}.log'
            started = time.monotonic()
            json_atomic(ROOT / 'active.json', dict(pid=os.getpid(), stage='p1', unit=unit,
                log=str(log), started=datetime.now(timezone.utc).isoformat()))
            with log.open('a') as output:
                result = subprocess.run(args, stdout=output, stderr=subprocess.STDOUT)
            receipt = dict(unit=unit, exit_code=result.returncode, seconds=time.monotonic() - started,
                           log=str(log), finished=datetime.now(timezone.utc).isoformat())
            with (ROOT / 'execution.jsonl').open('a') as f:
                f.write(json.dumps(receipt) + '\n')
            print(receipt, flush=True)
            if result.returncode:
                json_atomic(ROOT / 'active.json', dict(stage='attention_required', receipt=receipt))
                raise SystemExit(result.returncode)
        json_atomic(ROOT / 'active.json', dict(stage='p1_finished_or_cutoff',
                                             time=datetime.now(timezone.utc).isoformat()))


if __name__ == '__main__':
    main()
