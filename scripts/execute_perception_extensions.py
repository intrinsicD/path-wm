"""Sealed nine-fit adaptive encoder-extension comparison, one GPU fit at a time."""
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


def seal():
    root = ROOT / 'extensions'; root.mkdir(exist_ok=True)
    for kind in ('joint', 'conv', 'transformer'):
        path = ROOT / 'development/extensions/seed_9107' / kind / 'curriculum_result.json'
        if not path.exists() or json.loads(path.read_text())['status'] != 'completed':
            raise RuntimeError('complete extension development first')
    for seed in (9107, 9108, 9109):
        for encoder in ('cnn', 'vit'):
            p = ROOT / 'p1' / f'seed_{seed}' / encoder / 'curriculum_result.json'
            if not p.exists() or json.loads(p.read_text())['status'] != 'completed': raise RuntimeError('P1 is incomplete')
    files = [Path('world_model/curriculum') / name for name in (
        'perception_extensions.py', 'perception_continuation.py', 'perception_training.py',
        'perception_heads.py', 'perception_cache.py', 'encoder_variants.py', 'data.py', 'encoder_masks.py')]
    protocol = Path('docs/perception-extension-protocol-2026-09-08.md')
    identity = dict(code={str(p): file_hash(p) for p in files}, protocol_sha256=file_hash(protocol),
        source_sha256=file_hash('runs/encoder_study_2026-09-08/factorial/seed_7107/deeper/best.pt'),
        data_sha256=file_hash(ROOT / 'cache/cnn/manifest.json'), seeds=[9107, 9108, 9109],
        arms=['joint', 'conv', 'transformer'], updates=4000, batch_size=64, max_seconds_per_fit=1200,
        selector='minimum validation q; earliest exact tie')
    path = root / 'execution_plan.json'
    if path.exists():
        if json.loads(path.read_text()) != identity: raise ValueError('extension identity changed; explicit amendment required')
    else:
        json_atomic(path, identity)
        snapshot = root / 'frozen_source'; snapshot.mkdir(exist_ok=True)
        for p in [*files, protocol]: (snapshot / p.name).write_bytes(p.read_bytes())
        json_atomic(root / 'freeze_receipt.json', dict(time=datetime.now(timezone.utc).isoformat(),
            git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()))
    return identity


def main():
    identity = seal(); root = ROOT / 'extensions'
    import fcntl
    with (root / 'coordinator.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); lock.write(str(os.getpid())); lock.flush()
        for seed in identity['seeds']:
            for kind in identity['arms']:
                out = root / f'seed_{seed}' / kind
                if (out / 'curriculum_analysis.json').exists(): continue
                if datetime.now(timezone.utc) >= datetime(2026, 9, 9, 4, 0, tzinfo=timezone.utc): return
                seal()
                log = root / f'fit_{seed}_{kind}.log'; started = time.monotonic()
                json_atomic(ROOT / 'active.json', dict(stage='extensions', pid=os.getpid(), seed=seed,
                    kind=kind, log=str(log), started=datetime.now(timezone.utc).isoformat()))
                args = [sys.executable, '-m', 'scripts.run_perception_continuation', '--kind', kind, '--seed', str(seed)]
                if (out / 'curriculum_manifest.json').exists(): args.append('--resume')
                with log.open('a') as stream: result = subprocess.run(args, stdout=stream, stderr=subprocess.STDOUT)
                receipt = dict(seed=seed, kind=kind, exit_code=result.returncode, seconds=time.monotonic()-started,
                    log=str(log), finished=datetime.now(timezone.utc).isoformat())
                with (root / 'execution.jsonl').open('a') as f: f.write(json.dumps(receipt)+'\n')
                print(receipt, flush=True)
                if result.returncode:
                    json_atomic(ROOT / 'active.json', dict(stage='extension_attention_required', receipt=receipt))
                    raise SystemExit(result.returncode)
        json_atomic(ROOT / 'active.json', dict(stage='extensions_finished', time=datetime.now(timezone.utc).isoformat()))


if __name__ == '__main__': main()
