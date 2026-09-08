"""Seal and execute paired CPU category probes independently of GPU P1 fits."""
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


def main():
    root = ROOT / 'semantics'
    protocol = Path('docs/perception-semantic-protocol-2026-09-08.md')
    files = [Path('world_model/curriculum/perception_semantics.py'),
             Path('world_model/curriculum/perception_heads.py'),
             Path('world_model/curriculum/encoder_masks.py')]
    identity = dict(code={str(p): file_hash(p) for p in files}, protocol_sha256=file_hash(protocol),
        labels_sha256=file_hash(root / 'labels/manifest.json'),
        pooled={e: file_hash(root / 'pooled' / e / 'manifest.json') for e in ('cnn', 'vit')},
        seeds=[9107, 9108, 9109], encoders=['cnn', 'vit'], device='cpu', updates=2000, batch_size=64)
    frozen = root / 'execution_plan.json'
    if frozen.exists():
        if identity != json.loads(frozen.read_text()):
            raise ValueError('semantic execution identity changed')
    else:
        json_atomic(frozen, identity)
        snapshot = root / 'frozen_source'; snapshot.mkdir(exist_ok=True)
        for p in [*files, protocol]: (snapshot / p.name).write_bytes(p.read_bytes())
    import fcntl
    with (root / 'coordinator.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock.write(str(os.getpid())); lock.flush()
        for seed in identity['seeds']:
            for encoder in identity['encoders']:
                out = root / 'fits' / f'seed_{seed}' / encoder
                if (out / 'curriculum_analysis.json').exists(): continue
                if datetime.now(timezone.utc) >= datetime(2026, 9, 9, 4, 0, tzinfo=timezone.utc): return
                if any(file_hash(p) != h for p, h in identity['code'].items()):
                    raise ValueError('semantic source changed before fit')
                log = root / f'fit_{seed}_{encoder}.log'; started = time.monotonic()
                with log.open('a') as output:
                    result = subprocess.run([sys.executable, '-m', 'scripts.run_perception_semantics', 'train',
                        '--encoder', encoder, '--seed', str(seed), '--device', 'cpu'], stdout=output, stderr=subprocess.STDOUT)
                receipt = dict(seed=seed, encoder=encoder, exit_code=result.returncode,
                    seconds=time.monotonic() - started, log=str(log), finished=datetime.now(timezone.utc).isoformat())
                with (root / 'execution.jsonl').open('a') as f: f.write(json.dumps(receipt) + '\n')
                print(receipt, flush=True)
                if result.returncode: raise SystemExit(result.returncode)


if __name__ == '__main__': main()
