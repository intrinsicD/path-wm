"""Seal the read-only decoder diagnostic after development and all endpoints."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from world_model.curriculum.perception_cache import ROOT
from world_model.curriculum.data import file_hash
from world_model.pusht.checkpoints import json_atomic


def main():
    import fcntl
    root=ROOT/'decoder_reliance'; root.mkdir(exist_ok=True)
    dev=ROOT/'development/decoder_reliance'
    if not (dev/'curriculum_analysis.json').exists(): raise RuntimeError('finish CPU diagnostic development first')
    r=json.loads((dev/'evaluation.json').read_text()); m=json.loads((dev/'manifest.json').read_text())
    if r['status']!='completed' or len(r['evaluations'])!=5: raise RuntimeError('CPU diagnostic development incomplete')
    for path,sha in m['source_hashes'].items():
        if file_hash(path)!=sha: raise RuntimeError('decoder diagnostic source changed after development')
    protocol=Path('docs/perception-decoder-reliance-protocol-2026-09-08.md')
    initial=dict(source=m['source_hashes'],protocol_sha256=file_hash(protocol),
        cache_sha256=file_hash(ROOT/'cache/vit/manifest.json'),dev_sha256=file_hash(dev/'evaluation.json'))
    with (root/'coordinator.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);lock.write(str(os.getpid()));lock.flush()
        while not (ROOT/'decoders/completed.json').exists():
            if datetime.now(timezone.utc)>=datetime(2026,9,9,4,0,tzinfo=timezone.utc): return
            time.sleep(30)
        for path,sha in initial['source'].items():
            if file_hash(path)!=sha: raise RuntimeError('diagnostic source changed while queued')
        if file_hash(protocol)!=initial['protocol_sha256']: raise RuntimeError('diagnostic protocol changed while queued')
        if (root/'curriculum_analysis.json').exists(): return
        files=[ROOT/'decoders'/f'seed_{seed}'/kind/'last.pt' for seed in (9107,9108,9109) for kind in ('late','early','raw','conditioned')]
        identity=dict(**initial,checkpoints={str(p):file_hash(p) for p in files},rows=512,conditions=42,device='cpu',max_seconds=1200)
        plan=root/'execution_plan.json'
        if plan.exists() and json.loads(plan.read_text())!=identity: raise RuntimeError('diagnostic identity changed')
        json_atomic(plan,identity); snapshot=root/'frozen_source'; snapshot.mkdir(exist_ok=True)
        for path in [*(Path(p) for p in initial['source']),protocol]:(snapshot/path.name).write_bytes(path.read_bytes())
        json_atomic(root/'freeze_receipt.json',dict(time=datetime.now(timezone.utc).isoformat(),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
        start=time.monotonic()
        with (root/'evaluation.log').open('a') as stream:
            result=subprocess.run([sys.executable,'-m','scripts.run_perception_decoder_reliance'],stdout=stream,stderr=subprocess.STDOUT)
        json_atomic(root/'execution.json',dict(exit_code=result.returncode,seconds=time.monotonic()-start,finished=datetime.now(timezone.utc).isoformat()))
        raise SystemExit(result.returncode)


if __name__=='__main__': main()
