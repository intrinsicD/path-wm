"""Wait for sealed encoder fits, then evaluate each selected model independently."""
from datetime import datetime, timezone
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from world_model.curriculum.perception_cache import ROOT
from world_model.curriculum.data import file_hash
from world_model.pusht.checkpoints import json_atomic


def main():
    import fcntl
    root = ROOT / 'extension_audits'; root.mkdir(exist_ok=True)
    with (root / 'coordinator.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); lock.write(str(os.getpid())); lock.flush()
        files = [Path('world_model/curriculum') / p for p in ('perception_extension_audits.py',
            'perception_semantics.py', 'perception_fresh.py')]
        identity = dict(code={str(p): file_hash(p) for p in files},
            extension_plan_sha256=file_hash(ROOT/'extensions/execution_plan.json'),
            population_sha256=file_hash(ROOT/'fresh/population/manifest.json'),
            labels_sha256=file_hash(ROOT/'semantics/labels/manifest.json'),
            seeds=[9107, 9108, 9109], kinds=['joint', 'conv', 'transformer'])
        path = root/'execution_plan.json'
        if path.exists():
            if json.loads(path.read_text()) != identity: raise ValueError('audit identity changed')
        else:
            json_atomic(path, identity)
            snapshot=root/'frozen_source'; snapshot.mkdir(exist_ok=True)
            for p in files: (snapshot/p.name).write_bytes(p.read_bytes())
            json_atomic(root/'freeze_receipt.json', dict(time=datetime.now(timezone.utc).isoformat(),
                git_commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip()))
        while not all((ROOT/'extensions'/f'seed_{s}'/k/'curriculum_analysis.json').exists()
                      for s in identity['seeds'] for k in identity['kinds']):
            if datetime.now(timezone.utc) >= datetime(2026,9,9,4,0,tzinfo=timezone.utc): return
            time.sleep(30)
        for seed in identity['seeds']:
            for kind in identity['kinds']:
                for command, filename, device in [('pool','pooled/manifest.json','cuda'),
                        ('semantics','semantics/curriculum_analysis.json','cpu'),
                        ('fresh','fresh/curriculum_analysis.json','cuda')]:
                    if (root/f'seed_{seed}'/kind/filename).exists(): continue
                    for name,h in identity['code'].items():
                        if file_hash(name)!=h: raise ValueError('audit source changed after freeze')
                    log=root/f'{command}_{seed}_{kind}.log'; started=time.monotonic()
                    json_atomic(ROOT/'active.json', dict(stage='extension_audits',seed=seed,kind=kind,command=command,pid=os.getpid()))
                    with log.open('a') as stream:
                        result=subprocess.run([sys.executable,'-m','scripts.run_perception_extension_audit',command,
                            '--kind',kind,'--seed',str(seed),'--device',device],stdout=stream,stderr=subprocess.STDOUT)
                    receipt=dict(seed=seed,kind=kind,command=command,exit_code=result.returncode,
                        seconds=time.monotonic()-started,log=str(log),finished=datetime.now(timezone.utc).isoformat())
                    with (root/'execution.jsonl').open('a') as f: f.write(json.dumps(receipt)+'\n')
                    print(receipt,flush=True)
                    if result.returncode: raise SystemExit(result.returncode)
        json_atomic(root/'completed.json',dict(time=datetime.now(timezone.utc).isoformat()))
        json_atomic(ROOT/'active.json',dict(stage='extension_audits_finished',time=datetime.now(timezone.utc).isoformat()))


if __name__=='__main__': main()
