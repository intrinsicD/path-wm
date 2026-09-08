"""Seal the geometry objective comparison and wait for dense decoding to finish."""
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


def seal():
    root=ROOT/'localization'; root.mkdir(exist_ok=True)
    files=[Path('world_model/curriculum')/name for name in ('perception_localization.py','perception_training.py',
        'perception_cache.py','perception_heads.py','perception_fresh.py','encoder_reference.py','dino_reference.py','data.py')]
    protocol=Path('docs/perception-localization-protocol-2026-09-08.md')
    for encoder in ('cnn','vit'):
        for control in (True,False):
            dev=ROOT/'development/localization_v2/seed_9107'/(encoder+('_control' if control else ''))
            if not (dev/'curriculum_analysis.json').exists(): raise RuntimeError('finish all geometry development units first')
            result=json.loads((dev/'curriculum_result.json').read_text()); manifest=json.loads((dev/'curriculum_manifest.json').read_text())
            if result['status']!='completed' or result['step']!=50: raise RuntimeError('geometry development incomplete')
            if file_hash(files[0])!=manifest['source_sha256']: raise ValueError('geometry code changed after development')
            if control:
                check=json.loads((dev/'reference_check.json').read_text())
                if check['maximum_identifiable_parameters']>check['tolerance'] or check['maximum_output_difference']>check['tolerance']:
                    raise ValueError('unpaired development control')
    identity=dict(code={str(p):file_hash(p) for p in files},protocol_sha256=file_hash(protocol),
        data_sha256={e:file_hash(ROOT/f'cache/{e}/manifest.json') for e in ('cnn','vit')},
        fresh_population_sha256=file_hash(ROOT/'fresh/population/manifest.json'),
        seeds=[9107,9108,9109],arms=['cnn','vit'],updates=4000,batch_size=32,max_seconds_per_fit=900,
        selector='minimum validation q; earliest exact tie',location_kl_weight=.001)
    path=root/'execution_plan.json'
    if path.exists():
        if json.loads(path.read_text())!=identity: raise ValueError('geometry protocol identity changed')
    else:
        json_atomic(path,identity); snapshot=root/'frozen_source'; snapshot.mkdir(exist_ok=True)
        for p in [*files,protocol]:(snapshot/p.name).write_bytes(p.read_bytes())
        json_atomic(root/'freeze_receipt.json',dict(time=datetime.now(timezone.utc).isoformat(),
            git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
    return identity


def main():
    import fcntl
    identity=seal(); root=ROOT/'localization'
    with (root/'coordinator.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); lock.write(str(os.getpid())); lock.flush()
        while not (ROOT/'independent/completed.json').exists():
            if datetime.now(timezone.utc)>=datetime(2026,9,9,4,0,tzinfo=timezone.utc): return
            time.sleep(30)
        for seed in identity['seeds']:
            for encoder in identity['arms']:
                out=root/f'seed_{seed}'/encoder
                if (out/'curriculum_analysis.json').exists(): continue
                if datetime.now(timezone.utc)>=datetime(2026,9,9,4,0,tzinfo=timezone.utc): return
                seal(); log=root/f'fit_{seed}_{encoder}.log'; start=time.monotonic()
                json_atomic(ROOT/'active.json',dict(stage='localization',seed=seed,encoder=encoder,pid=os.getpid(),log=str(log),started=datetime.now(timezone.utc).isoformat()))
                args=[sys.executable,'-m','scripts.run_perception_localization','--encoder',encoder,'--seed',str(seed)]
                if (out/'curriculum_manifest.json').exists(): args.append('--resume')
                with log.open('a') as stream: result=subprocess.run(args,stdout=stream,stderr=subprocess.STDOUT)
                receipt=dict(seed=seed,encoder=encoder,exit_code=result.returncode,seconds=time.monotonic()-start,
                    log=str(log),finished=datetime.now(timezone.utc).isoformat())
                with (root/'execution.jsonl').open('a') as f:f.write(json.dumps(receipt)+'\n')
                print(receipt,flush=True)
                if result.returncode:
                    json_atomic(ROOT/'active.json',dict(stage='localization_attention_required',receipt=receipt)); raise SystemExit(result.returncode)
        json_atomic(root/'completed.json',dict(time=datetime.now(timezone.utc).isoformat()))
        json_atomic(ROOT/'active.json',dict(stage='localization_finished',time=datetime.now(timezone.utc).isoformat()))


if __name__=='__main__': main()
