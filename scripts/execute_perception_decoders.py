"""Seal and execute the four-arm, three-seed decoder comparison after development."""
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
    root=ROOT/'decoders'; root.mkdir(exist_ok=True)
    files=[Path('world_model/curriculum')/name for name in ('perception_decoder.py','perception_decoder_training.py',
        'perception_training.py','perception_cache.py','perception_heads.py','encoder_reference.py','dino_reference.py','encoder_masks.py','data.py')]
    protocol=Path('docs/perception-decoder-protocol-2026-09-08.md')
    for kind in ('late','early','raw','conditioned'):
        dev=ROOT/'development/decoders/seed_9107'/kind
        if not (dev/'curriculum_analysis.json').exists(): raise RuntimeError('complete and verify every decoder development arm first')
        result=json.loads((dev/'curriculum_result.json').read_text()); manifest=json.loads((dev/'curriculum_manifest.json').read_text())
        if result['status']!='completed' or result['step']!=50: raise RuntimeError('decoder development did not complete')
        for name,h in manifest['source_hashes'].items():
            if file_hash(name)!=h: raise ValueError('decoder code changed after development')
    identity=dict(code={str(p):file_hash(p) for p in files},protocol_sha256=file_hash(protocol),
        data_sha256=file_hash(ROOT/'cache/vit/manifest.json'),seeds=[9107,9108,9109],
        arms=['late','early','raw','conditioned'],updates=4000,batch_size=64,max_seconds_per_fit=1200,
        selector='fixed update endpoint; validation monitoring only')
    path=root/'execution_plan.json'
    if path.exists():
        if json.loads(path.read_text())!=identity: raise ValueError('decoder protocol identity changed')
    else:
        json_atomic(path,identity); snapshot=root/'frozen_source'; snapshot.mkdir(exist_ok=True)
        for p in [*files,protocol]:(snapshot/p.name).write_bytes(p.read_bytes())
        json_atomic(root/'freeze_receipt.json',dict(time=datetime.now(timezone.utc).isoformat(),
            git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
    return identity


def main():
    import fcntl
    if not (ROOT/'extension_audits/completed.json').exists(): raise RuntimeError('finish existing GPU queue first')
    identity=seal(); root=ROOT/'decoders'
    with (root/'coordinator.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); lock.write(str(os.getpid())); lock.flush()
        for seed in identity['seeds']:
            for kind in identity['arms']:
                out=root/f'seed_{seed}'/kind
                if (out/'curriculum_analysis.json').exists(): continue
                if datetime.now(timezone.utc)>=datetime(2026,9,9,4,0,tzinfo=timezone.utc): return
                seal(); log=root/f'fit_{seed}_{kind}.log'; started=time.monotonic()
                json_atomic(ROOT/'active.json',dict(stage='decoders',seed=seed,kind=kind,pid=os.getpid(),log=str(log),started=datetime.now(timezone.utc).isoformat()))
                args=[sys.executable,'-m','scripts.run_perception_decoder','--kind',kind,'--seed',str(seed)]
                if (out/'curriculum_manifest.json').exists(): args.append('--resume')
                with log.open('a') as stream: result=subprocess.run(args,stdout=stream,stderr=subprocess.STDOUT)
                receipt=dict(seed=seed,kind=kind,exit_code=result.returncode,seconds=time.monotonic()-started,
                    log=str(log),finished=datetime.now(timezone.utc).isoformat())
                with (root/'execution.jsonl').open('a') as f:f.write(json.dumps(receipt)+'\n')
                print(receipt,flush=True)
                if result.returncode:
                    json_atomic(ROOT/'active.json',dict(stage='decoder_attention_required',receipt=receipt)); raise SystemExit(result.returncode)
        json_atomic(root/'completed.json',dict(time=datetime.now(timezone.utc).isoformat()))
        json_atomic(ROOT/'active.json',dict(stage='decoders_finished',time=datetime.now(timezone.utc).isoformat()))


if __name__=='__main__': main()
