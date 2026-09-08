"""Queue bounded decoder development after the active encoder work and audits."""
from datetime import datetime,timezone
import json
import os
import subprocess
import sys
import time
from world_model.curriculum.perception_cache import ROOT
from world_model.pusht.checkpoints import json_atomic


def main():
    import fcntl
    root=ROOT/'development/decoders'; root.mkdir(parents=True,exist_ok=True)
    with (root/'coordinator.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); lock.write(str(os.getpid())); lock.flush()
        while not (ROOT/'extension_audits/completed.json').exists():
            if datetime.now(timezone.utc)>=datetime(2026,9,9,4,0,tzinfo=timezone.utc): return
            time.sleep(30)
        for kind in ('late','early','raw','conditioned'):
            if (root/'seed_9107'/kind/'curriculum_analysis.json').exists(): continue
            log=root/f'{kind}.log'; start=time.monotonic()
            json_atomic(ROOT/'active.json',dict(stage='decoder_development',kind=kind,pid=os.getpid()))
            with log.open('a') as stream:
                result=subprocess.run([sys.executable,'-m','scripts.run_perception_decoder','--kind',kind,'--development'],
                    stdout=stream,stderr=subprocess.STDOUT)
            receipt=dict(kind=kind,exit_code=result.returncode,seconds=time.monotonic()-start,log=str(log))
            with (root/'execution.jsonl').open('a') as f: f.write(json.dumps(receipt)+'\n')
            print(receipt,flush=True)
            if result.returncode: raise SystemExit(result.returncode)
        json_atomic(root/'completed.json',dict(time=datetime.now(timezone.utc).isoformat()))
        json_atomic(ROOT/'active.json',dict(stage='decoder_development_complete',time=datetime.now(timezone.utc).isoformat()))


if __name__=='__main__': main()
