"""Wait for the first conditioned endpoint, then run CPU diagnostic development."""
from datetime import datetime,timezone
import os
import subprocess
import sys
import time
from world_model.curriculum.perception_cache import ROOT
from world_model.pusht.checkpoints import json_atomic


def main():
    import fcntl
    root=ROOT/'development/decoder_reliance'; root.mkdir(parents=True,exist_ok=True)
    with (root/'coordinator.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); lock.write(str(os.getpid()));lock.flush()
        while not (ROOT/'decoders/seed_9107/conditioned/curriculum_analysis.json').exists():
            if datetime.now(timezone.utc)>=datetime(2026,9,9,4,0,tzinfo=timezone.utc): return
            time.sleep(30)
        if (root/'curriculum_analysis.json').exists(): return
        start=time.monotonic()
        with (root/'development.log').open('a') as stream:
            result=subprocess.run([sys.executable,'-m','scripts.run_perception_decoder_reliance','--development'],stdout=stream,stderr=subprocess.STDOUT)
        json_atomic(root/'execution.json',dict(exit_code=result.returncode,seconds=time.monotonic()-start,finished=datetime.now(timezone.utc).isoformat()))
        raise SystemExit(result.returncode)


if __name__=='__main__': main()
