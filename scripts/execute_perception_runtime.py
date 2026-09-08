"""Run the bounded frozen-package timing audit after all GPU training."""
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
    root=ROOT/'runtime'; root.mkdir(exist_ok=True)
    dev=ROOT/'development/runtime'
    if not (dev/'curriculum_analysis.json').exists(): raise RuntimeError('finish runtime development and HTML first')
    identity=json.loads((dev/'manifest.json').read_text())
    with (root/'coordinator.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); lock.write(str(os.getpid())); lock.flush()
        while not (ROOT/'localization/completed.json').exists():
            if datetime.now(timezone.utc)>=datetime(2026,9,9,4,40,tzinfo=timezone.utc): return
            time.sleep(30)
        files=[Path(p) for p in identity['source_sha256']]
        for path in files:
            if file_hash(path)!=identity['source_sha256'][str(path)]: raise ValueError('runtime code changed after development')
        snapshot=root/'frozen_source'; snapshot.mkdir(exist_ok=True)
        protocol=Path('docs/perception-runtime-protocol-2026-09-09.md')
        if file_hash(protocol)!=identity['protocol_sha256']: raise ValueError('runtime protocol changed after development')
        for path in [*files,protocol]: (snapshot/path.name).write_bytes(path.read_bytes())
        json_atomic(root/'freeze_receipt.json',dict(time=datetime.now(timezone.utc).isoformat(),
            git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_sha256=identity['source_sha256']))
        if (root/'curriculum_analysis.json').exists(): return
        with (root/'audit.log').open('a') as log:
            result=subprocess.run([sys.executable,'-m','scripts.run_perception_runtime'],stdout=log,stderr=subprocess.STDOUT)
        json_atomic(root/'execution.json',dict(exit_code=result.returncode,time=datetime.now(timezone.utc).isoformat()))
        if result.returncode: raise SystemExit(result.returncode)
        json_atomic(root/'completed.json',dict(time=datetime.now(timezone.utc).isoformat()))


if __name__=='__main__': main()
