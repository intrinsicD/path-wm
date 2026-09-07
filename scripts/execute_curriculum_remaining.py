"""Fixed B/C screen sequence; one GPU phase and verified report at a time."""
import subprocess
import sys
from pathlib import Path
from world_model.curriculum.training import source_hash
from world_model.pusht.checkpoints import json_atomic
ROOT=Path('runs/curriculum_2026-09-07')
if __name__=='__main__':
    for arm,phase in [('B','warmup'),('B','supervised'),('C','warmup'),('C','supervised')]:
        result=subprocess.run([sys.executable,'-u','-m','scripts.run_curriculum','phase','--arm',arm,'--phase',phase])
        if result.returncode:raise SystemExit(result.returncode)
