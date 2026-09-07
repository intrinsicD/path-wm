"""Frozen post-screen perception evaluation; preserve selected/final controls."""
import json
from pathlib import Path
import sys
import subprocess
ROOT=Path('runs/curriculum_2026-09-07')
if __name__=='__main__':
    frozen=json.loads((ROOT/'seed_4107/screen_decision.json').read_text())
    if frozen['status']!='frozen_before_test' or frozen['confirmation_candidates']:
        raise ValueError('Resolve confirmation before final test assessment')
    jobs=[]
    for arm in ('B','C'):
        jobs.append((f'validation_{arm}_selected',ROOT/f'seed_4107/{arm}/supervised/best.pt','validation'))
    for arm in ('A','B','C'):
        for label,file in [('selected','best.pt'),('final','last.pt')]:
            jobs.append((arm+'_'+label,ROOT/f'seed_4107/{arm}/supervised'/file,'test'))
        if arm=='A':jobs.append(('A_update2000',ROOT/'seed_4107/A/supervised/update_00002000.pt','test'))
        else:jobs.append((arm+'_warmup',ROOT/f'seed_4107/{arm}/warmup/last.pt','test'))
    for label,file in [('selected','best.pt'),('final','last.pt')]:
        jobs.append(('historical_'+label,Path('runs/pusht_world_model/baseline/perception')/file,'test'))
    for name,path,split in jobs:
        out=ROOT/'inspection'/name
        code=subprocess.run([sys.executable,'-u','-m','scripts.run_curriculum','python','-m','world_model.curriculum.inspection',
                             str(path),str(out),'--split',split]).returncode
        if code:raise SystemExit(code)
