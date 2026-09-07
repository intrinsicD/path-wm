"""Execute the accepted new-U predictor sequence, preserving the original gate."""
from pathlib import Path
import json
import sys
from run import run_experiment
from scripts.run_curriculum import refresh

ROOT=Path('runs/curriculum_2026-09-07/paddle_history')
CONFIG='configs/paddle/history_predictor_home.yaml'
PERCEPTION='runs/paddle/baseline/perception/best.pt'
MEMORY='runs/paddle/history_start_v1/memory/best.pt'
COMMON=['--config',CONFIG,'--data','data/paddle/baseline','--perception',PERCEPTION,'--memory',MEMORY]

def execute(command,args):
    return run_experiment(['-m','world_model',command,*args],refresh=refresh)

if __name__=='__main__':
    p1=ROOT/'predictor_1';p5=ROOT/'predictor_5'
    code=execute('train-predictor',COMMON+['--horizon','1','--run',str(p1)])
    result=json.loads((p1/'paddle_result.json').read_text())
    if not result['quality_gate']['passed']:
        # A failed gate is an expected scientific outcome. Verify reporting before
        # cleanly ending the authorized branch; no implicit budget extension.
        refresh()
        (ROOT/'downstream_decision.json').write_text(json.dumps(dict(
            status='stopped_at_original_gate',P1=result,skip=['P5 training','full controller comparison'],
            reason='Selected P1 must improve latent loss and every physical coordinate over matched copy.'),indent=2))
        print('P1 failed the original gate; P5/control are not launched.',flush=True)
        raise SystemExit(0)
    if code:raise SystemExit(code)
    code=execute('train-predictor',COMMON+['--horizon','5','--run',str(p5),'--initialize-from',str(p1/'best.pt')])
    if code:raise SystemExit(code)
    code=execute('evaluate',COMMON+['--predictor',str(p5/'best.pt'),'--output',str(ROOT/'evaluation')])
    if code:raise SystemExit(code)
    (ROOT/'downstream_decision.json').write_text(json.dumps(dict(status='completed',P1=result,
        P5=json.loads((p5/'paddle_result.json').read_text()),evaluation=str(ROOT/'evaluation/metrics.json')),indent=2))
