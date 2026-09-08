"""Run each declared unit once, with a verified dashboard after every unit."""
import json,subprocess,sys,time
from pathlib import Path
from world_model.curriculum.encoder_factorial import ARMS
from world_model.curriculum.data import file_hash

ROOT=Path('runs/encoder_study_2026-09-08')


def main():
    required=[ROOT/'reference'/arm/'curriculum_result.json' for arm in ('custom','dino')]
    for p in required:
        if json.loads(p.read_text())['status']!='completed':raise RuntimeError('reference phase incomplete')
    dino=json.loads(required[-1].read_text())
    if dino['selected']['q']>1 and not (ROOT/'reference/native/curriculum_result.json').exists():
        raise RuntimeError('declared native-width control still pending')
    if not (ROOT/'coco_masks/manifest.json').exists():raise RuntimeError('COCO audit data missing')
    plan=[]
    for arm in ('custom','warmup','dino'):
        plan.append(dict(args=['probe','--arm',arm],result=str(ROOT/'probes'/arm/'curriculum_result.json')))
    for seed in (7107,7108,7109):
        for arm in ARMS:
            plan.append(dict(args=['factorial','--arm',arm,'--seed',str(seed)],result=str(ROOT/'factorial'/f'seed_{seed}'/arm/'curriculum_result.json')))
        for arm in ARMS:
            plan.append(dict(args=['probe','--arm',arm,'--seed',str(seed)],result=str(ROOT/'probes'/f'{seed}_{arm}'/'curriculum_result.json')))
    identity=dict(protocol_sha256=file_hash('docs/encoder-study-protocol-2026-09-08.md'),
                  code={str(p):file_hash(p) for p in sorted(Path('world_model/curriculum').glob('*.py'))},plan=plan)
    frozen=ROOT/'execution_plan.json'
    if frozen.exists():
        if json.loads(frozen.read_text())!=identity:raise ValueError('execution plan or training code changed; explicit protocol amendment required')
    else:frozen.write_text(json.dumps(identity,indent=2)+'\n')
    for unit in plan:
        result=Path(unit['result']);args=unit['args'];label='_'.join(args)
        if result.exists():
            r=json.loads(result.read_text())
            if r.get('status')!='completed':raise RuntimeError('incomplete run requires explicit recovery')
            print('Preserve completed',label,flush=True);continue
        log=ROOT/(label+'.log');start=time.monotonic()
        with log.open('w') as output:
            completed=subprocess.run([sys.executable,'-m','scripts.run_encoder_study',*args],stdout=output,stderr=subprocess.STDOUT)
        receipt=dict(args=args,exit_code=completed.returncode,seconds=time.monotonic()-start,log=str(log))
        with (ROOT/'execution.jsonl').open('a') as f:f.write(json.dumps(receipt)+'\n')
        print(receipt,flush=True)
        if completed.returncode:raise SystemExit(completed.returncode)

if __name__=='__main__':main()
