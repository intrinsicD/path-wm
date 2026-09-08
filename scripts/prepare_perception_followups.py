"""Wait for the sealed GPU study, then audit fresh cases and profile extensions."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
from world_model.curriculum.perception_cache import ROOT
from world_model.pusht.checkpoints import json_atomic


def main():
    while True:
        active = json.loads((ROOT / 'active.json').read_text())
        if active['stage'] == 'attention_required': raise RuntimeError('P1 requires repair before follow-ups')
        if active['stage'] == 'p1_finished_or_cutoff': break
        if datetime.now(timezone.utc) >= datetime(2026, 9, 9, 4, 0, tzinfo=timezone.utc): return
        time.sleep(15)
    for seed in (9107, 9108, 9109):
        for encoder in ('cnn', 'vit'):
            path = ROOT / 'p1' / f'seed_{seed}' / encoder / 'curriculum_result.json'
            if not path.exists() or json.loads(path.read_text())['status'] != 'completed':
                raise RuntimeError('P1 incomplete; do not launch dependent follow-ups')
    for seed in (9107, 9108, 9109):
        for encoder in ('cnn', 'vit'):
            output = ROOT / 'fresh/evaluations' / f'seed_{seed}' / encoder
            if (output / 'curriculum_analysis.json').exists(): continue
            log = ROOT / f'fresh_{seed}_{encoder}.log'; started = time.monotonic()
            with log.open('a') as stream:
                result = subprocess.run([sys.executable, '-m', 'scripts.run_perception_fresh', 'evaluate',
                    '--encoder', encoder, '--seed', str(seed)], stdout=stream, stderr=subprocess.STDOUT)
            receipt = dict(stage='fresh', seed=seed, encoder=encoder, exit_code=result.returncode,
                           seconds=time.monotonic()-started, log=str(log))
            with (ROOT / 'followup_execution.jsonl').open('a') as stream: stream.write(json.dumps(receipt)+'\n')
            print(receipt, flush=True)
            if result.returncode: raise SystemExit(result.returncode)
    for kind in ('joint', 'conv', 'transformer'):
        output = ROOT / 'development/extensions/seed_9107' / kind
        if (output / 'curriculum_analysis.json').exists(): continue
        log = ROOT / f'dev_extension_{kind}.log'; started = time.monotonic()
        with log.open('a') as stream:
            result = subprocess.run([sys.executable, '-m', 'scripts.run_perception_continuation',
                '--kind', kind, '--development'], stdout=stream, stderr=subprocess.STDOUT)
        receipt = dict(stage='extension_development', kind=kind, exit_code=result.returncode,
                       seconds=time.monotonic()-started, log=str(log))
        with (ROOT / 'followup_execution.jsonl').open('a') as stream: stream.write(json.dumps(receipt)+'\n')
        print(receipt, flush=True)
        if result.returncode: raise SystemExit(result.returncode)
    json_atomic(ROOT / 'followups_ready.json', dict(status='development_and_fresh_evaluation_complete',
        time=datetime.now(timezone.utc).isoformat(), next='Inspect extension profiles, commit development, then seal formal extension plan.'))


if __name__ == '__main__': main()
