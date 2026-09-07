"""Run one curriculum command through the shared completion/reporting wrapper."""
import sys
from pathlib import Path
import shutil
from run import run_experiment
from viewer.dashboard import write_experiment_dashboard

ROOT=Path('runs/curriculum_2026-09-07')

def refresh():
    ROOT.mkdir(parents=True,exist_ok=True)
    previous=ROOT/'prior_reports';previous.mkdir(exist_ok=True)
    for name in ('experiment_dashboard.html','experiment_dashboard.artifact.json','experiment_dashboard.receipt.json'):
        source=Path('runs')/name;target=previous/('home_'+name)
        if source.exists() and not target.exists():shutil.copy2(source,target)
    return write_experiment_dashboard(ROOT,
        artifact_path=Path('runs/experiment_dashboard.artifact.json'),
        html_path=Path('runs/experiment_dashboard.html'))

if __name__=='__main__':
    arguments=sys.argv[1:]
    command=arguments[1:] if arguments and arguments[0]=='python' else ['-m','world_model.curriculum',*arguments]
    raise SystemExit(run_experiment(command,refresh=refresh))
