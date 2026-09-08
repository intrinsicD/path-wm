"""Run one pose-accessibility stage through the canonical verified dashboard."""
import sys
import shutil
from pathlib import Path
from run import run_experiment
from viewer.dashboard import write_experiment_dashboard
from world_model.curriculum.pose_accessibility import ROOT


def refresh():
    previous=ROOT/'prior_reports';previous.mkdir(parents=True,exist_ok=True)
    for name in ('experiment_dashboard.html','experiment_dashboard.artifact.json','experiment_dashboard.receipt.json'):
        source=Path('runs')/name;target=previous/name
        if source.exists() and not target.exists():shutil.copy2(source,target)
    scope=ROOT/('development' if '--development' in sys.argv else 'experiment')
    return write_experiment_dashboard(scope,artifact_path=Path('runs/experiment_dashboard.artifact.json'),html_path=Path('runs/experiment_dashboard.html'))

if __name__=='__main__':
    raise SystemExit(run_experiment(['-m','world_model.curriculum.pose_accessibility',*sys.argv[1:]],refresh=refresh))
