"""Refresh and browser-verify the instrument panel after each overnight unit."""
import sys
from pathlib import Path
from run import run_experiment
from viewer.dashboard import write_experiment_dashboard
from world_model.curriculum.perception_cache import ROOT


def refresh():
    return write_experiment_dashboard(ROOT,
        artifact_path=Path('runs/experiment_dashboard.artifact.json'),
        html_path=Path('runs/experiment_dashboard.html'))


if __name__ == '__main__':
    raise SystemExit(run_experiment(['-m', 'world_model.curriculum.perception_program',
                                    *sys.argv[1:]], refresh=refresh))
