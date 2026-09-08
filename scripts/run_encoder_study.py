"""Run a single encoder-study unit, then refresh and browser-verify its dashboard."""
import sys
from pathlib import Path
from run import run_experiment
from viewer.dashboard import write_experiment_dashboard

ROOT=Path('runs/encoder_study_2026-09-08')


def refresh():
    return write_experiment_dashboard(ROOT,artifact_path=Path('runs/experiment_dashboard.artifact.json'),
                                      html_path=Path('runs/experiment_dashboard.html'))


if __name__=='__main__':
    raise SystemExit(run_experiment(['-m','world_model.curriculum.encoder_study',*sys.argv[1:]],refresh=refresh))
