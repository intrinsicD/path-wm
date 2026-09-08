"""Render declared perception panels, then rebuild and browser-verify their report."""
from run import run_experiment
from scripts.run_perception_program import refresh

if __name__=='__main__':
    raise SystemExit(run_experiment(['-m','world_model.curriculum.perception_figures'],refresh=refresh))
