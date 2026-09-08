"""Read-only attention diagnostic with the shared completion/report contract."""
import sys
from run import run_experiment
from scripts.run_perception_program import refresh

if __name__=='__main__':
    raise SystemExit(run_experiment(['-m','world_model.curriculum.perception_internal',*sys.argv[1:]],refresh=refresh))
