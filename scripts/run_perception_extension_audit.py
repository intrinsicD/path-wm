"""Publish and verify the canonical report after every extension audit unit."""
import sys
from run import run_experiment
from scripts.run_perception_program import refresh

if __name__ == '__main__':
    raise SystemExit(run_experiment(['-m', 'world_model.curriculum.perception_extension_audits',
                                    *sys.argv[1:]], refresh=refresh))
