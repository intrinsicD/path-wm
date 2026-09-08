"""Fresh perception audit with canonical dashboard verification after every unit."""
import sys
from run import run_experiment
from scripts.run_perception_program import refresh

if __name__ == '__main__':
    raise SystemExit(run_experiment(['-m', 'world_model.curriculum.perception_fresh',
                                    *sys.argv[1:]], refresh=refresh))
