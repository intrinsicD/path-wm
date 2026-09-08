"""Decoder reliance diagnostic with the mandatory canonical report refresh."""
import sys
from run import run_experiment
from scripts.run_perception_program import refresh

if __name__=='__main__':
    raise SystemExit(run_experiment(['-m','world_model.curriculum.perception_decoder_reliance',*sys.argv[1:]],refresh=refresh))
