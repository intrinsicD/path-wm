"""Verify and freeze a source-scale training/evaluation candidate without a model."""
import json
from pathlib import Path
import subprocess
import h5py
import numpy as np
import yaml
from world_model.protocol import prepare_training_data
from world_model.evaluation import sample_source_cases
from world_model.train import write_json
from scripts.diagnose_pusht_control import sha256


def prepare():
    config=Path('configs/reproduction/pusht_source_scale.yaml')
    cfg=yaml.safe_load(config.read_text())
    ds=yaml.safe_load(Path(cfg['dataset']).read_text())
    source=Path(ds['path'])
    extraction=json.loads((source.parent/'extraction.json').read_text())
    if any(extraction[k]!=ds[k] for k in ('repo','revision','sha256')):
        raise ValueError('Full-source extraction receipt mismatch')
    train,val,tr,va,stats,receipt=prepare_training_data(cfg,ds)
    batches=len(train)//cfg['batch_size'];steps=batches*cfg['epochs']
    if len(train)!=cfg['expected_train_windows'] or steps!=cfg['expected_total_steps']:
        raise ValueError('Reproduction source/window budget changed')
    with h5py.File(source,'r') as data:
        lengths,offsets=data['ep_len'][:],data['ep_offset'][:]
        cases=sample_source_cases(lengths,offsets,count=50,seed=42)
    out=Path('runs/reproduction/pusht_source_scale_preparation');out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(out/'window_indices.npz',train=np.asarray(train.indices,np.int64),val=np.asarray(val.indices,np.int64))
    control=dict(dataset=ds,cases=cases,sampling_seed=42,solver_and_reset_seeds=list(range(1234,1284)),
                 population='Full source; source-training population for the reproduction and released weights',
                 protocol='Deterministic paired local evaluation; distinct from upstream unseeded batch-50 evaluation',
                 samples=300,iterations=30,elites=30,horizon=5,action_block=5,receding_horizon=5,goal_offset=25,budget=50,
                 precision='float32',success_threshold=None)
    write_json(out/'control_cases.json',control)
    manifest=dict(config=cfg,config_sha256=sha256(config),dataset=ds,source_verification=extraction,
                  source_episodes=len(lengths),source_frames=int(lengths.sum()),data_protocol=receipt,
                  action_stats=stats,batches_per_epoch=batches,total_steps=steps,warmup_steps=max(1,int(steps*.01)),
                  checkpoint_steps=cfg['checkpoint_steps'],window_indices_sha256=sha256(out/'window_indices.npz'),
                  control_cases_sha256=sha256(out/'control_cases.json'),
                  code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  training_launched=False,model_constructed=False,
                  pilot_rate_extrapolated_hours=1506.89/1000*steps/3600,
                  limitation='Runtime extrapolation excludes full-source I/O and changes in activation-checkpointing cost.')
    write_json(out/'manifest.json',manifest)
    print(json.dumps({k:manifest[k] for k in ('source_episodes','source_frames','batches_per_epoch','total_steps','warmup_steps','training_launched','model_constructed','pilot_rate_extrapolated_hours')}),flush=True)


if __name__=='__main__':prepare()
