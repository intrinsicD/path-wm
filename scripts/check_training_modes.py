"""Checkpoint diagnostics on fixed training and held-out windows.

Evaluation uses unmodified running BatchNorm buffers. A separate train-mode
probe shows batch-statistic behavior and is discarded after measurement.
"""
import argparse,copy,json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from world_model.data import TrajectoryDataset,preprocess_pixels,normalize_actions
from world_model.evaluation import evaluate_prediction
from world_model.model import build_model
from world_model.train import write_json

@torch.no_grad()
def check(run):
    torch.set_num_threads(4)
    run=Path(run);meta=json.loads((run/'manifest.json').read_text())
    saved=torch.load(run/'checkpoint.pt',weights_only=True,map_location='cuda')
    model=build_model(saved['model_config']).cuda();model.load_state_dict(saved['model'],strict=True)
    output={'step':saved['step'],'seed':meta['seed']}
    for split,key in [('train','train_episodes'),('heldout','val_episodes')]:
        ds=TrajectoryDataset(meta['dataset']['path'],meta[key],frameskip=meta['dataset']['frameskip'],num_steps=4)
        n=min(len(ds),512)
        indices=torch.randperm(len(ds),generator=torch.Generator().manual_seed(103072)).tolist()[:n]
        loader=DataLoader(ds,batch_size=128,sampler=indices,num_workers=2)
        output[split]=evaluate_prediction(model,loader,saved['action_stats'],torch.device('cuda'),batches=4)
        batch=next(iter(loader));x=preprocess_pixels(batch['pixels'].cuda());a=normalize_actions(batch['action'].cuda(),saved['action_stats'])
        model.eval();z=model.encode(x);prediction=model.predict(z[:,:-1],a[:,:-1])
        output[split]['eval_mode_batch_mse']=float((prediction-z[:,1:]).square().mean())
        # Same images; only BatchNorm uses current-batch statistics. Disable
        # dropout to avoid attributing its noise to normalization mismatch.
        probe=copy.deepcopy(model)
        for m in probe.modules():
            if isinstance(m,torch.nn.modules.batchnorm._BatchNorm):m.train()
        z=probe.encode(x);prediction=probe.predict(z[:,:-1],a[:,:-1])
        output[split]['batch_statistics_mse']=float((prediction-z[:,1:]).square().mean())
        del probe
    write_json(run/'diagnostics.json',output);print(json.dumps(output),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run');check(p.parse_args().run)
