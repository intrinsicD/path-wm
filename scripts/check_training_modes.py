"""Matched normalization diagnostics on restored training/validation windows.

Original checkpoints stay immutable. Calibrated buffers belong to disposable
clones; batch-statistic probes are not single-state deployment evaluators.
"""
import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from scripts.bn_calibration import recalibrate_bn_statistics
from scripts.inspect_checkpoint import inspection_datasets,load,sha256
from world_model.data import preprocess_pixels,normalize_actions
from world_model.evaluation import evaluate_prediction
from world_model.introspection import state_digest
from world_model.training import autocast_context
from world_model.train import write_json


def source_rows(dataset,indices):
    rows=[]
    for index in indices:
        base=dataset
        while isinstance(base,Subset):index=base.indices[index];base=base.dataset
        episode,start=base.locate(int(index));rows.append(int(base.offsets[episode])+start)
    return rows


def batch_statistics(model,loader,stats,device,image_size):
    """BN uses current batches; dropout stays off, and the clone is discarded."""
    probe=copy.deepcopy(model).eval()
    for layer in probe.modules():
        if isinstance(layer,torch.nn.modules.batchnorm._BatchNorm):layer.train()
    totals={};count=0
    with torch.no_grad():
        for batch in loader:
            x=preprocess_pixels(batch['pixels'].to(device),image_size)
            actions=normalize_actions(batch['action'].to(device),stats)
            z=probe.encode(x);target=z[:,1:]
            pred=probe.predict(z[:,:-1],actions[:,:-1])
            zero=probe.predict(z[:,:-1],torch.zeros_like(actions[:,:-1]))
            shuffled=probe.predict(z[:,:-1],actions.roll(1,0)[:,:-1])
            rollout=probe.rollout(z[:,:1],actions[:,:-1])
            values=dict(pred_mse=(pred-target).square().mean(),identity_mse=(z[:,:-1]-target).square().mean(),
                zero_action_mse=(zero-target).square().mean(),shuffled_action_mse=(shuffled-target).square().mean(),
                rollout_mse=(rollout-target).square().mean(),embedding_std=z.std(dim=0).mean(),
                action_effect=(pred-shuffled).square().mean())
            count+=len(z)
            for key,value in values.items():totals[key]=totals.get(key,0.)+float(value)*len(z)
    if not count:raise ValueError('Empty diagnostic sample')
    return {**{k:v/count for k,v in totals.items()},'examples':count}


@torch.no_grad()
def check(run,output_path=None,*,device='cuda',windows=512,calibration_windows=512,batch_size=128,
          checkpoint=None,reference=None,save_calibrated=None,calibration_only=False):
    if calibration_only and save_calibrated is None:raise ValueError('calibration_only requires save_calibrated')
    if min(windows,calibration_windows,batch_size)<2:raise ValueError('Diagnostic sample and batch sizes must be at least two')
    torch.set_num_threads(4);device=torch.device(device);run=Path(run)
    output=Path(output_path) if output_path else run/'diagnostics.json'
    if output.exists() or (output.parent/'variants').exists():raise FileExistsError('Diagnostic output already exists')
    if save_calibrated is not None and Path(save_calibrated).exists():raise FileExistsError('Calibrated clone output already exists')
    meta=json.loads((run/'manifest.json').read_text());checkpoint=Path(checkpoint) if checkpoint else run/'checkpoint.pt'
    digest=sha256(checkpoint)
    model,stats,step,_=load(checkpoint,meta,reference is not None,reference)
    model=model.to(device).eval();initial_state=state_digest(model)
    train,val,_=inspection_datasets(meta)
    train_indices=torch.randperm(len(train),generator=torch.Generator().manual_seed(103072)).tolist()[:calibration_windows]
    val_indices=list(meta['validation_window_indices'][:windows])
    if len(train_indices)<2 or len(val_indices)<2:raise ValueError('Too few recorded diagnostic windows')
    # Unequal final batches remain counted and explicit; BN averages batch statistics.
    def batches(dataset,indices):
        return DataLoader(dataset,batch_size=batch_size,sampler=indices,num_workers=0,
                          generator=torch.Generator().manual_seed(103072))
    def forward(m,batch):
        pixels=preprocess_pixels(batch['pixels'].to(device),m.encoder.config.image_size)
        actions=normalize_actions(batch['action'].to(device),stats)
        z=m.encode(pixels);return m.predict(z[:,:-1],actions[:,:-1])
    calibrated,calibration=recalibrate_bn_statistics(model,lambda:batches(train,train_indices),forward)
    calibrated.eval()
    train_rows,val_rows=source_rows(train,train_indices),source_rows(val,val_indices)
    if set(train_rows)&set(val_rows):raise ValueError('Calibration and validation source windows overlap')
    output.parent.mkdir(parents=True,exist_ok=True)
    variants=[('saved_float32',model,'float32'),('calibrated_float32',calibrated,'float32')]
    if device.type=='cuda':variants += [('saved_bf16',model,'bf16'),('calibrated_bf16',calibrated,'bf16')]
    result=dict(step=step,seed=meta['seed'],checkpoint=str(checkpoint),checkpoint_sha256=digest,
        calibration_window_indices=train_indices,validation_window_indices=val_indices,
        calibration_source_rows=train_rows,validation_source_rows=val_rows,
        calibration=asdict(calibration),calibration_method='Layerwise cumulative mean of per-batch statistics; upstream BN and dropout eval',
        population=meta.get('population','Recorded episode-disjoint training/validation split'),
        batch_size=batch_size,variants={},batch_statistics={},
        batch_statistics_scope='Disposable BN-only current-batch probe; dropout disabled; batch-coupled outputs, not deployable control')
    if calibration_only: variants=[]
    for name,variant,precision in variants:
        result['variants'][name]={}
        for split,ds,indices,rows in [('training',train,train_indices,train_rows),('validation',val,val_indices,val_rows)]:
            metrics=evaluate_prediction(variant,batches(ds,indices),stats,device,
                model.encoder.config.image_size,batches=len(indices),precision=precision)
            result['variants'][name][split]=metrics
            directory=output.parent/'variants'/name/split;directory.mkdir(parents=True,exist_ok=False)
            write_json(directory/'manifest.json',dict(dataset=meta['dataset'],step=step,checkpoint=str(checkpoint),
                checkpoint_sha256=digest,source_run_manifest=str(run/'manifest.json'),precision=precision,
                model_state_sha256=state_digest(variant),window_indices=indices,source_rows=rows,
                batchnorm='saved unchanged' if name.startswith('saved') else 'training-only calibrated diagnostic clone',
                normalization='Saved model-specific action statistics',action_stats=stats,
                population=result['population']+'; '+split,protocol='Matched diagnostic modes; '+name,
                code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
            write_json(directory/'prediction.json',metrics)
            print(json.dumps(dict(mode=name,split=split,**metrics)),flush=True)
    if not calibration_only:
        for split,ds,indices in [('training',train,train_indices),('validation',val,val_indices)]:
            result['batch_statistics'][split]=batch_statistics(model,batches(ds,indices),stats,device,model.encoder.config.image_size)
    if sha256(checkpoint)!=digest or state_digest(model)!=initial_state:raise RuntimeError('Source checkpoint/model changed')
    result['checkpoint_unchanged']=True
    if save_calibrated is not None:
        if reference is not None:raise ValueError('Released clone export requires its own training-compatible manifest')
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        directory=Path(save_calibrated);directory.mkdir(parents=True,exist_ok=False)
        clone={k:saved[k] for k in ('fingerprint','fingerprint_version','model_config','action_stats','step','validation_step') if k in saved}
        clone.update(model={k:v.cpu() for k,v in calibrated.state_dict().items()},diagnostic_only=True,
                     parent_checkpoint_sha256=digest,calibration=result['calibration'])
        torch.save(clone,directory/'checkpoint.pt')
        write_json(directory/'manifest.json',{**meta,'diagnostic_only':True,
            'calibration_parent_checkpoint':str(checkpoint),'calibration_parent_sha256':digest,
            'calibration_receipt':str(output),'calibration_source_rows':train_rows})
        result['calibrated_checkpoint']=str(directory/'checkpoint.pt');result['calibrated_checkpoint_sha256']=sha256(directory/'checkpoint.pt')
    write_json(output,result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run');p.add_argument('--output')
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--windows',type=int,default=512)
    p.add_argument('--calibration-windows',type=int,default=512);p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--calibration-only',action='store_true',help='Export the same calibrated clone without auxiliary mode probes')
    p.add_argument('--checkpoint');p.add_argument('--reference');p.add_argument('--save-calibrated')
    args=p.parse_args();check(args.run,args.output,device=args.device,windows=args.windows,
        calibration_windows=args.calibration_windows,batch_size=args.batch_size,checkpoint=args.checkpoint,
        reference=args.reference,save_calibrated=args.save_calibrated,calibration_only=args.calibration_only)
