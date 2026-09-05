"""One batch-128 backward on a discarded clone; no optimizer step or training run."""
import json
from pathlib import Path
import time
import torch
from torch.utils.data import DataLoader
from world_model.data import TrajectoryDataset,preprocess_pixels,normalize_actions
from world_model.model import build_model
from world_model.training import backward_batch
from third_party.lewm.module import SIGReg
from world_model.train import write_json
from scripts.diagnose_pusht_control import sha256


def check():
    out=Path('runs/diagnostics/pusht_control_diagnosis/reproduction_memory.json')
    if out.exists(): raise FileExistsError(out)
    checkpoint=Path('runs/diagnostics/pusht_broader_pilot/checkpoint_001000.pt')
    digest=sha256(checkpoint)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    meta=json.loads((checkpoint.parent/'manifest.json').read_text())
    ds=TrajectoryDataset(meta['dataset']['path'],meta['train_episodes'])
    indices=torch.randperm(len(ds),generator=torch.Generator().manual_seed(9201))[:128].tolist()
    batch=next(iter(DataLoader(ds,batch_size=128,sampler=indices,num_workers=0)))
    torch.set_num_threads(4);torch.manual_seed(9202)
    model=build_model(saved['model_config'])
    model.load_state_dict(saved['model'],strict=True)
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model=model.cuda().train();reg=SIGReg(knots=17,num_proj=1024).cuda()
    torch.cuda.reset_peak_memory_stats();started=time.monotonic()
    result=dict(checkpoint_sha256=digest,batch_size=128,precision='bf16',encoder_chunk=0,
                encoder_gradient_checkpointing=True,window_indices=indices,optimizer_steps=0,
                limitation='One full-size backward on a clone; excludes optimizer state and long-run source I/O.')
    try:
        terms=backward_batch(model,preprocess_pixels(batch['pixels'].cuda()),
                             normalize_actions(batch['action'].cuda(),saved['action_stats']),reg,
                             encoder_chunk=0,precision='bf16')
        torch.cuda.synchronize()
        result.update(status='completed',terms={k:float(v) for k,v in terms.items()},
                      all_gradients_finite=all(p.grad is not None and torch.isfinite(p.grad).all().item()
                                              for p in model.parameters()),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(),seconds=time.monotonic()-started)
    except torch.OutOfMemoryError as error:
        result.update(status='cuda_out_of_memory',error=str(error),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),seconds=time.monotonic()-started)
    result['checkpoint_unchanged']=sha256(checkpoint)==digest
    write_json(out,result)
    print(json.dumps({k:v for k,v in result.items() if k!='window_indices'}),flush=True)
    if result['status']!='completed' or not result['checkpoint_unchanged']:
        raise RuntimeError('Reproduction memory preflight did not pass; raw diagnostic preserved')


if __name__=='__main__':check()
