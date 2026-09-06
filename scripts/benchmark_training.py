"""Bounded full-batch throughput/memory probe on a discarded training clone."""
import argparse
import json
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from world_model.data import TrajectoryDataset, preprocess_pixels, normalize_actions
from world_model.model import build_model
from world_model.objective import SIGReg
from world_model.training import backward_batch
from world_model.train import write_json
from scripts.diagnose_pusht_control import sha256


def benchmark(mode, output):
    output = Path(output)
    if output.exists(): raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = Path('runs/diagnostics/pusht_source_10min/checkpoint_000000.pt')
    digest = sha256(checkpoint)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    meta = json.loads((checkpoint.parent / 'manifest.json').read_text())
    ds = TrajectoryDataset(meta['dataset']['path'], meta['train_episodes'])
    indices = torch.randperm(len(ds), generator=torch.Generator().manual_seed(9201))[:128].tolist()
    batch = next(iter(DataLoader(ds, batch_size=128, sampler=indices, num_workers=0)))
    torch.set_num_threads(4); torch.manual_seed(9202)
    model = build_model(saved['model_config']).cuda().train()
    model.load_state_dict(saved['model'], strict=True)
    if mode == 'checkpointed':
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    reg = SIGReg(knots=17, num_proj=1024).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=.001)
    torch.cuda.reset_peak_memory_stats()
    result = dict(mode=mode, batch_size=128, precision='bf16', encoder_chunk=0,
        checkpoint_sha256=digest, window_indices=indices, optimizer_steps=0,
        scope='Discarded clone; two warmup and five timed updates on one repeated source batch. Includes preprocessing and optimizer state; excludes data loading.')
    durations = []
    try:
        for i in range(7):
            torch.cuda.synchronize(); started = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            pixels = preprocess_pixels(batch['pixels'].cuda())
            actions = normalize_actions(batch['action'].cuda(), saved['action_stats'])
            terms = backward_batch(model, pixels, actions, reg, encoder_chunk=0, precision='bf16')
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            if not all(torch.isfinite(v) for v in terms.values()): raise FloatingPointError('Non-finite loss')
            optimizer.step(); result['optimizer_steps'] += 1
            torch.cuda.synchronize()
            if i >= 2: durations.append(time.monotonic()-started)
        result.update(status='completed', timed_seconds=durations,
            median_step_seconds=float(torch.tensor(durations).median()),
            all_gradients_finite=all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters()),
            final_terms={k:float(v) for k,v in terms.items()}, grad_norm=float(norm))
    except torch.OutOfMemoryError as error:
        result.update(status='cuda_out_of_memory', error=str(error))
    result.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(), checkpoint_unchanged=sha256(checkpoint)==digest)
    write_json(output, result)
    print(json.dumps({k:v for k,v in result.items() if k!='window_indices'}), flush=True)
    if not result['checkpoint_unchanged']: raise RuntimeError('Reference checkpoint changed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['native', 'checkpointed'])
    parser.add_argument('output')
    args = parser.parse_args(); benchmark(args.mode, args.output)
