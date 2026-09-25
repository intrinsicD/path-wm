"""Compare invocation-local K/V reuse on trained native image producers.

    python -m experiments.context_cache --output runs/native_context_cache_v1
"""
import argparse
import json
import os
import subprocess
import shutil
import statistics
import time
from pathlib import Path
from unittest.mock import patch

import torch
from pathwm.data import rule_world as rw
from pathwm.io import atomic_json, environment, file_hash, source_record, state_hash
from pathwm.evaluation.report import write_report
from pathwm.models import conditional_image as ci
from pathwm.models.image_code import load_image_code, decode_image_code
from pathwm.models.slots import pointer
from experiments.unified_session import load_decoder, load_edit_run, paired_requests, _generated


def request_inputs(decoder, population, seed):
    """Saved observed codes; native slots selected at the original machine locations.

    Direct percept binding is sufficient for a cache equivalence benchmark; this
    deliberately makes no new memory-retrieval or task-quality claim.
    """
    path = Path(f"runs/native_code_refinement_confirmation_v1/{population}-0-0.code.pt")
    code = load_image_code(path)
    percept = decode_image_code(decoder, code)
    g = torch.Generator().manual_seed(seed)
    kinds = torch.tensor(rw.KIND_SPLIT[population])[torch.randint(len(rw.KIND_SPLIT[population]), (64, 2), generator=g)]
    scenes = rw.sample_scenes(g, kinds)
    slots = torch.stack([pointer(percept.alpha, scenes.machine_xy[:16, m]) for m in range(2)], 1)
    source, machine, state = (x.cuda() for x in paired_requests(16))
    return code['fine']['values'].cuda()[source], percept.slots[source, slots[source, machine]], state, path


def generated(generator, context, sample_id, cached, trace=None):
    original = ci.integrate
    def capture(field, noise, steps):
        def record(x, t):
            velocity = field(x, t)
            trace.append((x['fine'].detach().cpu(), velocity['fine'].detach().cpu()))
            return velocity
        return original(record, noise, steps)
    kwargs = dict(sample_ids=[sample_id] * len(context), cache_context=cached)
    if trace is None:
        return generator.features(context, **kwargs)['fine']
    with patch.object(ci, 'integrate', capture):
        return generator.features(context, **kwargs)['fine']


def difference(a, b):
    return dict(max_abs=(a - b).abs().max().item(), close=torch.allclose(a, b, atol=2e-5, rtol=2e-5))


@torch.no_grad()
def run(out):
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.perf_counter()
    def budget():
        if time.perf_counter() - started > 900:
            raise TimeoutError('Cache benchmark exceeded 900 seconds')
        if torch.cuda.max_memory_reserved() > 6 * 1024**3:
            raise MemoryError('Cache benchmark exceeded 6 GiB')
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out/'status.json', dict(result='running', report='pending', step=0, error=None))
    decoder = load_decoder('runs/native_decoder_subpixel_3601_v1').cuda().eval()
    source = source_record(__file__, decoder)
    helper = str(Path('experiments/unified_session.py').resolve())
    source['files'][helper] = file_hash(helper)
    settings = dict(threads=2, matmul_tf32=False, cudnn_tf32=False, seeds=[3711, 3712], steps=16, samples=[0, 1], images_per_population=16,
                    batch_sizes=[1, 8], warmups=5, repeats=30, atol=2e-5, rtol=2e-5,
                    rgb_max_abs=2e-5, max_seconds=900, max_reserved_gib=6,
                    direct_percept_binding=True, new_task_quality_claim=False)
    atomic_json(out/'run.json', dict(schema='pathwm-run-v1', identity=dict(settings=settings, environment=environment('cuda')), source=source))
    shutil.copyfile(__file__, out/'recipe.py')
    rows, checks, timings, evidence = [], {}, {}, {}
    inputs = [request_inputs(decoder, pop, seed) for pop, seed in [('train', 3602), ('validation', 3603)]]
    evidence['inputs'] = {str(v[3]): file_hash(v[3]) for v in inputs}
    examples = None
    equivalence_peaks = {}
    def gpu_processes():
        raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader'], text=True)
        if any(int(line.split(',')[0]) != os.getpid() for line in raw.splitlines() if line.strip()):
            raise RuntimeError('Concurrent GPU job invalidates timing')
        return raw
    evidence['gpu_before'] = gpu_processes()
    try:
        for seed in settings['seeds']:
            edit_run = Path(f'runs/native_memory_edit_{seed}_flow_v1')
            model, edit_settings = load_edit_run(edit_run, decoder, 'runs/real_visual_joint_repair_3501_u6000_v1')
            assert edit_settings['edit_objective'] == 'flow' and edit_settings['edit_steps'] == 16
            model = model.cuda().eval()
            checkpoint = edit_run/'last.pt'
            evidence[str(checkpoint)] = file_hash(checkpoint)
            before = (state_hash(model), state_hash(decoder))
            rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
            for population, (values, slots, states, _) in zip(['train', 'validation'], inputs):
                context = model['request'].context(values, slots, states)
                for offset in range(0, len(context), 8):
                    for sample_id in settings['samples']:
                        budget()
                        ctx, src = context[offset:offset+8], values[offset:offset+8]
                        traces = [[], []]
                        deltas = [generated(model['generator'], ctx, sample_id, bool(k), traces[k]) for k in range(2)]
                        comparisons = [difference(a, b) for left, right in zip(*traces) for a, b in zip(left, right)]
                        codes = [src + x.flatten(2).transpose(1, 2) for x in deltas]
                        comparison = difference(*codes)
                        images = [_generated(decoder, x, 'cache-comparison') for x in codes]
                        rgb = difference(*images)
                        passed = len(traces[0]) == len(traces[1]) == 16 and all(v['close'] for v in comparisons) and comparison['close'] and rgb['max_abs'] <= 2e-5
                        row = dict(split='equivalence', phase='equivalence', seed=seed, population=population, offset=offset, sample_id=sample_id,
                                   passed=passed, trajectory_max_abs=max(v['max_abs'] for v in comparisons), code=comparison, rgb=rgb)
                        rows.append(row)
                        checks[f'{seed}_{population}_{offset}_{sample_id}'] = passed
                        if examples is None:
                            examples = [x[:4].cpu() for x in images]
                if seed == 3711 and population == 'train':
                    timing_context, timing_source = context, values
            equivalence_peaks[str(seed)] = torch.cuda.max_memory_reserved()
            budget()
            checks[f'{seed}_frozen'] = before == (state_hash(model), state_hash(decoder))
            checks[f'{seed}_rng'] = torch.equal(rng, torch.get_rng_state()) and torch.equal(cuda_rng, torch.cuda.get_rng_state())
            if seed != 3711:
                continue
            for batch in settings['batch_sizes']:
                ctx, src = timing_context[:batch], timing_source[:batch]
                evidence[f'gpu_timing_{batch}_before'] = gpu_processes()
                prepared = model['generator']._context(ctx, None)
                blocks = [*model['generator'].blocks['fine'], *model['generator'].fusion]
                pairs = [block._context_kv(prepared[0]) for block in blocks]
                kv_bytes = sum(t.numel()*t.element_size() for pair in pairs for t in pair)
                prepared_bytes = sum(t.numel()*t.element_size() for t in prepared)
                del prepared, pairs

                for complete in [False, True]:
                    def call(cached):
                        delta = generated(model['generator'], ctx, 0, cached)
                        return _generated(decoder, src + delta.flatten(2).transpose(1, 2), 'cache-timing') if complete else delta
                    for _ in range(5):
                        call(False); call(True)
                    measured = {False: [], True: []}
                    memory = {}
                    for cached in [False, True]:
                        torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                        call(cached); torch.cuda.synchronize()
                        memory[str(cached)] = dict(allocated=torch.cuda.max_memory_allocated(), reserved=torch.cuda.max_memory_reserved())
                    for repeat in range(30):
                        budget()
                        for cached in ([False, True] if repeat % 2 == 0 else [True, False]):
                            torch.cuda.synchronize(); t = time.perf_counter()
                            call(cached); torch.cuda.synchronize()
                            seconds = time.perf_counter() - t
                            measured[cached].append(seconds)
                            rows.append(dict(split='timing', phase='timing', batch=batch, complete=complete, repeat=repeat, cached=cached, seconds=seconds))
                    base, cache = (statistics.median(measured[k]) for k in [False, True])
                    timings[f'{batch}_{complete}'] = dict(uncached_seconds=base, cached_seconds=cache, ratio=cache/base, memory=memory,
                                                         kv_bytes=kv_bytes, prepared_bytes=prepared_bytes, total_cache_bytes=kv_bytes+prepared_bytes)
                evidence[f'gpu_timing_{batch}_after'] = gpu_processes()
            checks['timing_frozen'] = before == (state_hash(model), state_hash(decoder))
            checks['timing_rng'] = torch.equal(rng, torch.get_rng_state()) and torch.equal(cuda_rng, torch.cuda.get_rng_state())
        checks['decoder_final_frozen'] = before[1] == state_hash(decoder)
        checks['source_unchanged'] = all(file_hash(f) == h for f, h in source['files'].items())
        checks['time_budget'] = time.perf_counter()-started <= 900
        checks['memory_budget'] = max(equivalence_peaks.values()) <= 6*1024**3 and all(m['reserved'] <= 6*1024**3 for v in timings.values() for m in v['memory'].values())
        ratios = [timings[f'{b}_False']['ratio'] for b in [1,8]]
        result = dict(valid=all(checks.values()), checks=checks, timings=timings, evidence=evidence, equivalence_peak_reserved=equivalence_peaks,
                      default_gate=all(checks.values()) and min(ratios)<=.95 and max(ratios)<=1.05,
                      seconds=time.perf_counter()-started,
                      limitations=['Cache numerical/performance validation only; no generation-quality claim.',
                                   'Native full configuration; direct percept bindings, not a new memory evaluation.',
                                   'Existing report renderer; structural QA.'])
        atomic_json(out/'result.json', result)
        (out/'metrics.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
        atomic_json(out/'status.json', dict(result='completed', report='pending', step=16, error=None))
    except Exception as error:
        (out/'metrics.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
        atomic_json(out/'result.json', dict(valid=False, error=str(error), checks=checks, timings=timings))
        atomic_json(out/'status.json', dict(result='failed', report='incomplete', step=0, error=str(error)))
        raise
    try:
        write_report(out, batch=dict(rgb=examples[0]), outputs=dict(cached_rgb=examples[1]))
    except Exception as error:
        atomic_json(out/'status.json', dict(result='completed', report='failed', step=16, error=str(error)))
        raise
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
