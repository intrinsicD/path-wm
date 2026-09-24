"""Frozen detail-code affine transfer diagnostic; no learned-induction claim.

python -m experiments.representation_transfer --checkpoint runs/evidence_loop_final_s17/last.pt --output runs/my_transfer
See docs/representation-transfer-plan.md for gates, populations and limits.
"""
import argparse
import json
import math
from pathlib import Path
import resource
import time

import torch
from torch import nn
from pathwm.data.detail_views import sample_tiles, target_view
from pathwm.models.detail_memory import DetailCodec, to_parts
from pathwm.io import (atomic_json, atomic_torch, environment, file_hash,
                       seed_everything, source_record, state_hash)
from pathwm.evaluation.report import write_report
from pathwm.models.belief import BeliefAgent, BeliefCorrection, BeliefDynamics
from pathwm.models.agent import Thinker, ActionHead, ErrorMonitor
from pathwm.models.hybrid_memory import HybridMemory
from pathwm.world_state.modules import AssociationBinder, ReplaceUpdater, ContextEncoder
from pathwm.world_state.session import WorldSession
from pathwm.world_state.episodes import EpisodeClient
from pathwm.world_state.context import WorkingContext

RIDGE = .001
ARMS = ('identity', 'nearest', 'displacement', 'ridge', 'shuffled', 'pixel_ridge')


def _matrix(x):
    if not isinstance(x, torch.Tensor) or x.ndim != 2 or not x.numel() or not x.is_floating_point() or not torch.isfinite(x).all():
        raise ValueError('Expected a nonempty finite floating matrix')


def fit_affine(x, y, *, ridge=RIDGE):
    """Claude public reference adapted: sum-form ridge, unpenalized intercept.

    Returns [input_width+1, output_width] float64, intercept in the last row.
    All statistics belong to support only. No neural weight updates.
    """
    _matrix(x); _matrix(y)
    if len(x) != len(y) or x.device != y.device or not math.isfinite(ridge) or ridge <= 0:
        raise ValueError('Matching rows/device and finite positive ridge required')
    x, y = x.detach().double(), y.detach().double()
    xm, ym = x.mean(0), y.mean(0)
    xc, yc = x-xm, y-ym
    weight = torch.linalg.solve(xc.T @ xc + ridge * torch.eye(x.shape[1], dtype=x.dtype, device=x.device), xc.T @ yc)
    code = torch.cat([weight, (ym-xm @ weight)[None]], 0)
    if not torch.isfinite(code).all():
        raise ValueError('Nonfinite fitted map')
    return code


def apply_affine(x, code):
    _matrix(x); _matrix(code)
    if x.shape[1]+1 != len(code) or x.device != code.device:
        raise ValueError('Incompatible query and affine code')
    return x.detach().double() @ code[:-1].double() + code[-1].double()


def transform(rgb, relation):
    """Evaluator/data generator only; never called by an inference arm."""
    if relation == 'cycle': return rgb[:, [1, 2, 0]]
    if relation == 'mix_cycle': return .5*rgb + .5*rgb[:, [1, 2, 0]]
    if relation == 'reverse': return rgb[:, [2, 1, 0]]
    if relation == 'mix_reverse': return .7*rgb + .3*rgb[:, [2, 1, 0]]
    if relation == 'threshold': return (rgb > .4).to(rgb.dtype)
    raise ValueError('Unknown relation')


def diagnostics(x):
    x = x.double()-x.double().mean(0)
    s = torch.linalg.svdvals(x)
    rank = int((s > max(x.shape)*torch.finfo(x.dtype).eps*s[0]).sum())
    return dict(rows=len(x), width=x.shape[1], rank=rank, sigma_max=float(s[0]),
                sigma_min=float(s[-1]), condition=float(s[0]/s[-1]) if s[-1] > 0 else None,
                ridge_to_mean_eigenvalue=RIDGE/max(float(s.square().sum()/x.shape[1]), 1e-300),
                ridge_to_min_eigenvalue=RIDGE/max(float(s[-1].square()), 1e-300))


def _session(version):
    # Existing generic owners; this shell is not a trained belief/language model.
    contracts = {n: ('affine-state', version) for n in ('map', 'unrelated')}
    agent = BeliefAgent(width=16, context_tokens=4, latent_groups=8, latent_codes=8,
        evidence_tokens=4, encoders={}, decoders={}, updater=BeliefCorrection(16,8,8,2),
        dynamics=BeliefDynamics(16,8,8,2), thinker=Thinker(16),
        memory=HybridMemory(16,recent=2,block=2,blocks=1), action_head=ActionHead(16), monitor=ErrorMonitor(16)).eval()
    binder = AssociationBinder(); binder.scorer.eval()
    modules = dict(agent=agent, binder=binder, updater=ReplaceUpdater(1).eval(),
        context_encoder=ContextEncoder(16,{n: nn.Linear(1,16) for n in contracts},contracts).eval())
    return modules, WorldSession(**modules), contracts


def memory_check(x, y, q, version, directory):
    """Live session integration on supplied features, separate from population fit.

    One source batch is replaced explicitly, not autonomously judged false. Raw
    image evidence is in the run's raw.pt; this software path uses feature inputs.
    """
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    modules, session, contracts = _session(version)
    client = EpisodeClient(session, representations=contracts)
    client.create('create', 'relation', kind='concept-candidate')
    wrong = y.clone(); wrong[:len(y)//2] += .5
    proof = client.observe('initial', 'relation', source='supplied-features', modality='tensor',
        occurred_at=1., available_at=1., data=dict(x=x.tolist(), y=wrong.tolist()))
    original = fit_affine(x, wrong)
    cid = client.publish('fit-initial', 'relation', 'map', original, evidence=(proof,))
    keep = client.observe('separate', 'relation', source='supplied-features', modality='tensor',
        occurred_at=2., available_at=2., data=dict(value=7))
    untouched = client.publish('keep', 'relation', 'unrelated', torch.tensor([7.]), evidence=(keep,))
    old = client.load('relation', ['map'], max_values=original.numel())
    context = WorkingContext(session, representations=contracts, owner='transfer')
    context.retain(cid, task='predict')
    corrected = client.observe('replacement', 'relation', source='supplied-features', modality='tensor',
        occurred_at=3., available_at=3., supersedes=proof, data=dict(x=x.tolist(), y=y.tolist()))
    checks = {}
    for name, action in [('stale_episode', lambda: client.validate(old)),
                          ('stale_context', lambda: context.read('predict', max_values=original.numel()))]:
        try: action(); checks[name] = False
        except ValueError: checks[name] = True
    checks['invalidated_map'] = client.load('relation', ['map']).omitted == ('map',)
    source = next(e for e in session.store.evidence() if e.id == corrected).data['detail']
    rebuilt = fit_affine(torch.tensor(source['x'], dtype=x.dtype), torch.tensor(source['y'], dtype=y.dtype))
    checks['corrected_from_scratch'] = torch.equal(rebuilt, fit_affine(x, y))
    checks['correction_changes_prediction'] = not torch.equal(apply_affine(q, original), apply_affine(q, rebuilt))
    cid = client.publish('refit', 'relation', 'map', rebuilt, evidence=(corrected,), data=dict(ridge=RIDGE))
    context.reset(); context.retain(cid, task='predict')
    selected = context.read('predict', max_values=rebuilt.numel())[0].tensor(dtype=torch.float64)
    checks['stored_map_exact'] = torch.equal(selected, rebuilt)
    checks['unrelated_preserved'] = session.store.latest('relation','unrelated').id == untouched
    before = apply_affine(q, selected)
    atomic_torch(directory/'session.pt', session.snapshot())
    atomic_json(directory/'context.json', context.snapshot())
    restored = WorldSession.restore(torch.load(directory/'session.pt', weights_only=True), **modules)
    ctx = WorkingContext.restore(restored, json.loads((directory/'context.json').read_text()), owner='transfer', representations=contracts)
    after = ctx.read('predict', max_values=rebuilt.numel())[0].tensor(dtype=torch.float64)
    checks['restart_map_exact'] = torch.equal(selected, after)
    checks['restart_prediction_exact'] = torch.equal(before, apply_affine(q, after))
    incompatible = WorkingContext(restored, representations={n: ('affine-state','different') for n in contracts})
    try: incompatible.retain(cid, task='predict'); checks['wrong_version_rejected'] = False
    except ValueError: checks['wrong_version_rejected'] = True
    atomic_json(directory/'checks.json', checks)
    return checks


def load_codec(checkpoint):
    saved = torch.load(checkpoint, weights_only=True, map_location='cpu')
    if saved['phase'] != 'complete' or saved['step'] != saved['settings']['steps']:
        raise ValueError('A completed codec checkpoint is required')
    model = DetailCodec(hidden=saved['settings']['hidden'], variant=saved['settings']['variant'])
    model.load_state_dict(saved['model'])
    return model.eval().requires_grad_(False)


def encode(model, rgb):
    return model.encode(to_parts(rgb), torch.ones(len(rgb),4,dtype=torch.bool))


def decode(model, codes, pose=0):
    return model.decode(codes.float(), torch.ones(len(codes),4,dtype=torch.bool), torch.full((len(codes),),pose,dtype=torch.long))[0]


def pixels(rgb):
    return rgb.permute(0,2,3,1).reshape(-1,3)


def predict(arm, x, y, q, code, pixel_code, query_rgb):
    if arm == 'pixel_ridge':
        return apply_affine(pixels(query_rgb), pixel_code).reshape(-1,16,16,3).permute(0,3,1,2).float()
    if arm == 'identity': return q
    if arm == 'nearest': return y[torch.cdist(q.double(), x.double()).argmin(1)]
    if arm == 'displacement': return q + (y-x).mean(0)
    if arm in ('ridge', 'shuffled'): return apply_affine(q, code)
    raise ValueError('Unknown arm')


@torch.no_grad()
def execute(output, checkpoint, *, seed=925101, count=64, support=64, final=False):
    if final and (seed not in (925117,925129) or count != 64 or support != 64):
        raise ValueError('Final requires the registered population and sample counts')
    if not final and seed != 925101:
        raise ValueError('Use the registered development population')
    if count < 2 or support < 2:
        raise ValueError('At least two support/query tiles required')
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    start, cpu_start = time.perf_counter(), time.process_time()
    seed_everything(seed)
    model = load_codec(checkpoint); version = state_hash(model)
    settings = dict(seed=seed, count=count, support=support, final=final, ridge=RIDGE,
        checkpoint=str(checkpoint), checkpoint_sha256=file_hash(checkpoint), codec_version=version,
        scope='Frozen codec affine hypothesis-class screen; supplied identity/grouping; no neural training or general induction',
        example_labels=dict(input='True target',rgb='Inferred latent affine prediction'))
    source = source_record(__file__, model)
    atomic_json(output/'run.json', dict(identity=dict(settings=settings,environment=environment('cpu')),source=source))
    (output/'recipe.py').write_text(Path(__file__).read_text())
    for i, filename in enumerate(source['files']):
        dest=output/'source'/f'{i:03d}_{Path(filename).name}'; dest.parent.mkdir(exist_ok=True); dest.write_bytes(Path(filename).read_bytes())
    (output/'metrics.jsonl').write_text('')
    atomic_json(output/'status.json',dict(result='running',report='pending',step=0))
    raw, metrics, gates, timings = {}, {}, {}, {}
    try:
        g = torch.Generator().manual_seed(seed)
        relations = ('reverse','mix_reverse','threshold') if final else ('cycle','mix_cycle','threshold')
        old_images = sample_tiles(g, count)
        old_codes = encode(model, old_images)
        old_recon = [decode(model,old_codes,k) for k in range(4)]
        reconstruction = [float((old_recon[k]-target_view(old_images,torch.full((count,),k,dtype=torch.long))).square().mean()) for k in range(4)]
        raw['retention'] = dict(images=old_images,codes=old_codes,reconstruction=torch.stack(old_recon))
        for relation in relations:
            t=time.perf_counter()
            s,q,d = [sample_tiles(g,n) for n in (support,count,count)]
            sy,qy,dy = [transform(a,relation) for a in (s,q,d)]
            sx,sz = encode(model,s).flatten(0,1), encode(model,sy).flatten(0,1)
            qx,dx = encode(model,q).flatten(0,1), encode(model,d).flatten(0,1)
            fit_start=time.perf_counter()
            code=fit_affine(sx,sz)
            # Derange whole source images so no shuffled pair shares an instance.
            shuffled=fit_affine(sx,sz.reshape(support,4,-1).roll(1,0).flatten(0,1))
            pc=fit_affine(pixels(s),pixels(sy))
            timings[relation]=dict(preparation_seconds=fit_start-t,fit_seconds=time.perf_counter()-fit_start)
            floor=decode(model,encode(model,qy))
            negative_relation=relations[1] if relation == relations[0] else relations[0]
            candidates=torch.stack([dy,transform(d,negative_relation),dy.roll(1,0)],1)
            record=dict(instance_ids={role: [f'{seed}:{relation}:{role}:{i}' for i in range(n)]
                        for role,n in (('support',support),('query',count),('discrimination',count))},
                        support=s,support_target=sy,query=q,target=qy,discrimination=d,
                        candidates=candidates,latent_map=code,shuffled_map=shuffled,pixel_map=pc,
                        source_codes=sx,target_codes=sz,reconstruction_floor=floor,predictions={},choices={})
            m=dict(floor_mse=float((floor-qy).square().mean()),input_min=float(s.min()),input_max=float(s.max()),
                threshold_fraction=float((s>.4).float().mean()),latent_fit=diagnostics(sx),pixel_fit=diagnostics(pixels(s)),
                arms={},map_bytes=code.numel()*code.element_size(),example_code_bytes=(sx.numel()+sz.numel())*sx.element_size(),
                pixel_map_bytes=pc.numel()*pc.element_size())
            pred_start=time.perf_counter()
            for arm in ARMS:
                p=predict(arm,sx,sz,qx,shuffled if arm=='shuffled' else code,pc,q)
                dp=predict(arm,sx,sz,dx,shuffled if arm=='shuffled' else code,pc,d)
                if arm != 'pixel_ridge':
                    p,dp=[decode(model,a.reshape(count,4,model.width)) for a in (p,dp)]
                errors=(p-qy).square().flatten(1).mean(1)
                distances=(dp[:,None]-candidates).square().flatten(2).mean(2)
                # Strict winning margin: ties cannot earn a correct answer.
                if not torch.isfinite(distances).all() or not torch.isfinite(errors).all():
                    raise ValueError('Nonfinite prediction or discrimination distance')
                correct=(distances[:,0] < distances[:,1:].min(1).values)
                reversed_distances=(dp[:,None]-candidates.flip(1)).square().flatten(2).mean(2)
                if not torch.equal(correct,reversed_distances[:,-1] < reversed_distances[:,:-1].min(1).values):
                    raise ValueError('Candidate order changed scoring')
                m['arms'][arm]=dict(mse=float(errors.mean()),excess_mse=float(errors.mean())-m['floor_mse'],
                    discrimination=float(correct.float().mean()),
                    ties=int((distances[:,0]==distances[:,1:].min(1).values).sum()),
                    near_duplicate_candidates=int(((candidates[:,1:]-candidates[:,:1]).square().flatten(2).mean(2)<1e-10).any(1).sum()))
                record['predictions'][arm]=p
                record['choices'][arm]=dict(distances=distances,correct=correct,prediction=dp)
                with (output/'metrics.jsonl').open('a') as f:
                    f.write(json.dumps(dict(step=0,split=f'{relation}/{arm}',loss=float(errors.mean()),**m['arms'][arm]))+'\n')
            timings[relation]['prediction_seconds']=time.perf_counter()-pred_start
            if relation != 'threshold':
                a=m['arms']; gates[relation]=dict(quality=a['ridge']['mse']<=.002,
                    beats_examples=a['ridge']['mse']<=.5*min(a['nearest']['mse'],a['displacement']['mse']),
                    support_dependence=a['ridge']['mse']<=.5*a['shuffled']['mse'],
                    discrimination=a['ridge']['discrimination']>=.9,
                    reconstruction_floor=m['floor_mse']<=.001,pixel_reference=a['pixel_ridge']['mse']<=1e-5)
            raw[relation],metrics[relation]=record,m
            if relation==relations[0]:
                software=memory_check(sx,sz,qx,version,output/'memory')
                examples=(qy[:8],record['predictions']['ridge'][:8])
            if time.perf_counter()-start>120:
                raise RuntimeError('Declared wall budget exceeded')
        integrity=dict(weights_unchanged=state_hash(model)==version,
            reconstruction_unchanged=all(torch.equal(a,decode(model,encode(model,old_images),k)) for k,a in enumerate(old_recon)),
            **software)
        passed=all(all(v.values()) for v in gates.values()) and all(integrity.values()) and all(v<=.001 for v in reconstruction)
        atomic_torch(output/'raw.pt',raw)
        atomic_torch(output/'last.pt',dict(model=model.state_dict(),settings=settings,source=source,
            maps={r:raw[r]['latent_map'] for r in relations},phase='evaluation_complete',neural_updates=0))
        result=dict(metrics=metrics,gates=gates,integrity=integrity,passed=passed,gate=passed,
            evaluation_scope=settings['scope'],
            reconstruction_by_pose=reconstruction,scope=settings['scope'],timings=timings,
            neural_updates=0,parameters=sum(p.numel() for p in model.parameters()),
            parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
            prior_training='Inherited checkpoints: 3000 mean + 600 variance updates per seed; not charged as new runtime work',
            metric_route='Population inference on tensors; separate live EpisodeClient/WorkingContext correction and restart demonstration')
        atomic_json(output/'result.json',result)
        atomic_json(output/'status.json',dict(result='completed',report='pending',step=0))
        try: write_report(output,batch={'rgb':examples[0]},outputs={'rgb':examples[1]})
        except Exception as e:
            atomic_json(output/'status.json',dict(result='completed',report='failed',step=0,error=str(e))); raise
        elapsed=time.perf_counter()-start
        atomic_json(output/'timing.json',dict(wall_seconds=elapsed,cpu_seconds=time.process_time()-cpu_start,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            includes='load, generation, encoding, fitting, all baselines, persistence, raw saves and report rendering',
            wall_budget_pass=elapsed<=120,rss_budget_pass=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=2*1024**3))
        print(json.dumps(dict(passed=passed,gates=gates,integrity=integrity,seconds=elapsed)),flush=True)
        return result
    except Exception as e:
        status=json.loads((output/'status.json').read_text())
        if status['result']!='completed':
            atomic_json(output/'status.json',dict(result='failed',report='pending',step=0,error=str(e)))
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--seed',type=int,default=925101)
    p.add_argument('--count',type=int,default=64)
    p.add_argument('--support',type=int,default=64)
    p.add_argument('--final',action='store_true')
    args=p.parse_args()
    execute(args.output,args.checkpoint,seed=args.seed,count=args.count,support=args.support,final=args.final)


if __name__=='__main__': main()
