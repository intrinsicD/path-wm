"""Frozen nonlinear output diagnostic; see docs/nonlinear-fidelity-plan.md.

python -m experiments.nonlinear_fidelity --checkpoint runs/evidence_loop_final_s17/last.pt --output runs/my_fidelity
The target projection is an evaluator-only oracle, never an inference arm.
"""
import argparse
import json
from pathlib import Path
import resource
import time

import torch
from experiments.representation_transfer import load_codec, encode, decode, pixels, fit_affine, apply_affine
from pathwm.data.detail_views import sample_tiles, target_view
from pathwm.io import atomic_json, atomic_torch, environment, file_hash, seed_everything, source_record, state_hash, digest
from pathwm.evaluation.report import write_report


def finite_matrix(x):
    if not isinstance(x, torch.Tensor) or x.ndim != 2 or not x.numel() or not x.is_floating_point() or not torch.isfinite(x).all():
        raise ValueError('Expected nonempty finite floating matrix')


def range_basis(weight, bias):
    finite_matrix(weight)
    if bias.shape != (weight.shape[0],) or not bias.is_floating_point() or not torch.isfinite(bias).all() or bias.device != weight.device:
        raise ValueError('Bias must match output dimension and device')
    w, b = weight.detach().double(), bias.detach().double()
    u, s, vh = torch.linalg.svd(w, full_matrices=False)
    tol = max(w.shape)*torch.finfo(w.dtype).eps*s[0]
    rank = int((s > tol).sum())
    return dict(weight=w, bias=b, u=u[:, :rank], singular=s, vh=vh[:rank], rank=rank, tolerance=float(tol))


def project_basis(basis, target):
    """Evaluator only: unrestricted latent optimum in the unclipped affine range."""
    finite_matrix(target)
    w, b, u = (basis[k] for k in ('weight', 'bias', 'u'))
    if target.shape[1] != len(b) or target.device != w.device:
        raise ValueError('Target must match output dimension and device')
    centered = target.detach().double()-b
    coordinates = centered @ u
    prediction = coordinates @ u.T+b
    codes = (coordinates/basis['singular'][:basis['rank']]) @ basis['vh']
    return dict(prediction=prediction, codes=codes,
                normal_residual=float(((target-prediction) @ w).abs().max()))


def project_range(weight, bias, target):
    return project_basis(range_basis(weight, bias), target)


def fit_threshold(x, y):
    """Strict monotone rule, identified only on a support-derived interval."""
    finite_matrix(x); finite_matrix(y)
    if x.shape != y.shape or x.device != y.device or not ((y == 0) | (y == 1)).all():
        raise ValueError('Matching matrices and binary labels required')
    if not ((y == 0).any(0) & (y == 1).any(0)).all():
        raise ValueError('Both classes required per channel')
    lower = x.double().masked_fill(y != 0, -torch.inf).max(0).values
    upper = x.double().masked_fill(y != 1, torch.inf).min(0).values
    if not (lower < upper).all():
        raise ValueError('Inconsistent monotone support')
    middle = lower/2+upper/2
    if not ((lower <= middle) & (middle < upper)).all():
        raise ValueError("Threshold midpoint is not representable")
    return torch.stack([lower, upper])


def apply_threshold(x, bounds):
    finite_matrix(x); finite_matrix(bounds)
    if bounds.shape != (2, x.shape[1]) or bounds.device != x.device or not (bounds[0] < bounds[1]).all():
        raise ValueError('Valid channel bounds required')
    middle = bounds[0].double()/2+bounds[1].double()/2
    if not ((bounds[0] <= middle) & (middle < bounds[1])).all():
        raise ValueError('Threshold midpoint is not representable')
    return (x.double() > middle).to(x.dtype)


def threshold_image(x, bounds):
    return apply_threshold(pixels(x), bounds).reshape(-1,16,16,3).permute(0,3,1,2)


@torch.no_grad()
def infer(model, support, labels, query):
    """Support only; query targets are not accepted or generated here."""
    sx, sy = encode(model, support).flatten(0,1), encode(model, labels).flatten(0,1)
    qx = encode(model, query)
    code = fit_affine(sx, sy)
    bounds = fit_threshold(pixels(support), pixels(labels))
    nearest = torch.cdist(query.flatten(1).double(), support.flatten(1).double()).argmin(1)
    predictions = dict(nearest_raw=labels[nearest],
        latent_ridge=decode(model, apply_affine(qx.flatten(0,1), code).reshape(len(query),4,model.width)),
        raw_threshold=threshold_image(query, bounds),
        reconstructed_threshold=threshold_image(decode(model, qx), bounds))
    try:
        shuffled = fit_threshold(pixels(support), pixels(labels.roll(1,0)))
        shuffled_control = dict(status='fit', bounds=shuffled, prediction=threshold_image(query, shuffled))
    except ValueError as e:
        shuffled_control = dict(status='rejected', reason=str(e))
    return dict(predictions=predictions, bounds=bounds, map=code, nearest=nearest, shuffled=shuffled_control)


def scores(prediction, target):
    error = (prediction.double()-target.double()).square()
    per_image = error.flatten(1).mean(1)
    truth, estimate = target.bool(), prediction > .5
    rates = [(estimate[truth] != truth[truth]).double().mean(),
             (estimate[~truth] != truth[~truth]).double().mean()]
    return dict(mse=float(per_image.mean()), p95_image_mse=float(torch.quantile(per_image,.95)),
                balanced_pixel_error=float(torch.stack(rates).mean()) if truth.any() and (~truth).any() else None,
                per_image_mse=per_image.tolist())


def relations(final):
    return {'threshold_a': (.3,.45,.6), 'threshold_b': (.55,.35,.5)} if final else {'threshold_dev': (.4,.4,.4)}


def target_for(x, thresholds):
    return (x > torch.tensor(thresholds, dtype=x.dtype)[None,:,None,None]).to(x.dtype)


@torch.no_grad()
def execute(output, checkpoint, *, seed=926101, count=64, support=64, final=False):
    if final and (seed not in (926117,926129) or count != 64 or support != 64):
        raise ValueError('Final requires registered seed and sample counts')
    if not final and seed != 926101:
        raise ValueError('Use registered development population')
    if min(count,support) < 2:
        raise ValueError('At least two support/query images required')
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    start, cpu = time.perf_counter(), time.process_time()
    seed_everything(seed)
    model = load_codec(checkpoint)
    if model.variant != 'linear':
        raise ValueError('The range diagnostic requires the affine decoder')
    version = state_hash(model)
    settings = dict(seed=seed,count=count,support=support,final=final,checkpoint=str(checkpoint),
        checkpoint_sha256=file_hash(checkpoint),codec_version=version,relations=relations(final),
        purpose='frozen nonlinear output diagnostic',
        example_labels={'input':'Binary target (evaluator only)', 'rgb':'Optimal affine-range projection (oracle)'})
    source = source_record(__file__, model)
    helper = str(Path('experiments/representation_transfer.py').resolve())
    source['files'][helper] = file_hash(helper); source['sha256'] = digest(source['files'])
    atomic_json(output/'run.json',dict(identity=dict(settings=settings,environment=environment('cpu')), source=source))
    for i, filename in enumerate(source['files']):
        dest = output/'source'/f'{i:03d}_{Path(filename).name}'; dest.parent.mkdir(exist_ok=True); dest.write_bytes(Path(filename).read_bytes())
    (output/'metrics.jsonl').write_text('')
    atomic_json(output/'status.json',dict(result='running',report='pending',step=0))
    try:
        basis = range_basis(model.mean.weight[:768], model.mean.bias[:768])
        # Independent Householder basis; no truncated-range blocker claims.
        qr_basis = torch.linalg.qr(basis['weight'], mode='reduced').Q
        g = torch.Generator().manual_seed(seed)
        old = sample_tiles(g,count); old_codes = encode(model,old)
        old_outputs = torch.stack([decode(model,old_codes,k) for k in range(4)])
        old_mse = [float((old_outputs[k]-target_view(old,torch.full((count,),k,dtype=torch.long))).square().mean()) for k in range(4)]
        identity = project_basis(basis,old.flatten(1))
        constants = {str(v): project_basis(basis,torch.full((1,768),float(v))) for v in (0,1)}
        raw = dict(basis=basis,retention=dict(images=old,outputs=old_outputs),identity=identity,constants=constants)
        metrics, gates, integrity = {}, {}, {}
        # Check actual flattening/head selection, not just assumed affine algebra.
        probes = torch.randn(3,4,model.width,generator=g)
        actual = decode(model,probes).flatten(1).double()
        affine_error = float((actual-(probes.flatten(1).double() @ basis['weight'].T+basis['bias'])).abs().max())
        for name, thresholds in relations(final).items():
            s, q = sample_tiles(g,support), sample_tiles(g,count)
            sy = target_for(s,thresholds)
            fitted = infer(model,s,sy,q)  # No query target yet exists in this path.
            y = target_for(q,thresholds)
            projection = project_basis(basis,y.flatten(1))
            encoded = decode(model,encode(model,y))
            oracle = projection['prediction'].reshape_as(y)
            native = decode(model,projection['codes'].reshape(count,4,model.width))
            # Independent algorithm; requires full column rank for gels.
            if basis['rank'] != basis['weight'].shape[1]:
                raise ValueError('Checkpoint decoder is rank deficient; independent gels audit unavailable')
            ls = torch.linalg.lstsq(basis['weight'],(y.flatten(1).double()-basis['bias']).T,driver='gels').solution.T
            ls_prediction = ls @ basis['weight'].T+basis['bias']
            predictions = {**fitted['predictions'],'target_encode_decode':encoded,'range_oracle':oracle,'native_range_oracle':native}
            m = {arm: scores(pred,y) for arm,pred in predictions.items()}
            ls_error = abs(float((ls_prediction-y.flatten(1)).square().mean())-m['range_oracle']['mse'])
            residual = y.flatten(1).double()-projection['prediction']
            qr_gap = float((residual @ qr_basis).square().sum(1).mean()/768)
            qr_prediction = (y.flatten(1).double()-basis['bias']) @ qr_basis @ qr_basis.T+basis['bias']
            qr_difference = float((qr_prediction-projection['prediction']).abs().max())
            certificate = (basis['rank']==basis['weight'].shape[1]
                and float(basis['singular'][-1])>100*basis['tolerance']
                and qr_gap<=1e-12 and qr_difference<=1e-8 and ls_error<=1e-9)
            gate = dict(numerical_certificate=certificate,decoder_range_quality=m['range_oracle']['mse']<=.002,
                latent_inference_quality=m['latent_ridge']['mse']<=.002,
                raw_identifiability=m['raw_threshold']['mse']<=.001,
                normal_residual=projection['normal_residual']<=1e-8,
                independent_lstsq=ls_error<=1e-9,
                projection_below_encoded=m['range_oracle']['mse']<=m['target_encode_decode']['mse']+1e-9)
            raw[name] = dict(support=s,support_target=sy,query=q,target=y,fitted=fitted,
                predictions=predictions,projection=projection,least_squares_prediction=ls_prediction)
            metrics[name] = dict(arms=m,normal_residual=projection['normal_residual'],lstsq_mse_difference=ls_error,
                qr_residual_gap=qr_gap,qr_prediction_max_difference=qr_difference,
                minimum_norm_code_l2=projection['codes'].norm(dim=1).tolist(),
                conservative_range_mse=m['range_oracle']['mse']-1e-8,
                native_max_difference=float((native.double()-oracle).abs().max()),
                foreground_fraction_by_channel=y.mean((0,2,3)).tolist(),
                shuffled_status=fitted['shuffled']['status'],
                threshold_interval_widths=(fitted['bounds'][1]-fitted['bounds'][0]).tolist(),
                threshold_state_bytes=fitted['bounds'].numel()*fitted['bounds'].element_size(),
                map_bytes=fitted['map'].numel()*fitted['map'].element_size(),
                raw_example_bytes=(s.numel()+sy.numel())*s.element_size())
            gates[name] = gate
            for arm, values in m.items():
                with (output/'metrics.jsonl').open('a') as f:
                    f.write(json.dumps(dict(step=0,split=f'{name}/{arm}',**values))+'\n')
            if time.perf_counter()-start>120: raise RuntimeError('Wall budget exceeded')
        atomic_torch(output/'raw.pt',raw)
        atomic_torch(output/'last.pt',dict(model=model.state_dict(),settings=settings,source=source,
            phase='evaluation_complete',neural_updates=0,
            states={n:{k:raw[n]['fitted'][k] for k in ('bounds','map')} for n in relations(final)}))
        saved = torch.load(output/'last.pt',map_location='cpu',weights_only=True)
        reloaded = load_codec(checkpoint); reloaded.load_state_dict(saved['model'])
        restart = True
        for name in relations(final):
            q, state = raw[name]['query'], saved['states'][name]
            restart &= torch.equal(threshold_image(q,state['bounds']),raw[name]['predictions']['raw_threshold'])
            z = apply_affine(encode(reloaded,q).flatten(0,1),state['map']).reshape(count,4,model.width)
            restart &= torch.equal(decode(reloaded,z),raw[name]['predictions']['latent_ridge'])
        integrity = dict(weights_unchanged=state_hash(model)==version==state_hash(reloaded),restart_exact=restart,
            old_outputs_exact=all(torch.equal(old_outputs[k],decode(model,encode(model,old),k)) for k in range(4)),
            familiar_quality=all(m<=.001 for m in old_mse),actual_decoder_affine=affine_error<=1e-5,
            identity_projection_below_encoded=float((identity['prediction']-old.flatten(1)).square().mean())<=old_mse[0]+1e-9)
        result = dict(metrics=metrics,gates=gates,integrity=integrity,
            gate=all(all(v.values()) for v in gates.values()) and all(integrity.values()),
            decoder_blocked=any(v['numerical_certificate'] and metrics[n]['conservative_range_mse']>.002 for n,v in gates.items()),
            evaluation_scope='Fixed unclipped affine decoder output range; target oracle is not inference. Supplied identity; threshold-family prior; no neural updates or general concept acquisition.',
            rank=basis['rank'],rank_tolerance=basis['tolerance'],singular_values=basis['singular'].tolist(),
            familiar_mse=old_mse,affine_check_max_error=affine_error,
            constant_target_mse={k:float((v['prediction']-float(k)).square().mean()) for k,v in constants.items()},
            parameters=sum(p.numel() for p in model.parameters()),neural_updates=0,
            prior_training='Inherited codec: 3000 mean + 600 variance updates per seed; sunk cost, not zero-cost learning')
        atomic_json(output/'result.json',result)
        atomic_json(output/'status.json',dict(result='completed',report='pending',step=0))
        try: write_report(output,batch={'rgb':y[:8]},outputs={'rgb':oracle[:8].float()})
        except Exception as e:
            atomic_json(output/'status.json',dict(result='completed',report='failed',step=0,error=str(e))); raise
        wall = time.perf_counter()-start
        atomic_json(output/'timing.json',dict(wall_seconds=wall,cpu_seconds=time.process_time()-cpu,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            wall_budget_pass=wall<=120,rss_budget_pass=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=2*1024**3,
            includes='Loading, all comparisons, persistence and report; process wrapper separately includes imports'))
        print(json.dumps(dict(gate=result['gate'],decoder_blocked=result['decoder_blocked'],gates=gates,integrity=integrity,seconds=wall)))
        return result
    except Exception as e:
        status = json.loads((output/'status.json').read_text())
        if status['result'] != 'completed':
            atomic_json(output/'status.json',dict(result='failed',report='pending',step=0,error=str(e)))
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=926101); p.add_argument('--count',type=int,default=64)
    p.add_argument('--support',type=int,default=64); p.add_argument('--final',action='store_true')
    args = p.parse_args()
    execute(args.output,args.checkpoint,seed=args.seed,count=args.count,support=args.support,final=args.final)


if __name__ == '__main__': main()
