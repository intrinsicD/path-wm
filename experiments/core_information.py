"""Nonlinguistic property diagnostics; see integrated architecture plan CI1.

python -m experiments.core_information --perception RUN/last.pt --core RUN/last.pt \
    --data data/core_information --output runs/core_information --device cuda
No runtime oracle, no resume; fresh directories protect completed evidence.
"""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from pathwm.data import rule_world as rw
from pathwm.models.slots import SlotPerception, pointer, perception_loss, match_slots
from pathwm.models.latent_core import LatentCore, ROLE_A
from pathwm.io import atomic_json, atomic_torch, environment, file_hash, load_component, seed_everything, source_record, state_hash
from pathwm.evaluation.report import write_report


def make_batch(seed, count, heldout=False):
    g = torch.Generator().manual_seed(seed)
    kinds = torch.tensor(rw.KIND_SPLIT['validation' if heldout else 'train'])
    pair = kinds[torch.randint(len(kinds), (count, 2), generator=g)]
    attrs = torch.randint(4, (count, 4, 4), generator=g)
    # Both marginal color and shape remain uniform; only joint combinations differ.
    offset = torch.zeros(count, 4, dtype=torch.long) if heldout else torch.randint(1, 4, (count, 4), generator=g)
    attrs[:,:,1] = (offset-attrs[:,:,0])%4
    scenes = rw.sample_scenes(g, pair, attrs)
    lamps = torch.randint(2, (count, 2), generator=g)
    target = torch.randint(4, (count,), generator=g)
    return pack(scenes, lamps, target)


def pack(scenes, lamps, target):
    rgb, entity = rw.render(scenes, lamps)
    xy=scenes.object_xy[torch.arange(len(target)),target].long()
    if not torch.equal(entity[torch.arange(len(target)),xy[:,1],xy[:,0]],target+3):
        raise ValueError('query coordinate does not identify intended entity')
    rows = torch.arange(len(target))
    return dict(scenes=scenes, lamps=lamps, target=target, rgb=rgb, entity=entity,
                xy=scenes.object_xy[rows,target], y=scenes.attrs[rows,target,0])


def counterfactual(batch, kind):
    s = rw.Scenes(*(x.clone() for x in vars(batch['scenes']).values()))
    rows, target = torch.arange(len(s)), batch['target']
    if kind == 'color':
        s.attrs[rows,target,0] = (s.attrs[rows,target,0]+1)%4
    elif kind == 'motion':
        s.object_xy[rows,target,1] += 3
    else:
        raise ValueError(kind)
    return pack(s, batch['lamps'].clone(), target.clone())


class PropertyReader(nn.Module):
    """Diagnostic new query through existing tied block, not old lamp decision."""
    def __init__(self, adapt=False):
        super().__init__()
        self.perception = SlotPerception()
        self.core = LatentCore()
        self.query = nn.Parameter(torch.randn(1,1,64)*.02)
        self.color = nn.Linear(64,4)
        self.requires_grad_(False)
        self.core.block.requires_grad_(True)
        self.core.head_norm.requires_grad_(True)
        self.query.requires_grad_(True)
        self.color.requires_grad_(True)
        self.perception.encoder.requires_grad_(adapt)
        self.adapt = adapt

    def read(self, token):
        x = self.query.expand(len(token),-1,-1)
        context = (token+self.core.types.weight[ROLE_A])[:,None]
        for _ in range(self.core.loops):
            x = self.core.block(x, context)
        return self.color(self.core.head_norm(x[:,0]))

    def encode(self, rgb, xy):
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.adapt):
            p = self.perception(rgb)
        slot = pointer(p.alpha, xy)
        token = p.slots[torch.arange(len(rgb),device=rgb.device),slot]
        return token, p

    def forward(self, rgb, xy):
        token, p = self.encode(rgb, xy)
        return self.read(token), p


def claude_paired_metrics(base_logits, color_logits, motion_logits, base_targets, color_targets):
    """The color intervention should move the answer to color_targets; the motion intervention should
    keep base_targets. Pair rates require BOTH halves correct: a consistently wrong model is
    'invariant' but scores 0. Pair chance depends on prediction dependence; motion_pair <= base_acc.
    Returns (summary, raw); the caller stores the per-example raw arrays."""
    pb, pc, pm = (t.detach().float().cpu().softmax(-1) for t in (base_logits, color_logits, motion_logits))
    yb, yc = base_targets.detach().long().cpu(), color_targets.detach().long().cpu()
    at = lambda p, y: p.gather(1, y[:, None]).squeeze(1)
    ok_b, ok_c, ok_m = pb.argmax(-1) == yb, pc.argmax(-1) == yc, pm.argmax(-1) == yb
    changed = yc != yb  # no-op color edits are excluded from color metrics
    raw = {"base_correct": ok_b, "color_correct": ok_c, "motion_correct": ok_m, "color_changed": changed,
           "color_gain_new": at(pc, yc) - at(pb, yc), "color_drop_old": at(pb, yb) - at(pc, yb),
           "motion_shift_true": at(pm, yb) - at(pb, yb),
           "color_tv": 0.5 * (pc - pb).abs().sum(-1), "motion_tv": 0.5 * (pm - pb).abs().sum(-1)}

    def mean(v, mask=None):
        v = v if mask is None else v[mask]
        return v.double().mean().item() if v.numel() else float("nan")

    summary = {"n": len(yb), "n_color_changed": int(changed.sum()),
               "base_acc": mean(ok_b), "color_acc": mean(ok_c, changed),
               "color_pair_rate": mean(ok_b & ok_c, changed),
               "motion_acc": mean(ok_m), "motion_pair_rate": mean(ok_b & ok_m),
               "motion_argmax_stable_NOT_success": mean(pm.argmax(-1) == pb.argmax(-1)),
               "color_gain_new": mean(raw["color_gain_new"], changed),
               "color_drop_old": mean(raw["color_drop_old"], changed),
               "motion_shift_true": mean(raw["motion_shift_true"]),
               "color_tv": mean(raw["color_tv"], changed), "motion_tv": mean(raw["motion_tv"])}
    return summary, raw

def paired_metrics(base,color,motion,target,color_target):
    m,_ = claude_paired_metrics(base,color,motion,target,color_target)
    names={'base_acc':'accuracy','color_acc':'color_accuracy','color_pair_rate':'color_pair','motion_acc':'motion_accuracy','motion_pair_rate':'motion_pair'}
    return {names.get(k,k):v for k,v in m.items()}


def accuracy_screen(m):
    return m['accuracy'] >= .95 and m['color_pair'] >= .90 and m['motion_pair'] >= .90


@torch.no_grad()
def extract(model, batch, device, chunk=16):
    result = dict(token=[], logits=[], old_attributes=[], old_lamp=[], rgb_mean=[], binding=[])
    for start in range(0,len(batch['y']),chunk):
        sl = slice(start,start+chunk)
        rgb, xy = batch['rgb'][sl].to(device), batch['xy'][sl].to(device)
        token, p = model.encode(rgb,xy)
        result['token'].append(token.cpu())
        assign=match_slots(p.alpha,batch['entity'][sl].to(device))
        chosen=pointer(p.alpha,xy)
        result['binding'].append((assign[torch.arange(len(rgb),device=device),chosen].cpu()==batch['target'][sl]+3))
        result['logits'].append(model.read(token).cpu())
        attrs=[]
        for i in range(4):
            ptr=pointer(p.alpha,batch['scenes'].object_xy[sl,i].to(device))
            attrs.append(p.attributes[torch.arange(len(rgb),device=device),ptr].argmax(-1).cpu())
        result['old_attributes'].append(torch.stack(attrs,1))
        lamps=[]
        for i in range(2):
            ptr=pointer(p.alpha,batch['scenes'].machine_xy[sl,i].to(device))
            lamps.append((p.lamp[torch.arange(len(rgb),device=device),ptr]>0).long().cpu())
        result['old_lamp'].append(torch.stack(lamps,1))
        patches=[]
        for im,pos in zip(rgb,xy.long()):
            x,y=pos.tolist();patches.append(im[:,y-2:y+3,x-2:x+3].mean((1,2)).cpu())
        result['rgb_mean'].append(torch.stack(patches))
    return {k:torch.cat(v) for k,v in result.items()}


def retention(data,batch):
    m={f'attribute_{i}':float((data['old_attributes'][:,:,i]==batch['scenes'].attrs[:,:,i]).float().mean()) for i in range(4)}
    m['pointer']=float(data['binding'].float().mean())
    m['lamp']=float((data['old_lamp']==batch['lamps']).float().mean())
    return m


# CPU probe authored by Claude Opus 5.5 high; local adapter below.
class LinearProbe(nn.Module):
    """Keeps its train-set standardization, so it applies unchanged to counterfactual features."""

    def __init__(self, mu, sd, n_classes):
        super().__init__()
        self.register_buffer("mu", mu)
        self.register_buffer("sd", sd)
        self.weight = nn.Parameter(torch.zeros(n_classes, mu.numel(), dtype=torch.float64))
        self.bias = nn.Parameter(torch.zeros(n_classes, dtype=torch.float64))

    def forward(self, x):
        z = (x.detach().to("cpu", torch.float64).flatten(1) - self.mu) / self.sd
        return F.linear(z, self.weight, self.bias)

def claude_fit_probe(train_x, train_y, eval_x, eval_y, steps=300, seed=0, *, n_classes=4,
              lr=0.05, weight_decay=1e-3, shuffle_labels=False, eval_every=25):
    """Full-batch float64 CPU, zero init (convex, no RNG). The seed is used only by the
    shuffled-label control. Tune hyperparameters on a split other than eval_x."""
    X = train_x.detach().to("cpu", torch.float64).flatten(1)
    y, ey = train_y.detach().long().cpu(), eval_y.detach().long().cpu()
    if shuffle_labels:
        y = y[torch.randperm(len(y), generator=torch.Generator().manual_seed(int(seed)))]
    probe = LinearProbe(X.mean(0), X.std(0).clamp_min(1e-6), n_classes)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    majority = torch.bincount(ey, minlength=n_classes).max().item() / len(ey)
    history = []
    for step in range(steps):
        loss = F.cross_entropy(probe(X), y) + weight_decay * probe.weight.square().sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (step + 1) % eval_every == 0 or step + 1 == steps:
            with torch.no_grad():
                history.append({"step": step, "loss": loss.item(), "eval_majority": majority,
                                "train_acc": (probe(X).argmax(-1) == y).double().mean().item(),
                                "eval_acc": (probe(eval_x).argmax(-1) == ey).double().mean().item()})
    return probe.eval(), history

def fit_probe(x,y,steps=300,shuffle_labels=False):
    return claude_fit_probe(x,y,x,y,steps=steps,seed=6101,lr=.03,shuffle_labels=shuffle_labels)


def setup(path, args, model, arm):
    path.mkdir(parents=True,exist_ok=False)
    settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    atomic_json(path/'run.json',dict(identity=dict(settings={**settings,'arm':arm},environment=environment(args.device)),
                source=source_record(__file__,model),checkpoints={k:file_hash(getattr(args,k)) for k in ('perception','core')}))
    (path/'recipe.py').write_text(Path(__file__).read_text())
    (path/'metrics.jsonl').write_text('')
    atomic_json(path/'status.json',dict(result='running',report='pending',step=0,error=None))


def complete(path,result,step):
    atomic_json(path/'result.json',result)
    atomic_json(path/'status.json',dict(result='completed',report='pending',step=step,error=None))
    try:
        write_report(path)
    except Exception as e:
        atomic_json(path/'status.json',dict(result='completed',report='failed',step=step,error=str(e)))
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--perception',required=True,type=Path);p.add_argument('--core',required=True,type=Path)
    p.add_argument('--data',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--device',default='cuda')
    p.add_argument('--steps',type=int,default=400);p.add_argument('--batch',type=int,default=16)
    p.add_argument('--eval-scenes',type=int,default=256);p.add_argument('--probe-scenes',type=int,default=512)
    p.add_argument('--probe-steps',type=int,default=300)
    args=p.parse_args();seed_everything(1101)
    if args.output.exists() or args.data.exists():raise FileExistsError('output and data must be fresh')
    args.output.mkdir(parents=True)
    args.data.mkdir(parents=True)
    initial=PropertyReader()
    load_component(initial.perception,args.perception,'perception');load_component(initial.core,args.core,'core')
    batch=make_batch(4201,args.eval_scenes,True)
    panels=[batch,counterfactual(batch,'color'),counterfactual(batch,'motion')]
    model=copy.deepcopy(initial).to(args.device).eval()
    base=[extract(model,b,args.device) for b in panels]
    old=retention(base[0],batch)
    train=make_batch(4101,args.probe_scenes)
    dataset={}
    for name,b in [('probe_train',train),*zip(('base','color','motion'),panels)]:
        for k in ('rgb','entity','xy','y','target','lamps'):
            dataset[name+'_'+k]=b[k].numpy()
        for k,v in vars(b['scenes']).items():dataset[name+'_scene_'+k]=v.numpy()
    np.savez_compressed(args.data/'dataset.npz',**dataset)
    del dataset
    # Materialize the exact inputs once; both arms read these same saved batches.
    (args.data/'train').mkdir()
    train_hashes={}
    for step in range(1,args.steps+1):
        b=make_batch(5101+step,args.batch)
        file=args.data/'train'/f'{step:05d}.pt'
        atomic_torch(file,{**{k:b[k] for k in ('rgb','entity','xy','y','lamps')},'attrs':b['scenes'].attrs})
        train_hashes[file.name]=file_hash(file)
    atomic_json(args.data/'dataset.json',dict(train_files=train_hashes,sha256=file_hash(args.data/'dataset.npz'),train_step_seeds=[5101+i for i in range(1,args.steps+1)],schema='CI1-v1',scope='synthetic static pairs; no temporal tracking',split='(color+shape)%4==0 held out; color edits can cross this split'))
    train_data=extract(model,train,args.device)
    probe_path=args.output/'probes';setup(probe_path,args,model,'probes')
    probe_metrics={};arrays={};checkpoints={}
    for field in ('token','rgb_mean'):
        probe,history=fit_probe(train_data[field],train['y'],args.probe_steps)
        shuffled,control_history=fit_probe(train_data[field],train['y'],args.probe_steps,True)
        with torch.no_grad():logits=[probe(d[field]) for d in base]
        probe_metrics[field]=paired_metrics(*logits,batch['y'],panels[1]['y'])
        probe_metrics[field]['screen']=accuracy_screen(probe_metrics[field])
        with torch.no_grad():
            probe_metrics[field]['shuffled_label_accuracy']=float((shuffled(base[0][field]).argmax(-1)==batch['y']).float().mean())
        probe_metrics[field]['train_accuracy']=history[-1]['train_acc']
        atomic_json(probe_path/(field+'_history.json'),dict(real=history,shuffled=control_history))
        checkpoints[field]=probe.state_dict()
        checkpoints[field+'_shuffled']=shuffled.state_dict()
        for i,l in enumerate(logits):arrays[f'{field}_{i}']=l.numpy()
    arrays.update(target=batch['y'].numpy(),color_target=panels[1]['y'].numpy(),train_token=train_data['token'].numpy(),train_target=train['y'].numpy(),eval_token=base[0]['token'].numpy())
    np.savez_compressed(probe_path/'raw.npz',**arrays);atomic_torch(probe_path/'last.pt',checkpoints)
    complete(probe_path,dict(evaluation_scope='CI1 frozen checkpoint inputs, one seed, synthetic property task; composition holdout applies only to this diagnostic, not checkpoint pretraining',metrics=probe_metrics,initial_retention=old),args.probe_steps)
    del model,train_data
    if args.device.startswith('cuda'):torch.cuda.empty_cache()
    summary={'probes':probe_metrics}
    for arm,adapt in [('frozen',False),('adaptive',True)]:
        model=copy.deepcopy(initial).to(args.device)
        model.adapt=adapt;model.perception.encoder.requires_grad_(adapt)
        path=args.output/arm;setup(path,args,model,arm)
        initial_hash=state_hash(model)
        buffers={n:v.detach().cpu().clone() for n,v in model.named_buffers()}
        frozen={n:v.detach().cpu().clone() for n,v in model.named_parameters() if not v.requires_grad}
        opt=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=3e-4)
        if args.device.startswith('cuda'):torch.cuda.reset_peak_memory_stats()
        start=time.monotonic();model.train()
        for step in range(1,args.steps+1):
            train_file=args.data/'train'/f'{step:05d}.pt'
            if file_hash(train_file)!=train_hashes[train_file.name]:raise RuntimeError('training data changed')
            b=torch.load(train_file,weights_only=True)
            logits,percept=model(b['rgb'].to(args.device),b['xy'].to(args.device))
            prior,_=perception_loss(percept,b['rgb'].to(args.device),b['entity'].to(args.device),b['attrs'].to(args.device),b['lamps'].to(args.device))
            ce=F.cross_entropy(logits,b['y'].to(args.device));loss=ce+.2*prior
            if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
            opt.zero_grad(set_to_none=True);loss.backward();
            grads={name:sum(float(v.grad.detach().square().sum()) for v in module.parameters() if v.grad is not None)**.5 for name,module in [('encoder',model.perception.encoder),('core',model.core.block)]}
            if any(v.grad is not None for v in model.parameters() if not v.requires_grad):raise RuntimeError('frozen gradient')
            if not all(np.isfinite(v) for v in grads.values()):raise RuntimeError('nonfinite gradient')
            opt.step()
            if step==1 or step%25==0:
                row=dict(step=step,split='train',loss=float(loss.detach()),color_ce=float(ce.detach()),retention_loss=float(prior.detach()),**{k+'_gradient_norm':v for k,v in grads.items()})
                with (path/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                print(arm,row,flush=True)
            reserved=torch.cuda.max_memory_reserved()/2**30 if args.device.startswith('cuda') else 0
            if reserved>6 or time.monotonic()-start>1800:raise RuntimeError('declared resource budget exceeded')
        elapsed=time.monotonic()-start;model.eval()
        before_eval=state_hash(model)
        data=[extract(model,b,args.device) for b in panels]
        metrics=paired_metrics(*(d['logits'] for d in data),batch['y'],panels[1]['y'])
        with torch.no_grad():
            tok=data[0]['token'].to(args.device)
            controls={'zero':model.read(torch.zeros_like(tok)).cpu(),'shuffled':model.read(tok.roll(1,0)).cpu()}
        for name,l in controls.items():metrics[name+'_accuracy']=float((l.argmax(-1)==batch['y']).float().mean())
        if state_hash(model)!=before_eval:raise RuntimeError('evaluation changed model')
        now=retention(data[0],batch);retain=all(now[k]>=old[k]-.02 for k in old)
        screen=accuracy_screen(metrics) and all(metrics['accuracy']-metrics[n+'_accuracy']>=.30 for n in controls)
        unchanged=all(torch.equal(v,dict(model.named_parameters())[n].detach().cpu()) for n,v in frozen.items())
        unchanged=unchanged and all(torch.equal(v,dict(model.named_buffers())[n].detach().cpu()) for n,v in buffers.items())
        if not unchanged:raise RuntimeError('frozen parameter changed')
        # Fresh probe on final representation, using training-only examples.
        final_train=extract(model,train,args.device)
        final_probe,probe_history=fit_probe(final_train['token'],train['y'],args.probe_steps)
        with torch.no_grad(): final_probe_logits=[final_probe(d['token']) for d in data]
        final_probe_metrics=paired_metrics(*final_probe_logits,batch['y'],panels[1]['y'])
        atomic_json(path/'final_probe.json',dict(metrics=final_probe_metrics,history=probe_history))
        atomic_torch(path/'final_probe.pt',final_probe.state_dict())
        arrays={f'{k}_{i}':v.numpy() for i,d in enumerate(data) for k,v in d.items()}
        arrays.update({f'final_probe_{i}':v.numpy() for i,v in enumerate(final_probe_logits)})
        arrays.update({f'control_{k}':v.numpy() for k,v in controls.items()});arrays.update(target=batch['y'].numpy(),color_target=panels[1]['y'].numpy())
        np.savez_compressed(path/'raw.npz',**arrays)
        atomic_torch(path/'last.pt',dict(model=model.state_dict(),optimizer=opt.state_dict(),step=args.steps,torch_rng=torch.get_rng_state()))
        result=dict(evaluation_scope='CI1 newly trained property query through tied core; not original lamp policy, no temporal tracking; composition holdout applies only to this diagnostic, not checkpoint pretraining',gate=screen and retain,property_gate=screen,metrics={'property':metrics,'retention':now},retention_pass=retain,final_probe=final_probe_metrics,initial_retention=old,frozen_unchanged=unchanged,initial_hash=initial_hash,elapsed_seconds=elapsed,peak_reserved_gib=reserved)
        complete(path,result,args.steps);summary[arm]=result
        del model,opt
        if args.device.startswith('cuda'):torch.cuda.empty_cache()
    atomic_json(args.output/'comparison.json',summary)


if __name__=='__main__':main()
