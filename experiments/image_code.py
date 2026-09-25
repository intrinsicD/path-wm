"""Native image-code replay in a fresh process, using the existing report renderer.

Verify: python -m experiments.image_code --checkpoint RUN/last.pt --output NEW_RUN
Decode: python -m experiments.image_code --checkpoint RUN/last.pt --code CODE.pt --decoded OUT.pt
The decode process accepts only weights and a code, never RGB or a source store.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import torch
import torch.utils.serialization.config  # initialize runtime code before auditing data reads
from pathwm.io import atomic_json, atomic_torch, environment, file_hash, load_component, seed_everything, source_record, state_hash
from pathwm.models.slots import SlotPerception
from pathwm.models.image_code import make_image_code, save_image_code, load_image_code, code_pyramid


def model(checkpoint, device):
    p=SlotPerception().eval()
    p.decoder.enable_pyramid_connections()
    load_component(p,checkpoint,'perception')
    return p.to(device).eval().requires_grad_(False)


def tensor_hash(*values):
    h=hashlib.sha256()
    for v in values:
        h.update(f'{v.dtype}{tuple(v.shape)}'.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def decode(args):
    reads=[]
    def audit(event,values):
        if event=='open' and isinstance(values[0],(str,bytes)):
            mode=values[1]
            if mode is not None and 'r' in mode:reads.append(str(values[0]))
    sys.addaudithook(audit)
    p=model(args.checkpoint,args.device)
    code=load_image_code(args.code)
    before=state_hash(p)
    def forbidden(*a,**kw):raise AssertionError('RGB encoding forbidden during code-only replay')
    p.encoder.forward=forbidden
    p.pyramid=forbidden
    rng=torch.get_rng_state().clone()
    cuda=torch.cuda.get_rng_state().clone() if args.device=='cuda' else None
    pyramid=code_pyramid(p,code)
    percept=p.from_pyramid(pyramid)
    assert torch.equal(rng,torch.get_rng_state())
    assert cuda is None or torch.equal(cuda,torch.cuda.get_rng_state())
    assert before==state_hash(p)
    atomic_torch(args.decoded,dict(coarse=pyramid.scales[1].values.cpu(),**{k:v.cpu() for k,v in vars(percept).items()}))
    atomic_json(str(args.decoded)+'.reads.json',dict(read_paths=reads,weights_unchanged=True,rng_unchanged=True,encoder_forward_forbidden=True))


@torch.no_grad()
def verify(args):
    from pathwm.data import rule_world as rw
    from pathwm.evaluation.report import write_report
    out=Path(args.output);out.mkdir(exist_ok=False)
    p=model(args.checkpoint,args.device)
    source=source_record(__file__,p)
    settings=dict(stage='native-image-code-replay',device=args.device,checkpoint=str(args.checkpoint),checkpoint_sha256=file_hash(args.checkpoint),
                  configuration='SlotPerception64/7/3/32, native fine16x16x64, coarse8x8x64',
                  seeds=dict(train=3602,validation=3603),scenes_per_population=64,
                  scope='First original64-image evaluation chunk per population; fresh-process exact software replay, not fidelity qualification')
    atomic_json(out/'run.json',dict(schema='pathwm-run-v1',identity=dict(settings=settings,environment=environment(args.device)),source=source))
    atomic_json(out/'status.json',dict(result='running',report='pending',step=0,error=None))
    original=json.loads((Path(args.checkpoint).parent/'evaluation_final.json').read_text())
    checks={};metrics={};images=None
    for population,seed in settings['seeds'].items():
        g=torch.Generator().manual_seed(seed);kinds=rw.KIND_SPLIT[population]
        picked=torch.tensor(kinds)[torch.randint(len(kinds),(64,2),generator=g)]
        scene=rw.sample_scenes(g,picked);lamps=torch.randint(2,(64,2),generator=g)
        scene=rw.Scenes(*(v.to(args.device) for v in vars(scene).values()))
        rgb,entity=rw.render(scene,lamps.to(args.device))
        sha=tensor_hash(rgb,entity)
        checks[population+'_same_inputs']=sha==original['summary']['inputs'][population]['chunk_sha256'][0]
        pyramid=p.pyramid(rgb);expected=p.from_pyramid(pyramid)
        code=make_image_code(p,pyramid,provenance=[dict(kind='observed',id=f'{population}/{i}',available_at=0.) for i in range(64)])
        codepath=out/f'{population}.code.pt';save_image_code(codepath,code)
        decoded=out/f'{population}.decoded.pt'
        command=[sys.executable,'-m','experiments.image_code','--checkpoint',str(args.checkpoint),'--device',args.device,'--code',str(codepath),'--decoded',str(decoded)]
        subprocess.run(command,check=True)
        result=torch.load(decoded,map_location='cpu',weights_only=True)
        checks[population+'_coarse_exact']=torch.equal(result.pop('coarse'),pyramid.scales[1].values.cpu())
        for k,v in result.items():checks[population+'_'+k+'_exact']=torch.equal(v,getattr(expected,k).cpu())
        mse=(result['recon'].to(args.device)-rgb).square().mean((1,2,3)).tolist()
        checks[population+'_original_mse']=mse==original['per_image']['true'][population]['mse'][:64]
        reads=json.loads(Path(str(decoded)+'.reads.json').read_text())
        checks[population+'_code_only']=all(Path(f).resolve() in (Path(args.checkpoint).resolve(),codepath.resolve()) for f in reads['read_paths'])
        metrics[population]=dict(mse=sum(mse)/len(mse),per_image_mse=mse,input_sha256=sha,canonical_values=64*256*64,coarse_values=64*64*64,read_receipt=reads)
        if population=='validation':images=(rgb[:8].cpu(),result['recon'][:8])
    checks['source_unchanged']=all(file_hash(k)==v for k,v in source['files'].items())
    result=dict(gate=all(checks.values()),checks=checks,metrics=metrics,limitations=['Software replay only; inherited decoder pixel fidelity remains poor.','128images; actual trained full model, not a downscaled architecture.','Existing renderer; structural QA only.'])
    atomic_json(out/'result.json',result)
    (out/'metrics.jsonl').write_text(''.join(json.dumps(dict(step=0,split=k,mse=v['mse']))+'\n' for k,v in metrics.items()))
    atomic_json(out/'status.json',dict(result='completed',report='pending',step=0,error=None))
    print(write_report(out,batch=dict(rgb=images[0]),outputs=dict(rgb=images[1])))
    print(json.dumps(dict(gate=result['gate'],checks=checks)))
    assert result['gate']


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--checkpoint',type=Path,required=True)
    a.add_argument('--device',default='cuda',choices=('cpu','cuda'))
    a.add_argument('--code',type=Path);a.add_argument('--decoded',type=Path);a.add_argument('--output',type=Path)
    args=a.parse_args()
    if bool(args.code)!=bool(args.decoded) or (bool(args.code)==bool(args.output)):
        a.error('Provide either --output for verification or --code plus --decoded for code-only replay')
    seed_everything(812)
    decode(args) if args.code else verify(args)


if __name__=='__main__':main()
