"""Frozen custom/DINO references from aligned source views and audited caches."""
import json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .data import task_frames,file_hash,digest
from .dino_reference import FrozenDino,DinoAdapter
from .pose_accessibility import setup,features,batch,pose_metrics,analyze,PARENT
from .training import initial_models
from world_model.paddle.types import ObservationLatent
from world_model.paddle.training import optimizer_for
from world_model.pusht.checkpoints import json_atomic,atomic_checkpoint,fingerprint_modules,versions

ROOT=Path('runs/encoder_study_2026-09-08')
OLD_CACHE=Path('runs/bottlenecks_2026-09-08/pose/experiment')


def pinned_backbone(device):
    p=ROOT/'upstream/manifest.json';m=json.loads(p.read_text())
    proof=json.loads((p.parent/'official_weights_verification.json').read_text())
    if not proof['matches_cached_weights'] or file_hash(m['weights'])!=proof['sha256']:raise ValueError('DINO weights differ from official bytes')
    for name,h in m['files'].items():
        if file_hash(Path(m['source'])/name)!=h:raise ValueError('DINO source changed')
    return FrozenDino(m['source'],m['weights']).to(device)


def prepare(development=False,device='cuda'):
    setup();out=ROOT/('development/dino_features' if development else 'dino_features');out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():raise FileExistsError('preserve completed extraction')
    backbone=pinned_backbone(device);before=fingerprint_modules({'backbone':backbone});splits={};begin=time.monotonic()
    for split in ('train','validation','test'):
        ds=task_frames('data/pusht_world_model/cchi_v1',split)
        ids=np.arange(min(len(ds),48 if split=='train' else 16)) if development else np.arange(len(ds))
        path=out/f'{split}.npy'
        cache=np.lib.format.open_memmap(path,mode='w+',dtype='float16',shape=(len(ids),256,384));norms=[];error_num=0.;error_den=0.
        for start in range(0,len(ids),32):
            rows=ids[start:start+32];rgb,_=ds.batch(rows,device);z=backbone(rgb)
            cache[start:start+len(rows)]=z.cpu().half().numpy()
            if start<256:
                norms.extend(z.norm(dim=-1).flatten().cpu().tolist())
                error_num+=float((z-z.half().float()).square().sum());error_den+=float(z.square().sum())
        cache.flush();del cache
        labels=out/f'{split}_labels.npz'
        np.savez_compressed(labels,targets=ds.targets[ids],indices=ids,source_rows=ds.rows[ids],groups=np.array([ds.metadata[i]['group'] for i in ids]))
        # Nonconsecutive live/cache check also audits source-row and patch ordering.
        check_ids=np.unique(np.array([0,len(ids)//2,len(ids)-1]));rgb,_=ds.batch(ids[check_ids],device)
        live=backbone(rgb).cpu().numpy();stored=np.load(path,mmap_mode='r')[check_ids].astype('float32')
        live_relative=float(np.square(live-stored).mean()/np.square(live).mean())
        # Different inference batch sizes change low-order GPU summation bits.
        # Preserve the predeclared relative-MSE gate; allow 3e-4 near zero.
        if not np.allclose(live,stored,rtol=6e-4,atol=3e-4) or live_relative>1e-6:
            raise RuntimeError('live/cache frame or token mismatch')
        splits[split]=dict(frames=len(ids),dataset=ds.fingerprint,source_rows_sha256=digest(ds.rows[ids].tolist()),features_sha256=file_hash(path),
                           labels_sha256=file_hash(labels),relative_cache_mse=error_num/error_den,
                           sampled_patch_norm_percentiles=np.percentile(norms,[0,50,95,99,100]).tolist(),
                           live_cache_relative_mse=live_relative,live_cache_within_fp16_tolerance=True)
        if error_num/error_den>1e-6:raise RuntimeError('cache precision exceeds declared tolerance')
        print('DINO cached',split,len(ids),'seconds',round(time.monotonic()-begin,1),flush=True)
    if before!=fingerprint_modules({'backbone':backbone}):raise RuntimeError('frozen DINO changed')
    manifest=dict(status='completed',development=development,splits=splits,seconds=time.monotonic()-begin,frozen_fingerprint=before,
                  upstream_manifest_sha256=file_hash(ROOT/'upstream/manifest.json'),source_sha256=file_hash(__file__),versions=versions(),
                  preprocessing='RGB64 to224 bicubic antialias clamp[0,1], full field of view, ImageNet mean/std; FP32 compute, FP16 cache',
                  scope='Exploratory reference; pretraining and adapter capacity differ from custom E.')
    json_atomic(out/'manifest.json',manifest)
    analyze(ROOT,out,'Frozen DINO feature extraction',{'frames':sum(s['frames'] for s in splits.values())},
            '## Frozen DINO reference features\n\nSame source RGB64 and grouped CCHI splits. Official source/weights pinned. Cache quantization and live feature alignment passed; no model-quality claim yet.',[out/'manifest.json'])


def data(arm,split,development=False):
    if arm=='custom':return features(OLD_CACHE,split)
    out=ROOT/('development/dino_features' if development else 'dino_features');m=json.loads((out/'manifest.json').read_text())
    path=out/f'{split}.npy';label=out/f'{split}_labels.npz'
    if file_hash(path)!=m['splits'][split]['features_sha256'] or file_hash(label)!=m['splits'][split]['labels_sha256']:raise ValueError('DINO cache mutated')
    return np.load(path,mmap_mode='r'),dict(np.load(label))


def models_for(arm,seed,device):
    baseline=initial_models(seed);models={'D':baseline['D'],'H':baseline['H']}
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed+30011)
        if arm=='dino':models['A']=DinoAdapter()
        elif arm in ('native','native_scaled'):models={'H':nn.Linear(256*384,6)}
        elif arm!='custom':raise ValueError('reference arm')
    return {k:m.to(device) for k,m in models.items()}


def forward(models,arm,array,ids,device):
    x=torch.from_numpy(np.asarray(array[ids]).copy()).float().to(device)
    if arm in ('native','native_scaled'):
        # Fixed invertible feature scaling preserves linear readout capacity.
        # It controls the update magnitude when Adam sees 98,304 coordinates.
        scale=(256*384)**-.5 if arm=='native_scaled' else 1.
        return models['H'](x.flatten(1)*scale),None
    z=ObservationLatent.from_tokens(x) if arm=='custom' else models['A'](x)
    return models['H'](z),models['D'](z)


@torch.no_grad()
def evaluate(models,arm,cache,labels,ds,ids,device):
    for m in models.values():m.eval()
    predictions=[];pixels=[]
    for start in range(0,len(ids),128):
        rows=ids[start:start+128];pred,recon=forward(models,arm,cache,rows,device)
        predictions.append(pred.cpu().double().numpy())
        if recon is not None:
            rgb,_=ds.batch(labels['indices'][rows],device)
            pixels.extend((recon-rgb).square().mean((1,2,3)).cpu().double().tolist())
    metrics,raw=pose_metrics(np.concatenate(predictions),labels['targets'][ids].astype('float64'))
    if pixels:metrics['image_mse']=float(np.mean(pixels));raw['image_mse']=np.array(pixels)
    metrics['loss']=metrics['pose_mse']+metrics.get('image_mse',0.)
    return metrics,raw


def train(arm,development=False,device='cuda'):
    setup();out=ROOT/('development/reference' if development else 'reference')/arm;out.mkdir(parents=True,exist_ok=True)
    if (out/'last.pt').exists():raise FileExistsError('preserve reference run')
    if arm in ('native','native_scaled') and not development:
        prior=json.loads((ROOT/'reference/dino/curriculum_result.json').read_text())
        if prior['selected']['q']<=1:raise ValueError('native-width conditional control not triggered')
    train_x,tl=data(arm,'train',development);val_x,vl=data(arm,'validation',development)
    datasets={s:task_frames('data/pusht_world_model/cchi_v1',s) for s in ('train','validation','test')}
    if not np.array_equal(tl['targets'],datasets['train'].targets[tl['indices']]):raise ValueError('feature/pose targets misaligned')
    ids=np.random.default_rng(6107).choice(len(val_x),min(16 if development else 2048,len(val_x)),replace=False)
    config=dict(seed=6107,arm=arm,phase='supervised',updates=3 if development else (6000 if arm in ('native','native_scaled') else 4000),
                batch_size=128,learning_rate=.0003,weight_decay=.0001,grad_clip=1.,validate_every=1 if development else 100,
                max_seconds=1800 if arm in ('native','native_scaled') else 1200,development=development,
                input_feature_scale=(256*384)**-.5 if arm=='native_scaled' else 1.,
                protocol='A2: frozen representation with fresh D/H; DINO has trainable adapter; native control pose-only')
    models=models_for(arm,6107,device);opt=optimizer_for(models,config);rng=np.random.default_rng(26110)
    fm=OLD_CACHE/'features/manifest.json' if arm=='custom' else ROOT/('development/dino_features/manifest.json' if development else 'dino_features/manifest.json')
    manifest=dict(config=config,dataset=datasets['train'].fingerprint,feature_manifest=str(fm),feature_manifest_sha256=file_hash(fm),
                  parent_sha256=file_hash(PARENT) if arm=='custom' else None,validation_indices=ids.tolist(),
                  initial=fingerprint_modules(models),parameters={k:sum(p.numel() for p in m.parameters()) for k,m in models.items()},
                  source_sha256=file_hash(__file__),versions=versions())
    json_atomic(out/'curriculum_manifest.json',manifest);begin=time.monotonic();curve=[];best=None;selected=None
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    def validate(step):
        nonlocal best,selected
        metric,_=evaluate(models,arm,val_x,vl,datasets['validation'],ids,device)
        row=dict(step=step,elapsed_seconds=time.monotonic()-begin,**metric);curve.append(row)
        with (out/'validation.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        state=dict(schema='encoder-reference-v1',models={k:{n:t.detach().cpu().clone() for n,t in m.state_dict().items()} for k,m in models.items()},
                   config=config,step=step,metrics=row,feature_manifest_sha256=manifest['feature_manifest_sha256'])
        atomic_checkpoint(out/'last.pt',state)
        key=(metric['q'],metric.get('image_mse',metric['pose_mse']),step)
        if best is None or key<best:best=key;selected=row;atomic_checkpoint(out/'best.pt',state)
        print(arm,step,'q',round(metric['q'],4),'angle',round(metric['angle_mae_deg'],3),flush=True)
    validate(0)
    for step in range(1,config['updates']+1):
        if time.monotonic()-begin>config['max_seconds']:raise TimeoutError('reference wall budget')
        draw=rng.integers(len(train_x),size=128);rgb,targets=datasets['train'].batch(tl['indices'][draw],device)
        for m in models.values():m.train()
        opt.zero_grad(set_to_none=True);pred,recon=forward(models,arm,train_x,draw,device)
        pose=(pred-targets).square().mean();pixel=pose.new_zeros(()) if recon is None else (recon-rgb).square().mean();loss=pose+pixel
        loss.backward();grad=nn.utils.clip_grad_norm_([p for m in models.values() for p in m.parameters()],1.)
        if not torch.isfinite(loss) or not torch.isfinite(grad):raise RuntimeError('nonfinite reference')
        opt.step()
        with (out/'training.jsonl').open('a') as f:f.write(json.dumps(dict(step=step,loss=float(loss.detach()),pose_mse=float(pose.detach()),
            image_mse=float(pixel.detach()),examples=step*128,sample_indices_sha256=digest(draw.tolist()),elapsed_seconds=time.monotonic()-begin))+'\n')
        if step%config['validate_every']==0 or step==config['updates']:validate(step)
    result=dict(status='completed',step=step,selected_step=selected['step'],selected=selected,final=curve[-1],examples=128*step,
                elapsed_seconds=time.monotonic()-begin,peak_cuda_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None)
    json_atomic(out/'curriculum_result.json',result)
    saved=torch.load(out/'best.pt',map_location='cpu',weights_only=False)
    for k,m in models.items():m.load_state_dict(saved['models'][k])
    metrics={};sources=[out/'curriculum_manifest.json',out/'curriculum_result.json',out/'training.jsonl',out/'validation.jsonl']
    for split in ('train','validation','test'):
        x,l=data(arm,split,development)
        rows=np.random.default_rng(6107).choice(len(x),min(16 if development else 2048,len(x)),replace=False) if split!='test' or development else np.arange(len(x))
        m,r=evaluate(models,arm,x,l,datasets[split],rows,device);metrics[split]=m
        path=out/f'{split}_errors.npz';np.savez_compressed(path,**r,indices=l['indices'][rows],groups=l['groups'][rows]);sources.append(path)
    json_atomic(out/'evaluation.json',metrics);sources.append(out/'evaluation.json')
    analyze(ROOT,out,'Frozen representation comparator',{'test_q':metrics['test']['q'],'selected_q':selected['q']},
            f"## {arm} frozen representation comparator\n\n{'Development. ' if development else 'Exploratory reused grouped holdouts. '}Validation-selected q={selected['q']:.4f}; test q={metrics['test']['q']:.4f}. Backbone frozen, fresh readouts. DINO includes a trained projection; native-width arm is a separate pose-only diagnostic. No planning-quality conclusion.",sources)
    return result
