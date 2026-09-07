"""Read-only Paddle prediction and observer internals after the bounded follow-up."""
from pathlib import Path
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from world_model.paddle.checkpoints import load_system,read_checkpoint,fingerprint_modules,json_atomic
from world_model.paddle.data import EpisodeDataset
from world_model.paddle.evaluation import prediction_diagnostics,_encode_episode,matched_rollout
from world_model.paddle.types import PlanningState
from .inspection import pca_fit,pca_transform,attention_weights
from .data import file_hash,digest

ROOT=Path('runs/curriculum_2026-09-07')
RUN=ROOT/'paddle_history'

@torch.inference_mode()
def memory_panels(system,train,test,output,device):
    fit=[]
    for i in range(8):
        _,m=_encode_episode(system,train[i],device);fit.append(m.cpu().numpy())
    fit=np.concatenate(fit);basis=pca_fit(fit);std=np.maximum(fit.std(0),1e-6)
    np.savez_compressed(output/'memory_train_pca.npz',**basis,standard_deviation=std,training_episodes=np.arange(8))
    fig,axes=plt.subplots(4,4,figsize=(13,9),layout='constrained')
    raw={}
    for row,episode_id in enumerate(range(4)):
        ep=test[episode_id];obs,mem=_encode_episode(system,ep,device);values=mem.cpu().numpy()
        state=(system['R'](mem)*torch.tensor([64,64,6,6,64],device=device)).cpu().numpy()
        coords=pca_transform(values,basis);raw[f'episode{episode_id}_memory']=values
        raw[f'episode{episode_id}_state_readout']=state;raw[f'episode{episode_id}_truth']=ep['states']
        t=min(5,len(mem)-1);prior=mem[t-1:t] if t else torch.zeros_like(mem[:1])
        weights=attention_weights(system['U'].attention,prior[:,None],obs[t:t+1].tokens()).mean((0,1,2)).cpu().numpy()
        raw[f'episode{episode_id}_attention']=weights
        ax=axes[row,0];handle=ax.imshow(((values-basis['mean'])/std).T,aspect='auto',cmap='coolwarm',vmin=-3,vmax=3)
        ax.set(title=f'Test episode{episode_id} · memory',xlabel='Observation index',ylabel='Memory coordinate')
        ax=axes[row,1];ax.plot(coords[:,0],coords[:,1],color='#9b9b9b',lw=1)
        dots=ax.scatter(coords[:,0],coords[:,1],c=np.arange(len(coords)),cmap='viridis',s=12)
        ax.set(title='PCA · purple → yellow over time',xlabel='PC1',ylabel='PC2')
        ax.title.set_fontsize(9)
        ax=axes[row,2]
        for k,color in [(2,'#2463a6'),(3,'#db7923')]:
            ax.plot(ep['states'][:,k],color=color,label=('true vx' if k==2 else 'true vy'))
            ax.plot(state[:,k],color=color,ls='--',label=('R vx' if k==2 else 'R vy'))
        ax.axvspan(0,1,color='#cccccc',alpha=.3)
        ax.set(title='True and read-out velocity',xlabel='Observation index',ylabel='pixels / interval')
        if row==0:ax.legend(fontsize=6,ncol=2)
        ax=axes[row,3];h=ax.imshow(weights[:256].reshape(16,16),vmin=0,cmap='viridis')
        ax.set(title=f'U→fine attention at t={t}\nCoarse attention mass: {weights[256:].sum():.3f}',xlabel='Fine token column',ylabel='Fine token row')
        ax.title.set_fontsize(9)
        fig.colorbar(h,ax=ax,fraction=.04)
    fig.colorbar(handle,ax=list(axes[:,0]),fraction=.015,label='Memory z-score; training mean/SD')
    fig.suptitle('Mixed-history U · fixed first four test episodes\nMemory PCA/standardization fitted to first eight training episodes; attention is descriptive',fontsize=12)
    path=output/'memory_states.png';fig.savefig(path,dpi=110);fig.savefig(output/'memory_states.svg');plt.close(fig)
    np.savez_compressed(output/'memory_panel_states.npz',**raw)
    return path

@torch.inference_mode()
def rollout_panel(system,test,output,device,horizon_trained):
    fig,axes=plt.subplots(4,6,figsize=(10,7),layout='constrained');raw={}
    eligible=[i for i in range(len(test)) if len(test[i]['frames'])>=8][:2]
    raw['selected_episode_ids']=np.asarray(eligible)
    for row,episode_id in enumerate(eligible):
        ep=test[episode_id];obs,memory=_encode_episode(system,ep,device);source=2
        initial=PlanningState(obs[source:source+1],memory[source:source+1])
        imagined=matched_rollout(system,initial,torch.as_tensor(ep['actions'][source:source+5][None],device=device))['prediction']
        decoded=torch.cat([system['D'](initial.observation),*[system['D'](s.observation) for s in imagined]])
        decoded=decoded.permute(0,2,3,1).cpu().numpy();actual=ep['frames'][source:source+6]/255
        raw[f'episode{episode_id}_actual']=actual;raw[f'episode{episode_id}_decoded']=decoded
        raw[f'episode{episode_id}_actions']=ep['actions'][source:source+5]
        for t in range(6):
            axes[row*2,t].imshow(actual[t]);axes[row*2+1,t].imshow(decoded[t])
            for ax in (axes[row*2,t],axes[row*2+1,t]):ax.set_xticks([]);ax.set_yticks([])
            axes[row*2,t].set_title('source' if t==0 else f'+{t} interval')
        axes[row*2,0].set_ylabel(f'Episode{episode_id}\nActual RGB')
        axes[row*2+1,0].set_ylabel('Decoded model')
    fig.suptitle(f'Paddle fixed diagnostic rollouts · predictor trained at horizon{horizon_trained}\nSame recorded executable actions; source decoding then imagined future states',fontsize=12)
    path=output/'paddle_rollouts.png';fig.savefig(path,dpi=120);fig.savefig(output/'paddle_rollouts.svg');plt.close(fig)
    np.savez_compressed(output/'rollout_states.npz',**raw);return path

def main():
    ready=RUN/'predictor_5/paddle_result.json'
    if not ready.exists() and not (RUN/'downstream_decision.json').exists():raise ValueError('Finish the bounded training branch first')
    if ready.exists() and json.loads(ready.read_text())['status']!='completed':raise ValueError('P5 training is unfinished')
    output=RUN/'inspection';output.mkdir(exist_ok=True)
    if (output/'curriculum_analysis.json').exists():raise ValueError('preserve completed inspection')
    torch.set_num_threads(2);device='cpu'
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    predictor=RUN/('predictor_5' if (RUN/'predictor_5/best.pt').exists() else 'predictor_1')/'best.pt'
    saved=read_checkpoint(predictor);system=load_system('runs/paddle/baseline/perception/best.pt',
        'runs/paddle/history_start_v1/memory/best.pt',predictor,device)
    models={k:v for k,v in system.items() if k!='statistics'};before=fingerprint_modules(models)
    test=EpisodeDataset('data/paddle/baseline','test');train=EpisodeDataset('data/paddle/baseline','train')
    cached=RUN/'evaluation/diagnostics/prediction.json'
    if cached.exists():
        report=json.loads((RUN/'evaluation/evaluation_manifest.json').read_text());record=json.loads(cached.read_text())
        assert report['identity']['checkpoint_sha256']['predictor']==file_hash(predictor)
        assert record['evaluation_fingerprint']==report['fingerprint']
        assert record['payload_fingerprint']==digest(record['payload'])
        diagnostics=record['payload'];raw=cached
        assert diagnostics['window_count']==1024 and diagnostics['window_seed']==7300
    else:
        diagnostics=prediction_diagnostics(system,test,device,window_limit=1024,seed=7300)
        raw=output/'prediction_records.json';json_atomic(raw,diagnostics)
    memory=memory_panels(system,train,test,output,device)
    rollout=rollout_panel(system,test,output,device,saved['horizon'])
    after=fingerprint_modules(models)
    if before!=after:raise RuntimeError('inspection changed models')
    summary=diagnostics['summary']
    fig,axes=plt.subplots(1,3,figsize=(11,3),layout='constrained')
    for k,name in enumerate(('ball x','ball y','paddle x')):
        for method,color in [('prediction','#2463a6'),('copy','#db7923'),('reset','#6d7278')]:
            axes[k].plot(range(1,6),[summary[method][str(h)]['all']['h_mae'][k] for h in range(1,6)],marker='o',label=method,color=color)
        axes[k].set(title=name,xlabel='Prediction horizon',ylabel='Mean absolute pixels',xticks=range(1,6))
    axes[0].legend(fontsize=8);fig.suptitle(f'Paddle held-out prediction · {diagnostics["window_count"]} matched windows · P trained at horizon{saved["horizon"]}')
    overview=output/'paddle_prediction.png';fig.savefig(overview,dpi=120);fig.savefig(output/'paddle_prediction.svg');plt.close(fig)
    actual=summary['actual']['0']['all'];metrics=dict(windows=diagnostics['window_count'],actual_frames=actual['count'],trained_horizon=saved['horizon'])
    for h in (1,5):
        for method in ('prediction','copy','reset'):
            r=summary[method][str(h)]['all']
            metrics[f'{method}_h{h}_latent']=r['latent_error']
            for i,v in enumerate(r['h_mae']):metrics[f'{method}_h{h}_position_{i}']=v
    narrative=f'## Paddle observer and prediction diagnosis\n\nSelected P trained at horizon{saved["horizon"]}; evaluated1024matched test windows and all{actual["count"]}test observations after two-frame warmup. '
    narrative+=f'Actual-frame H position MAE: {actual["h_mae"]}. U/R state MAE [x,y,vx,vy,paddle_x]: {actual["r_mae"]}. '
    narrative+='Five-step rollouts from a one-step-trained model are diagnostic if the P1 gate stopped expansion; they do not imply P5 training or a controller pass. Full memory heatmaps, train-fit PCA trajectories, U attention, velocity traces and decoded rollouts are standalone PNG/SVGs beside this report.'
    sources=[raw,memory,rollout,overview,output/'memory_train_pca.npz',output/'memory_panel_states.npz',output/'rollout_states.npz',RUN/'evaluation/evaluation_manifest.json',predictor.parent/'paddle_result.json']
    json_atomic(output/'curriculum_analysis.json',dict(status='completed',purpose='Paddle held-out observer and predictor diagnosis',
        metrics=metrics,narrative=narrative,checkpoint=str(predictor),checkpoint_sha256=file_hash(predictor),
        model_fingerprint=before,checkpoint_unchanged=True,figure_device='cpu',figure_threads=2,quantitative_diagnostics='reused verified completed GPU prediction cache',
        sources={p.relative_to(ROOT).as_posix():file_hash(p) for p in sources},
        panels=[dict(file=str(p),embed=(p==overview),title=p.stem,caption='Fixed held-out observations; training-only memory basis. Same-source predictions, copy and memory reset controls.') for p in [overview,memory,rollout]]))
    print(json.dumps(metrics),flush=True)
if __name__=='__main__':main()
