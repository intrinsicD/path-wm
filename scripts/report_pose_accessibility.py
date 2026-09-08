"""Audit completed pose records and render row-scaled spatial heatmap detail."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from world_model.curriculum.pose_accessibility import ROOT, PARENT, HEADS, pose_metrics, analyze
from world_model.curriculum.data import file_hash
from world_model.pusht.checkpoints import json_atomic


def main():
    root=ROOT/'experiment';out=root/'evaluation';m=json.loads((out/'metrics.json').read_text());fm=json.loads((root/'features/manifest.json').read_text())
    assert file_hash(PARENT)==m['parent_sha256']==fm['parent_sha256']
    groups={};streams=[];audits={};components={}
    for split in ('train','validation','test'):
        p=root/'features'/f'{split}_labels.npz';f=root/'features'/f'{split}.npy'
        assert file_hash(p)==fm['splits'][split]['labels_sha256']
        assert file_hash(f)==fm['splits'][split]['features_sha256']
        groups[split]=set(np.load(p)['groups'].tolist())
    assert not(groups['train']&groups['validation'] or groups['train']&groups['test'] or groups['validation']&groups['test'])
    for arm in HEADS:
        result=json.loads((root/arm/'result.json').read_text());manifest=json.loads((root/arm/'manifest.json').read_text())
        rows=[json.loads(s) for s in (root/arm/'training.jsonl').read_text().splitlines()]
        valid=[json.loads(s) for s in (root/arm/'validation.jsonl').read_text().splitlines()]
        assert [r['step'] for r in rows]==list(range(1,2001))
        assert result['examples']==256000 and result['step']==2000
        assert result['selected_step']==min(valid,key=lambda r:(r['q'],r['pose_mse'],r['step']))['step']
        streams.append([r['sample_indices_sha256'] for r in rows])
        audits[arm]=dict(parameters=manifest['parameters'],seconds=result['seconds'],selected_step=result['selected_step'],updates=2000,examples=256000,
            best_sha256=file_hash(root/arm/'best.pt'),final_sha256=file_hash(root/arm/'last.pt'))
    assert all(s==streams[0] for s in streams[1:])
    for name in m['metrics']:
        for split,metric in m['metrics'][name].items():
            d=np.load(out/f'{name}_{split}.npz');ref=np.load(out/f'original_{split}.npz')
            assert np.array_equal(d['indices'],ref['indices']) and np.array_equal(d['targets'],ref['targets'])
            actual,_=pose_metrics(d['predictions'],d['targets'])
            for k in metric:assert np.allclose(actual[k],metric[k],rtol=1e-12,atol=1e-12)
            if split=='train':components[name]=dict(normalized_coordinate_mse=((d['predictions']-d['targets'])**2).mean(0).tolist())
    json_atomic(out/'integrity_audit.json',dict(status='passed',parent_sha256=m['parent_sha256'],arms=audits,
        disjoint_group_counts={k:len(v) for k,v in groups.items()},same_training_streams=True,raw_metrics_reconciled=True,
        selected_by_declared_validation_key=True,training_loss_components=components))
    raw=np.load(out/'spatial_panels.npz');maps=raw['heatmaps'];n=len(maps)
    fig,axes=plt.subplots(3,n,figsize=(12,6),layout='constrained')
    for row,label in enumerate(['Pusher','Object center','Orientation landmark']):
        vmax=float(maps[:,row].max())
        for c in range(n):
            im=axes[row,c].imshow(maps[c,row],cmap='Greys',vmin=0,vmax=vmax)
            axes[row,c].set_xticks([]);axes[row,c].set_yticks([])
            if row==0:axes[row,c].set_title(f'Test row {raw["indices"][c]}',fontsize=8)
        axes[row,0].set_ylabel(label,fontsize=9)
        fig.colorbar(im,ax=list(axes[row]),shrink=.8,label='Probability per grid cell')
    fig.suptitle('Spatial pose head · heatmap detail\nScale shared within each row; different rows use different probability ranges',fontsize=12)
    panel=out/'spatial_heatmaps_detail.png';fig.savefig(panel,dpi=120);fig.savefig(panel.with_suffix('.svg'));plt.close(fig)
    analyze(root,out/'audit','Pose record integrity and heatmap detail',{'audited_heads':3,'optimizer_updates':6000,'training_presentations':768000},
        '## Pose audit and spatial detail\n\nAll declared training budgets, selections, paired frame identities, frozen-cache hashes and raw aggregate metrics reconcile. '
        'Training/validation/test configuration groups remain disjoint (164/20/22). '
        'The heatmap detail uses a separate, explicitly labeled probability range for each output, exposing diffuse object maps without implying equal concentration.',
        [out/'integrity_audit.json',out/'spatial_panels.npz',panel],
        [dict(file=str(panel),title='Spatial probability detail',caption='Same six examples as the common-scale panel. Scales differ between rows.',embed=True)])

if __name__=='__main__':main()
