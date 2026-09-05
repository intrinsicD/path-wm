import json
import numpy as np
import pytest
from scripts.prepare_pusht_pilot import configuration_groups
from world_model.data import explicit_episode_split


def test_related_configurations_cannot_leak_across_frozen_split(tmp_path):
    # Duplicate variants, transitive near variants, and the 0/2pi boundary.
    states=np.array([[0,0,50,50,.01], [0,0,50,50,2*np.pi-.01],
        [4,0,50,50,.01], [8,0,50,50,.01], [200,200,300,300,1.]])
    groups,audit=configuration_groups(states)
    assert groups[0]==groups[1]==groups[2]==groups[3]
    assert groups[0]!=groups[4]
    path=tmp_path/'split.json'
    receipt=dict(train_episodes=[0,1,2,3],val_episodes=[4],group_ids=groups.tolist())
    path.write_text(json.dumps(receipt))
    assert explicit_episode_split(path,5)[:2]==([0,1,2,3],[4])
    receipt.update(train_episodes=[0,1,2],val_episodes=[3,4])
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='cross the split'): explicit_episode_split(path,5)
    receipt.update(train_episodes=[0,1,2,3],val_episodes=[3,4])
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='exactly once'): explicit_episode_split(path,5)


@pytest.mark.parametrize("protocol", ["episodes", "random_windows"])
def test_training_retains_checkpoints_and_resume_respects_spent_budget(tmp_path, monkeypatch, protocol):
    import h5py
    import torch
    import yaml
    from world_model.train import train
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    source=tmp_path/'data.h5'
    rng=np.random.default_rng(4)
    with h5py.File(source,'w') as f:
        f['ep_len']=[12]*4;f['ep_offset']=[0,12,24,36]
        f['pixels']=rng.integers(0,256,(48,28,28,3),dtype=np.uint8)
        f['action']=rng.normal(size=(48,2)).astype(np.float32)
    dataset=tmp_path/'dataset.yaml'
    dataset.write_text(yaml.safe_dump(dict(kind='action_trajectory',path=str(source),frameskip=1,history=3,action_dim=2)))
    split=tmp_path/'split.json'
    split.write_text(json.dumps(dict(train_episodes=[0,1],val_episodes=[2,3],group_ids=[0,1,2,3])))
    run=tmp_path/'run'
    cfg=dict(dataset=str(dataset),episode_split=str(split),run_dir=str(run),seed=3,
        max_steps=2,max_seconds=60,epochs=1,batch_size=2,workers=0,encoder_chunk=0,
        precision='float32',eval_precision='float32',lr=1e-4,weight_decay=.001,
        warmup_fraction=.01,grad_clip=1.,sigreg_weight=.09,sigreg_projections=8,sigreg_knots=5,
        log_every=1,eval_every=1,eval_batches=1,checkpoint_steps=[0,1,2],introspect=protocol=='episodes',
        model=dict(width=12,image_size=28,encoder_depth=1,encoder_heads=3,
            predictor_depth=1,predictor_heads=2,head_dim=4,mlp_dim=24,projector_dim=24,dropout=0.))
    if protocol=='random_windows':
        cfg.pop('episode_split')
        cfg.update(split_protocol='random_windows',normalization_population='full_source',
                   train_fraction=.9,encoder_gradient_checkpointing=True)
    config=tmp_path/'run.yaml';config.write_text(yaml.safe_dump(cfg))
    train(config)
    snapshots=[torch.load(run/f'checkpoint_{i:06d}.pt',weights_only=True) for i in range(3)]
    assert [s['step'] for s in snapshots]==[0,1,2]
    assert any(not torch.equal(snapshots[0]['model'][k],snapshots[2]['model'][k]) for k in snapshots[0]['model'])
    rows=[json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
    captured=[r for r in rows if r['kind']=='internals']
    if protocol=='episodes':
        assert [r['step'] for r in captured]==[0,1,2] and all('effective_rank' in r and 'gate_msa_mean' in r for r in captured)
    else:
        assert not captured
    # Simulate recovering step 1 after the whole time budget has been spent.
    recovery=snapshots[1];recovery['elapsed_seconds']=61.
    torch.save(recovery,run/'recovery.tmp');(run/'recovery.tmp').replace(run/'checkpoint.pt')
    train(config,resume=True)
    saved=torch.load(run/'checkpoint.pt',weights_only=True)
    assert saved['step']==1
    assert json.loads((run/'status.json').read_text())['kind']=='time_limit'
    for k,value in recovery['model'].items(): torch.testing.assert_close(saved['model'][k],value)
    assert torch.load(run/'checkpoint_000001.pt',weights_only=True)['elapsed_seconds']<60
