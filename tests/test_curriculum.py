"""Essential invariants for the new perception curriculum."""
import copy
import numpy as np
import pytest
import torch
from world_model.curriculum.data import group_hashes, split_groups
from world_model.curriculum.training import (
    initial_models, objective, backward_batch, selection_key, phase_sampler)

def test_generic_reconstruction_never_calls_task_head():
    models = initial_models(17)
    class Forbidden(torch.nn.Module):
        def forward(self, _):
            raise AssertionError("unlabelled images must not receive pose loss")
    models['H'] = Forbidden()
    loss, metrics = objective(models, torch.rand(2,3,64,64), None)
    loss.backward()
    assert set(metrics) == {'loss','image_mse'}
    assert any(p.grad is not None for p in models['E'].parameters())

def test_physical_selector_cannot_hide_bad_pose_with_good_pixels():
    good = {'position_mae':[4,4,4,4], 'angle_mae_deg':5, 'image_mse':.1}
    bad = {'position_mae':[4,4,4,16], 'angle_mae_deg':5, 'image_mse':.0001}
    assert selection_key(good, 100) < selection_key(bad, 200)
    assert selection_key(good, 100) < selection_key(good, 200)
    with pytest.raises(ValueError):
        selection_key({**good,'angle_mae_deg':float('nan')},0)

@pytest.mark.parametrize('labelled',[False, True])
def test_ragged_microbatches_match_full_batch_gradients(labelled):
    torch.set_num_threads(1)
    full=initial_models(42); micro=copy.deepcopy(full)
    x=torch.rand(3,3,64,64); y=torch.rand(3,6) if labelled else None
    backward_batch(full,x,y,3)
    backward_batch(micro,x,y,2)
    for key in full:
        for a,b in zip(full[key].parameters(),micro[key].parameters()):
            if a.grad is None:
                assert b.grad is None
            else:
                torch.testing.assert_close(a.grad,b.grad,atol=2e-6,rtol=1e-4)

def test_warmup_loads_only_encoder_decoder_and_preserves_fresh_head():
    original=initial_models(2)
    warmup={k:{name:torch.zeros_like(t) for name,t in model.state_dict().items()}
            for k,model in original.items()}
    adapted=initial_models(2,warmup)
    for k in ('E','D'):
        assert all(torch.count_nonzero(t)==0 for t in adapted[k].state_dict().values())
    for name,t in original['H'].state_dict().items():
        assert torch.equal(t,adapted['H'].state_dict()[name])

def test_private_supervised_stream_is_phase_independent_and_resumable():
    a=phase_sampler(4107,'supervised'); b=phase_sampler(4107,'supervised')
    np.testing.assert_array_equal(a.integers(200, size=128),b.integers(200,size=128))
    snapshot=copy.deepcopy(a.bit_generator.state)
    phase_sampler(4107,'warmup').integers(1000,size=5000)
    expected=a.integers(200,size=128)
    restored=phase_sampler(4107,'supervised');restored.bit_generator.state=snapshot
    np.testing.assert_array_equal(expected,restored.integers(200,size=128))

def test_duplicate_components_are_transitive_and_split_as_units():
    # 0->15->255 is a radius-four chain; first/last are distance eight.
    hashes=[0,15,255,2**64-1,0xAAAAAAAAAAAAAAAA,0x5555555555555555]
    groups,edges=group_hashes(hashes,[str(i) for i in range(len(hashes))])
    assert groups[0]==groups[1]==groups[2]
    assert len(set(groups))==4
    split=split_groups(groups,4107)
    membership={i:k for k,ids in split.items() for i in ids}
    assert len(membership)==len(groups)
    assert membership[0]==membership[1]==membership[2]
    assert all(split.values())

def test_archived_panel_resolves_in_current_runs_without_rewriting_source(tmp_path):
    from viewer.ledger import resolve_evidence_path
    current=tmp_path/'runs';p=current/'trial/visuals/frame.png'
    p.parent.mkdir(parents=True);p.write_bytes(b'image evidence')
    original='/old/computer/project/runs/trial/visuals/frame.png'
    assert resolve_evidence_path(original,current)==p
    with pytest.raises(FileNotFoundError):
        resolve_evidence_path('/old/computer/project/runs/missing.png',current)

def test_atomic_resume_matches_uninterrupted_actual_training(tmp_path):
    from world_model.curriculum.data import FrameSet
    from world_model.curriculum.training import train_phase
    from world_model.pusht.checkpoints import read_checkpoint
    rng=np.random.default_rng(7)
    data=FrameSet(rng.integers(0,256,(5,64,64,3),dtype=np.uint8),range(5),rng.random((5,6)).astype('float32'),
                  fingerprint='fixed-test-source')
    config=dict(phase='supervised',seed=7,arm='test',device='cpu',cpu_threads=1,
                updates=4,batch_size=3,microbatch=2,validate_every=2,validation_batch=3,
                max_seconds=120,learning_rate=3e-4,weight_decay=1e-4,grad_clip=1.)
    train_phase(config,data,data,[0,1,2],tmp_path/'full')
    train_phase(config,data,data,[0,1,2],tmp_path/'resumed',stop_after=2)
    train_phase(config,data,data,[0,1,2],tmp_path/'resumed',resume=True)
    full=read_checkpoint(tmp_path/'full/last.pt');resumed=read_checkpoint(tmp_path/'resumed/last.pt')
    assert full['model_fingerprint']==resumed['model_fingerprint']
    assert full['rng']['sampler']==resumed['rng']['sampler']
    assert full['best_key']==resumed['best_key']
    changed=FrameSet(data.frames,list(reversed(data.rows)),data.targets,fingerprint='fixed-test-source')
    with pytest.raises(ValueError,match='population'):
        train_phase(config,changed,data,[0,1,2],tmp_path/'resumed',resume=True)


def test_curriculum_dashboard_rejects_rewritten_selected_metrics(tmp_path):
    import json
    from viewer.curriculum import collect_curriculum_results
    from viewer.ledger import DashboardDataError
    root=tmp_path/'runs'; run=root/'phase';run.mkdir(parents=True)
    config={'batch_size':2,'phase':'supervised','seed':7,'arm':'A'}
    manifest={'config':config,'dataset':'data'}
    (run/'curriculum_manifest.json').write_text(json.dumps(manifest))
    (run/'training.jsonl').write_text(json.dumps({'step':1})+'\n')
    (run/'validation.jsonl').write_text(json.dumps({'step':1,'q':2.})+'\n')
    result={'status':'completed','step':1,'examples':2,'selected_step':1,'selected':{'q':2.},'final':{'q':2.}}
    path=run/'curriculum_result.json';path.write_text(json.dumps(result))
    records,_=collect_curriculum_results(root)
    assert all(not s.startswith('/') for s in records[0].source_paths)
    result['selected']['q']=.5;path.write_text(json.dumps(result))
    with pytest.raises(DashboardDataError,match='differs'):
        collect_curriculum_results(root)

def test_evidence_paths_resolve_with_nested_dashboard_scope(tmp_path):
    from viewer.ledger import resolve_evidence_path
    current = tmp_path / 'runs' / 'curriculum'
    panel = current / 'paddle/evaluation/visuals/frame.png'
    panel.parent.mkdir(parents=True)
    panel.write_bytes(b'unchanged evidence')
    assert resolve_evidence_path('runs/curriculum/paddle/evaluation/visuals/frame.png', current) == panel
    assert resolve_evidence_path('/old/computer/runs/curriculum/paddle/evaluation/visuals/frame.png', current) == panel
    assert panel.read_bytes() == b'unchanged evidence'
    with pytest.raises(ValueError, match='traversal'):
        resolve_evidence_path('../frame.png', current)
