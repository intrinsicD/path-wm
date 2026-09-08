"""Decoder repair must never silently retrain its frozen encoder/task head."""
import numpy as np
import pytest
import torch
from world_model.curriculum.data import FrameSet
from world_model.curriculum.training import initial_models, train_phase
from world_model.pusht.checkpoints import (
    SCHEMA_VERSION, TENSOR_SCHEMA, ACTION_SCHEMA_VERSION,
    atomic_checkpoint, read_checkpoint, fingerprint_modules)


def parent_file(tmp_path):
    models=initial_models(19)
    value=dict(schema_version=SCHEMA_VERSION,tensor_schema=TENSOR_SCHEMA,
               action_schema=ACTION_SCHEMA_VERSION,models={k:m.state_dict() for k,m in models.items()},
               model_fingerprint=fingerprint_modules(models),normalization={'test':'preserve'})
    path=tmp_path/'parent.pt';atomic_checkpoint(path,value)
    return path,value


@pytest.mark.parametrize('initialization',['fresh','parent'])
def test_decoder_recovery_freezes_state_and_resumes_exactly(tmp_path,initialization):
    torch.set_num_threads(1)
    parent,original=parent_file(tmp_path)
    data=FrameSet(np.random.default_rng(8).integers(0,256,(5,64,64,3),dtype=np.uint8),range(5))
    config=dict(phase='warmup',decoder_only=True,decoder_initialization=initialization,
                seed=7,arm='decoder-test',device='cpu',cpu_threads=1,updates=4,
                batch_size=3,microbatch=2,validate_every=2,validation_batch=3,
                max_seconds=120,learning_rate=3e-4,weight_decay=1e-4,grad_clip=1.)
    train_phase(config,data,data,[0,1,2],tmp_path/'full',warmup=parent)
    saved=read_checkpoint(tmp_path/'full/last.pt')
    for k in ('E','H'):
        for name,t in original['models'][k].items():
            assert torch.equal(t,saved['models'][k][name]), f'frozen {k}.{name} changed'
    assert saved['normalization']==original['normalization']
    assert saved['stage']=='decoder_refit'
    initial=read_checkpoint(tmp_path/'full/update_00000000.pt')
    expected=original['models']['D'] if initialization=='parent' else initial_models(7)['D'].state_dict()
    for name,t in expected.items():assert torch.equal(t,initial['models']['D'][name])
    assert any(not torch.equal(t,saved['models']['D'][name]) for name,t in expected.items())
    train_phase(config,data,data,[0,1,2],tmp_path/'resume',warmup=parent,stop_after=2)
    train_phase(config,data,data,[0,1,2],tmp_path/'resume',warmup=parent,resume=True)
    resumed=read_checkpoint(tmp_path/'resume/last.pt')
    assert saved['model_fingerprint']==resumed['model_fingerprint']
    assert saved['rng']['sampler']==resumed['rng']['sampler']
    assert saved['best_key']==resumed['best_key']
    assert saved['frozen_fingerprint']==initial['frozen_fingerprint']


def test_decoder_recovery_rejects_labelled_or_missing_parent(tmp_path):
    data=FrameSet(np.zeros((2,64,64,3),dtype=np.uint8),range(2))
    config=dict(phase='supervised',decoder_only=True,seed=7,device='cpu')
    with pytest.raises(ValueError,match='image-only.*parent'):
        train_phase(config,data,data,[0],tmp_path/'invalid')


def test_paired_recovery_bootstrap_preserves_groups_and_difference():
    from world_model.curriculum.decoder_recovery import paired_group_bootstrap
    a=np.array([1.,2.,3.,4.]);b=a-.25
    result,draws=paired_group_bootstrap(a,b,[0,0,0,1],seed=12,draws=50)
    assert result['groups']==2
    np.testing.assert_allclose(draws,.25)
    assert result['delta_mse']==.25
    result,draws=paired_group_bootstrap(a,a,[0,0,0,1],seed=12,draws=50)
    np.testing.assert_array_equal(draws,np.zeros(50))
    with pytest.raises(ValueError,match='aligned'):
        paired_group_bootstrap(a,b,[0],seed=12)
