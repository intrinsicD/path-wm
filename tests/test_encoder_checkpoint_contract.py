"""An exchange-off checkpoint must not silently load as the legacy encoder."""
import pytest
import torch
from world_model.curriculum.encoder_variants import matched_models
from world_model.pusht.checkpoints import (SCHEMA_VERSION,TENSOR_SCHEMA,ACTION_SCHEMA_VERSION,
    fingerprint_modules,atomic_checkpoint,load_observer,load_bundle,export_bundle)
from world_model.pusht.models import MemoryUpdater,StateReadout,Predictor


def save(path,models,stage,dependencies=None,config=None):
    states={k:m.state_dict() for k,m in models.items()}
    payload=dict(schema_version=SCHEMA_VERSION,tensor_schema=TENSOR_SCHEMA,action_schema=ACTION_SCHEMA_VERSION,
        stage=stage,models=states,model_fingerprint=fingerprint_modules(states),config=config or {},dependencies=dependencies or {},
        normalization={},dataset_fingerprint='contract-test',statistics={'v_fine':1.,'v_coarse':1.})
    atomic_checkpoint(path,payload);return payload


@pytest.mark.parametrize('depth,exchange',[(0,False),(2,False),(2,True)])
def test_standard_observer_loader_preserves_declared_behavior(tmp_path,depth,exchange):
    torch.set_num_threads(1);models=matched_models(7107,depth,exchange)
    p=tmp_path/'perception.pt';save(p,models,'perception',config={'encoder_variant':{'depth':depth,'exchange':exchange}})
    restored=load_observer(p,device='cpu');x=torch.rand(2,3,64,64)
    torch.testing.assert_close(restored['E'](x).tokens(),models['E'](x).tokens(),rtol=0,atol=0)


def test_legacy_inspection_rejects_unsupported_variant_before_evaluation(tmp_path,monkeypatch):
    from world_model.curriculum import inspection
    def legacy_constructor(*args,**kwargs):
        raise AssertionError('unsupported variant reached legacy encoder construction')
    models=matched_models(7107,0,False)
    p=tmp_path/'perception.pt';save(p,models,'perception',config={'encoder_variant':{'depth':0,'exchange':False}})
    monkeypatch.setattr(inspection,'initial_models',legacy_constructor)
    with pytest.raises(ValueError,match='encoder.variant'):
        inspection.inspect_checkpoint(p,tmp_path/'inspection')


def test_legacy_refit_rejects_variant_parent_before_training(tmp_path,monkeypatch):
    from world_model.curriculum import training
    def legacy_constructor(*args,**kwargs):
        raise AssertionError('unsupported variant reached legacy encoder construction')
    models=matched_models(7107,0,False)
    p=tmp_path/'perception.pt';save(p,models,'perception',config={'encoder_variant':{'depth':0,'exchange':False}})
    monkeypatch.setattr(training,'initial_models',legacy_constructor)
    with pytest.raises(ValueError,match='encoder.variant'):
        training.train_phase({'phase':'warmup','seed':1,'decoder_only':True},None,None,[],tmp_path/'refit',warmup=p)


def test_export_keeps_encoder_variant_separate_from_predictor_config(tmp_path):
    torch.set_num_threads(1);models=matched_models(7107,0,False)
    a=save(tmp_path/'e.pt',models,'perception',config={'encoder_variant':{'depth':0,'exchange':False}})
    b=save(tmp_path/'u.pt',{'U':MemoryUpdater(),'R':StateReadout()},'memory',{'perception':a['model_fingerprint']})
    save(tmp_path/'p.pt',{'P':Predictor()},'predictor',{'perception':a['model_fingerprint'],'memory':b['model_fingerprint']},config={'horizon':1})
    export_bundle(tmp_path/'e.pt',tmp_path/'u.pt',tmp_path/'p.pt',tmp_path/'bundle.pt')
    restored=load_bundle(tmp_path/'bundle.pt',device='cpu');x=torch.rand(2,3,64,64)
    torch.testing.assert_close(restored['E'](x).tokens(),models['E'](x).tokens(),rtol=0,atol=0)
