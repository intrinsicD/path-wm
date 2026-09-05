"""Controlled physical A/V must obey the existing data ABI without leaking audit truth (DDR §38)."""
import importlib
from pathlib import Path
import pytest
import torch
import yaml
from contracts import RepresentationData
from training.av_data import build_representation_data

def _module():
    try:
        return importlib.import_module('training.controlled_av')
    except ModuleNotFoundError:
        pytest.fail('no implementation: controlled physical A/V generator and raw sensor readout')

def _config():
    cfg=yaml.safe_load((Path(__file__).resolve().parents[2]/'configs/dev/common_base_controlled_r0.yaml').read_text())
    cfg['data']['source'].update(train_groups=2,eval_groups=2,clips_per_group=2,clip_duration_seconds=4.)
    cfg['data']['video']['resolution']=32
    return cfg

def test_controlled_source_is_reproducible_disjoint_and_truth_free(tmp_path):
    module=_module();cfg=_config();rng=torch.get_rng_state().clone()
    records=module.generate_controlled_av(cfg,tmp_path/'first')
    replay=module.generate_controlled_av(cfg,tmp_path/'second')
    assert torch.equal(rng,torch.get_rng_state())
    assert records==replay and len(records)==8
    a,b=[root/cfg['data']['manifest'] for root in (tmp_path/'first',tmp_path/'second')]
    assert a.read_bytes()==b.read_bytes()
    data=build_representation_data(cfg,tmp_path/'first');assert isinstance(data,RepresentationData)
    groups={s:{r.group_id for r in data.records[s]} for s in ('train','eval')}
    assert len(groups['train'])==len(groups['eval'])==2 and groups['train'].isdisjoint(groups['eval'])
    batch=data.sample('eval','representation_av',4,torch.Generator().manual_seed(19))
    assert set(vars(batch))=={'current','future','shifted'}
    for view in (batch.current,batch.future,batch.shifted):
        assert set(view)=={'video','audio'}
        assert set(vars(view['video']))=={'values','timestamps','valid_mask'}
    assert torch.equal(batch.current['video'].timestamps,batch.shifted['video'].timestamps)
    # Refuse a different source definition at a published path rather than silently changing evidence.
    cfg['data']['source']['seed']+=1
    with pytest.raises(ValueError,match='source|config|different|identity'):
        module.generate_controlled_av(cfg,tmp_path/'first')

def test_pixels_and_waveform_expose_shared_motion_with_shift_and_constant_controls(tmp_path):
    module=_module();cfg=_config()
    records=module.generate_controlled_av(cfg,tmp_path/'dynamic')
    for record in records:
        payload=torch.load(tmp_path/'dynamic'/cfg['data']['shard_root']/record.shard,weights_only=True)
        video,audio=module.read_physical_coordinates(payload,cfg)
        truth=torch.load(tmp_path/'dynamic'/cfg['data']['shard_root']/record.source['audit_truth'],weights_only=True)['coordinates']
        assert video.shape==audio.shape==truth.shape==(32,8)
        assert (video-truth).abs().mean()<0.03
        torch.testing.assert_close(audio,truth,rtol=1e-4,atol=1e-5)
    audit=module.audit_controlled_av(cfg,tmp_path/'dynamic')
    assert audit['raw_balanced_accuracy']>=.95 and audit['swapped_balanced_accuracy']<=.05
    assert audit['coordinate_mean_absolute_error']<.03
    cfg['data']['source']['motion']='constant'
    module.generate_controlled_av(cfg,tmp_path/'constant')
    null=module.audit_controlled_av(cfg,tmp_path/'constant')
    assert null['raw_balanced_accuracy']==null['swapped_balanced_accuracy']==.5
    assert null['tie_fraction']==1.
