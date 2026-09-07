"""Coverage-only static observations must not leak held-out poses or invent labels."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from world_model.pusht import perception_data as subject
from world_model.pusht.data import canonical_frame


def source_metadata(path, count):
    manifest={'complete':True,'fingerprint':'source-fixture-fingerprint',
              'splits':{'train':{'source_episode_ids':[0,2],'group_ids':[7,9]},
                        'validation':{'source_episode_ids':[991],'group_ids':[991]},
                        'test':{'source_episode_ids':[992],'group_ids':[992]}},
              'episodes':[{'source_episode':0,'group_id':7,'frames':count//2,'split':'train'},
                          {'source_episode':2,'group_id':9,'frames':count-count//2,'split':'train'}],
              'source':{'path':'/never-open-source-or-heldout-data.h5','sha256':'source-only-provenance'}}
    path.write_text(json.dumps(manifest))
    return manifest


def requested_poses(count,seed=73107):
    return np.random.default_rng(seed).uniform([16,16,96,96,0],[496,496,416,416,2*np.pi],size=(count,5))


def rgb96(pose):
    y,x=np.indices((96,96));offset=int(pose[0])
    return np.stack([(x+offset)%256,(3*y+offset)%256,(x+2*y)%256],axis=-1).astype(np.uint8)


class FakeEnv:
    calls=[]
    def reset(self,pose5_world=None,goal_pose5_world=None,seed=0):
        assert goal_pose5_world is None
        requested=np.asarray(pose5_world,dtype=np.float64)
        type(self).calls.append((requested.copy(),seed))
        self.pose=requested+np.array([1.,-2.,3.,-4.,.125])
        return canonical_frame(rgb96(requested)),{}
    def close(self):pass


@pytest.fixture
def fake(monkeypatch):
    FakeEnv.calls=[]
    monkeypatch.setattr(subject,'PushTEnv',FakeEnv)
    return FakeEnv


def test_requests_are_independent_private_rng_and_labels_use_actual_reset(tmp_path,fake):
    source=tmp_path/'source_manifest.json';source_metadata(source,8)
    previous=copy.deepcopy(np.random.get_state())
    manifest=subject.prepare_supplement(tmp_path/'supplement',count=8,seed=73107,source_manifest=source)
    after=np.random.get_state()
    assert previous[0]==after[0] and previous[2:]==after[2:]
    np.testing.assert_array_equal(previous[1],after[1])
    requests=requested_poses(8)
    np.testing.assert_array_equal(np.stack([p for p,_ in fake.calls]),requests)
    assert [s for _,s in fake.calls]==list(range(73107,73115))
    assert manifest['count']==8 and manifest['complete']
    supplement=subject.SupplementFrames(tmp_path/'supplement',manifest['fingerprint'])
    assert isinstance(supplement.frames,np.memmap) and not supplement.frames.flags.writeable
    assert not supplement.targets.flags.writeable and not supplement.poses.flags.writeable
    actual=requests+np.array([1.,-2.,3.,-4.,.125])
    np.testing.assert_array_equal(supplement.poses,actual)
    target=np.column_stack([actual[:,:4]/512,np.sin(actual[:,4]),np.cos(actual[:,4])]).astype(np.float32)
    np.testing.assert_array_equal(supplement.targets,target)
    for index,request in enumerate(requests):
        np.testing.assert_array_equal(supplement.frames[index],cv2.resize(rgb96(request),(64,64),interpolation=cv2.INTER_AREA))
    assert manifest['source_training']['dataset_fingerprint']=='source-fixture-fingerprint'
    assert manifest['source_training']['source_episode_ids']==[0,2]
    assert manifest['source_training']['frame_count']==8
    # Changing held-out metadata cannot change any generated sample or request.
    altered=json.loads(source.read_text());altered['splits']['test']['source_episode_ids']=[999999]
    source.write_text(json.dumps(altered))
    second=subject.prepare_supplement(tmp_path/'second',count=8,seed=73107,source_manifest=source)
    other=subject.SupplementFrames(tmp_path/'second',second['fingerprint'])
    np.testing.assert_array_equal(other.frames,supplement.frames)
    np.testing.assert_array_equal(other.poses,supplement.poses)


def test_real_reset_render_uses_same_canonical_pixels_and_actual_pose(tmp_path):
    from world_model.pusht.env import PushTEnv
    source=tmp_path/'source_manifest.json';source_metadata(source,2)
    manifest=subject.prepare_supplement(tmp_path/'real',count=2,seed=73107,source_manifest=source)
    saved=subject.SupplementFrames(tmp_path/'real',manifest['fingerprint'])
    env=PushTEnv()
    try:
        for index,request in enumerate(requested_poses(2)):
            frame,_=env.reset(pose5_world=request,seed=73107+index)
            expected=cv2.resize(env.simulator.render().astype(np.uint8),(64,64),interpolation=cv2.INTER_AREA)
            np.testing.assert_array_equal(saved.frames[index],frame)
            np.testing.assert_array_equal(saved.frames[index],expected)
            np.testing.assert_array_equal(saved.poses[index],env.pose)
    finally:env.close()


def test_mixture_routes_exact_indices_duplicates_and_source_normalization(tmp_path,fake):
    source=tmp_path/'source_manifest.json';metadata=source_metadata(source,4)
    manifest=subject.prepare_supplement(tmp_path/'supplement',count=4,seed=73107,source_manifest=source)
    supplement=subject.SupplementFrames(tmp_path/'supplement',manifest['fingerprint'])
    class Source:
        frames=list(range(4));lengths=[2,2]
        dataset=SimpleNamespace(split='train',fingerprint='source-fixture-fingerprint',manifest=metadata)
        calls=[]
        def frame_batch(self,indices,device):
            self.calls.append(list(indices))
            return (torch.stack([torch.full((3,64,64),(int(i)+10)/255,device=device) for i in indices]),
                    torch.stack([torch.full((6,),float(i),device=device) for i in indices]))
    original=Source();mixed=subject.MixedPerceptionSamples(original,supplement)
    assert len(mixed.frames)==8 and mixed.dataset is original.dataset
    assert mixed.lengths==original.lengths
    x,y=mixed.frame_batch(np.array([0,4,3,7,4,1]),'cpu')
    assert original.calls==[[0,3,1]]
    assert x.dtype==torch.float32 and y.dtype==torch.float32
    for batch,index in enumerate([0,4,3,7,4,1]):
        if index<4:
            torch.testing.assert_close(x[batch],torch.full((3,64,64),(index+10)/255))
            torch.testing.assert_close(y[batch],torch.full((6,),float(index)))
        else:
            torch.testing.assert_close(x[batch],torch.from_numpy(supplement.frames[index-4].copy()).permute(2,0,1).float()/255)
            torch.testing.assert_close(y[batch],torch.from_numpy(supplement.targets[index-4].copy()))
    with pytest.raises(ValueError,match='index|indices'):
        mixed.frame_batch([-1],'cpu')
    original.dataset=SimpleNamespace(split='validation',fingerprint='source-fixture-fingerprint',manifest=metadata)
    with pytest.raises(ValueError,match='train'):
        subject.MixedPerceptionSamples(original,supplement)
    original.dataset=SimpleNamespace(split='train',fingerprint='different-source',manifest=metadata)
    with pytest.raises(ValueError,match='fingerprint|provenance|source'):
        subject.MixedPerceptionSamples(original,supplement)
    original.dataset=SimpleNamespace(split='train',fingerprint='source-fixture-fingerprint',manifest=metadata)
    original.frames=[0,1,2]
    with pytest.raises(ValueError,match='count|equal|length'):
        subject.MixedPerceptionSamples(original,supplement)


@pytest.mark.parametrize('name',['frames.npy','poses.npy','pose_targets.npy'])
def test_completed_data_identity_and_byte_corruption_are_enforced(tmp_path,fake,monkeypatch,name):
    source=tmp_path/'source_manifest.json';source_metadata(source,4)
    output=tmp_path/'supplement';manifest=subject.prepare_supplement(output,count=4,seed=73107,source_manifest=source)
    report=subject.verify_supplement(output)
    assert report['passed'] and report['fingerprint']==manifest['fingerprint'] and report['count']==4
    with pytest.raises(ValueError,match='fingerprint|identity'):
        subject.SupplementFrames(output,'wrong-identity')
    class NoRender:
        def __init__(self):raise AssertionError('Verified completed data must not be regenerated')
    monkeypatch.setattr(subject,'PushTEnv',NoRender)
    assert subject.prepare_supplement(output,count=4,seed=73107,source_manifest=source)['fingerprint']==manifest['fingerprint']
    with pytest.raises(ValueError,match='seed|identity|protocol|incompatible'):
        subject.prepare_supplement(output,count=4,seed=73108,source_manifest=source)
    values=np.load(output/name,mmap_mode='r+');values.reshape(-1)[0]+=1;values.flush();del values
    with pytest.raises(ValueError,match='hash|checksum|corrupt'):
        subject.verify_supplement(output)
    with pytest.raises(ValueError,match='hash|checksum|corrupt'):
        subject.prepare_supplement(output,count=4,seed=73107,source_manifest=source)


def test_interrupted_generation_resumes_verified_prefix_and_rejects_corruption(tmp_path,fake,monkeypatch):
    source=tmp_path/'source_manifest.json';source_metadata(source,6)
    monkeypatch.setattr(subject,'CHUNK_SIZE',2)
    class Interrupted(FakeEnv):
        calls=[]
        def reset(self,*args,**kwargs):
            if len(type(self).calls)==2:raise RuntimeError('injected interruption')
            return super().reset(*args,**kwargs)
    output=tmp_path/'resume'
    monkeypatch.setattr(subject,'PushTEnv',Interrupted)
    with pytest.raises(RuntimeError,match='injected interruption'):
        subject.prepare_supplement(output,count=6,seed=73107,source_manifest=source)
    assert not (output/'manifest.json').exists()
    first=np.load(output/'poses.npy',mmap_mode='r')[:2].copy()
    monkeypatch.setattr(subject,'PushTEnv',fake);fake.calls=[]
    manifest=subject.prepare_supplement(output,count=6,seed=73107,source_manifest=source)
    assert len(fake.calls)==4  # The verified two-row prefix is not rendered again.
    np.testing.assert_array_equal(np.load(output/'poses.npy')[:2],first)
    uninterrupted=subject.prepare_supplement(tmp_path/'whole',count=6,seed=73107,source_manifest=source)
    for name in ['frames.npy','poses.npy','pose_targets.npy']:
        assert (output/name).read_bytes()==(tmp_path/'whole'/name).read_bytes()
    assert manifest['fingerprint']==uninterrupted['fingerprint']
    # A corrupted committed prefix must never be accepted or silently replaced.
    Interrupted.calls=[];monkeypatch.setattr(subject,'PushTEnv',Interrupted)
    broken=tmp_path/'broken'
    with pytest.raises(RuntimeError,match='injected interruption'):
        subject.prepare_supplement(broken,count=6,seed=73107,source_manifest=source)
    poses=np.load(broken/'poses.npy',mmap_mode='r+');poses[0,0]+=10;poses.flush();del poses
    monkeypatch.setattr(subject,'PushTEnv',fake)
    with pytest.raises(ValueError,match='prefix|hash|checksum|corrupt'):
        subject.prepare_supplement(broken,count=6,seed=73107,source_manifest=source)
