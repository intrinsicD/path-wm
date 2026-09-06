"""Saved-checkpoint diagnostics must use the training run's actual window split."""
import copy
import h5py
import numpy as np
import pytest
from torch.utils.data import Subset
from world_model.protocol import prepare_training_data
from scripts.inspect_checkpoint import inspection_datasets, frame_rows


def test_inspection_restores_random_window_indices_and_rejects_changed_split(tmp_path):
    path = tmp_path / 'source.h5'
    with h5py.File(path, 'w') as f:
        f['ep_len'] = [60, 60]
        f['ep_offset'] = [0, 60]
        f['pixels'] = np.zeros((120, 8, 8, 3), dtype=np.uint8)
        f['action'] = np.arange(240, dtype=np.float32).reshape(120, 2)
    cfg = dict(seed=3072, train_fraction=0.9, split_protocol='random_windows', normalization_population='full_source')
    ds = dict(path=str(path), frameskip=1, history=3)
    train, val, tr, va, stats, receipt = prepare_training_data(cfg, ds)
    manifest = dict(config=cfg, dataset=ds, train_episodes=tr, val_episodes=va,
                    action_stats=stats, data_protocol=receipt, train_windows=len(train),
                    val_windows=len(val), validation_window_indices=[0, 2])
    actual_train, actual_val, long = inspection_datasets(manifest)
    assert isinstance(actual_val, Subset)
    assert actual_val.indices == val.indices and actual_train.indices == train.indices
    assert set(actual_train.indices).isdisjoint(actual_val.indices)
    items = [actual_val[i] for i in manifest['validation_window_indices']]
    meta = [(int(item['episode']), int(item['start'])) for item in items]
    assert frame_rows(actual_val, meta, t=1) == [ep * 60 + start + 1 for ep, start in meta]
    bad = copy.deepcopy(manifest)
    bad['data_protocol']['val_indices_sha256'] = 'wrong'
    with pytest.raises(ValueError, match='split'):
        inspection_datasets(bad)
    bad = copy.deepcopy(manifest)
    bad['validation_window_indices'] = [len(val)]
    with pytest.raises(ValueError, match='validation window'):
        inspection_datasets(bad)


def test_probe_labels_use_the_dataset_state_and_preserve_requested_row_order(tmp_path):
    from scripts.inspect_checkpoint import read_probe_targets
    path=tmp_path/'labels.h5'
    with h5py.File(path,'w') as f:
        f['proprio']=np.array([[1,10],[2,20],[3,30],[4,40]],np.float32)
        state=np.zeros((4,7),np.float32);state[:,0]=[1,2,3,4];state[:,4]=np.pi/2
        f['state']=state
    with h5py.File(path,'r') as f:
        values,names=read_probe_targets(f,[3,0,3,1],'tworoom')
        np.testing.assert_array_equal(values,[[4,40],[1,10],[4,40],[2,20]])
        assert names==['agent x','agent y']
        values,names=read_probe_targets(f,[3,0],'pusht')
        assert values.shape==(2,8) and names[4:6]==['block angle sin','block angle cos']
        np.testing.assert_allclose(values[:,4],1.,atol=1e-6)
        with pytest.raises(ValueError,match='probe'):
            read_probe_targets(f,[0],'unknown')


def test_nearest_frame_panel_uses_actual_history_and_excludes_query_window(tmp_path,monkeypatch):
    import torch
    from matplotlib.axes import Axes
    from scripts.inspect_checkpoint import neighbour_panel
    captured=[]
    original=Axes.imshow
    def capture(self,array,*args,**kwargs):
        captured.append(np.asarray(array).copy())
        return original(self,array,*args,**kwargs)
    monkeypatch.setattr(Axes,'imshow',capture)
    class Predictor:
        def predict(self,z,a):
            assert z.shape[1]==a.shape[1]==1
            return z
    pixels=torch.arange(8,dtype=torch.uint8).reshape(4,2,1,1,1).expand(4,2,3,8,8)
    z=torch.arange(8,dtype=torch.float32).reshape(4,2,1)
    path=tmp_path/'neighbours.png'
    neighbour_panel(Predictor(),pixels,z,torch.zeros(4,2,10),torch.device('cpu'),path,'history one')
    assert path.exists() and len(captured)==16
    assert np.all(captured[0]==0) and np.all(captured[1]==1)
    assert captured[2].min()>=2 and captured[3].min()>=2
