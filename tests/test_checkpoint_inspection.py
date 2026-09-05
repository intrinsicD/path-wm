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
