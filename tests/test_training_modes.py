"""Keep diagnostic calibration within the recorded split and outside source state."""
import json
import torch
import yaml
from test_training_lifecycle import setup_run


def test_modes_restore_random_window_population_and_preserve_checkpoint(setup_run,tmp_path):
    from world_model.train import train
    from scripts.check_training_modes import check
    path,run,cfg=setup_run('mode_source',split_protocol='random_windows',
        normalization_population='full_source',batch_size=4,eval_batches=2)
    train(path)
    original=(run/'checkpoint.pt').read_bytes()
    out=tmp_path/'modes';clone=tmp_path/'calibrated'
    result=check(run,out/'diagnostics.json',device='cpu',windows=8,calibration_windows=8,
                 batch_size=4,save_calibrated=clone)
    meta=json.loads((run/'manifest.json').read_text())
    assert result['validation_window_indices']==meta['validation_window_indices']
    assert set(result['calibration_source_rows']).isdisjoint(result['validation_source_rows'])
    assert result['checkpoint_unchanged'] and (run/'checkpoint.pt').read_bytes()==original
    for mode in ('saved_float32','calibrated_float32'):
        prediction=json.loads((out/'variants'/mode/'validation'/'prediction.json').read_text())
        assert prediction['examples']==8
    saved=torch.load(run/'checkpoint.pt',weights_only=True)
    calibrated=torch.load(clone/'checkpoint.pt',weights_only=True)
    assert calibrated['diagnostic_only'] and calibrated['fingerprint']==saved['fingerprint']
    assert 'optimizer' not in calibrated
    for k in saved['model']:
        if not any(s in k for s in ('running_mean','running_var','num_batches_tracked')):
            assert torch.equal(saved['model'][k],calibrated['model'][k]),k
    assert result['batch_statistics']['validation']['examples']==8
