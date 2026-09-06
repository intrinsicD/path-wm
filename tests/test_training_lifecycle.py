"""Scientific invariants across instrumentation and bounded-run recovery."""
import json
import h5py
import numpy as np
import pytest
import torch
import yaml


@pytest.fixture
def setup_run(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    rng = np.random.default_rng(4)
    source = tmp_path / 'data.h5'
    with h5py.File(source, 'w') as f:
        f['ep_len'] = [12] * 4
        f['ep_offset'] = [0, 12, 24, 36]
        f['pixels'] = rng.integers(0, 256, (48, 28, 28, 3), dtype=np.uint8)
        f['action'] = rng.normal(size=(48, 2)).astype(np.float32)
    dataset = tmp_path / 'dataset.yaml'
    dataset.write_text(yaml.safe_dump(dict(kind='action_trajectory', path=str(source),
        frameskip=1, history=3, action_dim=2)))
    cfg = dict(dataset=str(dataset), seed=3, train_fraction=.5, max_steps=3, max_seconds=60,
        epochs=1, batch_size=2, workers=0, encoder_chunk=0, precision='float32',
        eval_precision='float32', lr=1e-4, weight_decay=.001, warmup_fraction=.01,
        grad_clip=1., sigreg_weight=.09, sigreg_projections=8, sigreg_knots=5,
        log_every=1, eval_every=1, eval_batches=1, checkpoint_steps=[0, 1, 3],
        model=dict(width=12, image_size=28, encoder_depth=1, encoder_heads=3,
            predictor_depth=1, predictor_heads=2, head_dim=4, mlp_dim=24,
            projector_dim=24, dropout=.2))
    def configure(name, **overrides):
        value = {**cfg, 'run_dir': str(tmp_path / name), **overrides}
        path = tmp_path / f'{name}.yaml'
        path.write_text(yaml.safe_dump(value))
        return path, tmp_path / name, value
    return configure


def assert_same_weights(left, right):
    assert left.keys() == right.keys()
    for key in left:
        torch.testing.assert_close(left[key], right[key], rtol=0, atol=0, msg=key)


def test_instrumentation_does_not_change_cpu_optimization(setup_run):
    from world_model.train import train
    results = []
    for name, enabled in [('plain', False), ('observed', True)]:
        path, run, _ = setup_run(name, introspect=enabled)
        train(path)
        results.append(torch.load(run / 'checkpoint.pt', weights_only=True)['model'])
    assert_same_weights(*results)


@pytest.mark.parametrize('sketch_seed', [None, 700003])
def test_resume_extends_time_budget_but_preserves_exact_optimization(setup_run, sketch_seed):
    from world_model.train import train
    options = {} if sketch_seed is None else {'sigreg_seed': sketch_seed}
    path, run, cfg = setup_run('recovery', introspect=True, **options)
    train(path)
    expected = torch.load(run / 'checkpoint.pt', weights_only=True)
    assert ('regularizer' in expected) == (sketch_seed is not None)
    saved = torch.load(run / 'checkpoint_000001.pt', weights_only=True)
    saved['elapsed_seconds'] = 61.
    torch.save(saved, run / 'checkpoint.pt.tmp')
    (run / 'checkpoint.pt.tmp').replace(run / 'checkpoint.pt')
    original_manifest = (run / 'manifest.json').read_bytes()
    # A cumulative budget extension must be explicit, recorded, and RNG-neutral.
    cfg.update(max_seconds=120, log_every=2, checkpoint_steps=[0, 1, 2, 3])
    path.write_text(yaml.safe_dump(cfg))
    train(path, resume=True)
    actual = torch.load(run / 'checkpoint.pt', weights_only=True)
    assert actual['step'] == expected['step'] == 3
    assert_same_weights(expected['model'], actual['model'])
    assert torch.equal(expected['rng'], actual['rng'])
    if sketch_seed is not None:
        assert torch.equal(expected['regularizer']['_extra_state']['generator_state'],
                           actual['regularizer']['_extra_state']['generator_state'])
    assert (run / 'manifest.json').read_bytes() == original_manifest
    receipts = [json.loads(line) for line in (run / 'resumes.jsonl').read_text().splitlines()]
    assert receipts[-1]['operational_overrides']['max_seconds'] == {'previous': 60, 'current': 120}
    cfg['lr'] *= 2
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match='configuration or dataset changed'):
        train(path, resume=True)


def test_time_limit_validates_the_actual_final_checkpoint(setup_run, monkeypatch):
    import world_model.train as training
    clock = [0.]
    real_backward = training.backward_batch
    def finish_one_update(*args, **kwargs):
        result = real_backward(*args, **kwargs)
        clock[0] = 61.
        return result
    monkeypatch.setattr(training.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(training, 'backward_batch', finish_one_update)
    path, run, _ = setup_run('timed', eval_every=3, checkpoint_steps=[0, 3])
    training.train(path)
    saved = torch.load(run / 'checkpoint.pt', weights_only=True)
    rows = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines()]
    assert saved['step'] == 1
    assert [r['step'] for r in rows if r['kind'] == 'validation'] == [0, 1]
    assert rows[-1]['kind'] == 'time_limit'
    assert rows[-1]['validation_step'] == saved['step']
    assert (run / 'checkpoint_000001.pt').exists()


def test_legacy_fingerprints_keep_the_original_resume_contract():
    import hashlib
    from world_model.train import configuration_fingerprint
    signature = dict(config=dict(seed=3, lr=.001, max_seconds=60), dataset=dict(path='data.h5'))
    legacy = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    assert configuration_fingerprint(signature, version=1) == legacy
    modern = configuration_fingerprint(signature)
    signature['config']['max_seconds'] = 120
    assert configuration_fingerprint(signature, version=1) != legacy
    assert configuration_fingerprint(signature) == modern
    signature['dataset']['path'] = 'other.h5'
    assert configuration_fingerprint(signature) != modern


def test_graceful_stop_retains_the_last_update_and_all_interval_timing_counts(setup_run, monkeypatch):
    import world_model.train as training
    path, run, _ = setup_run('requested', eval_every=3, checkpoint_steps=[0,3], log_every=3)
    real_backward = training.backward_batch
    calls = [0]
    def request_stop(*args, **kwargs):
        result = real_backward(*args, **kwargs)
        calls[0] += 1
        if calls[0] == 2:
            (run/'STOP').write_text('Finish the current update and checkpoint.')
        return result
    monkeypatch.setattr(training,'backward_batch',request_stop)
    training.train(path)
    saved=torch.load(run/'checkpoint.pt',weights_only=True)
    rows=[json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
    assert saved['step']==2 and rows[-1]['kind']=='stop_requested'
    assert rows[-1]['validation_step']==2 and (run/'checkpoint_000002.pt').exists()
    assert sum(r['timing_updates'] for r in rows if r['kind'] in ('train','stop_requested'))==2
    assert all(r['mean_data_wait_seconds']>=0 and r['mean_step_seconds']>=0
               for r in rows if r['kind'] in ('train','stop_requested'))


def test_fork_preserves_parent_and_exact_optimizer_trajectory(setup_run):
    from world_model.train import train
    path,parent,cfg=setup_run('parent',checkpoint_steps=[0,1,2,3])
    train(path)
    frozen={p.name:p.read_bytes() for p in parent.iterdir() if p.is_file()}
    child_path,child,child_cfg=setup_run('child',checkpoint_steps=[0,1,2,3],stop_at_step=2)
    train(child_path,fork_from=parent/'checkpoint_000001.pt')
    child_saved=torch.load(child/'checkpoint.pt',weights_only=True)
    expected=torch.load(parent/'checkpoint_000002.pt',weights_only=True)
    assert child_saved['step']==child_saved['validation_step']==2
    assert_same_weights(expected['model'],child_saved['model'])
    assert torch.equal(expected['rng'],child_saved['rng'])
    manifest=json.loads((child/'manifest.json').read_text())
    assert manifest['total_steps']==3 and manifest['initialization']=='checkpoint_continuation'
    assert manifest['parent']['checkpoint']==str((parent/'checkpoint_000001.pt').resolve())
    assert json.loads((child/'status.json').read_text())['kind']=='step_limit'
    child_cfg['stop_at_step']=3;child_path.write_text(yaml.safe_dump(child_cfg))
    train(child_path,resume=True)
    result=torch.load(child/'checkpoint.pt',weights_only=True)
    assert_same_weights(torch.load(parent/'checkpoint.pt',weights_only=True)['model'],result['model'])
    assert {p.name:p.read_bytes() for p in parent.iterdir() if p.is_file()}==frozen
    with pytest.raises(FileExistsError):train(child_path,fork_from=parent/'checkpoint_000001.pt')


def test_fork_rejects_scientific_changes_before_any_update(setup_run):
    from world_model.train import train
    path,parent,_=setup_run('parent_bad');train(path)
    child_path,child,_=setup_run('child_bad',lr=.002)
    with pytest.raises(ValueError,match='configuration or dataset'):
        train(child_path,fork_from=parent/'checkpoint_000001.pt')
    assert not (child/'checkpoint.pt').exists()
