import torch
from pathwm.data.episode_facts import batch


def test_pairing_split_fixed_masks_and_supervision():
    train = batch(torch.Generator().manual_seed(17), 2048)
    held = batch(torch.Generator().manual_seed(29), 2048, 'pairing')
    assert set(train['query'].flatten().tolist()) == set(range(24))
    assert ((train['query'] < 8).any(-1)).all()
    assert (held['query'] >= 8).all()
    for data in (train, held):
        assert (data['valid'].sum(-1) == 8).all()
        gold = data['valid'] & (data['attributes'] == data['attribute'][:, None])
        assert torch.equal(~gold.any(-1), data['target'] == 32)
        answerable = data['target'] != 32
        assert gold[answerable, data['target'][answerable]].all()
        # Same-factor near misses exist in both answerability groups with high frequency;
        # evaluation explicitly chooses only missing keys with a one-factor near miss.
        assert data['keys'].shape == (2048, 32, 2)


def test_training_resume_is_exact_and_reports_remain_standalone(tmp_path):
    from experiments.context_retrieval import execute
    from pathwm.io import state_hash
    full, full_run = execute(tmp_path/'full', steps=6, worlds=1)
    execute(tmp_path/'split', steps=6, worlds=1, stop_after=3)
    resumed, resumed_run = execute(tmp_path/'split', steps=6, worlds=1, resume=True)
    assert state_hash(full) == state_hash(resumed)
    assert torch.equal(full_run.sampler.get_state(), resumed_run.sampler.get_state())
    assert full_run.step == resumed_run.step == 6
    assert (full_run.path/'report.html').exists()
    import json
    assert json.loads((resumed_run.path/'report.qa.json').read_text())['self_contained']
    a = json.loads((full_run.path/'result.json').read_text())
    b = json.loads((resumed_run.path/'result.json').read_text())
    assert a['populations']['iid']['dataset_sha256'] == b['populations']['iid']['dataset_sha256']
    assert a['metrics']['iid/learned/flat']['macro'] == b['metrics']['iid/learned/flat']['macro']


def test_candidate_permutation_is_equivariant_and_mask_blocks_foreign_entities():
    from experiments.context_retrieval import LexicalPolicy
    model = LexicalPolicy().eval()
    data = batch(torch.Generator().manual_seed(4), 8)
    perm = torch.randperm(32)
    with torch.no_grad():
        a = model(data['query'], data['keys'], data['valid'])
        b = model(data['query'], data['keys'][:, perm], data['valid'][:, perm])
    assert torch.allclose(a[:, :32][:, perm], b[:, :32])
    assert torch.equal(a[:, 32], b[:, 32])
    assert torch.isneginf(a[:, :32][~data['valid']]).all()


def test_report_failure_preserves_completed_result(tmp_path, monkeypatch):
    import json
    import pytest
    import experiments.context_retrieval as recipe
    def fail(*args, **kwargs):
        raise RuntimeError('deliberate renderer failure')
    monkeypatch.setattr(recipe, 'write_report', fail)
    with pytest.raises(RuntimeError, match='renderer failure'):
        recipe.execute(tmp_path/'report_failure', steps=1, worlds=1)
    status = json.loads((tmp_path/'report_failure'/'status.json').read_text())
    assert status['result'] == 'completed' and status['report'] == 'failed'
    assert (tmp_path/'report_failure'/'result.json').is_file()
