"""Range failures are inspectable completed evaluations, never partial accuracy."""
import json
import numpy as np
import pytest
import torch
from pathwm.data.memory_output import MemoryOutputEpisodes
from pathwm.models.memory_output import PixelMedianCentering
from pathwm.models.modalities import Observation


def test_range_inspection_is_pure_and_agrees_with_forward():
    class Capture(torch.nn.Module):
        code_width = 8
        def __init__(self):
            super().__init__()
            self.calls = 0
        def forward(self, observation):
            self.calls += 1
            return observation
    base = Capture()
    model = PixelMedianCentering(base, [0.2] * 3)
    x = torch.full((2, 3, 3, 8, 8), 0.8)
    x[0, 1, :, 0, 0] = 0.0
    times = torch.arange(3).float().expand(2, -1)
    expected = torch.tensor([[True, False, True], [True, True, True]])
    assert torch.equal(model.range_validity(Observation(x, times)), expected)
    assert base.calls == 0
    with pytest.raises(ValueError, match="valid RGB range"):
        model(Observation(x, times))
    assert base.calls == 0
    clean = model(Observation(x[1:], times[1:]))
    assert torch.equal(clean.times, times[1:]) and base.calls == 1
    x[0, 1] = float('nan')
    assert model.range_validity(Observation(x, times, expected)).all()
    assert base.calls == 1
    with pytest.raises(ValueError, match='finite'):
        model.range_validity(Observation(x, times))


@pytest.mark.parametrize('broken_report', [False, True])
@pytest.mark.parametrize('offset', [48, 64])
def test_rejected_export_records_coverage_without_running_model(tmp_path, monkeypatch, broken_report, offset):
    import experiments.memory_output as recipe
    from pathwm.io import file_hash
    from tests.test_memory_output import model_fixture
    m = model_fixture()
    before = [p.clone() for p in m.parameters()]
    training = MemoryOutputEpisodes(128, seed=7701, curriculum='relocation')
    settings = dict(recipe.default_settings(37), width=16, levels=2, depth=1, fusion_depth=0, curriculum='relocation')
    origin = tmp_path/'origin'; origin.mkdir()
    torch.save(dict(model=m.state_dict(), settings=settings), origin/'weights.pt')
    (origin/'run.json').write_text(json.dumps(dict(identity=dict(settings=settings, data=dict(train=training.identity)))))
    source_hash = file_hash(origin/'weights.pt')
    data = MemoryOutputEpisodes(64, seed=25073, split='test', curriculum='relocation').with_scene(background_offset=(offset,)*3)
    # Independent reference, without calling the model's range implementation.
    x = data.batch(range(len(data)))['images'].numpy()
    ref = (np.median(training.images.reshape(-1,3),axis=0)/255).astype('float32')
    median = np.sort(x.reshape(len(data),3,3,-1),axis=-1)[...,2047,None,None]
    z = x-median+ref[None,None,:,None,None]
    valid = ((z>=0)&(z<=1)).all((2,3,4))
    expected_accepted = int(valid.all(1).sum())
    assert (0 < expected_accepted < len(data)) if offset == 48 else expected_accepted == 0
    monkeypatch.setattr(recipe, 'load_model', lambda *args: m)
    monkeypatch.setattr(m, 'observe_history', lambda *a, **kw: pytest.fail('Rejected evaluation ran model queries'))
    monkeypatch.setattr(torch.optim, 'AdamW', lambda *a, **kw: pytest.fail('Evaluation created optimizer'))
    out = tmp_path/'evaluation'
    if broken_report:
        def fail(*args, **kwargs):
            raise RuntimeError('report failure')
        monkeypatch.setattr(recipe, 'write_report', fail)
        with pytest.raises(RuntimeError, match='report failure'):
            recipe.evaluate_export(origin/'weights.pt', data, output=out, center_input=True)
    else:
        assert recipe.evaluate_export(origin/'weights.pt', data, output=out, center_input=True) is None
    result = json.loads((out/'result.json').read_text())
    coverage = json.loads((out/'input_coverage.json').read_text())
    assert result['completed'] and result['gate'] is False and result['task_scored'] is False
    assert result['coverage'] == expected_accepted/len(data)
    assert 'metrics' not in result
    assert coverage['accepted_histories'] == expected_accepted
    assert coverage['attempted_histories'] == len(data)
    assert coverage['valid_frames'] == valid.tolist()
    assert not (out/'predictions.npz').exists()
    assert file_hash(origin/'weights.pt') == source_hash
    assert all(torch.equal(p,v) for p,v in zip(m.parameters(), before))
    status = json.loads((out/'status.json').read_text())
    assert status['result']=='completed' and status['report']==('failed' if broken_report else 'structural-only')
    if not broken_report:
        html = (out/'report.html').read_text()
        assert 'not scored' in html and 'coverage' in html and 'not passed' in html
