import copy

import numpy as np
import pytest
import torch


def test_partial_capability_gates_cannot_pass():
    from pathwm.evaluation.capabilities import case_record, score

    result = case_record('partial', {}, torch.zeros(1), {},
                         {'a': score(1., .8), 'b': score(None, .8)}, 'tiny', 'raw.npz')
    assert result['status'] == 'not_measured'


def test_cartesian_intervention_factors_are_conditionally_balanced():
    from pathwm.data.modality_readout import dataset, observations

    d = dataset('intervention', 7201)
    y = d['targets']['factors'].numpy()
    assert len(y) == 72 and len(set(d['ids'])) == 72
    for col, cardinality in enumerate((3, 3, 2)):
        rest = np.delete(y, col, axis=1)
        for key in np.unique(rest, axis=0):
            group = y[(rest == key).all(1), col]
            assert np.array_equal(np.bincount(group, minlength=cardinality),
                                  np.full(cardinality, 4))
    assert set(observations(d, 'complementary', omit='video')) == {'text', 'image', 'audio'}
    for split in ('train', 'validation', 'seen', 'heldout'):
        other = dataset(split, 7201)
        assert not set(other['ids']) & set(d['ids'])


def test_factor_diagnostics_validate_and_keep_failures_per_draw():
    from pathwm.evaluation.modality_suite import factor_diagnostics

    y = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 0]])
    pred = np.repeat(y[None], 3, axis=0)
    pred[2, 0, 0] = 1
    r = factor_diagnostics(pred, y, ['a', 'b', 'c'])
    assert r['min_joint_accuracy'] == pytest.approx(2 / 3)
    assert r['min_factor_accuracy'][0] == pytest.approx(2 / 3)
    assert r['failures'][0]['example_id'] == 'a'
    assert r['failures'][0]['draw'] == 2
    assert len(r['by_combination']) == 3
    for broken in (pred.astype(float) + .5, pred[:, :-1], np.full_like(pred, 4),
                   np.full(pred.shape, np.nan)):
        with pytest.raises(ValueError):
            factor_diagnostics(broken, y, ['a', 'b', 'c'])
    with pytest.raises(ValueError):
        factor_diagnostics(pred, y, ['a', 'a', 'c'])


def test_suite_keeps_missing_cases_and_does_not_promote_probes():
    from pathwm.evaluation.modality_suite import build_suite

    report = build_suite({}, [], source={'checkpoint': 'abc'}, seed=7201)
    assert report['coverage']['passed'] == 0
    assert report['coverage']['not_implemented'] > 0
    assert report['coverage']['not_run'] > 0
    ids = [c['id'] for c in report['cases']]
    assert len(ids) == len(set(ids))
    assert {'text', 'image', 'audio', 'video', 'core', 'action'} <= {c['modality'] for c in report['cases']}
    bogus = [{'stage': 'encoder', 'factor_accuracy': [1, 1, 1], 'all_correct': 1}]
    other = build_suite({}, bogus, source={'checkpoint': 'abc'}, seed=7201)
    assert other['coverage'] == report['coverage']
    assert all(c['assessment'] == 'not_scored' for c in other['cases'])


def test_suite_requires_each_draw_and_every_factor_and_separates_omission():
    from pathwm.evaluation.modality_suite import build_suite, factor_diagnostics

    y = np.array([[a,b,c] for a in range(3) for b in range(3) for c in range(2)])
    ids = [str(i) for i in range(len(y))]
    pred = np.repeat(y[None], 3, axis=0)
    good = factor_diagnostics(pred, y, ids)
    measured = {f'{split}.{mode}': copy.deepcopy(good)
                for split in ('seen', 'heldout')
                for mode in ('text', 'image', 'audio', 'video', 'all', 'complementary')}
    broken = pred.copy(); broken[1, :, 2] = 1 - broken[1, :, 2]
    measured['heldout.video'] = factor_diagnostics(broken, y, ids)
    measured['intervention.complementary'] = good
    missing = pred.copy(); missing[:, :, 0] = 0
    measured['intervention.without_image'] = factor_diagnostics(missing, y, ids)
    report = build_suite(measured, [], source={'checkpoint': 'abc'}, seed=7201)
    by_id = {c['id']: c for c in report['cases']}
    assert by_id['VID.symbolic.heldout']['assessment'] == 'fail'
    assert by_id['TXT.symbolic.heldout']['assessment'] == 'pass'
    assert by_id['CORE.source_image']['assessment'] == 'pass'
    assert by_id['CORE.source_audio']['assessment'] == 'not_scored'
    assert by_id['VID.natural_motion']['implementation'] == 'not_implemented'
