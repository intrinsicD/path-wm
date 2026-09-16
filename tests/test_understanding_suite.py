import copy

import numpy as np
import pytest
import torch


def test_contrast_records_are_balanced_and_temporal_controls_are_identical():
    from pathwm.data.understanding import synthetic_records, validate_records

    records, arrays, cases = synthetic_records('quick')
    validate_records(records, arrays)
    assert len(cases) == 10
    for case in cases:
        rows = [r for r in records if r['case'] == case['id'] and r['split'] == 'test']
        assert {r['answer'] for r in rows} == {0, 1}
        for a, b in zip(rows[::2], rows[1::2]):
            assert a['pair'] == b['pair'] and a['question'] == b['question']
            assert a['choices'] == b['choices'] and a['answer'] != b['answer']
            if case['id'] in ('VID.order', 'VID.history'):
                assert np.array_equal(arrays[a['evidence']['video']][-1], arrays[b['evidence']['video']][-1])
    broken = copy.deepcopy(records)
    broken[1]['answer'] = broken[0]['answer']
    with pytest.raises(ValueError, match='pair'):
        validate_records(broken, arrays)


def test_source_groups_cannot_cross_calibration_and_test():
    from pathwm.data.understanding import synthetic_records, validate_records

    records, arrays, _ = synthetic_records('quick')
    a = next(r for r in records if r['split'] == 'calibration')
    b = next(r for r in records if r['split'] == 'test')
    b['groups'] = a['groups']
    with pytest.raises(ValueError, match='group'):
        validate_records(records, arrays)


def test_prior_guessing_cannot_pass_grounded_pairs_and_probes_cannot_promote():
    from pathwm.evaluation.understanding import task_metrics

    labels = np.array([0, 1, 0, 1])
    pairs = ['a', 'a', 'b', 'b']
    guessed = np.zeros((3, 4), dtype=int)
    score = task_metrics(guessed, guessed, guessed, labels, pairs, 2)
    assert score['accuracy_min'] == .5 and score['paired_min'] == 0
    assert not score['passed']
    correct = np.tile(labels, (3, 1))
    good = task_metrics(correct, guessed, guessed, labels, pairs, 2)
    assert good['passed']
    with pytest.raises(ValueError):
        task_metrics(correct[:1], guessed, guessed, labels, pairs, 2)


def test_reference_refuses_changed_fixtures_profile_or_readout_contract():
    from pathwm.evaluation.understanding import compare_understanding

    a = {'contract': {'fixtures': 'a', 'profile': 'quick', 'readout': 'mc-v1'},
         'cases': [{'id': 'T', 'metrics': {'accuracy_min': .5, 'paired_min': 0.0}, 'passed': False}]}
    assert compare_understanding(a, a)[0]['status'] == 'unchanged'
    for field in a['contract']:
        b = copy.deepcopy(a); b['contract'][field] = 'changed'
        with pytest.raises(ValueError, match='contract'):
            compare_understanding(a, b)


def test_diagnostic_capture_is_observational_and_choice_scoring_uses_actual_decoder():
    from experiments.modality_readout import Model
    from pathwm.data.modality_readout import dataset, observations
    from pathwm.evaluation.understanding import capture_stages, choice_scores
    from pathwm.models.modalities import bytes_batch

    model = Model('native').eval()
    inputs = observations(dataset('train'), 'audio', [0])
    rng = torch.get_rng_state().clone()
    expected = model.core(inputs); after = torch.get_rng_state().clone()
    torch.set_rng_state(rng)
    tokens, stages = capture_stages(model.core, inputs)
    assert torch.equal(tokens, expected) and torch.equal(torch.get_rng_state(), after)
    assert set(stages) == {'encoder', 'posterior', 'working'}
    assert all(not t.requires_grad for t in stages.values())
    candidates = ['ja', 'nein']
    actual = choice_scores(model.outputs, tokens, candidates)
    for i, answer in enumerate(candidates):
        target, valid = bytes_batch([answer])
        logits = model.outputs('text', tokens, target[:, :-1])
        expected_score = logits.log_softmax(-1).gather(-1, target[:, 1:, None]).squeeze(-1)[valid[:, 1:]].mean()
        torch.testing.assert_close(actual[i], expected_score)
