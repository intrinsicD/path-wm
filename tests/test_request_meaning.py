from collections import Counter, defaultdict
import copy

import numpy as np
import torch

from pathwm.data.request_meaning import request_corpus
from pathwm.evaluation.request_meaning import capture_request_stages, request_probe
from experiments import modality_readout as recipe
from pathwm.data.understanding import UnderstandingData, synthetic_records


def test_corpus_balances_each_prefix_length_and_holds_out_templates():
    rows = request_corpus()
    assert len({r['question'] for r in rows}) == len(rows)
    direct = [r for r in rows if r['style'] == 'direct']
    families, pairs = defaultdict(set), defaultdict(list)
    for r in direct:
        families[r['split']].add(r['family'])
        pairs[r['pair']].append(r)
    assert families['calibration'].isdisjoint(families['test'] | families['validation'])
    assert families['validation'].isdisjoint(families['test'])
    assert [len(families[s]) for s in ('calibration','validation','test')] == [6,2,4]
    for pair in pairs.values():
        assert {r['label'] for r in pair} == {0,1}
        assert len({len(r['question'].encode()) for r in pair}) == 1
    stress = [r for r in rows if r['style'] == 'order']
    for family in {r['family'] for r in stress}:
        pair = [r for r in stress if r['family'] == family]
        assert len(pair)==2 and Counter(pair[0]['question'].encode()) == Counter(pair[1]['question'].encode())


def test_probe_test_labels_cannot_affect_fitted_weights_or_predictions():
    rows = [r for r in request_corpus() if r['style']=='direct']
    x = torch.randn(len(rows), 8, generator=torch.Generator().manual_seed(71)).double()
    result, fitted = request_probe(x, rows)
    changed = [r | dict(label=1-r['label']) if r['split']=='test' else r for r in rows]
    other, weights = request_probe(x, changed)
    assert np.array_equal(result['predictions'], other['predictions'])
    for key in fitted: torch.testing.assert_close(fitted[key], weights[key], rtol=0,atol=0)


def test_captured_request_stages_are_actual_path_and_preserve_physical_state():
    torch.set_num_threads(2)
    torch.manual_seed(93)
    model = recipe.Model('native')
    recipe.configure_request_readout(model,'instruction')
    model.eval()
    data = UnderstandingData.__new__(UnderstandingData)
    data.records, data.arrays, data.cases = synthetic_records('full')
    base = next(r for r in data.records if r['case']=='VID.order')
    inputs = data.inputs(base | dict(question='.'))
    _, state = model.core(inputs, return_state=True)
    before = copy.deepcopy(state)
    weights = recipe.state_hash(model)
    rng = torch.get_rng_state().clone()
    questions = [r['question'] for r in request_corpus()[:2]]
    captured = [capture_request_stages(model,state,q) for q in questions]
    assert recipe.state_hash(model)==weights and torch.equal(rng,torch.get_rng_state())
    assert torch.equal(before.tokens,state.tokens) and torch.equal(before.logits,state.logits)
    assert set(captured[0][0]) == {'encoder','instruction','task','working'}
    assert not torch.equal(captured[0][0]['task'],captured[1][0]['task'])
    # Diagnostic stores detached representations, never feeds labels/heads to generation.
    assert all(not x.requires_grad for x in captured[0][0].values())
