import torch
from torch import nn

from pathwm.evaluation.diagrams import CallFlow, architecture, mermaid, dot


def test_architecture_tracks_replacements_without_inventing_data_connections():
    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU())
    before = architecture(model, depth=2)
    assert before['nodes'][0]['parameters'] == 16
    assert before['edges'][0]['kind'] == 'contains'
    model[0] = nn.Linear(3, 7, bias=False)
    after = architecture(model, depth=2)
    assert after['nodes'][0]['parameters'] == 21
    assert before != after


def test_value_flow_preserves_forks_and_repeated_calls():
    model = nn.Linear(3, 3)
    flow = CallFlow()
    x = flow.input('input', torch.ones(1, 3))
    a = flow.call(model, x)
    b = flow.call(model, a)
    c = flow.call(model, a)
    flow.output('first branch', b)
    flow.output('second branch', c)
    graph = flow.graph()
    edges = {(e['source'], e['target']) for e in graph['edges']}
    assert {('n0', 'n1'), ('n1', 'n2'), ('n1', 'n3'), ('n2', 'n4'), ('n3', 'n5')} == edges
    assert ('n2', 'n3') not in edges
    assert [n['label'] for n in graph['nodes']][1:4] == ['Linear', 'Linear', 'Linear']


def test_diagram_bytes_do_not_depend_on_python_object_ids():
    def record():
        flow = CallFlow()
        x = flow.input('image "example"', torch.ones(1, 3))
        flow.output('result', flow.call(torch.neg, x))
        return flow.graph()
    first, second = record(), record()
    assert first == second
    assert mermaid(first) == mermaid(second)
    assert dot(first) == dot(second)
    assert '&quot;' in mermaid(first)
