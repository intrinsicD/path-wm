import copy

import pytest
import torch

from experiments import modality_readout as recipe
from pathwm.data.understanding import UnderstandingData, synthetic_records
from pathwm.evaluation.understanding import capture_stages


def fixtures():
    data = UnderstandingData.__new__(UnderstandingData)
    data.records, data.arrays, data.cases = synthetic_records("full")
    return data


def test_request_route_has_matched_weights_preserves_rng_and_no_request_behavior():
    torch.set_num_threads(2)
    torch.manual_seed(311)
    native = recipe.Model("native")
    constant, actual = copy.deepcopy(native), copy.deepcopy(native)
    rng = torch.get_rng_state().clone()
    recipe.configure_request_readout(constant, "constant")
    assert torch.equal(rng, torch.get_rng_state())
    recipe.configure_request_readout(actual, "instruction")
    assert recipe.state_hash(actual) == recipe.state_hash(constant)
    inputs = fixtures().inputs(fixtures().records[0])
    torch.set_rng_state(rng)
    expected = native.core(inputs)
    torch.set_rng_state(rng)
    torch.testing.assert_close(actual.core(inputs), expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="request"):
        actual.core(inputs, requests=["one", "two"])


def test_request_changes_thinking_not_belief_and_metadata_has_no_question_or_ids():
    torch.set_num_threads(2)
    torch.manual_seed(59)
    data = fixtures()
    record = next(r for r in data.records if r["case"] == "VID.order")
    model = recipe.Model("native")
    recipe.configure_request_readout(model, "instruction")
    recipe.configure_grounded_training(model, "core")
    inputs = data.inputs(record)
    traces, tokens, beliefs = [], [], []
    for question in ("Erste Farbe?", "Beide Farben"):
        trace = {}
        torch.manual_seed(72)
        out, state = model.core(
            inputs, requests=[question], trace=trace, return_state=True
        )
        tokens.append(out)
        beliefs.append(state)
        traces.append(trace)
    assert not torch.equal(tokens[0], tokens[1])
    assert torch.equal(beliefs[0].logits, beliefs[1].logits)
    assert torch.equal(beliefs[0].tokens, beliefs[1].tokens)
    assert torch.equal(traces[0]["task.metadata"], traces[1]["task.metadata"])
    assert (
        traces[0]["task.requests"][0]["task_id"]
        == traces[1]["task.requests"][0]["task_id"]
    )
    tokens[0].square().mean().backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.core.agent.task_interpreter.parameters()
    )
    assert all(p.grad is None for p in model.core.agent.encoders.parameters())
    # A second text-encoder call for instructions must not overwrite observation probes.
    torch.manual_seed(72)
    _, a = capture_stages(model.core, inputs, requests=["Erste Farbe?"])
    torch.manual_seed(72)
    _, b = capture_stages(model.core, inputs, requests=["Beide Farben"])
    assert torch.equal(a["encoder"], b["encoder"])


def test_order_requests_preserve_evidence_splits_and_balance_answer_length():
    data = fixtures()
    records = recipe.grounded_records(data, "VID.order", request_contrasts=True)
    assert len(records) == 64
    assert {r["split"] for r in records} == {"calibration"}
    assert len({r["id"] for r in records}) == 64
    base = next(r for r in data.records if r["case"] == "VID.order")
    variants = recipe.order_requests(base, novel=False)
    assert len(variants) == 4
    assert all(r["evidence"] == base["evidence"] for r in variants)
    assert (
        len({len(r["question"].encode()) for r in variants if r["wording"] == 1}) == 1
    )
    assert {r["format"] for r in variants} == {"first", "sequence"}
    for row in variants:
        value = row["choices"][row["answer"]]
        assert value.split()[0] == base["choices"][base["answer"]]
        assert len(value.split()) == (1 if row["format"] == "first" else 2)
    novel = recipe.order_requests(base, novel=True)
    assert set(r["question"] for r in novel).isdisjoint(r["question"] for r in variants)
    assert len({len(r["question"].encode()) for r in novel if r["wording"] == 0}) == 1
