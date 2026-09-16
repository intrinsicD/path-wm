import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments import modality_readout as recipe
from pathwm.data.understanding import UnderstandingData, synthetic_records
from pathwm.data.request_meaning import request_corpus, apply_request
from pathwm.models.modalities import bytes_text


def fixtures():
    data = UnderstandingData.__new__(UnderstandingData)
    data.records, data.arrays, data.cases = synthetic_records("quick")
    data.identity = "routing-fixture"
    data.manifest = dict(profile="unit", gaps=[], limits=[])
    return data


def test_question_routes_preserve_evidence_and_token_length_without_label_leakage():
    data = fixtures()
    base = next(r for r in data.records if r["case"] == "VID.order")
    row = base | dict(question="Beide Farbtöne?", evidence=base["evidence"] | {"text": "Zeuge sagt: grün."})
    old = copy.deepcopy(row)
    full = data.inputs(row)
    for mode in ("full", "neutral", "masked"):
        inputs = data.inputs(row, question_mode=mode)
        q = row["question"] if mode == "full" else "." if mode == "neutral" else "." * len(row["question"].encode())
        assert bytes_text(inputs["text"].values[0]) == "Zeuge sagt: grün.\nFrage: " + q
        for key in ("values", "times"):
            assert torch.equal(getattr(inputs["video"], key), getattr(full["video"], key))
        if mode != "neutral":
            assert inputs["text"].values.shape == full["text"].values.shape
            assert torch.equal(inputs["text"].valid, full["text"].valid)
            assert torch.equal(inputs["text"].times, full["text"].times)
        changed = row | dict(id="arbitrary", answer=1-row["answer"], choices=["invalid", "labels"])
        assert torch.equal(inputs["text"].values, data.inputs(changed, question_mode=mode)["text"].values)
    assert row == old
    with pytest.raises(ValueError, match="question"):
        data.inputs(row, question_mode="typo")


def test_neutral_and_masked_routes_keep_physical_state_but_allow_task_differences():
    torch.set_num_threads(2)
    torch.manual_seed(174)
    data = fixtures()
    base = next(r for r in data.records if r["case"] == "VID.order")
    requests = request_corpus()[:2]
    model = recipe.Model("native").eval()
    recipe.configure_request_readout(model, "instruction")
    original = recipe.state_hash(model)
    for mode in ("neutral", "masked"):
        states, working = [], []
        for request in requests:
            row = apply_request(base, request)
            torch.manual_seed(981)
            with torch.no_grad():
                tokens, state = model.core(data.inputs(row, question_mode=mode), requests=[row["question"]], return_state=True)
            states.append(state)
            working.append(tokens)
        assert torch.equal(states[0].tokens, states[1].tokens)
        assert torch.equal(states[0].logits, states[1].logits)
        assert not torch.equal(working[0], working[1])
    assert recipe.state_hash(model) == original


def test_non_instruction_route_fails_before_any_evaluation_files(tmp_path, monkeypatch):
    from pathwm.evaluation.understanding import evaluate_understanding
    model = recipe.Model("native")
    monkeypatch.setattr(recipe, "restore_readout", lambda *a: (model, {}, "fixture"))
    for stage in ("request", "understanding"):
        output = tmp_path / stage
        with pytest.raises(ValueError, match="instruction"):
            if stage == "request":
                recipe.request_evaluate(SimpleNamespace(core=tmp_path, seed=7, device="cpu", output=output, observation_question="neutral"))
            else:
                evaluate_understanding(model, fixtures(), output, source={}, seed=7, device="cpu", recipe=Path(__file__), question_mode="neutral")
        assert not output.exists()
