import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from experiments import modality_readout as recipe
from pathwm.data.request_meaning import request_corpus
from pathwm.data.understanding import UnderstandingData, synthetic_records
from pathwm.models.modalities import bytes_batch


def test_fresh_prefixes_keep_training_fixed_and_exclude_old_scored_questions():
    old = request_corpus()
    fresh = request_corpus(fresh=True)
    assert [r for r in fresh if r["split"] != "test"] == [
        r for r in old if r["split"] != "test"
    ]
    direct = [r for r in fresh if r["split"] == "test" and r["style"] == "direct"]
    assert len(direct) == 24
    assert not {r["question"] for r in old} & {r["question"] for r in direct}
    for i in range(0, len(direct), 2):
        first, sequence = direct[i : i + 2]
        assert first["pair"] == sequence["pair"] and (
            first["format"],
            sequence["format"],
        ) == ("first", "sequence")
        assert len(first["question"].encode()) == len(sequence["question"].encode())
    assert [r for r in fresh if r["style"] == "order"] == [
        r for r in old if r["style"] == "order"
    ]


def test_boundary_loss_uses_correct_utf8_positions_full_vocabulary_and_one_forward():
    texts = ["grün", "rot blau", "blau rot"]
    targets, _ = bytes_batch(texts)
    logits = torch.randn(3, targets.shape[1] - 1, 259, requires_grad=True)
    calls = []
    model = SimpleNamespace(outputs=lambda *a: calls.append(a) or logits)
    actual = recipe.grounded_objective(model, None, targets, boundary_weight=1.0)
    ordinary = F.cross_entropy(
        logits.flatten(0, 1), targets[:, 1:].flatten(), ignore_index=0
    ) / np.log(259)
    boundary = F.cross_entropy(
        logits[torch.arange(3), [5, 3, 4]], torch.tensor([2, 35, 35])
    ) / np.log(259)
    expected = ordinary + boundary
    assert len(calls) == 1
    assert torch.equal(actual, expected)
    assert torch.equal(
        torch.autograd.grad(actual, logits, retain_graph=True)[0],
        torch.autograd.grad(expected, logits)[0],
    )
    zero = recipe.grounded_objective(model, None, targets, boundary_weight=0.0)
    assert torch.equal(zero, ordinary)


@pytest.mark.parametrize("text", ["", " rot", "  "])
def test_boundary_training_rejects_empty_first_word(text):
    targets, _ = bytes_batch([text])
    model = SimpleNamespace(
        outputs=lambda *a: torch.zeros(1, targets.shape[1] - 1, 259)
    )
    with pytest.raises(ValueError, match="first word"):
        recipe.grounded_objective(model, None, targets, boundary_weight=1.0)


def test_paired_neutral_batch_preserves_evidence_and_frozen_parameters():
    torch.set_num_threads(2)
    torch.manual_seed(902)
    data = UnderstandingData.__new__(UnderstandingData)
    data.records, data.arrays, data.cases = synthetic_records("full")
    rows = recipe.grounded_records(
        data, "VID.order", request_contrasts=True, request_profile="paired"
    )
    assert len(rows) == 384 and all(r["split"] == "calibration" for r in rows)
    for i in range(0, len(rows), 2):
        a, b = rows[i : i + 2]
        assert (a["format"], b["format"]) == ("first", "sequence")
        assert a["choices"][a["answer"]] == b["choices"][b["answer"]].split()[0]
    inputs, targets = recipe.grounded_batch(
        data, rows, [0, 1], "cpu", question_mode="neutral"
    )
    full, _ = recipe.grounded_batch(data, rows, [0, 1], "cpu")
    assert torch.equal(inputs["video"].values, full["video"].values)
    assert torch.equal(inputs["text"].values[0], inputs["text"].values[1])
    model = recipe.Model("native")
    recipe.configure_request_readout(model, "instruction")
    recipe.configure_grounded_training(model, "interpreter")
    before = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=0.001
    )
    loss = recipe.grounded_objective(
        model,
        model.core(inputs, requests=[r["question"] for r in rows[:2]]),
        targets,
        boundary_weight=1.0,
    )
    loss.backward()
    optimizer.step()
    changed = {
        n for n, v in model.state_dict().items() if not torch.equal(before[n], v)
    }
    assert changed and all(
        n.startswith("core.agent.task_interpreter.") for n in changed
    )


@pytest.mark.parametrize('stage,scope', [('grounded', 'decoder'), ('request-diagnose', 'interpreter')])
def test_cli_rejects_fresh_profile_when_stage_cannot_use_it(monkeypatch, tmp_path, capsys, stage, scope):
    import sys

    monkeypatch.setattr(sys, 'argv', [
        'modality_readout', '--stage', stage, '--grounded-scope', scope,
        '--request-evaluation', 'fresh', '--core', str(tmp_path / 'missing'),
        '--output', str(tmp_path / 'output'), '--understanding-suite', str(tmp_path / 'missing'),
        '--steps', '1',
    ])
    with pytest.raises(SystemExit) as error:
        recipe.main()
    assert error.value.code == 2
    assert 'Fresh requests require' in capsys.readouterr().err
    assert not (tmp_path / 'output').exists()
