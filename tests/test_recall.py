import pytest
import torch

from pathwm.models.tasks import Actor, TaskRequest
from pathwm.models.recall import (
    SeenRecord, RecallQuery, historical_target, select_recall, verify_recall,
)
from pathwm.evaluation.recall import fit_temperature, recall_metrics


def query(cutoff=3, entity=7):
    return RecallQuery(TaskRequest("q", "Where was it last seen?", Actor("user", "test")),
                       "session", entity, cutoff)


def test_historical_cutoff_and_verification_do_not_conflate_absence_with_abstention():
    records = (SeenRecord(1, 7, 2), SeenRecord(2, 1, 0), SeenRecord(3, 7, 1))
    assert historical_target(records, query(1)) == 2
    assert historical_target(records, query()) == 1
    assert historical_target(records, query(entity=8)) == 4
    with pytest.raises(ValueError, match="ordered"):
        historical_target(records[::-1], query())
    decision = select_recall(torch.tensor([0., 0., 0., 0., 1.]), query())
    assert verify_recall(query(), decision, records, "session")["verification"] == "contradicted"
    with pytest.raises(ValueError, match="session"):
        verify_recall(query(), decision, records, "different")


def test_exact_cost_boundary_invalid_probabilities_and_zero_answer_denominator():
    q = query()
    tied = select_recall(torch.tensor([.75, .25, 0., 0., 0.]), q)
    assert tied.answer is None and tied.reason == "cost"
    assert select_recall(torch.tensor([.751, .249, 0., 0., 0.]), q).answer == 0
    assert select_recall(torch.tensor([float("nan")] * 5), q).reason == "invalid"
    metrics = recall_metrics(torch.zeros(2, 5), torch.tensor([0, 4]))
    assert metrics["coverage"] == 0 and metrics["answered_error"] is None
    assert metrics["task_loss"] == .25


def test_temperature_fits_only_supplied_labels_and_preserves_factual_argmax():
    logits = torch.tensor([[4., 0., 0., 0., 0.], [0., 4., 0., 0., 0.],
                           [0., 0., 4., 0., 0.], [0., 0., 0., 4., 0.]])
    labels = torch.tensor([0, 2, 2, 0])
    before = logits.clone()
    fit = fit_temperature(logits, labels)
    assert fit["status"] == "fitted" and fit["nll_after"] <= fit["nll_before"]
    assert torch.equal(logits, before)
    assert torch.equal(logits.argmax(-1), (logits / fit["temperature"]).argmax(-1))
    assert not logits.requires_grad
    failed = fit_temperature(torch.full((2, 5), float("nan")), labels[:2])
    assert failed["status"] == "failed" and failed["temperature"] == 1.
