import copy

import numpy as np
import torch


def test_probe_recovers_facts_and_uses_training_statistics():
    from pathwm.evaluation.modality_readout import (
        fit_factor_probe,
        predict_factor_probe,
    )

    rng = np.random.default_rng(12)
    labels = np.array([[a, b, c] for a in range(3) for b in range(3) for c in range(2)])
    facts = np.concatenate(
        [np.eye(n)[labels[:, i]] for i, n in enumerate((3, 3, 2))], 1
    )
    train = np.repeat(facts, 8, axis=0)
    y = np.repeat(labels, 8, axis=0)
    train = np.concatenate([train, rng.normal(size=(len(train), 2))], 1)
    validation = np.concatenate([facts, rng.normal(size=(len(facts), 2))], 1)
    reader, choices = fit_factor_probe(train, y, validation, labels)
    assert np.array_equal(predict_factor_probe(reader, validation), labels)
    np.testing.assert_allclose(reader["mean"], train.mean(0))
    assert len(choices) == 5
    assert np.isfinite(reader["weights"]).all()


