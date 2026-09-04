"""One-step smoke coverage for training, checkpoint reload, fixed probes, and panel output."""
from __future__ import annotations

import copy

import torch

from evaluation import evaluate_checkpoint
from training.data import EpisodeStore, PairedInterventionStore
from training.e1 import _encode_counterfactuals, train_e1
from training.model import build_models


def test_one_step_training_writes_reloadable_metrics(cfg, tmp_path):
    spec = copy.deepcopy(cfg)
    spec["env"].update(n_worlds=2, episode_len=1, n_transitions=2)
    spec["encoder"].update(dim=32, layers=1, heads=4, params_target=None)
    spec["predictor"].update(dim=32, layers=1, heads=4, params_target=None)
    spec["losses"]["reg"].update(weight=0.1, projections=8, knots=5, per_token=False)
    spec["losses"]["rollout"].update(h_train=1, delta_t_max=1)
    spec["train"].update(steps=1, batch_size=2)
    spec["train"]["diagnostic_checkpoint_steps"] = [0, 1]
    spec["probe_set"].update(count=2, horizon=1)
    spec["counterfactual_data"]["probe_groups"] = 2

    observations = torch.randint(
        0, 256, (2, 2, 3, 64, 64), dtype=torch.uint8, generator=torch.Generator().manual_seed(9)
    )
    actions = torch.rand(2, 1, 2, generator=torch.Generator().manual_seed(10)) * 2.0 - 1.0
    store = EpisodeStore(observations, actions)
    checkpoint, final_training = train_e1(spec, store, tmp_path, seed=0, device=torch.device("cpu"))
    metrics = evaluate_checkpoint(checkpoint, tmp_path, torch.device("cpu"))

    assert checkpoint.exists() and (tmp_path / "training.jsonl").exists()
    assert (tmp_path / "checkpoints" / "step_000000.pt").exists()
    assert (tmp_path / "checkpoints" / "step_000001.pt").exists()
    assert (tmp_path / "metrics.json").exists()
    assert final_training["step"] == 1
    assert {
        "counterfactual_accuracy",
        "transition_error_identity",
        "transition_error_zero_action",
        "transition_error_shuffled_action",
    } <= metrics.keys()
    assert all(torch.isfinite(torch.tensor(value)) for key, value in metrics.items() if "error" in key or "ratio" in key)


def test_stage_two_training_consumes_paired_interventions(cfg, tmp_path):
    spec = copy.deepcopy(cfg)
    spec["env"].update(n_worlds=2, episode_len=1, n_transitions=2)
    spec["encoder"].update(dim=32, layers=1, heads=4, params_target=None)
    spec["predictor"].update(dim=32, layers=1, heads=4, params_target=None)
    spec["losses"]["reg"].update(weight=0.0, projections=8, knots=5, per_token=False)
    spec["losses"]["rollout"].update(h_train=1, delta_t_max=1)
    spec["losses"]["inverse"]["weight"] = 0.0
    spec["losses"]["counterfactual"].update(weight=1.0, k=3, kappa=0.1, batch_size=2)
    spec["train"].update(steps=2, batch_size=2)

    observations = torch.randint(
        0, 256, (2, 2, 3, 64, 64), dtype=torch.uint8, generator=torch.Generator().manual_seed(12)
    )
    actions = torch.rand(2, 1, 2, generator=torch.Generator().manual_seed(13)) * 2.0 - 1.0
    paired_initial = torch.randint(
        0, 256, (3, 3, 64, 64), dtype=torch.uint8, generator=torch.Generator().manual_seed(14)
    )
    paired_next = torch.randint(
        0, 256, (3, 3, 3, 64, 64), dtype=torch.uint8, generator=torch.Generator().manual_seed(15)
    )
    paired_actions = torch.rand(3, 3, 2, generator=torch.Generator().manual_seed(16)) * 2.0 - 1.0

    _, final = train_e1(
        spec,
        EpisodeStore(observations, actions),
        tmp_path,
        seed=0,
        device=torch.device("cpu"),
        counterfactual_store=PairedInterventionStore(paired_initial, paired_next, paired_actions),
    )

    assert final["step"] == 2
    assert final["counterfactual"] > 0.0
    assert final["counterfactual_positive"] >= 0.0
    assert 0.0 <= final["counterfactual_accuracy"] <= 1.0


def test_counterfactual_context_gradient_can_be_routed_to_predictor_only(cfg):
    spec = copy.deepcopy(cfg)
    spec["encoder"].update(dim=32, layers=1, heads=4, params_target=None)
    spec["predictor"].update(dim=32, layers=1, heads=4, params_target=None)
    models = build_models(spec, torch.device("cpu"))
    initial = torch.randint(0, 256, (2, 3, 64, 64), dtype=torch.uint8)
    following = torch.randint(0, 256, (2, 3, 3, 64, 64), dtype=torch.uint8)

    joint = _encode_counterfactuals(models, initial, following, context_gradient=True)
    predictor_only = _encode_counterfactuals(models, initial, following, context_gradient=False)

    assert joint[:, 0].requires_grad
    assert not predictor_only.requires_grad
