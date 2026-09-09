import numpy as np
import torch
from pathwm.io import seed_everything, state_hash, trainable_parameters
from pathwm.models.encoders import CNNEncoder
from pathwm.models.temporal import MemoryUpdater, Predictor, WorldModel
from pathwm.training.dynamics import train_dynamics
from experiments.dynamics import objective
from tests.test_runs import equal_tree


class Sequences:
    identity = {"kind": "deterministic sequence fixture", "count": 4}

    def __init__(self):
        self.images = torch.from_numpy(
            np.random.default_rng(8).random((4, 4, 3, 64, 64), dtype="float32")
        )

    def __len__(self):
        return 4

    def batch(self, ids, device="cpu"):
        images = self.images[ids].to(device)
        n = len(ids)
        return dict(
            history=images[:, :2],
            history_actions=torch.full((n, 1, 2), 0.3, device=device),
            actions=torch.full((n, 2, 2), 0.7, device=device),
            initial_previous_action=torch.full((n, 2), -1.0, device=device),
            future=images[:, 2:],
        )


def fit(path, resume=False, stop_after=None):
    seed_everything(22)
    e = CNNEncoder(width=8).requires_grad_(False)
    m = WorldModel(
        e,
        MemoryUpdater(e.feature_spec, memory_width=16),
        Predictor(e.feature_spec, memory_width=16, depth=1),
    )
    initial = state_hash(e)
    opt = torch.optim.AdamW(trainable_parameters(m), lr=1e-3)
    settings = dict(
        seed=22, steps=3, batch_size=2, evaluate_every=2, grad_clip=1.0, horizon=2
    )
    train_dynamics(
        m,
        Sequences(),
        Sequences(),
        objective,
        opt,
        settings=settings,
        recipe=__file__,
        output=path,
        resume=resume,
        stop_after=stop_after,
    )
    assert state_hash(e) == initial
    assert all(p.grad is None for p in e.parameters())
    return torch.load(path / "last.pt", map_location="cpu", weights_only=True)


def test_dynamics_resume_preserves_all_trainable_state_and_frozen_encoder(tmp_path):
    full = fit(tmp_path / "full")
    fit(tmp_path / "resumed", stop_after=1)
    resumed = fit(tmp_path / "resumed", resume=True)
    for key in ("model", "optimizer", "sampler", "torch", "step"):
        equal_tree(full[key], resumed[key])
