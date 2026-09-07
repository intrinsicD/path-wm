"""Small controls guarding evaluation timing and aggregation, without training."""

from dataclasses import dataclass

import numpy as np
import torch

from world_model.paddle.evaluation import (
    assimilate_history,
    matched_rollout,
    summarize_prediction_records,
    fit_linear_velocity_probe,
)
from world_model.paddle.types import PlanningState


@dataclass
class ScalarLatent:
    fine: torch.Tensor
    coarse: torch.Tensor

    def clone(self):
        return ScalarLatent(self.fine.clone(), self.coarse.clone())


class ScalarEncoder:
    def __call__(self, frames):
        value = frames[:, :1, :1, :1].reshape(-1, 1, 1)
        return ScalarLatent(value, value)


class RecordingUpdater:
    def __init__(self):
        self.calls = []

    def __call__(self, memory, observation, action):
        self.calls.append((memory.clone(), observation.fine.clone(), action.clone()))
        return memory + observation.fine[:, 0, 0, None] + action[:, :1]


def test_observer_assimilates_each_real_frame_once_with_previous_action():
    updater = RecordingUpdater()
    frames = np.stack([np.full((64, 64, 3), v, dtype=np.uint8) for v in (0, 64, 128)])
    state = assimilate_history({"E": ScalarEncoder(), "U": updater}, frames, np.array([1, 0]), "cpu")
    assert len(updater.calls) == 3
    assert torch.equal(updater.calls[0][2], torch.zeros(1, 3))
    assert updater.calls[1][2].tolist() == [[0, 1, 0]]
    assert updater.calls[2][2].tolist() == [[1, 0, 0]]
    assert torch.allclose(state.memory, torch.full((1, 128), 192 / 255 + 1))


def test_copy_and_reset_controls_use_matched_actions_and_do_not_mutate_source():
    updater = RecordingUpdater()

    def predictor(observation, memory, action):
        value = observation.fine + memory[:, :1, None] + action[:, 2:3, None]
        return ScalarLatent(value, value)

    initial = PlanningState(ScalarLatent(torch.ones(1, 1, 1), torch.ones(1, 1, 1)), torch.full((1, 128), 4.0))
    result = matched_rollout({"P": predictor, "U": updater}, initial, torch.tensor([[2, 0]]))
    assert result["prediction"][0].observation.fine.item() == 6
    assert result["prediction"][1].observation.fine.item() == 16
    assert [s.observation.fine.item() for s in result["copy"]] == [1, 1]
    assert result["copy"][1].memory[0, 0] == 7
    assert result["reset"][0].observation.fine.item() == 3
    assert initial.memory[0, 0] == 4
    assert initial.observation.fine.item() == 1


def test_prediction_summary_preserves_horizon_collision_and_coordinate_denominators():
    rows = [
        {"method": "prediction", "horizon": 1, "collision": False, "h_abs_error": [1, 2, 3], "r_abs_error": [1, 2, 3, 4, 5], "latent_error": 2.0},
        {"method": "prediction", "horizon": 1, "collision": True, "h_abs_error": [3, 4, 5], "r_abs_error": [3, 4, 5, 6, 7], "latent_error": 4.0},
        {"method": "prediction", "horizon": 2, "collision": False, "h_abs_error": [9, 8, 7], "r_abs_error": [9, 8, 7, 6, 5], "latent_error": 8.0},
    ]
    result = summarize_prediction_records(rows)
    assert result["prediction"]["1"]["all"]["count"] == 2
    assert result["prediction"]["1"]["all"]["h_mae"] == [2, 3, 4]
    assert result["prediction"]["1"]["collision"]["h_mae"] == [3, 4, 5]
    assert result["prediction"]["2"]["collision"]["count"] == 0


def test_current_frame_linear_probe_cannot_separate_identical_images():
    features = torch.tensor([[0., 1.], [0., 1.], [1., 0.], [1., 0.]])
    velocity = torch.tensor([[-6., 3.], [6., 3.], [-2., 2.], [2., 2.]])
    probe = fit_linear_velocity_probe(features, velocity, ridge=1e-3)
    estimates = probe(features)
    assert torch.equal(estimates[0], estimates[1])
    assert torch.equal(estimates[2], estimates[3])
    assert torch.allclose(estimates[:, 0], torch.zeros(4), atol=1e-5)
