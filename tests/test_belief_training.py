import json
import torch

from experiments.multimodal import (
    build_model,
    SyntheticEpisodes,
    LearningState,
    objective,
    train,
)
from tests.test_multimodal_training import settings
from tests.test_runs import equal_tree


def test_new_objective_trains_grounding_memory_and_marking_without_teacher_gradients():
    torch.manual_seed(43)
    model = build_model(
        width=16, state_model="belief", memory_recent=2, memory_block=2, memory_blocks=1
    )
    data = SyntheticEpisodes(count=2, history=8, horizon=2)
    learner = LearningState(model, 2)
    losses, errors, raw = objective(
        learner, data.batch([0, 1]), history=8, horizon=2, dropout=0.25
    )
    assert errors.shape == (2,) and torch.isfinite(sum(losses.values()))
    assert {"image_mse", "audio_mse", "text_ce"} <= raw.keys()
    sum(losses.values()).backward()
    for module in (
        model.updater,
        model.dynamics,
        model.thinker,
        model.evidence_encoder,
        model.memory.compressors,
        model.memory.consolidators,
        model.memory.mark_score,
    ):
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()
        ), type(module).__name__
    assert all(p.grad is None for p in learner.target.parameters())
    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
    )
    assert "latent_nll" not in losses and raw["memory_tensor_bytes"] > 0


def test_belief_recipe_exact_resume_and_report(tmp_path):
    config = dict(
        settings(),
        state_model="belief",
        history=4,
        horizon=1,
        memory_recent=1,
        memory_block=1,
        memory_blocks=1,
        improve_every=2,
        validation_windows=2,
    )
    train(config, tmp_path / "full")
    train(config, tmp_path / "resume", stop_after=1)
    train(config, tmp_path / "resume", resume=True)
    full, resumed = [
        torch.load(tmp_path / name / "last.pt", weights_only=True)
        for name in ("full", "resume")
    ]
    for key in ("model", "optimizer", "sampler", "torch", "step"):
        equal_tree(full[key], resumed[key])
    assert [r for r in full["rows"] if r["split"] == "train"] == [
        r for r in resumed["rows"] if r["split"] == "train"
    ]
    inspection = torch.load(tmp_path / "resume/inspection.pt", weights_only=True)
    assert inspection["state"]["schema"] == "pathwm-belief-v1"
    assert inspection["state"]["memory"]["consolidated"] is not None
    assert (
        json.loads((tmp_path / "resume/status.json").read_text())["report"]
        == "structural_verified"
    )
