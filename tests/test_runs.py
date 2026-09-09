from pathlib import Path
import json
import random
import numpy as np
import pytest
import torch
from torch import nn
from pathwm.io import Run, seed_everything, evaluation_mode, training_mode, state_hash
from pathwm.models.encoders import CNNEncoder
from pathwm.models.decoders import ReconstructionDecoder
from pathwm.models.perception import Perception
from pathwm.data.images import Frames
from pathwm.training.perception import train_perception


def objective(output, batch):
    return {"rgb_mse": (output["rgb"] - batch["rgb"]).square().mean()}


def setup():
    seed_everything(111)
    model = Perception(
        CNNEncoder(width=16),
        {"rgb": ReconstructionDecoder(CNNEncoder(width=16).feature_spec)},
    )
    data = Frames(
        np.random.default_rng(10).integers(0, 256, (8, 64, 64, 3), dtype="uint8"),
        range(8),
        identity={"source": "test array"},
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    settings = dict(seed=111, steps=4, batch_size=2, grad_clip=1.0, evaluate_every=2)
    return model, data, opt, settings


def run(path, resume=False, stop_after=None):
    model, data, opt, settings = setup()
    train_perception(
        model,
        data,
        data,
        objective,
        opt,
        settings=settings,
        recipe=__file__,
        output=path,
        resume=resume,
        stop_after=stop_after,
    )
    return torch.load(path / "last.pt", map_location="cpu", weights_only=False)


def equal_tree(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            equal_tree(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal_tree(x, y)
    else:
        assert a == b


def test_full_and_resumed_training_are_identical(tmp_path):
    full = run(tmp_path / "full")
    run(tmp_path / "resume", stop_after=1)
    resumed = run(tmp_path / "resume", resume=True)
    for key in ("model", "optimizer", "sampler", "torch", "step"):
        equal_tree(full[key], resumed[key])
    train_full = [r for r in full["rows"] if r["split"] == "train"]
    train_resume = [r for r in resumed["rows"] if r["split"] == "train"]
    assert train_full == train_resume
    assert (
        json.loads((tmp_path / "resume/status.json").read_text())["result"]
        == "completed"
    )


def test_refuse_overwrite_and_changed_resume(tmp_path):
    run(tmp_path / "run", stop_after=1)
    with pytest.raises(FileExistsError):
        run(tmp_path / "run")
    model, data, opt, settings = setup()
    settings["batch_size"] = 3
    with pytest.raises(ValueError, match="Incompatible resume"):
        Run(
            tmp_path / "run",
            settings=settings,
            data={"train": data.identity, "validation": data.identity},
            recipe=__file__,
            model=model,
            optimizer=opt,
            device="cpu",
            resume=True,
        )


def test_partial_checkpoint_write_keeps_previous_transaction(tmp_path, monkeypatch):
    model, data, opt, settings = setup()
    path = tmp_path / "transaction"
    run = Run(
        path,
        settings=settings,
        data=data.identity,
        recipe=__file__,
        model=model,
        optimizer=opt,
        device="cpu",
    )
    old = (path / "last.pt").read_bytes()

    def interrupted_save(state, stream):
        stream.write(b"partial checkpoint")
        raise OSError("injected write interruption")

    with monkeypatch.context() as m:
        m.setattr(torch, "save", interrupted_save)
        run.step = 1
        with pytest.raises(OSError):
            run.save()
    assert (path / "last.pt").read_bytes() == old
    restored = Run(
        path,
        settings=settings,
        data=data.identity,
        recipe=__file__,
        model=model,
        optimizer=opt,
        device="cpu",
        resume=True,
    )
    expected = torch.Generator().manual_seed(settings["seed"] + 1009)
    assert np.array_equal(
        restored.sample(8, 4), torch.randint(8, (4,), generator=expected).numpy()
    )
    assert restored.step == 0


def test_validation_restores_rng_and_modes_and_rejects_buffer_mutation():
    model = nn.Sequential(nn.BatchNorm1d(4), nn.Dropout(0.5))
    seed_everything(3)
    before = state_hash(model)
    python_state, numpy_state, torch_state = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state().clone(),
    )
    with evaluation_mode(model):
        model(torch.rand(3, 4))
        random.random()
        np.random.rand()
    assert state_hash(model) == before and model.training
    assert random.getstate() == python_state
    assert np.array_equal(np.random.get_state()[1], numpy_state[1])
    assert torch.equal(torch.get_rng_state(), torch_state)
    with pytest.raises(RuntimeError, match="mutated model buffers"):
        with evaluation_mode(model):
            model[0].running_mean.add_(1)
    assert state_hash(model) == before


def test_frozen_parameters_and_buffers_stay_frozen():
    model = nn.Sequential(nn.BatchNorm1d(4), nn.Linear(4, 2))
    model[0].requires_grad_(False)
    before = state_hash(model[0])
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
    for _ in range(3):
        training_mode(model)
        opt.zero_grad()
        model(torch.randn(3, 4)).square().mean().backward()
        opt.step()
    assert state_hash(model[0]) == before
    assert all(p.grad is None for p in model[0].parameters())
    assert not any(
        id(p) in {id(x) for x in model[0].parameters()}
        for group in opt.param_groups
        for p in group["params"]
    )


def test_reporting_failure_does_not_hide_completed_training(tmp_path, monkeypatch):
    import pathwm.training.perception as module

    def fail(*args, **kwargs):
        raise RuntimeError("injected renderer error")

    with monkeypatch.context() as m:
        m.setattr(module, "write_report", fail)
        with pytest.raises(RuntimeError, match="injected renderer"):
            run(tmp_path / "failed_report")
    status = json.loads((tmp_path / "failed_report/status.json").read_text())
    assert status["result"] == "completed" and status["report"] == "failed"
    assert (tmp_path / "failed_report/last.pt").is_file()
    run(tmp_path / "failed_report", resume=True)
    assert (
        json.loads((tmp_path / "failed_report/status.json").read_text())["report"]
        == "structural_verified"
    )


def test_coco_mask_metrics_ignore_invalid_frames_independent_of_batch_size():
    from pathwm.training.perception import evaluate
    from experiments.perception import objective

    class MaskModel(nn.Module):
        def forward(self, rgb):
            return {"rgb": rgb, "mask": rgb[:, :1] * 4 - 2}

    frames = np.stack([np.full((4, 4, 3), v, dtype="uint8") for v in (0, 128, 255)])
    masks = np.ones((3, 1, 4, 4), dtype="float32")
    valid = np.ones_like(masks)
    valid[1] = 0
    data = Frames(frames, range(3), {"mask": masks, "valid": valid})
    a = evaluate(MaskModel(), data, objective, "cpu", batch_size=2)
    b = evaluate(MaskModel(), data, objective, "cpu", batch_size=3)
    assert a["mask_valid_frames"] == b["mask_valid_frames"] == 2
    assert a["mask_iou"] == b["mask_iou"] == 0.5
    assert a["mask_bce"] == pytest.approx(b["mask_bce"], abs=1e-7)


def test_checkpoint_is_safe_loadable_and_components_can_initialize_next_stage(tmp_path):
    from pathwm.io import load_component

    state = run(tmp_path / "perception")
    safe = torch.load(
        tmp_path / "perception/last.pt", map_location="cpu", weights_only=True
    )
    assert safe["step"] == 4
    encoder = CNNEncoder(width=16)
    load_component(encoder, tmp_path / "perception/last.pt", "encoder")
    for k, v in encoder.state_dict().items():
        assert torch.equal(v, state["model"]["encoder." + k])
    index = json.loads((tmp_path / "perception/source_index.json").read_text())
    assert str(Path(__file__).resolve()) in index
    assert (
        tmp_path / "perception" / index[str(Path(__file__).resolve())]
    ).read_bytes() == Path(__file__).read_bytes()
