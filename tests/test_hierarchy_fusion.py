import json
from types import SimpleNamespace

import numpy as np
import torch

from experiments.hierarchy_fusion import build_model, objective
from experiments.perception import build_model as perception_model
from pathwm.data.images import Frames
from pathwm.io import state_hash
from pathwm.training.perception import train_perception


def test_comparison_copies_all_shared_weights_and_recipe_uses_every_scale():
    shallow, anchor_hash = build_model(7498, "shallow")
    fused, other_hash = build_model(7498, "deep_fusion")
    assert anchor_hash == other_hash == state_hash(shallow)
    other = fused.state_dict()
    for name, value in shallow.state_dict().items():
        assert torch.equal(value, other[name]), name
    args = SimpleNamespace(
        encoder="pyramid",
        width=16,
        levels=3,
        stage_depth=2,
        fusion_depth=2,
        encoder_weights=None,
        freeze_encoder=False,
        dataset="coco",
        masks=True,
        rgb_only=False,
        diagnostic_rgb=False,
    )
    model = perception_model(args)
    assert tuple(model.heads["mask"].input.spec) == tuple(model.encoder.feature_spec)


def test_fused_hierarchy_training_resumes_exactly(tmp_path):
    frames = np.random.default_rng(9).integers(0, 256, (4, 64, 64, 3), dtype=np.uint8)
    data = Frames(
        frames,
        range(4),
        labels={
            "mask": (frames[:, :, :, 0] > 128)[:, None].astype("float32"),
            "valid": np.ones((4, 1, 64, 64), dtype="float32"),
        },
    )

    def run(path, resume=False, stop=None):
        model, _ = build_model(7498, "deep_fusion")
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003)
        settings = dict(
            seed=7498, steps=3, batch_size=1, evaluate_every=3, grad_clip=1.0
        )
        train_perception(
            model,
            data,
            data,
            objective,
            optimizer,
            settings=settings,
            recipe=__file__,
            output=path,
            resume=resume,
            stop_after=stop,
        )
        return torch.load(path / "last.pt", map_location="cpu", weights_only=True)

    full = run(tmp_path / "full")
    run(tmp_path / "resumed", stop=1)
    resumed = run(tmp_path / "resumed", resume=True)

    def same(a, b):
        if isinstance(a, torch.Tensor):
            assert torch.equal(a, b)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for k in a:
                same(a[k], b[k])
        elif isinstance(a, (tuple, list)):
            assert len(a) == len(b)
            for x, y in zip(a, b):
                same(x, y)
        else:
            assert a == b

    for key in ("model", "optimizer", "sampler", "torch", "step"):
        same(full[key], resumed[key])
    assert (
        json.loads((tmp_path / "resumed/status.json").read_text())["report"]
        == "structural_verified"
    )
