import json

import numpy as np
import torch
from torch.nn import functional as F


def test_rendered_pairs_are_visible_balanced_and_split_isolated():
    from pathwm.data.visual_memory import VisualMemoryEpisodes

    a = VisualMemoryEpisodes(8, seed=3101)
    b = VisualMemoryEpisodes(8, seed=3201)
    assert set(a.identity["scene_sha256"]).isdisjoint(b.identity["scene_sha256"])
    assert set(a.identity["final_sha256"]).isdisjoint(b.identity["final_sha256"])
    batch = a.batch(range(16))
    assert batch["images"].shape == (16, 4, 3, 32, 32)
    assert batch["labels"].tolist() == [0, 1] * 8
    for i in range(0, 16, 2):
        left, right = batch["images"][i : i + 2]
        if (i // 2) % 2 == 0:
            assert torch.equal(left[0], right[0])
        else:
            assert torch.equal(left[0], right[1])
            assert torch.equal(right[0], left[1])
        assert torch.equal(left[2:], right[2:])
        delta = (left[1] - right[1]).abs().sum(0)
        assert delta[:, :16].sum() > 0 and delta[:, 16:].sum() > 0
        # The target's bright red center must actually be visible on the named side.
        for side in (0, 1):
            frame = batch["images"][i + side, 1]
            red = (frame[0] > 0.75) & (frame[1] < 0.35) & (frame[2] < 0.35)
            assert red.any()
            assert (
                (red.nonzero()[:, 1] < 16).all()
                if side == 0
                else (red.nonzero()[:, 1] >= 16).all()
            )


def test_real_agent_history_gradients_and_erasure_invariance():
    from experiments.multimodal import build_visual_memory
    from pathwm.data.visual_memory import VisualMemoryEpisodes

    torch.manual_seed(5)
    model = build_visual_memory()
    batch = VisualMemoryEpisodes(1, seed=7).batch([0, 1])
    images = batch["images"].requires_grad_()
    loss = F.cross_entropy(model(images), batch["labels"])
    loss.backward()
    assert images.grad[:, 1].abs().sum() > 0
    for module in (
        model.agent.encoders["image"],
        model.agent.updater,
        model.agent.thinker,
        model.head,
    ):
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()
        )
    erased = images.detach()[:, -1:].expand_as(images)
    model.eval()
    torch.manual_seed(9)
    x = model(erased[:1])
    torch.manual_seed(9)
    y = model(erased[1:])
    torch.testing.assert_close(x, y, rtol=0, atol=0)


def test_direct_weights_save_reload_without_optimization(tmp_path, monkeypatch):
    from experiments.multimodal import direct_visual_weights, build_visual_memory
    from pathwm.io import file_hash
    from pathwm.data.visual_memory import VisualMemoryEpisodes

    def forbidden(*a, **k):
        raise AssertionError(
            "Direct construction must not use backpropagation or optimizer steps"
        )

    monkeypatch.setattr(torch.Tensor, "backward", forbidden)
    monkeypatch.setattr(torch.optim.SGD, "step", forbidden)
    opts = dict(
        fit_pairs=2, development_pairs=2, test_pairs=2, device="cpu", max_seconds=120
    )
    direct_visual_weights(tmp_path / "full", **opts)
    model = build_visual_memory()
    payload = torch.load(tmp_path / "full/weights.pt", weights_only=True)
    model.load_state_dict(payload["model"])
    model.eval()
    data = VisualMemoryEpisodes(2, seed=3301)
    result = json.loads((tmp_path / "full/visual_memory.json").read_text())
    torch.manual_seed(3401)
    with torch.no_grad():
        logits = model(data.batch([0, 2])["images"]).numpy()
    np.testing.assert_allclose(
        logits, np.asarray(result["evaluation"]["logits"])[::2], rtol=0, atol=0
    )
    before = file_hash(tmp_path / "full/last.pt")
    direct_visual_weights(tmp_path / "full", resume=True, **opts)
    assert file_hash(tmp_path / "full/last.pt") == before
    assert result["erased"]["accuracy"] == 0.5
    assert result["optimizer_updates"] == 0
    assert (
        json.loads((tmp_path / "full/status.json").read_text())["result"] == "completed"
    )
