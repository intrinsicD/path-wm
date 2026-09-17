import base64
import json
import zlib

import pytest
import torch
from torch import nn

from pathwm.evaluation.explorer import capture, write_explorer


class Branches(nn.Module):
    def __init__(self):
        super().__init__()
        self.left = nn.Linear(3, 3, bias=False)
        self.right = nn.Linear(3, 3, bias=False)
        self.unused = nn.Linear(3, 2)
        self.alias = self.left

    def run(self, x):
        left = self.left(x)
        right = self.right(x.detach())
        return left + right.detach()


def example():
    model = Branches()
    x = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    return model, x


def test_capture_hierarchy_shared_weights_and_real_gradients():
    model, x = example()
    data = capture(model, lambda m: m.run(x).square().mean())
    assert {"left", "right", "unused", "alias"} <= set(data["modules"])
    assert data["modules"]["alias"]["alias_of"] == "left"
    params = data["parameters"]
    assert params["left.weight"]["values"] == model.left.weight.flatten().tolist()
    assert params["left.weight"]["gradient"] is not None
    assert params["right.weight"]["gradient"] is None
    assert params["unused.weight"]["gradient"] is None
    assert "alias.weight" in params["left.weight"]["aliases"]
    assert data["modules"]["unused"]["calls"] == 0
    assert any(op["scope"] == "left" for op in data["operations"])
    assert any(op["scope"] == "right" for op in data["operations"])
    assert any(op["method"] == "run" for op in data["operations"])
    leaves = {n.get("parameter") for n in data["backward"]["nodes"]}
    assert "left.weight" in leaves
    assert "right.weight" not in leaves
    assert all(n["executed"] for n in data["backward"]["nodes"])


def test_tensor_dependencies_do_not_connect_independent_branches():
    model, x = example()
    data = capture(model, lambda m: m.run(x).sum())
    ops = {op["id"]: op for op in data["operations"]}
    tensors = data["tensors"]
    for op in data["operations"]:
        if op["scope"] == "right":
            for tensor in op["inputs"]:
                producer = tensors[tensor].get("producer")
                assert not producer or ops[producer]["scope"] != "left"
    assert any("detach" in op["name"] for op in ops.values())


def test_capture_preserves_original_state_gradients_rng_and_modes():
    model, x = example()
    model.train()
    model.right.eval()
    model.left.weight.grad = torch.full_like(model.left.weight, 7)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    rng = torch.random.get_rng_state().clone()
    data = capture(model, lambda m: (m.run(x) + torch.rand(1)).sum())
    assert data["metadata"]["copied_model"] is True
    assert torch.equal(rng, torch.random.get_rng_state())
    assert all(torch.equal(state[k], v) for k, v in model.state_dict().items())
    assert torch.equal(model.left.weight.grad, torch.full_like(model.left.weight, 7))
    assert x.grad is None
    assert model.training and not model.right.training


def test_failure_removes_instrumentation_and_restores_rng():
    import sys

    model, x = example()
    profile = sys.getprofile()
    rng = torch.random.get_rng_state().clone()

    def fail(m):
        m.run(x)
        torch.rand(1)
        raise RuntimeError("deliberate")

    with pytest.raises(RuntimeError, match="deliberate"):
        capture(model, fail)
    assert sys.getprofile() is profile
    assert torch.equal(rng, torch.random.get_rng_state())
    assert model.run(x).shape == (1, 3)


def test_standalone_contains_complete_roundtrippable_manifest(tmp_path):
    model, x = example()
    data = capture(model, lambda m: m.run(x).sum(), metadata={"title": "</script>"})
    path = write_explorer(data, tmp_path / "explorer.html")
    html = path.read_text()
    payload = html.split('<script id="snapshot" type="application/octet-stream">')[1].split("</script>")[0]
    decoded = json.loads(zlib.decompress(base64.b64decode(payload)))
    assert decoded["parameters"] == data["parameters"]
    assert decoded["metadata"]["title"] == "</script>"
    assert "https://" not in html and "http://" not in html
    assert "DecompressionStream" in html


def test_capture_rejects_non_scalar_or_non_differentiable_loss():
    model, x = example()
    with pytest.raises(ValueError, match="scalar"):
        capture(model, lambda m: m.run(x))
    with pytest.raises(ValueError, match="differentiable"):
        capture(model, lambda m: m.run(x).sum().detach())
