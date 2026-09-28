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
    # Backward is explicitly restricted to model parameters: external input
    # AccumulateGrad nodes exist in the graph but must not mutate x.grad.
    assert all(n["executed"] for n in data["backward"]["nodes"] if n.get("parameter"))
    assert any(n["executed"] for n in data["backward"]["nodes"])


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
    payload = html.split('<script id="snapshot" type="application/octet-stream">')[
        1
    ].split("</script>")[0]
    decoded = json.loads(zlib.decompress(base64.b64decode(payload)))
    assert decoded["parameters"] == data["parameters"]
    assert decoded["metadata"]["title"] == "</script>"
    assert "<script src=" not in html and '<link rel="stylesheet"' not in html
    assert "DecompressionStream" in html


def test_capture_rejects_non_scalar_or_non_differentiable_loss():
    model, x = example()
    with pytest.raises(ValueError, match="scalar"):
        capture(model, lambda m: m.run(x))
    with pytest.raises(ValueError, match="differentiable"):
        capture(model, lambda m: m.run(x).sum().detach())


def test_gradients_match_uninstrumented_reference_exactly():
    model, x = example()
    reference = torch.autograd.grad(model.run(x).square().mean(), model.left.weight)[0]
    result = capture(model, lambda m: m.run(x).square().mean())
    assert (
        result["parameters"]["left.weight"]["gradient"] == reference.flatten().tolist()
    )


def test_views_depend_on_intervening_inplace_writes():
    model = nn.Linear(3, 3)

    def execute(m):
        value = m(torch.ones(1, 3))
        view = value.view(3)
        value.add_(2)
        return view.sum()

    result = capture(model, execute)
    write = next(o for o in result["operations"] if o["name"] == "aten.add_.Tensor")
    read = next(o for o in result["operations"] if o["name"] == "aten.sum.default")
    assert set(write["outputs"]) & set(read["inputs"])
    assert all(
        int(result["tensors"][t]["producer"][1:]) < int(o["id"][1:])
        for o in result["operations"]
        for t in o["inputs"]
        if result["tensors"][t]["producer"] is not None
    )


def test_strict_checkpoint_rejects_missing_keys_and_loads_all_values(tmp_path):
    from pathwm.evaluation.explorer import load_explorer_weights

    model = nn.Linear(3, 2)
    saved = {
        key: torch.full_like(value, 0.25) for key, value in model.state_dict().items()
    }
    path = tmp_path / "checkpoint.pt"
    torch.save(saved, path)
    info = load_explorer_weights(model, path)
    assert info["sha256"] and all(
        torch.equal(saved[k], v) for k, v in model.state_dict().items()
    )
    torch.save({"weight": saved["weight"]}, path)
    with pytest.raises(ValueError, match="exactly match"):
        load_explorer_weights(model, path)


def test_live_server_rebuilds_source_and_surfaces_failures(tmp_path):
    import socket
    import subprocess
    import sys
    import time
    from urllib.request import urlopen

    root = tmp_path / "project"
    (root / "pathwm").mkdir(parents=True)
    source = root / "pathwm/model.py"
    source.write_text("first")
    output = tmp_path / "viewer.html"
    output.write_text("first snapshot")
    writer = tmp_path / "writer.py"
    writer.write_text(
        "from pathlib import Path\nimport sys\np=Path(sys.argv[1])\n"
        "text=Path(sys.argv[2]).read_text()\n"
        "assert text != 'broken', 'deliberate export failure'\n"
        "p.write_text(text+' snapshot')\n"
    )
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    command = [sys.executable, str(writer), str(output), str(source)]
    code = (
        "from pathwm.evaluation.explorer import serve_explorer; "
        f"serve_explorer({str(output)!r},root={str(root)!r},command={command!r},port={port})"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    url = f"http://127.0.0.1:{port}"

    def wait_for(expected):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                with urlopen(url + "/__explorer_status", timeout=1) as response:
                    status = json.load(response)
                if status["state"] == expected:
                    return status
            except OSError:
                pass
            time.sleep(0.05)
        pytest.fail(f"Preview did not reach {expected}")

    try:
        first = wait_for("current")
        source.write_text("second")
        wait_for("rebuilding")
        second = wait_for("current")
        assert first["version"] != second["version"]
        assert output.read_text() == "second snapshot"
        source.write_text("broken")
        status = wait_for("error")
        assert "deliberate export failure" in status["error"]
        assert output.read_text() == "second snapshot"
        source.write_text("repaired")
        wait_for("rebuilding")
        wait_for("current")
        assert output.read_text() == "repaired snapshot"
    finally:
        process.terminate()
        process.communicate(timeout=5)


def test_live_sources_include_world_state_recipe(tmp_path):
    from pathwm.evaluation.explorer import watched_sources

    assert tmp_path / "experiments/world_state.py" in watched_sources(tmp_path)
    assert tmp_path / "pathwm/evaluation/explorer-system.js" in watched_sources(
        tmp_path
    )


