"""Memory-conditioned residual editing of native fine codes: software contracts.

Actual native shapes (64-wide 16x16 fine codes, full SlotPerception). The store test
uses the real UnifiedAgent/WorldStore/ExactRetriever with the qualified J6000
perception; the request binding there is supplied by the caller, not learned.
"""
from pathlib import Path

import pytest
import torch
from torch import nn

from pathwm.data import rule_world as rw
from pathwm.models.features import FeatureSpec
from pathwm.models.slots import SlotPerception

J6000 = Path("runs/real_visual_joint_repair_3501_u6000_v1")


def native():
    torch.manual_seed(5)
    return SlotPerception().eval().requires_grad_(False)


def generator(objective="flow"):
    """The registered full producer configuration (depth 2, fusion 1, hidden 64, 16 steps)."""
    from experiments.unified_session import EDIT_STEPS, edit_model
    torch.manual_seed(6)
    return edit_model(objective, EDIT_STEPS)["generator"].eval()


def test_reconstruct_bypass_never_calls_generator_or_rng():
    from pathwm.models.conditional_image import edit_code

    class Raising(nn.Module):
        def features(self, *a, **k):
            raise AssertionError("generator called during reconstruct")

    z = torch.randn(2, 256, 64)
    state = torch.get_rng_state().clone()
    out = edit_code("reconstruct", z, generator=Raising())
    assert out is z and torch.equal(state, torch.get_rng_state())
    with pytest.raises(ValueError):
        edit_code("create", z, generator=Raising())


def test_edit_is_source_plus_generated_delta_and_valid_generated_code():
    from pathwm.models.conditional_image import ComponentRequest, edit_code
    from pathwm.models.image_code import code_from_values, decode_image_code
    p, g, request = native(), generator(), ComponentRequest(64)
    with torch.no_grad():
        z = p.pyramid(torch.rand(2, 3, 64, 64)).scales[0].values
        context = request.context(z, torch.randn(2, 64), torch.tensor([0, 1]))
        assert context.shape == (2, 257, 64)
        delta = g.features(context, sample_ids=[0, 1])["fine"].flatten(2).transpose(1, 2)
        edited = edit_code("edit", z, generator=g, context=context, sample_ids=[0, 1])
    assert torch.equal(edited, z + delta)
    with pytest.raises(ValueError):
        code_from_values(p, edited.double(), provenance=[dict(kind="generated", id="bad", available_at=0.)] * 2)
    code = code_from_values(p, edited, provenance=[dict(kind="generated", id=f"s{i}", available_at=1.) for i in range(2)])
    assert torch.equal(code["fine"]["values"], edited.cpu())
    with torch.no_grad():
        percept = decode_image_code(p, code)
    assert percept.recon.shape == (2, 3, 64, 64) and torch.isfinite(percept.recon).all()
    with pytest.raises(ValueError):
        code_from_values(p, edited[:, :255], provenance=[dict(kind="generated", id="x", available_at=0.)] * 2)


def test_request_context_rejects_bad_inputs_and_null_request_removes_binding():
    from pathwm.models.conditional_image import ComponentRequest
    request = ComponentRequest(64)
    z = torch.randn(3, 256, 64)
    with pytest.raises(ValueError):
        request.context(z, torch.randn(3, 32), torch.zeros(3, dtype=torch.long))
    with pytest.raises(ValueError):
        request.context(z, torch.randn(3, 64), torch.full((3,), 2))
    with pytest.raises(ValueError):
        request.context(z, torch.full((3, 64), float("nan")), torch.zeros(3, dtype=torch.long))
    with pytest.raises(ValueError):
        request.context(z, torch.randn(3, 64), torch.zeros(3))  # float states
    a = request.context(z, torch.randn(3, 64), torch.tensor([0, 1, 0]), null=True)
    b = request.context(z, torch.randn(3, 64), torch.tensor([1, 0, 1]), null=True)
    assert torch.equal(a, b)  # null request carries neither binding nor state


def test_paired_requests_are_balanced_same_source_counterfactuals():
    from experiments.unified_session import paired_requests
    source, machine, state = paired_requests(8)
    assert len(source) == 32
    for s in range(8):
        cases = {(int(m), int(v)) for m, v in zip(machine[source == s], state[source == s])}
        assert cases == {(0, 0), (0, 1), (1, 0), (1, 1)}


def test_edit_inputs_never_see_targets():
    from experiments.unified_session import edit_inputs, edit_sources, edit_targets
    p = native()
    g = torch.Generator().manual_seed(1)
    scenes, lamps, textures = edit_sources(g, torch.Generator().manual_seed(2), rw.KIND_SPLIT["train"], 2, 1.0)
    with torch.no_grad():
        inputs = edit_inputs(p, scenes, lamps, textures, "cpu")
        targets = edit_targets(p, scenes, lamps, textures, inputs, "cpu")
    assert inputs["source"].shape == (8, 256, 64) and targets["delta"].shape == (8, 256, 64)
    assert not {"delta", "post", "noop"} & set(inputs)
    noop = targets["noop"]
    assert noop.sum() == 4 and torch.equal(targets["delta"][noop], torch.zeros_like(targets["delta"][noop]))
    assert (targets["delta"][~noop].abs().sum((1, 2)) > 0).all()


def observed_agent(tmp_path):
    from experiments.unified_session import build, new_agent
    modules = build(0, identity_run=J6000, candidate_scope="predicted-machine")
    g = torch.Generator().manual_seed(3702)
    scene = rw.sample_scenes(g, torch.tensor(rw.KIND_SPLIT["validation"][:2])[None])
    rgb, _ = rw.render(scene, torch.tensor([[0, 1]]))
    agent = new_agent(modules, tmp_path / "case")
    view, _ = agent.observe(rgb[0])
    return agent, view, scene


def test_real_store_retrieval_matches_visual_memory_and_counts_failures(tmp_path):
    from experiments.unified_session import retrieve_request
    agent, view, scene = observed_agent(tmp_path)
    receipts = [retrieve_request(agent, view, scene.machine_xy[0, m]) for m in range(2)]
    ok = [r for r in receipts if r["ok"]]
    assert ok, receipts
    for r in ok:
        record = agent.visual_memory(r["instance"])
        assert r["component_id"] == record.id and torch.equal(r["values"], record.tensor())
        assert r["store_revision"] >= 0 and r["model_version"] == agent.memory.versions["perception"]
    miss = retrieve_request(agent, view, torch.tensor([0.5, 0.5]))  # background corner: no machine instance
    assert not miss["ok"] and miss["reason"] and miss["values"] is None


def test_real_store_invalidated_and_wrong_version_memory_fail_as_receipts(tmp_path):
    from experiments.unified_session import retrieve_request
    agent, view, scene = observed_agent(tmp_path)
    ok = [retrieve_request(agent, view, scene.machine_xy[0, m]) for m in range(2)]
    m = next(i for i, r in enumerate(ok) if r["ok"])
    versions = agent.memory.versions
    agent.memory.versions = dict(versions, perception="0" * 16)  # stored records now carry a foreign version
    wrong = retrieve_request(agent, view, scene.machine_xy[0, m])
    agent.memory.versions = versions
    assert not wrong["ok"] and "visual_memory" in wrong["reason"] and wrong["values"] is None
    frame = ok[m]["evidence"][0]  # withdraw the retained source frame through a real correction
    old = agent.memory.view()["evidence"][frame]
    tx = agent.session.store.begin(f"correction-{agent.session.store.revision + 1:06d}", occurred_at=old.occurred_at,
                                   available_at=agent.session.time + 1., kind="correction", payload=dict(source=old.source))
    tx.retract_evidence(frame)
    agent.correct(tx)
    stale = retrieve_request(agent, view, scene.machine_xy[0, m])
    assert not stale["ok"] and stale["values"] is None and stale["reason"]


@pytest.fixture(scope="module")
def edit_run(tmp_path_factory):
    """A real (1-update, full-configuration) edit training run on the actual J6000 perception."""
    from argparse import Namespace
    from experiments.unified_session import EDIT_STEPS, edit_train
    out = tmp_path_factory.mktemp("edit") / "run"
    edit_train(Namespace(seed=3711, device="cpu", identity_run=J6000, edit_objective="flow", edit_steps=EDIT_STEPS,
                         updates=1, lr=3e-4, texture_randomization=1.0, max_reserved_gib=6.0, output=out,
                         resume=None, max_minutes=None, stop_after=None))
    return out


def test_evaluation_rejects_incompatible_producer_before_output(edit_run, tmp_path):
    import json, shutil
    from experiments.unified_session import edit_evaluation
    from pathwm.world_state.modules import AssociationBinder
    bad = tmp_path / "bad"
    shutil.copytree(edit_run, bad)
    record = json.loads((bad / "run.json").read_text())
    record["identity"]["settings"]["frozen_perception_sha256"]["encoder"] = "0" * 64
    (bad / "run.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="frozen"):
        edit_evaluation(tmp_path / "out", identity_run=J6000, edit_run=bad, binding_calibration=None, sources=1,
                        binder=AssociationBinder(), policy={})
    assert not (tmp_path / "out").exists()


def test_generated_codes_are_identical_when_scorer_targets_are_corrupted(edit_run, tmp_path, monkeypatch):
    import json
    import experiments.unified_session as us
    from pathwm.world_state.modules import AssociationBinder

    def run(name):
        us.edit_evaluation(tmp_path / name, identity_run=J6000, edit_run=edit_run, binding_calibration=None,
                           sources=1, binder=AssociationBinder(), policy={})
        return [json.loads(line) for line in (tmp_path / name / "metrics.jsonl").read_text().splitlines()]

    clean = run("clean")
    original = us.edit_targets

    def corrupted(*args, **kwargs):  # same source/context/generator/noise; target frames/codes replaced
        t = original(*args, **kwargs)
        post = torch.randn_like(t["post"])
        return dict(t, post=post, delta=post - args[4]["source"])

    monkeypatch.setattr(us, "edit_targets", corrupted)
    dirty = run("dirty")
    assert [r["edited_sha256"] for r in clean] == [r["edited_sha256"] for r in dirty]
    assert [r["code_error"] for r in clean] != [r["code_error"] for r in dirty]  # only the scorer saw the change


def test_decoder_factory_enables_exactly_the_saved_optional_connections(tmp_path):
    from experiments.unified_session import load_decoder
    torch.manual_seed(1)
    p = SlotPerception()
    p.decoder.enable_pyramid_connections()
    p.decoder.enable_fine_subpixels()
    (tmp_path / "run").mkdir()
    torch.save(dict(schema="pathwm-run-v1", model={f"perception.{k}": v for k, v in p.state_dict().items()}),
               tmp_path / "run" / "last.pt")
    loaded = load_decoder(tmp_path / "run")
    assert hasattr(loaded.decoder, "fine_subpixel") and loaded.decoder.pyramid_enabled
    assert all(torch.equal(a, b) for a, b in zip(loaded.state_dict().values(), p.state_dict().values()))
    plain = load_decoder(Path("runs/native_pyramid_decoder_3601_pyramid_v1"))
    assert plain.decoder.pyramid_enabled and not hasattr(plain.decoder, "fine_subpixel")
