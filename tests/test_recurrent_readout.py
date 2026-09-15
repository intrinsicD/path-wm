import copy

import pytest
import torch

from pathwm.models.readout import RecurrentOutputAdapter, TemporalImageDecoder


def test_adapter_initial_identity_and_shared_iteration_parameters():
    torch.manual_seed(12)
    adapter = RecurrentOutputAdapter(16, iterations=1)
    x = torch.randn(2, 5, 16)
    original = x.clone()
    assert torch.equal(adapter(x), x)
    identities = [id(p) for p in adapter.parameters()]
    with torch.no_grad():
        adapter.gain.fill_(0.5)
    one = adapter(x)
    calls = []
    hook = adapter.cross.register_forward_hook(lambda *args: calls.append(1))
    adapter.iterations = 4
    four = adapter(x)
    hook.remove()
    assert len(calls) == 4
    assert identities == [id(p) for p in adapter.parameters()]
    assert not torch.allclose(one, four)
    assert torch.equal(x, original)


def test_adapter_mask_excludes_nan_payload_and_gradient():
    adapter = RecurrentOutputAdapter(16, iterations=2)
    with torch.no_grad():
        adapter.gain.fill_(0.5)
    x = torch.randn(2, 5, 16, requires_grad=True)
    valid = torch.tensor([[True, True, False, False, False]]).expand(2, -1)
    output = adapter(x.masked_fill(~valid[..., None], float("nan")), valid=valid)
    expected = adapter(x[:, :2])
    torch.testing.assert_close(output[:, :2], expected)
    assert output[:, 2:].count_nonzero() == 0
    output.square().sum().backward()
    assert x.grad[:, :2].abs().sum() > 0
    assert x.grad[:, 2:].count_nonzero() == 0
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in adapter.parameters()
    )
    with pytest.raises(ValueError, match="valid"):
        adapter(x, valid=torch.zeros_like(valid))


def test_adapter_trace_preserves_output_gradients_rng():
    torch.manual_seed(10)
    adapter = RecurrentOutputAdapter(16, iterations=2)
    with torch.no_grad():
        adapter.gain.fill_(0.5)
    twin = copy.deepcopy(adapter)
    x = torch.randn(2, 4, 16)
    state = torch.get_rng_state().clone()
    a = adapter(x)
    a.square().sum().backward()
    trace = {}
    b = twin(x, trace=trace)
    b.square().sum().backward()
    assert torch.equal(a, b)
    assert torch.equal(torch.get_rng_state(), state)
    assert "readout.loop.1.tokens" in trace
    for p, q in zip(adapter.parameters(), twin.parameters()):
        assert torch.equal(p.grad, q.grad)


def test_temporal_decoder_reads_only_context_and_requested_times():
    decoder = TemporalImageDecoder(16, image_size=16)
    context = torch.randn(2, 5, 16, requires_grad=True)
    before = context.detach().clone()
    times = torch.tensor([0.0, 1.0, 2.0, 3.0])
    result = decoder(context, times)
    assert result.shape == (2, 4, 3, 16, 16)
    assert not torch.equal(result[:, 0], result[:, -1])
    result.square().mean().backward()
    assert context.grad.abs().sum() > 0
    assert torch.equal(context.detach(), before)
    assert all(p.grad is not None for p in decoder.parameters())
    with pytest.raises(ValueError, match="increasing"):
        decoder(context, times.flip(0))


@pytest.mark.parametrize("iterations", [-1, 1.5, True])
def test_bad_iterations_rejected(iterations):
    with pytest.raises(ValueError, match="iterations"):
        RecurrentOutputAdapter(16, iterations=iterations)


def test_paired_data_splits_and_complementary_sources():
    from pathwm.data.modality_readout import dataset, observations

    train, seen, heldout = (dataset(s) for s in ("train", "seen", "heldout"))
    known = set(map(tuple, train["targets"]["factors"].tolist()))
    new = set(map(tuple, heldout["targets"]["factors"].tolist()))
    assert len(known) == 12 and len(new) == 6 and not known & new
    assert len(train["ids"]) == 144 and len(seen["ids"]) == len(heldout["ids"]) == 48
    for factor in range(3):
        assert {x[factor] for x in known} == {x[factor] for x in new}
    factors = train["targets"]["factors"]
    comp = observations(train, "complementary")
    for name, column in [("image", 0), ("audio", 1), ("video", 2)]:
        values = comp[name].values
        if name == "audio":
            values = values[:, 1:2]  # only this chunk is valid
        # Complete canonical complementary image/video carry exactly one factor;
        # audio has independent noise, so compare underlying target segment.
        if name == "audio":
            values = train["targets"]["audio"][:, 64:128]
        for value in factors[:, column].unique():
            group = values[factors[:, column] == value]
            assert torch.equal(group, group[:1].expand_as(group))
    assert len(torch.unique(comp["text"].values, dim=0)) == 1
    assert set(observations(train, "complementary", omit="audio")) == {
        "text",
        "image",
        "video",
    }


def test_perfect_outputs_pass_and_wrong_template_fails():
    import itertools
    from pathwm.data.modality_readout import canonical
    from pathwm.evaluation.modality_readout import scores

    target = canonical(list(itertools.product(range(3), range(3), range(2))))
    for kind in ("image", "audio", "video"):
        result, pred = scores(kind, target[kind], target, target, 1.0)
        assert result["gate"] and result["all_accuracy"] == 1.0 and result["mse"] == 0
        assert torch.equal(pred, target["factors"])
        wrong, _ = scores(kind, target[kind].roll(9, 0), target, target, 1.0)
        assert wrong["all_accuracy"] == 0 and not wrong["gate"]
    logits = torch.full((*target["text"][:, 1:].shape, 259), -50.0)
    logits.scatter_(-1, target["text"][:, 1:, None], 50.0)
    result, pred = scores("text", logits, target, target, generated=target["text"])
    assert result["free_exact"] == 1 and result["gate"]


def test_model_baseline_initialization_joint_reads_and_core_gradients():
    from experiments.modality_readout import (
        Model,
        output_objective,
        scales,
        target_batch,
    )
    from pathwm.data.modality_readout import dataset, observations, KINDS
    from pathwm.io import state_hash

    data = dataset("train")
    torch.manual_seed(23)
    native = Model("native")
    torch.manual_seed(23)
    adapted = Model("adapter2")
    assert state_hash(native.core) == state_hash(adapted.core)
    assert state_hash(native.outputs.decoders) == state_hash(adapted.outputs.decoders)
    target = target_batch(data, [0, 12], "cpu")
    tokens = native.core(observations(data, "all", [0, 12]))
    old = tokens.detach().clone()
    for k in KINDS:
        prefix = target["text"][:, :-1] if k == "text" else None
        assert torch.equal(
            native.outputs(k, tokens, prefix), adapted.outputs(k, tokens, prefix)
        )
    loss, _ = output_objective(native, tokens, target, KINDS, scales(data))
    loss.backward()
    assert torch.equal(tokens.detach(), old)
    for encoder in native.core.agent.encoders.values():
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in encoder.parameters()
        )
    for decoder in native.outputs.decoders.values():
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in decoder.parameters()
        )


def test_oracle_is_explicit_complete_information_not_an_encoder_path():
    from experiments.modality_readout import oracle_states
    from pathwm.data.modality_readout import dataset

    data = dataset("heldout")
    contexts = oracle_states({"heldout": data})["heldout"]["all"]
    decoded = torch.stack(
        [
            contexts[:, 0, :3].argmax(1),
            contexts[:, 0, 3:6].argmax(1),
            contexts[:, 0, 6:8].argmax(1),
        ],
        1,
    )
    assert torch.equal(decoded, data["targets"]["factors"])
    assert contexts.shape == (48, 12, 24)


def test_analysis_rows_without_loss_do_not_draw_empty_objective(tmp_path):
    import json
    from pathwm.evaluation.report import write_report

    (tmp_path / "run.json").write_text(
        json.dumps({"identity": {"settings": {"purpose": "diagnostic"}}})
    )
    (tmp_path / "status.json").write_text(
        json.dumps(
            {"result": "complete", "report": "pending", "step": 1, "error": None}
        )
    )
    (tmp_path / "metrics.jsonl").write_text(
        json.dumps({"step": 1, "split": "validation", "accuracy": 0.5}) + "\n"
    )
    report = write_report(tmp_path)
    assert report.exists()
    assert not (tmp_path / "learning_curve.png").exists()
