import pytest


def test_planner_known_unknown_absent_and_nonmutation():
    from pathwm.models.key_box import plan_key

    belief = (1.0, 0.0, 0.0)
    opened = (False, False)
    assert plan_key(belief, opened, 4)[0] == ("open", 0)
    assert plan_key(belief, (True, False), 3)[0] == ("retrieve", 0)
    assert plan_key((0.0, 0.0, 1.0), opened, 4)[0] == ("stop", -1)
    assert plan_key((0.5, 0.5, 0.0), opened, 4)[0][0] == "inspect"
    assert plan_key((0.5, 0.5, 0.0), opened, 0)[0] == ("stop", -1)
    assert plan_key((0.01, 0.0, 0.99), opened, 4)[0] == ("stop", -1)
    assert belief == (1.0, 0.0, 0.0) and opened == (False, False)
    with pytest.raises(ValueError):
        plan_key((1.0, 1.0, 0.0), opened, 4)


def make_model():
    from experiments.multimodal import build_model
    from pathwm.models.entities import EntityMatchReader
    from pathwm.models.entity_state import EntityStateCell
    from pathwm.models.key_box import KeyBoxReader

    return KeyBoxReader(
        build_model(
            width=16,
            state_model="belief",
            memory_recent=2,
            memory_block=2,
            memory_blocks=1,
        ),
        EntityMatchReader(),
        EntityStateCell(preserve_no_information=True),
    )


def test_workspace_gradient_and_session_retry_restore():
    import torch
    from pathwm.models.key_box import KeyBoxSession

    model = make_model()
    model.train()
    state = model.agent.initial_state(2)
    logits, _ = model(state, torch.randn(2, 16))
    logits.square().sum().backward()
    assert model.projection.weight.grad.abs().sum() > 0
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.agent.thinker.parameters()
    )
    assert all(p.grad is None for p in model.cell.parameters())
    model.eval()
    descriptors = torch.eye(8)[:2].tolist()
    session = KeyBoxSession(model, descriptors)
    session.observe("first", 0, 1)
    before = session.snapshot()
    session.observe("first", 0, 1)
    assert session.memory.snapshot() == before["memory"]
    torch.testing.assert_close(session.state.tokens, before["state"]["tokens"])
    with pytest.raises(ValueError):
        session.observe("first", 0, 0)
    restored = KeyBoxSession.restore(model, before)
    assert restored.memory.snapshot() == session.memory.snapshot()
    torch.testing.assert_close(restored.state.tokens, session.state.tokens)


def test_key_box_recipe_resume_and_action_budget(tmp_path, monkeypatch):
    import json
    import torch
    from experiments.multimodal import train_key_box

    from pathwm.models.entities import EntityMatchReader

    def exact_match(self, query, memory):
        distance = (query[:, None] - memory).square().sum(-1)
        return torch.cat((10 - 100 * distance, torch.zeros(len(query), 1)), -1)

    monkeypatch.setattr(EntityMatchReader, "match", exact_match)
    model = make_model()
    matcher, cell = tmp_path / "matcher.pt", tmp_path / "cell.pt"
    torch.save(
        {"model": {"agent." + k: v for k, v in model.matcher.state_dict().items()}},
        matcher,
    )
    torch.save({"model": model.cell.state_dict()}, cell)
    output = tmp_path / "key-box"
    train_key_box(matcher, cell, output, steps=2, families=1)
    raw = (output / "key_box.json").read_bytes()
    data = json.loads(raw)
    assert data["summary"]["supplied_state"]["success"] == 1
    assert data["summary"]["supplied_state"]["absent_stop"] == 1
    assert all(len(r["actions"]) <= 4 for r in data["episodes"])
    train_key_box(matcher, cell, output, steps=2, families=1, resume=True)
    assert (output / "key_box.json").read_bytes() == raw
    switched = tmp_path / "switched"
    train_key_box(
        matcher,
        cell,
        switched,
        steps=2,
        families=1,
        query_switch=True,
        history_pairs=4,
        eval_seed=2411,
        reference_weights=output / "last.pt",
    )
    comparison = json.loads((switched / "key_box.json").read_text())
    assert len(comparison["reference"]["episodes"]) == len(comparison["episodes"])
    frozen = (switched / "key_box.json").read_bytes()
    train_key_box(
        matcher,
        cell,
        switched,
        steps=2,
        families=1,
        query_switch=True,
        history_pairs=4,
        eval_seed=2411,
        reference_weights=output / "last.pt",
        resume=True,
    )
    assert (switched / "key_box.json").read_bytes() == frozen


def test_switched_query_targets_remain_aligned():
    import torch
    from pathwm.evaluation.key_box import second_key_query

    latent = torch.arange(12).reshape(3, 4)
    target = torch.tensor([0, 1, 0])
    values, labels = second_key_query(latent, target, True)
    assert torch.equal(values, latent.flip(0))
    assert torch.equal(labels, target.flip(0))
    assert torch.equal(second_key_query(latent, target, False)[0], latent)


def test_history_reads_keep_targets_and_interleave_events(monkeypatch):
    import torch
    import pathwm.evaluation.key_box as module

    seen = []

    class Reader:
        agent = object()

        def __call__(self, state, latent):
            seen.append((state, latent.clone()))
            return latent, state + 1

    monkeypatch.setattr(module, "neutral_event", lambda agent, state, time: state + 10)
    latent = torch.tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True)
    target = torch.tensor([0, 1])
    losses = module.key_read_losses(Reader(), 0, latent, target, True, 4, 8)
    assert [state for state, _ in seen] == [0, 1, 12, 13, 24, 25, 36, 37]
    assert len(losses) == 8
    torch.testing.assert_close(torch.stack(losses), losses[0].expand(8))
    torch.stack(losses).mean().backward()
    assert latent.grad.abs().sum() > 0
