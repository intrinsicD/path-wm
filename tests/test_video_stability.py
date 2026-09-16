from collections import Counter

import torch


def test_reflection_preserves_pair_indices_and_reverses_observed_motion():
    from pathwm.data.video_order import cyclic_pan_pairs, reflect_pairs
    from experiments.video_order import pixel_oracle

    torch.manual_seed(7600)
    original = cyclic_pan_pairs(torch.rand(1, 3, 48, 48))
    saved = {k: v.clone() for k, v in original.items()}
    reflected = reflect_pairs(original)
    assert torch.equal(reflected["frames"], original["frames"].flip(-1))
    assert torch.equal(reflected["views"][reflected["indices"]], reflected["frames"])
    assert torch.equal(reflected["metadata"], original["metadata"])
    assert torch.equal(reflected["indices"], original["indices"])
    assert torch.equal(reflected["labels"], 1 - original["labels"])
    predicted, ambiguous = pixel_oracle(reflected["frames"])
    assert torch.equal(predicted, reflected["labels"]) and not ambiguous.any()
    x = reflected["frames"]
    assert torch.equal(x[:, 0, -1], x[:, 1, -1])
    assert torch.equal(x[:, 0, :2], x[:, 1, :2].flip(1))
    for displacement in (2, 4):
        chosen = reflected["metadata"][:, -1] == displacement
        flat = x[chosen].flatten(0, 1)
        labels = reflected["labels"][chosen].flatten()
        for t in range(3):
            assert Counter(
                v.numpy().tobytes() for v in flat[labels == 0, t]
            ) == Counter(v.numpy().tobytes() for v in flat[labels == 1, t])
    assert all(torch.equal(original[k], v) for k, v in saved.items())

    marker = torch.zeros(1, 3, 48, 48)
    marker[..., 24, 24] = 1
    marked = reflect_pairs(cyclic_pan_pairs(marker))
    selected = marked["metadata"][:, 1] == 0  # no wrap near the marker
    columns = marked["frames"][selected].sum((-3, -2)).argmax(-1)
    direction = (columns[..., -1] > columns[..., -2]).long()
    assert torch.equal(direction, marked["labels"][selected])


def test_reflected_training_batch_uses_absolute_step_and_aligned_targets():
    from experiments.video_order import training_batch

    data = dict(
        features=torch.arange(6 * 2 * 3).reshape(6, 2, 3),
        labels=torch.tensor([0, 1]).expand(6, -1).clone(),
    )
    data["reflected_features"] = data["features"] + 1000
    data["reflected_labels"] = 1 - data["labels"]
    index = [4, 0, 3]
    for step in (1, 2, 5, 8):
        for reflect in (False, True):
            x, y = training_batch(data, index, step, reflect)
            prefix = "reflected_" if reflect and step % 2 else ""
            assert torch.equal(x, data[prefix + "features"][index].flatten(0, 1))
            assert torch.equal(y, data[prefix + "labels"][index].flatten())


def test_margin_metrics_distinguish_confident_wrong_and_undecided():
    from experiments.video_order import metrics

    logits = torch.tensor([[[0.0, 0.0], [10.0, 0.0]], [[10.0, 0.0], [0.0, 10.0]]])
    labels = torch.tensor([[0, 1], [0, 1]])
    score = metrics(logits, labels)
    assert score["margin_mean"] == 2.5
    assert score["confident_wrong_fraction"] == 0.25
    assert score["margin_p10"] < 0
