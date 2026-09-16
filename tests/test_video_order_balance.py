from collections import Counter

import torch


def test_cyclic_pairs_have_exact_class_conditional_frame_marginals():
    from pathwm.data.video_order import cyclic_pan_pairs

    # Unique-valued pixels make phase equality and sign independently checkable.
    images = torch.arange(2 * 3 * 6 * 8).reshape(2, 3, 6, 8).float()
    data = cyclic_pan_pairs(images, shifts=(1, 2))
    x, labels = data["frames"], data["labels"]
    assert torch.equal(x[:, 0, -1], x[:, 1, -1])
    assert torch.equal(x[:, 0, :2], x[:, 1, :2].flip(1))
    flat = x.flatten(0, 1)
    target = labels.flatten()
    for t in range(3):
        counts = [
            Counter(v.numpy().tobytes() for v in flat[target == label, t])
            for label in (0, 1)
        ]
        assert counts[0] == counts[1]
    for i, (_, phase, d) in enumerate(data["metadata"].tolist()):
        assert torch.equal(torch.roll(x[i, 0, -2], -d, dims=-1), x[i, 0, -1])
        assert labels[i, 0] == 0 and labels[i, 1] == 1
