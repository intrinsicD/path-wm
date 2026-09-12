"""Paired evaluation with common categorical draws and no model-state mutation."""

from time import perf_counter
import torch
from pathwm.io import evaluation_mode


@torch.no_grad()
def visual_features(
    model, data, *, device="cpu", seed=3401, erased=False, deadline=float("inf")
):
    result = torch.empty(len(data), model.agent.width)
    with evaluation_mode(model):
        for start in range(0, len(data) // 2, 16):
            count = min(16, len(data) // 2 - start)
            for side in (0, 1):
                if perf_counter() > deadline:
                    raise TimeoutError("Direct-weight evaluation budget exhausted")
                indices = [2 * i + side for i in range(start, start + count)]
                batch = data.batch(indices, device)["images"]
                if erased:
                    batch = batch[:, -1:].expand_as(batch)
                torch.manual_seed(seed + start)
                result[indices] = model.features(batch).cpu()
    return result


def visual_metrics(logits, data):
    logits = torch.as_tensor(logits).detach().cpu().double()
    labels = torch.from_numpy(data.labels)
    correct = logits.argmax(-1) == labels
    reversal = torch.tensor([r["reversal"] for r in data.records])
    result = dict(
        accuracy=float(correct.double().mean()),
        nll=float(torch.nn.functional.cross_entropy(logits, labels)),
        pair_both=float(correct.reshape(-1, 2).all(1).double().mean()),
        logits=logits.tolist(),
        labels=labels.tolist(),
    )
    for name, keep in [("center_start", ~reversal), ("reversal", reversal)]:
        result[name] = dict(
            count=int(keep.sum()),
            accuracy=float(correct[keep].double().mean()) if keep.any() else None,
        )
    return result
