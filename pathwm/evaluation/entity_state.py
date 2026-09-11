"""State readout and persistent-runtime agreement checks."""

from time import perf_counter
import torch
from pathwm.models.entity_state import EntityStateMemory


def state_metrics(cell, data):
    with torch.inference_mode():
        logits, hidden = cell(
            data["observations"], data["slots"], sources=data.get("sources")
        )
        nll = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1), data["targets"].flatten()
        ).item()
        accuracy = (logits.argmax(-1) == data["targets"]).all(-1).float().mean().item()
    return dict(pair_accuracy=accuracy, nll=nll, examples=len(logits)), logits, hidden


def state_runtime(cell, matcher, data, deadline=None):
    _, _, expected = state_metrics(cell, data)
    results = []
    for index, episode in enumerate(data["manifest"]):
        if deadline is not None and perf_counter() > deadline:
            raise TimeoutError("State runtime evaluation budget exhausted")
        store = EntityStateMemory(matcher, cell)
        restored = None
        replay = True
        receipts = []
        for t, descriptor in enumerate(episode["descriptors"]):
            args = (str(t), descriptor, t, data["observations"][index, t])
            source = episode.get("sources", [-1] * len(episode["descriptors"]))[t]
            kwargs = dict(source_id=source if source >= 0 else None)
            receipt = store.observe(*args, **kwargs)
            receipts.append(receipt)
            before = store.snapshot()
            replay &= (
                store.observe(*args, **kwargs) == receipt and store.snapshot() == before
            )
            if restored is not None:
                replay &= (
                    restored.observe(*args, **kwargs) == receipt
                    and restored.snapshot() == store.snapshot()
                )
            if t == 2:
                restored = EntityStateMemory.restore(matcher, cell, store.snapshot())
        complete = len(store.latents) == 2
        predicted = [store.read(i).argmax().item() for i in range(len(store.latents))]
        agrees = complete and torch.allclose(
            torch.tensor(store.latents), expected[index], atol=1e-6, rtol=1e-5
        )
        results.append(
            dict(
                predicted=predicted,
                target=episode["target"],
                correct=predicted == episode["target"],
                replay=replay,
                latent_agreement=agrees,
                receipts=receipts,
                snapshot=store.snapshot(),
            )
        )
    return dict(
        pair_accuracy=sum(r["correct"] for r in results) / len(results),
        transactions=all(r["replay"] for r in results),
        latent_agreement=all(r["latent_agreement"] for r in results),
        episodes=results,
    )
