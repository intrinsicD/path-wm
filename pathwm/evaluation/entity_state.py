"""State readout and persistent-runtime agreement checks."""

import torch
from pathwm.models.entity_state import EntityStateMemory


def state_metrics(cell, data):
    with torch.inference_mode():
        logits, hidden = cell(data["observations"], data["slots"])
        nll = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1), data["targets"].flatten()
        ).item()
        accuracy = (logits.argmax(-1) == data["targets"]).all(-1).float().mean().item()
    return dict(pair_accuracy=accuracy, nll=nll, examples=len(logits)), logits, hidden


def state_runtime(cell, matcher, data):
    _, _, expected = state_metrics(cell, data)
    results = []
    for index, episode in enumerate(data["manifest"]):
        store = EntityStateMemory(matcher, cell)
        restored = None
        replay = True
        receipts = []
        for t, descriptor in enumerate(episode["descriptors"]):
            args = (str(t), descriptor, t, data["observations"][index, t])
            receipt = store.observe(*args)
            receipts.append(receipt)
            before = store.snapshot()
            replay &= store.observe(*args) == receipt and store.snapshot() == before
            if restored is not None:
                replay &= (
                    restored.observe(*args) == receipt
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
