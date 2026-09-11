"""Destination-only recall after an earlier relation cue disappears."""

import itertools
from time import perf_counter
import torch
from pathwm.models.entity_relations import EntityRelationMemory


def relation_examples(families):
    return dict(
        cues=torch.tensor([f["revisits"][s] for f in families for s in range(3)]),
        candidates=torch.tensor(
            [f["descriptors"][:3] for f in families for _ in range(3)]
        ),
        labels=torch.tensor([s for _ in families for s in range(3)]),
    )


def evaluate_relations(matcher, cell, key_model, families, deadline):
    cohorts = {}
    for condition in ("reference", "rebind", "gap", "permuted", "erased"):
        rows = []
        logits_rows = []
        targets = []
        for family_id, f in enumerate(families):
            for destination in range(3):
                others = [e for e in range(3) if e != destination]
                for a, b, choice in itertools.product(range(2), repeat=3):
                    if perf_counter() > deadline:
                        raise TimeoutError("Relation evaluation budget exhausted")
                    source = others[choice]
                    other = others[1 - choice]
                    truth = [None] * 3
                    truth[destination] = a
                    truth[others[0]] = b
                    truth[others[1]] = 1 - b
                    initial = truth.copy()
                    bindings = {}
                    receipts = []
                    transactions = True
                    restored = None
                    store = EntityRelationMemory(matcher, cell, key_model)

                    def invoke(method, args):
                        nonlocal transactions
                        receipt = getattr(store, method)(*args)
                        before = store.snapshot()
                        transactions &= (
                            getattr(store, method)(*args) == receipt
                            and store.snapshot() == before
                        )
                        if restored is not None:
                            transactions &= (
                                getattr(restored, method)(*args) == receipt
                                and restored.snapshot() == before
                            )
                        receipts.append(receipt)
                        return receipt

                    t = 0
                    for e in [2, 0, 1] if condition == "permuted" else range(3):
                        receipt = invoke(
                            "observe",
                            (str(t), f["descriptors"][e], t, torch.eye(4)[truth[e]]),
                        )
                        bindings[e] = receipt["entity_id"]
                        t += 1
                    if any(v is None for v in bindings.values()):
                        raise ValueError("Unresolved initial entities")
                    if condition == "rebind":
                        invoke(
                            "bind",
                            (
                                str(t),
                                f["revisits"][destination],
                                f["revisits"][other],
                                t,
                            ),
                        )
                        t += 1
                    invoke(
                        "bind",
                        (str(t), f["revisits"][destination], f["revisits"][source], t),
                    )
                    t += 1
                    restored = EntityRelationMemory.restore(
                        matcher, cell, key_model, store.snapshot()
                    )
                    for e in range(3):
                        invoke(
                            "observe", (str(t), f["revisits"][e], t, torch.eye(4)[2])
                        )
                        t += 1
                        truth[e] ^= 1
                    if condition == "gap":
                        for i in range(31):
                            invoke(
                                "observe",
                                (str(t), f["revisits"][i % 3], t, torch.eye(4)[3]),
                            )
                            t += 1
                    if condition == "erased":
                        for memory in (store, restored):
                            memory.relations[str(bindings[destination])] = [
                                0.0
                            ] * key_model.width
                    before = store.snapshot()
                    args = ("read", f["revisits"][destination], t)
                    rejected = False
                    receipt = None
                    try:
                        receipt = store.recall(*args)
                    except LookupError:
                        rejected = True
                    after = store.snapshot()
                    if rejected:
                        transactions &= before == after
                        for memory in (store, restored):
                            try:
                                memory.recall(*args)
                            except LookupError:
                                pass
                            else:
                                transactions = False
                            transactions &= memory.snapshot() == before
                    else:
                        transactions &= (
                            store.recall(*args) == receipt and store.snapshot() == after
                        )
                        transactions &= (
                            restored.recall(*args) == receipt
                            and restored.snapshot() == after
                        )
                    selected = None if rejected else receipt["source_id"]
                    unchanged = before["relations"] == after["relations"] and all(
                        before["state"]["latents"][bindings[e]]
                        == after["state"]["latents"][bindings[e]]
                        for e in others
                    )
                    truth[destination] = truth[source]
                    with torch.inference_mode():
                        logits = torch.stack(
                            [store.state.read(bindings[e]) for e in range(3)]
                        )
                    logits_rows.append(logits)
                    targets.append(truth)
                    rows.append(
                        dict(
                            family=family_id,
                            initial=initial,
                            destination=destination,
                            source=source,
                            choice=choice,
                            bindings=bindings,
                            selected=selected,
                            rejected=rejected,
                            source_correct=selected == bindings[source],
                            transactions=transactions,
                            unchanged=unchanged,
                            target=truth,
                            predicted=logits.argmax(-1).tolist(),
                            pre_read=before,
                            snapshot=after,
                            receipts=receipts,
                        )
                    )
        logits = torch.stack(logits_rows)
        target = torch.tensor(targets)
        accuracy = (logits.argmax(-1) == target).all(-1).float().mean().item()
        nll = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1), target.flatten()
        ).item()
        source_accuracy = sum(r["source_correct"] for r in rows) / len(rows)
        integrity = all(r["transactions"] and r["unchanged"] for r in rows)
        passed = (
            source_accuracy <= 0.60
            if condition == "erased"
            else source_accuracy >= 0.95 and accuracy >= 0.95 and nll <= 0.15
        )
        cohorts[condition] = dict(
            examples=len(rows),
            accuracy=accuracy,
            nll=nll,
            source_accuracy=source_accuracy,
            rejection_rate=sum(r["rejected"] for r in rows) / len(rows),
            integrity=integrity,
            passed=passed and integrity,
            episodes=rows,
            logits=logits.tolist(),
        )
    return dict(cohorts=cohorts, passed=all(c["passed"] for c in cohorts.values()))
