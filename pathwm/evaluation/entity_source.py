"""Frozen source retrieval with an opposing-state distractor."""

import itertools
from time import perf_counter
import torch
from pathwm.models.entity_state import EntityStateMemory


def evaluate_sources(matcher, cell, families, max_seconds=180):
    deadline = perf_counter() + max_seconds
    cohorts = {}
    features = torch.eye(4)
    for condition in ("oracle", "lookup", "permuted", "wrong_query", "unknown"):
        rows, all_logits, all_targets = [], [], []
        for family_id, family in enumerate(families):
            for destination in range(3):
                for source in (i for i in range(3) if i != destination):
                    distractor = 3 - destination - source
                    for a, b in itertools.product(range(2), repeat=2):
                        if perf_counter() > deadline:
                            raise TimeoutError("Source retrieval budget exhausted")
                        truth = [None] * 3
                        truth[destination] = a
                        truth[source] = b
                        truth[distractor] = 1 - b
                        initial = truth.copy()
                        store = EntityStateMemory(matcher, cell, capacity=3)
                        bindings = {}
                        restored = None
                        transactions = True
                        observations = []
                        slots = []
                        sources = []
                        receipts = []
                        order = [2, 0, 1] if condition == "permuted" else [0, 1, 2]
                        events = [(e, truth[e]) for e in order] + [
                            (e, 2) for e in range(3)
                        ]
                        for t, (entity, op) in enumerate(events):
                            descriptor = family["descriptors" if t < 3 else "revisits"][
                                entity
                            ]
                            args = (str(t), descriptor, t, features[op])
                            receipt = store.observe(*args)
                            if t < 3:
                                bindings[entity] = receipt["entity_id"]
                            before = store.snapshot()
                            transactions &= (
                                store.observe(*args) == receipt
                                and store.snapshot() == before
                            )
                            if restored is not None:
                                transactions &= (
                                    restored.observe(*args) == receipt
                                    and restored.snapshot() == before
                                )
                            if t == 2:
                                restored = EntityStateMemory.restore(
                                    matcher, cell, before
                                )
                            observations.append(features[op])
                            slots.append(receipt["entity_id"])
                            sources.append(-1)
                            receipts.append(receipt)
                            if op == 2:
                                truth[entity] ^= 1
                        if any(v is None for v in bindings.values()):
                            raise ValueError(
                                "Initial recognition did not resolve all records"
                            )
                        before = store.snapshot()
                        expected = truth.copy()
                        query = family["revisits"][source]
                        if condition == "wrong_query":
                            query = family["revisits"][distractor]
                        if condition == "unknown":
                            query = family["descriptors"][8]
                        kwargs = (
                            dict(source_id=bindings[source])
                            if condition == "oracle"
                            else dict(source_query=query)
                        )
                        args = (
                            "copy",
                            family["revisits"][destination],
                            6,
                            torch.zeros(4),
                        )
                        rejected = False
                        receipt = None
                        try:
                            receipt = store.observe(*args, **kwargs)
                        except LookupError:
                            rejected = True
                        after = store.snapshot()
                        if rejected:
                            transactions &= before == after
                            for replay_store in (store, restored):
                                try:
                                    replay_store.observe(*args, **kwargs)
                                except LookupError:
                                    pass
                                else:
                                    transactions = False
                                transactions &= replay_store.snapshot() == before
                            selected = None
                            slots.append(-1)
                            sources.append(-1)
                        else:
                            transactions &= (
                                store.observe(*args, **kwargs) == receipt
                                and store.snapshot() == after
                            )
                            transactions &= (
                                restored.observe(*args, **kwargs) == receipt
                                and restored.snapshot() == after
                            )
                            selected = receipt.get("source_id", bindings[source])
                            slots.append(
                                receipt["entity_id"]
                                if receipt["entity_id"] is not None
                                else -1
                            )
                            sources.append(selected)
                        observations.append(torch.zeros(4))
                        receipts.append(receipt)
                        if condition != "unknown":
                            expected[destination] = truth[source]
                        unchanged = all(
                            after["latents"][bindings[e]]
                            == before["latents"][bindings[e]]
                            and after["memory"]["records"][bindings[e]]
                            == before["memory"]["records"][bindings[e]]
                            for e in range(3)
                            if e != destination
                        )
                        with torch.inference_mode():
                            _, hidden = cell(
                                torch.stack(observations)[None],
                                torch.tensor([slots]),
                                torch.tensor([sources]),
                                entity_count=3,
                            )
                            logits = torch.stack(
                                [store.read(bindings[e]) for e in range(3)]
                            )
                        agreement = torch.allclose(
                            hidden[0], torch.tensor(store.latents), atol=1e-6, rtol=1e-5
                        )
                        all_logits.append(logits)
                        all_targets.append(expected)
                        rows.append(
                            dict(
                                family=family_id,
                                initial=initial,
                                destination=destination,
                                source=source,
                                distractor=distractor,
                                bindings=bindings,
                                query=query,
                                selected=selected,
                                source_correct=selected == bindings[source],
                                rejected=rejected,
                                transactions=transactions,
                                non_target_unchanged=unchanged,
                                latent_agreement=agreement,
                                target=expected,
                                predicted=logits.argmax(-1).tolist(),
                                receipts=receipts,
                                snapshot=after,
                            )
                        )
        logits = torch.stack(all_logits)
        targets = torch.tensor(all_targets)
        accuracy = (logits.argmax(-1) == targets).all(-1).float().mean().item()
        nll = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1), targets.flatten()
        ).item()
        source_accuracy = sum(r["source_correct"] for r in rows) / len(rows)
        rejected = sum(r["rejected"] for r in rows) / len(rows)
        integrity = all(
            r["transactions"] and r["non_target_unchanged"] and r["latent_agreement"]
            for r in rows
        )
        if condition == "wrong_query":
            passed = accuracy <= 0.05 and source_accuracy <= 0.05 and rejected == 0
        elif condition == "unknown":
            passed = rejected == 1 and accuracy >= 0.95
        else:
            passed = (
                accuracy >= 0.95
                and nll <= 0.15
                and source_accuracy >= 0.95
                and rejected == 0
            )
        cohorts[condition] = dict(
            examples=len(rows),
            accuracy=accuracy,
            nll=nll,
            source_accuracy=source_accuracy,
            rejection_rate=rejected,
            integrity=integrity,
            passed=passed and integrity,
            logits=logits.tolist(),
            episodes=rows,
        )
    return dict(cohorts=cohorts, passed=all(c["passed"] for c in cohorts.values()))
