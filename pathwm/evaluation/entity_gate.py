"""Controlled context relevance, with frozen relation consumers."""

import copy
from time import perf_counter
import torch
from torch.nn import functional as F
from pathwm.models.entity_relations import EntityRelationMemory


def gate_examples(families, seed):
    rng = torch.Generator().manual_seed(seed)
    rows = []

    def unit():
        return F.normalize(torch.randn(4, generator=rng), dim=0)

    for fi, family in enumerate(families):
        for old in range(3):
            for new in range(3):
                if old == new:
                    continue
                active = unit()
                same = F.normalize(active + 0.03 * torch.randn(4, generator=rng), dim=0)
                other = unit()
                while (active - other).norm() < 0.9:
                    other = unit()
                for accept, cue in ((False, other), (True, same)):
                    rows.append(
                        dict(
                            old=family["revisits"][old],
                            new=family["revisits"][new],
                            candidates=family["descriptors"][:3],
                            active=active.tolist(),
                            cue=cue.tolist(),
                            labels=new if accept else old,
                            old_id=old,
                            new_id=new,
                            family=fi,
                            accept=accept,
                        )
                    )
    return {k: torch.tensor([r[k] for r in rows]) for k in rows[0]}


def gate_logits(gate, key, matcher, data):
    probability = gate(data["active"], data["cue"])
    old, new = key.encode(data["old"]), key.encode(data["new"])
    mixed = old + probability[:, None] * (new - old)
    return matcher.match(key.decode(mixed), data["candidates"]), probability


def evaluate_gate(matcher, cell, key, gate, families, data, deadline):
    cohorts = {}
    with torch.inference_mode():
        soft, probabilities = gate_logits(gate, key, matcher, data)
    for condition in ("reference", "permuted", "repeated", "always", "never"):
        model = copy.deepcopy(gate)
        if condition in ("always", "never"):
            with torch.no_grad():
                for p in model.parameters():
                    p.zero_()
                model.network[-1].bias.fill_(20 if condition == "always" else -20)
        rows = []
        source_logits = []
        for i in range(len(data["labels"])):
            if perf_counter() > deadline:
                raise TimeoutError("Gate evaluation budget exhausted")
            fi, old, new = (int(data[k][i]) for k in ("family", "old_id", "new_id"))
            dest = 3 - old - new
            family = families[fi]
            store = EntityRelationMemory(matcher, cell, key, gate_model=model)
            bindings = {}
            truth = [0, 0, 0]
            truth[new] = 1
            for t, entity in enumerate(
                (2, 0, 1) if condition == "permuted" else (0, 1, 2)
            ):
                receipt = store.observe(
                    str(t),
                    family["descriptors"][entity],
                    t,
                    torch.eye(4)[truth[entity]],
                )
                bindings[entity] = receipt["entity_id"]
            store.bind("bind", family["revisits"][dest], data["old"][i], 3)
            before = store.snapshot()
            args = (
                "proposal",
                family["revisits"][dest],
                data["new"][i],
                data["active"][i],
                data["cue"][i],
                4,
            )
            receipt = store.consider(*args)
            snapshot = store.snapshot()
            integrity = (
                store.consider(*args) == receipt and store.snapshot() == snapshot
            )
            restored = EntityRelationMemory.restore(
                matcher, cell, key, snapshot, gate_model=model
            )
            integrity &= (
                restored.consider(*args) == receipt and restored.snapshot() == snapshot
            )
            if not receipt["write"]:
                integrity &= store.relations == before["relations"]
            integrity &= store.state.latents == before["state"]["latents"]
            t = 5
            if condition == "repeated" and not bool(data["accept"][i]):
                keys = copy.deepcopy(store.relations)
                for j in range(31):
                    store.consider("repeat" + str(j), *args[1:-1], t)
                    t += 1
                integrity &= store.relations == keys
            with torch.inference_mode():
                query = key.decode(torch.tensor(store.relations[str(bindings[dest])]))
                logits = matcher.match(query[None], data["candidates"][i : i + 1])[0]
            source_logits.append(logits)
            read = store.recall("read", family["revisits"][dest], t)
            after = store.snapshot()
            integrity &= (
                store.recall("read", family["revisits"][dest], t) == read
                and store.snapshot() == after
            )
            for entity in (old, new):
                identity = bindings[entity]
                integrity &= (
                    store.state.latents[identity]
                    == before["state"]["latents"][identity]
                )
            target = truth.copy()
            target[dest] = int(data["accept"][i])
            predicted = [int(store.state.read(bindings[e]).argmax()) for e in range(3)]
            rows.append(
                dict(
                    family=fi,
                    old=old,
                    new=new,
                    destination=dest,
                    accept=bool(data["accept"][i]),
                    probability=receipt["write_probability"],
                    write=receipt["write"],
                    source_correct=read["source_id"]
                    == bindings[int(data["labels"][i])],
                    predicted=predicted,
                    target=target,
                    integrity=bool(integrity),
                )
            )
        hard = torch.stack(source_logits)
        scores = dict(
            source_accuracy=sum(r["source_correct"] for r in rows) / len(rows),
            accuracy=sum(r["predicted"] == r["target"] for r in rows) / len(rows),
            nll=F.cross_entropy(hard, data["labels"]).item(),
            soft_hard_agreement=(hard.argmax(-1) == soft.argmax(-1))
            .float()
            .mean()
            .item(),
            integrity=all(r["integrity"] for r in rows),
            episodes=rows,
            source_logits=hard.tolist(),
        )
        scores["passed"] = scores["integrity"] and (
            scores["source_accuracy"] == 0.5 and scores["accuracy"] == 0.5
            if condition in ("always", "never")
            else scores["source_accuracy"] >= 0.95
            and scores["accuracy"] >= 0.95
            and scores["nll"] <= 0.15
            and scores["soft_hard_agreement"] >= 0.95
        )
        cohorts[condition] = scores
    # Unlabelled diagnostic: interpolation along a unit-circle arc, no quality gate.
    angles = torch.linspace(0, torch.pi, 21)
    active = torch.tensor([1.0, 0, 0, 0]).expand(21, -1)
    cue = torch.stack((angles.cos(), angles.sin(), angles * 0, angles * 0), -1)
    with torch.inference_mode():
        sweep = gate(active, cue).tolist()
    return dict(
        cohorts=cohorts,
        passed=all(c["passed"] for c in cohorts.values()),
        probabilities=probabilities.tolist(),
        soft_logits=soft.tolist(),
        distance_sweep=dict(
            distance=(active - cue).norm(dim=-1).tolist(), probability=sweep
        ),
    )
