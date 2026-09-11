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


def gate_shift_examples(seed=801, pairs=128):
    rng = torch.Generator().manual_seed(seed)
    active, prototypes, noise, separation = [], [], [], []
    for _ in range(pairs):
        a = F.normalize(torch.randn(4, generator=rng), dim=0)
        b = F.normalize(torch.randn(4, generator=rng), dim=0)
        while (a - b).norm() < 0.9:
            b = F.normalize(torch.randn(4, generator=rng), dim=0)
        eps = torch.randn(4, generator=rng)
        active.extend([a, a])
        prototypes.extend([a, b])
        noise.extend([eps, eps])
        separation.extend([(a - b).norm()] * 2)
    return dict(
        active=torch.stack(active),
        prototypes=torch.stack(prototypes),
        noise=torch.stack(noise),
        separation=torch.stack(separation),
        labels=torch.tensor([True, False] * pairs),
    )


def score_gate_shift(gate, data):
    labels = data["labels"]

    def scores(predicted):
        return dict(
            accuracy=(predicted == labels).float().mean().item(),
            positive_recall=predicted[labels].float().mean().item(),
            negative_recall=(~predicted[~labels]).float().mean().item(),
        )

    cohorts = {}
    with torch.inference_mode():
        for sigma in (0.03, 0.15, 0.30, 0.60):
            cue = F.normalize(data["prototypes"] + sigma * data["noise"], dim=-1)
            probability = gate(data["active"], cue)
            distance = (data["active"] - cue).norm(dim=-1)
            thresholds = {str(t): scores(probability > t) for t in (0.4, 0.5, 0.6)}
            primary = thresholds["0.5"]
            cohorts[str(sigma)] = dict(
                thresholds=thresholds,
                brier=(probability - labels.float()).square().mean().item(),
                distance_baseline=scores(distance < 0.5),
                noise_to_separation=(
                    sigma * data["noise"].norm(dim=-1) / data["separation"]
                ).tolist(),
                distance=distance.tolist(),
                cue=cue.tolist(),
                probability=probability.tolist(),
                passed=primary["positive_recall"] >= 0.95
                and primary["negative_recall"] >= 0.95,
            )
    return dict(
        cohorts=cohorts,
        passed=all(c["passed"] for c in cohorts.values()),
        data={k: v.tolist() for k, v in data.items()},
    )


def augmented_gate_examples(families, seed, augmented):
    data = gate_examples(families, seed)
    rng = torch.Generator().manual_seed(seed + 10000)
    noise = torch.randn(len(data["labels"]) // 2, 4, generator=rng).repeat_interleave(
        2, 0
    )
    prototypes = torch.where(data["accept"][:, None], data["active"], data["cue"])
    chunks = []
    for sigma in (0.03, 0.15, 0.30, 0.60) if augmented else (0.03,) * 4:
        chunk = {k: v.clone() for k, v in data.items()}
        chunk["cue"] = F.normalize(prototypes + sigma * noise, dim=-1)
        chunks.append(chunk)
    return {k: torch.cat([c[k] for c in chunks]) for k in data}


def gate_retention_loss(probability, teacher):
    """Bernoulli KL from frozen teacher to current gate, averaged over examples."""
    target = teacher.detach().clamp(1e-6, 1 - 1e-6)
    current = probability.clamp(1e-6, 1 - 1e-6)
    return (
        target * (target.log() - current.log())
        + (1 - target) * ((1 - target).log() - (1 - current).log())
    ).mean()


def gate_reobserve_examples(pairs=128, correlation=None):
    if correlation is not None and not 0 <= correlation <= 1:
        raise ValueError("Correlation must be finite and in [0,1]")
    data = gate_shift_examples(1301 if correlation is None else 1401, pairs)
    rng = torch.Generator().manual_seed(1302 if correlation is None else 1402)
    innovation = torch.randn(pairs, 4, generator=rng).repeat_interleave(2, 0)
    rho = 0 if correlation is None else correlation
    data["innovation"] = innovation
    data["second_noise"] = rho * data["noise"] + (1 - rho * rho) ** 0.5 * innovation
    return data


def score_gate_reobserve(gate, data):
    result = score_gate_shift(gate, data)
    labels = data["labels"]
    with torch.inference_mode():
        for sigma, c in result["cohorts"].items():
            first = torch.tensor(c["cue"])
            p = torch.tensor(c["probability"])
            defer = (p >= 0.2) & (p <= 0.8)
            second = F.normalize(
                data["prototypes"] + float(sigma) * data["second_noise"], dim=-1
            )
            averaged = F.normalize(first + second, dim=-1)
            p_two = gate(data["active"], averaged)
            p_duplicate = gate(data["active"], F.normalize(first + first, dim=-1))
            strategies = {}
            for name, prob, reread in (
                ("first", p, torch.zeros_like(defer)),
                ("selective", torch.where(defer, p_two, p), defer),
                ("duplicate", torch.where(defer, p_duplicate, p), defer),
                ("always_two", p_two, torch.ones_like(defer)),
            ):
                predicted = prob > 0.5
                accuracy = (predicted == labels).float().mean().item()
                rate = reread.float().mean().item()
                strategies[name] = dict(
                    accuracy=accuracy,
                    positive_recall=predicted[labels].float().mean().item(),
                    negative_recall=(~predicted[~labels]).float().mean().item(),
                    reread_rate=rate,
                    utility=accuracy - 0.02 * rate,
                    probability=prob.tolist(),
                )
            c.update(
                strategies=strategies,
                second_cue=second.tolist(),
                defer=defer.tolist(),
                duplicate_exact=torch.equal(
                    p > 0.5, torch.tensor(strategies["duplicate"]["probability"]) > 0.5
                ),
            )
    high = result["cohorts"]["0.6"]["strategies"]
    result["passed"] = (
        high["selective"]["accuracy"] >= high["first"]["accuracy"] + 0.02
        and high["selective"]["utility"] > high["first"]["utility"]
        and high["selective"]["negative_recall"]
        >= high["first"]["negative_recall"] - 0.02
        and all(c["duplicate_exact"] for c in result["cohorts"].values())
        and all(
            result["cohorts"][n]["strategies"]["selective"]["accuracy"]
            >= result["cohorts"][n]["strategies"]["first"]["accuracy"] - 0.01
            for n in ("0.03", "0.15")
        )
    )
    result["reobserve"] = True
    return result
