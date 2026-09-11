"""Training observations and executed key-box episodes; labels stay in harness."""

import torch
from pathwm.models.key_box import KeyBoxSession, neutral_event, plan_key
from pathwm.evaluation.entity_growth import growth_inputs


@torch.no_grad()
def key_training_batch(model, generator, batch=32):
    state = model.agent.initial_state(batch)
    latent = torch.zeros(batch, model.cell.width)
    bits = torch.randint(2, (batch,), generator=generator)
    length = int(torch.randint(2, 9, (), generator=generator))
    for t in range(length):
        operation = (
            bits.clone() if t == 0 else torch.randint(4, (batch,), generator=generator)
        )
        bits = torch.where(
            operation < 2, operation, torch.where(operation == 2, 1 - bits, bits)
        )
        latent = model.cell.update(torch.eye(4)[operation], latent)
        state = neutral_event(model.agent, state, t + 1)
    return state, latent, bits


@torch.no_grad()
def evaluate_key_box(model, families=16, seed=2401):
    rows = []
    inputs = growth_inputs(seed, families)
    for family, descriptors in enumerate(inputs):
        for condition in ("remembered", "moved", "uncertain", "absent"):
            for start in (0, 1):
                for policy in ("integrated", "no_history", "supplied_state"):
                    torch.manual_seed(seed + 100 * family + start)
                    session = KeyBoxSession(
                        model, descriptors["descriptors"][:2], f"key-{family}-{start}"
                    )
                    truth = start
                    for box in (0, 1) if family % 2 == 0 else (1, 0):
                        session.observe(f"initial-{box}", box, int(box == truth))
                    if condition in ("moved", "uncertain"):
                        truth = 1 - start
                        if condition == "moved":
                            for box in range(2):
                                session.observe(
                                    f"correction-{box}", box, int(box == truth)
                                )
                        else:
                            session.observe("unobserved-change", invalidate=True)
                    if condition == "absent":
                        truth = None
                        for box in range(2):
                            session.observe(f"empty-{box}", box, 0)
                    for t in range(2 + family % 7):
                        session.observe(f"delay-{t}")
                    if policy == "no_history":
                        session = KeyBoxSession(
                            model,
                            descriptors["descriptors"][:2],
                            f"reset-{family}-{start}",
                        )
                    opened = [False, False]
                    actions = []
                    total_cost = 0.0
                    success = False
                    outcome = "budget_exhausted"
                    initial_correct = []
                    for step in range(4):
                        if policy == "supplied_state":
                            q = tuple(
                                float(i == (2 if truth is None else truth))
                                for i in range(3)
                            )
                            probabilities = list(q[:2])
                        else:
                            q, probabilities = session.probabilities()
                        if step == 0 and policy == "integrated":
                            initial_correct = [
                                int((p > 0.5) == (box == truth))
                                for box, p in enumerate(probabilities)
                                if session.known[box]
                            ]
                        action, prediction = plan_key(q, tuple(opened), 4 - step)
                        kind, box = action
                        record = dict(
                            step=step,
                            action=list(action),
                            belief=list(q),
                            known=session.known.copy(),
                            opened=opened.copy(),
                            search=prediction,
                            agent_time=float(session.state.time[0]),
                        )
                        if kind == "stop":
                            success = truth is None
                            outcome = "correct_stop" if success else "false_stop"
                            actions.append(record)
                            break
                        cost = 0.25 if kind == "inspect" else 1.0
                        total_cost += cost
                        if kind == "inspect":
                            bit = int(box == truth)
                            session.observe(f"action-{step}", box, bit)
                            record["observed_bit"] = bit
                        elif kind == "open":
                            opened[box] = True
                            session.observe(f"action-{step}")
                        elif kind == "retrieve":
                            if opened[box] and truth == box:
                                success, outcome = True, "retrieved"
                                session.observe(f"action-{step}", box, 0)
                            else:
                                session.observe(f"action-{step}", box, 0)
                            record["retrieved"] = success
                        record["cost"] = cost
                        record["after_time"] = float(session.state.time[0])
                        actions.append(record)
                        if success:
                            break
                    rows.append(
                        dict(
                            family=family,
                            start=start,
                            condition=condition,
                            policy=policy,
                            truth=truth,
                            success=success,
                            outcome=outcome,
                            cost=total_cost,
                            utility=float(success) - 0.05 * total_cost,
                            initial_correct=initial_correct,
                            actions=actions,
                        )
                    )
    summary = {}
    for policy in ("integrated", "no_history", "supplied_state"):
        chosen = [r for r in rows if r["policy"] == policy]
        reachable = [r for r in chosen if r["truth"] is not None]
        absent = [r for r in chosen if r["truth"] is None]
        summary[policy] = dict(
            success=sum(r["success"] for r in reachable) / len(reachable),
            absent_stop=sum(r["success"] for r in absent) / len(absent),
            utility=sum(r["utility"] for r in chosen) / len(chosen),
            mean_cost=sum(r["cost"] for r in chosen) / len(chosen),
        )
    known = [v for r in rows for v in r["initial_correct"]]
    accuracy = sum(known) / len(known)
    a, b, c = (summary[p] for p in ("integrated", "no_history", "supplied_state"))
    return dict(
        episodes=rows,
        summary=summary,
        known_accuracy=accuracy,
        passed=accuracy >= 0.95
        and a["success"] >= 0.9
        and a["absent_stop"] >= 0.9
        and a["utility"] >= b["utility"] + 0.01
        and c["success"] == 1
        and c["absent_stop"] == 1,
        limitations="Supplied descriptors and action dynamics; frozen learned state donor, trained workspace readout. No learned dynamics or visual discovery.",
    )


def second_key_query(latent, target, switch):
    """Permute retrieved records and their labels together, without new information."""
    return (latent.flip(0), target.flip(0)) if switch else (latent, target)


def key_read_losses(model, state, latent, target, switch, pairs, start_time):
    """Supervise recurrent reads across observation gaps; targets never enter state."""
    losses = []
    other, labels = second_key_query(latent, target, switch)
    for pair in range(pairs):
        if pair:
            state = neutral_event(model.agent, state, start_time + pair)
        for values, expected in ((latent, target), (other, labels)):
            logits, state = model(state, values)
            losses.append(torch.nn.functional.cross_entropy(logits, expected))
    return losses
