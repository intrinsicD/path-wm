"""Static-source adaptation from calibration outcomes; held-out policy frozen."""

import torch
from torch.nn import functional as F
from pathwm.models.source_choice import SourceChoice
from pathwm.evaluation.entity_gate import gate_shift_examples


def evaluate_source_choice(gate, worlds=16, seed=1601):
    rows = []
    for world in range(worlds):
        data = gate_shift_examples(seed + world, 384)
        rng = torch.Generator().manual_seed(seed + 1000 + world)
        innovation = torch.randn(384, 4, generator=rng).repeat_interleave(2, 0)
        first = F.normalize(data["prototypes"] + 0.6 * data["noise"], dim=-1)
        with torch.inference_mode():
            probability = gate(data["active"], first)
            predicted = probability > 0.5
            defer = (probability >= 0.2) & (probability <= 0.8)
            outcomes = []
            for source in range(2):
                rho = 0.0 if source == world % 2 else 0.9
                noise = rho * data["noise"] + (1 - rho * rho) ** 0.5 * innovation
                second = F.normalize(data["prototypes"] + 0.6 * noise, dim=-1)
                outcomes.append(
                    gate(data["active"], F.normalize(first + second, dim=-1)) > 0.5
                )
        baseline = (predicted == data["labels"]).float()
        correct = torch.stack([(p == data["labels"]).float() for p in outcomes], dim=-1)
        policy, permuted = SourceChoice(), SourceChoice()
        cal_indices = defer[:512].nonzero().flatten()
        schedule = (torch.arange(len(cal_indices)) % 2)[
            torch.randperm(len(cal_indices), generator=rng)
        ]
        feedback = []
        cal_reward = baseline[:512].sum().item()
        for index, source in zip(cal_indices.tolist(), schedule.tolist()):
            gain = (correct[index, source] - baseline[index]).item() - 0.05
            policy.observe(source, gain)
            permuted.observe(1 - source, gain)
            cal_reward += gain
            feedback.append(dict(index=index, source=source, gain=gain))
        snapshot = policy.snapshot()
        choice = policy.choose()
        permutation = permuted.choose() == (None if choice is None else 1 - choice)
        scores = {}
        for name, selected in (
            ("learned", choice),
            ("fixed_a", 0),
            ("fixed_b", 1),
            ("stop", None),
            ("no_feedback", SourceChoice().choose()),
        ):
            reward = baseline[512:].clone()
            decision = predicted[512:].clone()
            count = 0
            if selected is not None:
                mask = defer[512:]
                reward[mask] = correct[512:, selected][mask] - 0.05
                decision[mask] = outcomes[selected][512:][mask]
                count = int(mask.sum())
            labels = data["labels"][512:]
            scores[name] = dict(
                utility=reward.mean().item(),
                accuracy=(decision == labels).float().mean().item(),
                positive_recall=decision[labels].float().mean().item(),
                negative_recall=(~decision[~labels]).float().mean().item(),
                acquisition_rate=count / 256,
            )
        assert policy.snapshot() == snapshot
        # Store observations and outcomes for independent auditing; policy receives only feedback.
        rows.append(
            dict(
                world=world,
                seed=seed + world,
                choice=choice,
                useful_source=world % 2,
                values=snapshot,
                feedback=feedback,
                permutation=permutation,
                scores=scores,
                calibration_cost=0.05 * len(cal_indices),
                calibration_utility=cal_reward / 512,
                combined_utility=(cal_reward + 256 * scores["learned"]["utility"])
                / 768,
                combined_stop=baseline.mean().item(),
                baseline=baseline.tolist(),
                correct=correct.tolist(),
                defer=defer.tolist(),
                labels=data["labels"].tolist(),
                first_predictions=predicted.tolist(),
                source_predictions=[x.tolist() for x in outcomes],
                active=data["active"].tolist(),
                first_cue=first.tolist(),
            )
        )

    def mean(vals):
        return sum(vals) / len(vals)

    summary = {
        name: {
            k: mean([r["scores"][name][k] for r in rows])
            for k in rows[0]["scores"][name]
        }
        for name in rows[0]["scores"]
    }
    learned = summary["learned"]["utility"]
    useful = mean([r["choice"] == r["useful_source"] for r in rows])
    combined = mean([r["combined_utility"] for r in rows])
    stop = mean([r["combined_stop"] for r in rows])
    return dict(
        worlds=rows,
        summary=summary,
        useful_source_rate=useful,
        calibration_cost_mean=mean([r["calibration_cost"] for r in rows]),
        combined_utility=combined,
        combined_stop=stop,
        integrity=all(r["permutation"] for r in rows),
        passed=all(r["permutation"] for r in rows)
        and learned >= max(summary[n]["utility"] for n in ("fixed_a", "fixed_b")) + 0.01
        and learned > summary["stop"]["utility"]
        and useful >= 0.75
        and combined >= stop,
    )


def evaluate_source_drift(gate, worlds=16):
    calibration = evaluate_source_choice(gate, worlds, seed=1801)
    rows = []
    for base in calibration["worlds"]:
        labels = torch.tensor(base["labels"])
        first = torch.tensor(base["first_predictions"])
        sources = torch.tensor(base["source_predictions"]).T
        mask = torch.tensor(base["defer"])
        rng = torch.Generator().manual_seed(3801 + base["world"])
        coins = torch.rand(256, generator=rng)
        explore = torch.randint(2, (256,), generator=rng)
        for swapped in (False, True):
            outcomes = sources[:, [1, 0]] if swapped else sources
            for name in ("frozen", "cumulative", "window", "triggered", "no_feedback"):
                policy = SourceChoice(
                    window=32 if name == "window" else None,
                    change_block=32 if name == "triggered" else None,
                )
                for event in base["feedback"]:
                    policy.observe(event["source"], event["gain"] - 0.005)
                initial = policy.snapshot()
                actions = []
                rewards = []
                for j in range(256):
                    i = j + 512
                    before = policy.snapshot()
                    action = None
                    pred = bool(first[i])
                    fee = 0.0
                    if bool(mask[i]):
                        action = (
                            int(explore[j])
                            if name != "frozen" and coins[j] < 0.2
                            else policy.choose()
                        )
                    feedback = None
                    if action is not None:
                        pred = bool(outcomes[i, action])
                        fee = 0.05
                        if name in ("cumulative", "window", "triggered"):
                            fee += 0.005
                            feedback = (
                                float(pred == bool(labels[i]))
                                - float(bool(first[i]) == bool(labels[i]))
                                - 0.055
                            )
                            policy.observe(action, feedback)
                    reward = float(pred == bool(labels[i])) - fee
                    rewards.append(reward)
                    actions.append(
                        dict(
                            index=i,
                            source=action,
                            feedback=feedback,
                            reward=reward,
                            before=before,
                            after=policy.snapshot(),
                        )
                    )
                cal_utility = (
                    base["calibration_utility"] - 0.005 * len(base["feedback"]) / 512
                )
                rows.append(
                    dict(
                        world=base["world"],
                        swapped=swapped,
                        policy=name,
                        initial=initial,
                        feedback_calibration=base["feedback"],
                        actions=actions,
                        calibration_resets=sum(initial.get("resets", [])),
                        post_resets=sum(policy.snapshot().get("resets", []))
                        - sum(initial.get("resets", [])),
                        first_reset_case=next(
                            (
                                j
                                for j, event in enumerate(actions)
                                if event["before"].get("resets")
                                != event["after"].get("resets")
                            ),
                            None,
                        ),
                        early_utility=sum(rewards[:128]) / 128,
                        late_utility=sum(rewards[128:]) / 128,
                        utility=sum(rewards) / 256,
                        combined_utility=(512 * cal_utility + sum(rewards)) / 768,
                    )
                )
    summary = {}
    for swapped in (False, True):
        summary["drift" if swapped else "static"] = {}
        for name in ("frozen", "cumulative", "window", "triggered", "no_feedback"):
            chosen = [
                r for r in rows if r["swapped"] == swapped and r["policy"] == name
            ]
            summary["drift" if swapped else "static"][name] = {
                k: sum(r[k] for r in chosen) / len(chosen)
                for k in (
                    "early_utility",
                    "late_utility",
                    "utility",
                    "combined_utility",
                )
            }
    d = summary["drift"]
    s = summary["static"]
    static_resets = (
        sum(
            r["post_resets"] > 0
            for r in rows
            if not r["swapped"] and r["policy"] == "triggered"
        )
        / worlds
    )
    return dict(
        detector=dict(
            block=32, threshold=0.15, static_reset_episode_rate=static_resets
        ),
        summary=summary,
        episodes=rows,
        environment=calibration["worlds"],
        passed=d["triggered"]["late_utility"] >= d["frozen"]["late_utility"] + 0.01
        and d["triggered"]["late_utility"] >= d["cumulative"]["late_utility"] + 0.01
        and d["triggered"]["utility"] >= d["frozen"]["utility"] - 0.01
        and s["triggered"]["utility"] >= s["frozen"]["utility"] - 0.02
        and static_resets <= 0.25,
    )
