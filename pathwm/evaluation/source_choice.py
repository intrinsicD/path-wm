"""Static-source adaptation from calibration outcomes; held-out policy frozen."""

import torch
from torch.nn import functional as F
from pathwm.models.source_choice import SourceChoice
from pathwm.evaluation.entity_gate import gate_shift_examples


def evaluate_source_choice(gate, worlds=16):
    rows = []
    for world in range(worlds):
        data = gate_shift_examples(1601 + world, 384)
        rng = torch.Generator().manual_seed(2601 + world)
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
                seed=1601 + world,
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
