"""Development diagnosis of a frozen RuleWorld perception checkpoint (validation only).

Standalone analysis companion to experiments/latent_agent.py; not library code.
Reports the C1 screen, lamp accuracy per validation texture x lamp state, a colour-
intervention panel and an appearance proxy (leave-one-out 1-NN texture identification
from machine slot tokens). With --control it applies the predeclared adoption rule of
the texture-repair comparison. If the checkpoint exports an identity key (S1
`--identity`), the same proxy is also reported on the learned keys and, with a control
that has keys too, the predeclared identity screen (plan §19) is applied; the raw rule
stays recorded. The sealed test textures are never rendered.

    .venv/bin/python -m experiments.perception_diagnosis --run runs/<arm> --output runs/<arm>/diagnosis
    .venv/bin/python -m experiments.perception_diagnosis --run runs/<arm> --control runs/<control>/diagnosis --output ...
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from pathwm.data import rule_world as rw
from pathwm.evaluation import rule_world as ev
from pathwm.evaluation.report import write_report
from pathwm.io import atomic_json, environment, file_hash, load_component, source_record
from pathwm.models.latent_core import key_head
from pathwm.models.slots import SlotPerception, pointer

YELLOW = (0.945, 0.895, 0.391)  # validation texture 53 interior, near the lamp-on colour
INTERVENTIONS = (
    ("kind53_interior_non_lamp", 53, 0, "kind48"),
    ("kind48_colour0_lamp_yellow", 48, 0, YELLOW),
    ("kind48_colour0_lamp_off", 48, 0, rw.LAMP[0]),
    ("kind55_colour0_lamp_yellow", 55, 0, YELLOW),
)
# Appearance must not decline by more than 0.02 overall AND across lamp states
# (strengthened before either comparison arm ran).
ADOPTION = dict(lamp=0.99, per_texture=0.97, attribute_pointer_decline=0.005, appearance_decline=0.02)
# Identity screen of the paired S1 comparison (plan §19): joint candidate vs detached control.
IDENTITY = dict(lamp=0.99, per_texture=0.97, attribute=0.95, machine_pointer=0.99, object_pointer=0.97,
                decline=0.005, key_cross_lamp=0.80, key_overall_decline=0.02, key_cross_lamp_gain=0.10)


def load(run):
    settings = json.loads((Path(run) / "run.json").read_text())["identity"]["settings"]["sizes"]
    model = SlotPerception(settings["width"], settings["slots"], settings["iterations"], decoder_width=settings["decoder_width"])
    load_component(model, Path(run) / "last.pt", "perception")
    return model.eval()


def load_key(run):
    """The exported identity key of an S1 `--identity` run, else None."""
    record = json.loads((Path(run) / "run.json").read_text())["identity"]["settings"]
    if not record.get("identity"):
        return None
    head = key_head(record["sizes"]["width"], record["sizes"]["key_width"])
    load_component(head, Path(run) / "last.pt", "key")
    return head.eval()


@torch.no_grad()
def cells(model, g, per_cell, kinds=None):
    """Machine-token lamp predictions and tokens per validation texture x lamp x side."""
    kinds = kinds or list(rw.KIND_SPLIT["validation"])
    pool = list(rw.KIND_SPLIT["validation"])
    out = dict(kind=[], lamp=[], correct=[], token=[])
    for kind in kinds:
        for lamp in (0, 1):
            for side in (0, 1):
                other = [k for k in pool if k != kind]
                partner = [other[int(i)] for i in torch.randint(len(other), (per_cell,), generator=g)]
                pair = torch.tensor([[kind, p] if side == 0 else [p, kind] for p in partner])
                scenes = rw.sample_scenes(g, pair)
                lamps = torch.randint(2, (per_cell, 2), generator=g)
                lamps[:, side] = lamp
                rgb, _ = rw.render(scenes, lamps)
                p = model(rgb)
                slot = pointer(p.alpha, scenes.machine_xy[:, side])
                rows = torch.arange(per_cell)
                out["kind"] += [kind] * per_cell
                out["lamp"] += [lamp] * per_cell
                out["correct"] += ((p.lamp[rows, slot] > 0).long() == lamp).tolist()
                out["token"].append(p.slots[rows, slot])
    out = {k: (torch.cat(v).numpy() if k == "token" else np.array(v)) for k, v in out.items()}
    return out


def appearance_proxy(data):
    """Leave-one-out cosine 1-NN texture identification; also across lamp states."""
    x = data["token"] / np.linalg.norm(data["token"], axis=1, keepdims=True)
    similarity = x @ x.T
    np.fill_diagonal(similarity, -np.inf)
    nearest = similarity.argmax(1)
    crossed = np.where(data["lamp"][None] != data["lamp"][:, None], similarity, -np.inf).argmax(1)
    return dict(
        nn_texture_accuracy=float((data["kind"][nearest] == data["kind"]).mean()),
        nn_texture_accuracy_across_lamp_states=float((data["kind"][crossed] == data["kind"]).mean()),
    )


def intervene(model, per_cell, seed, arrays=None):
    original = rw.KIND_COLORS.clone()
    result = {}
    for name, kind, index, value in INTERVENTIONS:
        colour = original[48][0] if value == "kind48" else torch.tensor(value)
        rw.KIND_COLORS[kind][index] = colour
        try:
            data = cells(model, torch.Generator().manual_seed(seed + kind), per_cell, [kind])
        finally:
            rw.KIND_COLORS.copy_(original)
        if arrays is not None:
            arrays.update({f"{name}/{key}": value for key, value in data.items()})
        result[name] = dict(
            lamp_acc=float(data["correct"].mean()),
            off_acc=float(data["correct"][data["lamp"] == 0].mean()),
            on_acc=float(data["correct"][data["lamp"] == 1].mean()),
        )
    return result


def adoption(candidate, control):
    """Predeclared rule of the texture-repair comparison (validation, development)."""
    c, k = candidate, control
    pointers = ("machine_pointer_accuracy", "object_pointer_accuracy", "machine_detection")
    checks = dict(
        lamp=c["c1"]["lamp_accuracy"] >= ADOPTION["lamp"],
        each_texture=min(c["per_texture"].values()) >= ADOPTION["per_texture"],
        attributes=all(a >= b - ADOPTION["attribute_pointer_decline"]
                       for a, b in zip(c["c1"]["attribute_accuracy"], k["c1"]["attribute_accuracy"])),
        pointers=all(c["c1"][p] >= k["c1"][p] - ADOPTION["attribute_pointer_decline"] for p in pointers),
        appearance=c["appearance"]["nn_texture_accuracy"] >= k["appearance"]["nn_texture_accuracy"] - ADOPTION["appearance_decline"],
        appearance_across_lamp_states=c["appearance"]["nn_texture_accuracy_across_lamp_states"]
        >= k["appearance"]["nn_texture_accuracy_across_lamp_states"] - ADOPTION["appearance_decline"],
    )
    return dict(criteria=ADOPTION, checks=checks, adopt=all(checks.values()))


def identity_screen(candidate, control):
    """Predeclared paired identity screen (plan §19); every check required."""
    c, k = candidate["c1"], control["c1"]
    key, base = candidate["appearance_key"], control["appearance_key"]
    cross = "nn_texture_accuracy_across_lamp_states"
    checks = dict(
        lamp=c["lamp_accuracy"] >= IDENTITY["lamp"],
        each_texture=min(candidate["per_texture"].values()) >= IDENTITY["per_texture"],
        attributes=all(a >= IDENTITY["attribute"] and a >= b - IDENTITY["decline"]
                       for a, b in zip(c["attribute_accuracy"], k["attribute_accuracy"])),
        machine_pointer=c["machine_pointer_accuracy"] >= IDENTITY["machine_pointer"]
        and c["machine_pointer_accuracy"] >= k["machine_pointer_accuracy"] - IDENTITY["decline"],
        object_pointer=c["object_pointer_accuracy"] >= IDENTITY["object_pointer"]
        and c["object_pointer_accuracy"] >= k["object_pointer_accuracy"] - IDENTITY["decline"],
        key_cross_lamp=key[cross] >= IDENTITY["key_cross_lamp"],
        key_overall_versus_control=key["nn_texture_accuracy"] >= base["nn_texture_accuracy"] - IDENTITY["key_overall_decline"],
        key_cross_lamp_gain=key[cross] - base[cross] >= IDENTITY["key_cross_lamp_gain"],
    )
    return dict(criteria=IDENTITY, checks=checks, **{"pass": all(checks.values())})


def status(output, result, report, step=0, error=None):
    atomic_json(output / "status.json", dict(result=result, report=report, step=step, error=error))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control", type=Path, help="diagnosis output of the control arm")
    parser.add_argument("--seed", type=int, default=1101)
    parser.add_argument("--scenes", type=int, default=256, help="C1 screen scenes (training recipe uses 256)")
    parser.add_argument("--per-cell", type=int, default=32)
    args = parser.parse_args()
    torch.set_num_threads(4)
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    (output / "metrics.jsonl").write_text("")
    status(output, "running", "pending")  # visible before anything can fail
    step = 0
    try:
        checkpoint = args.run / "last.pt"
        parent_settings = json.loads((args.run / "run.json").read_text())["identity"]["settings"]
        settings = dict(
            stage="perception-diagnosis", purpose="development", device="cpu", seed=args.seed,
            scenes=args.scenes, per_cell=args.per_cell, run=str(args.run),
            control=None if args.control is None else str(args.control),
            streams=dict(c1="seed+17", cells="seed+29", interventions="seed+kind"),
            adoption=ADOPTION,
        )
        data_identity = dict(
            checkpoint=str(checkpoint), checkpoint_sha256=file_hash(checkpoint),
            run_seed=parent_settings.get("seed"), warm_start=parent_settings.get("warm_start"),
            texture_randomization=parent_settings.get("texture_randomization", 0.0),
            validation_kinds=list(rw.KIND_SPLIT["validation"]),
        )
        atomic_json(output / "run.json", dict(
            schema="pathwm-run-v1",
            identity=dict(settings=settings, data=data_identity, environment=environment("cpu")),
            source=source_record(__file__, torch.nn.Module()),
        ))
        model = load(args.run)
        step = torch.load(checkpoint, map_location="cpu", weights_only=True)["step"]
        # Same C1 screen stream as the recipe's final perception evaluation (seed + 17).
        c1 = ev.perception_metrics(model, torch.Generator().manual_seed(args.seed + 17), "validation", args.scenes, "cpu")
        data = cells(model, torch.Generator().manual_seed(args.seed + 29), args.per_cell)
        arrays = {f"cells/{key}": value for key, value in data.items()}
        key = load_key(args.run)
        if key is not None:
            with torch.no_grad():
                arrays["cells/key"] = F.normalize(key(torch.from_numpy(data["token"])), dim=-1).numpy()
        interventions = intervene(model, args.per_cell, args.seed, arrays)
        np.savez_compressed(output / "per_example.npz", **arrays)
        per_texture = {int(k): float(data["correct"][data["kind"] == k].mean()) for k in rw.KIND_SPLIT["validation"]}
        summary = dict(
            scope="Development diagnosis on validation textures only; no test textures; not a formal gate.",
            checkpoint=dict(path=str(checkpoint), sha256=data_identity["checkpoint_sha256"], step=step),
            settings=settings, data=data_identity,
            per_example=dict(file="per_example.npz", sha256=file_hash(output / "per_example.npz"),
                             arrays=sorted(arrays)),
            c1=c1, per_texture=per_texture,
            per_texture_lamp={f"{k}/{l}": float(data["correct"][(data["kind"] == k) & (data["lamp"] == l)].mean())
                              for k in rw.KIND_SPLIT["validation"] for l in (0, 1)},
            appearance=appearance_proxy(data),
            interventions=interventions,
        )
        if key is not None:
            summary["appearance_key"] = appearance_proxy(dict(data, token=arrays["cells/key"]))
        control = json.loads((args.control / "diagnosis.json").read_text()) if args.control else None
        if control:
            summary["adoption"] = adoption(summary, control)  # raw-cosine rule, always recorded
            if "appearance_key" in summary and "appearance_key" in control:
                summary["identity_screen"] = identity_screen(summary, control)
        atomic_json(output / "diagnosis.json", summary)
        (output / "metrics.jsonl").write_text(json.dumps(dict(split="validation", lamp_accuracy=c1["lamp_accuracy"])) + "\n")
        screen = summary.get("identity_screen")
        atomic_json(output / "result.json", dict(
            evaluation_scope=summary["scope"],
            gate=screen["pass"] if screen else summary.get("adoption", {}).get("adopt"),
            metrics=dict(c1={k: v for k, v in c1.items() if not isinstance(v, list)},
                         attributes={f"attribute_{i}": v for i, v in enumerate(c1["attribute_accuracy"])},
                         per_texture={f"kind_{k}": v for k, v in per_texture.items()},
                         appearance=summary["appearance"], interventions=summary["interventions"],
                         **({"appearance_key": summary["appearance_key"]} if "appearance_key" in summary else {}),
                         **({"adoption": summary["adoption"]["checks"]} if args.control else {}),
                         **({"identity_screen": screen["checks"]} if screen else {})),
            limitations=["Development validation population; colour interventions edit the in-process texture table only."]
            + (["Gate = identity screen (plan §19); the raw-cosine adoption rule is reported, not replaced."] if screen else []),
        ))
    except Exception as error:
        status(output, "failed", "incomplete", step, f"{type(error).__name__}: {error}")
        raise
    status(output, "completed", "pending", step)
    try:
        write_report(output)  # sets report status to structural_verified
    except Exception as error:
        status(output, "completed", "failed", step, f"report: {type(error).__name__}: {error}")
        raise
    print(json.dumps({k: summary[k] for k in ("per_texture", "appearance", "interventions")}
                     | dict(c1_lamp=c1["lamp_accuracy"], adoption=summary.get("adoption"),
                            appearance_key=summary.get("appearance_key"), identity_screen=summary.get("identity_screen")), indent=1))


if __name__ == "__main__":
    main()
