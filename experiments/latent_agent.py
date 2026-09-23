"""R1 RuleWorld-64 reference integration: perceive, induce, remember, plan, correct.

Stages (see docs/integrated-architecture-plan.md):
  audit       S0 generator/split/floor/strata checks (CPU, no model)
  perception  S1 multiscale encoder + slots + decoder + training-only label heads
  core        S2 shared latent G/T core on frozen perception (prepare-once frames)
  symbolic    S2s structured-input diagnostic: same core on supplied render symbols
  evaluate    S3 frozen-weight pixel lives with controls; point-estimate screen
  gates       formal one-sided t lower bounds over >=2 evaluations (3 seeds planned)
  check       tiny CPU run of every stage above, for software verification only

Only frames, action records/receipts, goals and correction messages reach the
agent. Hidden rules/kinds/labels are generator, loss and evaluator knowledge.
"""

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from pathwm.data import rule_world as rw
from pathwm.evaluation import rule_world as ev
from pathwm.evaluation.report import write_report
from pathwm.io import (
    Run,
    atomic_json,
    digest,
    environment,
    evaluation_mode,
    load_component,
    resume_arguments,
    seed_everything,
    source_record,
    state_hash,
    training_mode,
)
from pathwm.models.latent_core import LatentCore, episode_loss
from pathwm.models.slots import SlotPerception, SymbolicSlots, perception_loss

SIZES = dict(
    full=dict(width=64, heads=4, slots=7, iterations=3, decoder_width=32, loops=2, code_tokens=4, key_width=32,
              perception_batch=32, episodes=16, support=(8, 16, 32, 64, 128), queries=32,
              validation_scenes=256, validation_episodes=48, validate_every=250,
              life_supports=(8, 32, 128), lives_per_support=4, life_queries=64, life_goals=16,
              distract=32, counter=24),
    check=dict(width=16, heads=2, slots=7, iterations=2, decoder_width=8, loops=2, code_tokens=2, key_width=8,
               perception_batch=4, episodes=2, support=(4, 8), queries=4,
               validation_scenes=8, validation_episodes=4, validate_every=2,
               life_supports=(8,), lives_per_support=1, life_queries=4, life_goals=4,
               distract=8, counter=8),
)


class RuleModel(nn.Module):
    """Perception (pixel or symbolic diagnostic) + shared core + fixed target variance."""

    def __init__(self, s, symbolic=False):
        super().__init__()
        self.perception = (
            SymbolicSlots(s["width"])
            if symbolic
            else SlotPerception(s["width"], s["slots"], s["iterations"], decoder_width=s["decoder_width"])
        )
        self.core = LatentCore(s["width"], s["heads"], s["loops"], s["code_tokens"], s["key_width"])
        self.register_buffer("variance", torch.ones(s["width"]))


def resources(device):
    result = {}
    if torch.device(device).type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        result = dict(
            torch_max_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30,
            torch_max_reserved_gib=torch.cuda.max_memory_reserved(device) / 2**30,
            device_used_gib_at_end=(total - free) / 2**30,
            device_total_gib=total / 2**30,
            reserved_within_6gib=torch.cuda.max_memory_reserved(device) <= 6 * 2**30,
        )
    return result


class ResourceCeiling(RuntimeError):
    pass


def enforce_ceiling(args, runner=None):
    """Cheap stage/update-boundary check of the declared reserved-memory ceiling."""
    if torch.device(args.device).type != "cuda":
        return
    reserved = torch.cuda.max_memory_reserved(args.device) / 2**30
    if reserved > args.max_reserved_gib:
        if runner is not None:
            runner.save()  # preserve the checkpoint for inspection/resume
        raise ResourceCeiling(f"torch reserved {reserved:.2f} GiB exceeds declared {args.max_reserved_gib} GiB")


def stop_reason(runner, args, started):
    if runner.step >= args.updates:
        return "updates_complete"
    if args.stop_after is not None and runner.step >= args.stop_after:
        return "stop_after (deliberate pause; resumable)"
    return "time_cap (deliberate bounded development run; resumable)"


def parameters(module):
    return dict(total=sum(p.numel() for p in module.parameters()),
                trainable=sum(p.numel() for p in module.parameters() if p.requires_grad))


def finish(runner_or_path, result, *, complete, images=None):
    """Write the result first, then the report; report failure stays visible."""
    runner = runner_or_path if isinstance(runner_or_path, Run) else None
    path = runner.path if runner else Path(runner_or_path)
    atomic_json(path / "result.json", result)
    status = "completed" if complete else "paused"
    mark = runner.status if runner else (lambda r, rep, error=None: write_status(path, r, rep, error))
    mark(status, "pending")
    try:
        if images is not None:
            write_report(path, batch=dict(rgb=images[0]), outputs=dict(rgb=images[1]))
        else:
            write_report(path)
    except Exception as error:
        mark(status, "failed", str(error))
        raise


def write_status(path, result, report, error=None):
    step = json.loads((Path(path) / "status.json").read_text()).get("step", 0) if (Path(path) / "status.json").exists() else 0
    atomic_json(Path(path) / "status.json", dict(result=result, report=report, step=step, error=error))


def record_directory(path, settings, data, model=None):
    """Run record for stages without an optimizer (audit/evaluate/gates)."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    source = source_record(__file__, model if model is not None else nn.Module())
    atomic_json(path / "run.json", dict(schema="pathwm-run-v1", identity=dict(settings=settings, data=data, environment=environment(settings["device"])), source=source))
    (path / "recipe.py").write_text(Path(__file__).read_text())
    (path / "metrics.jsonl").write_text("")
    atomic_json(path / "status.json", dict(result="running", report="pending", step=0, error=None))
    return path


def log_rows(path, rows):
    with (Path(path) / "metrics.jsonl").open("a") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def flat(prefix, values):
    out = {}
    for k, v in values.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
            out[f"{prefix}{k}"] = float(v)
    return out


# ---------------------------------------------------------------- S0 audit


def audit(args, s):
    settings = dict(stage="audit", seed=args.seed, device="cpu", purpose="development")
    path = record_directory(args.output, settings, rw.manifest())
    g = torch.Generator().manual_seed(args.seed)
    rules = rw.grammar()
    domain = torch.cartesian_prod(*[torch.arange(4)] * 8, torch.arange(2))
    tensors = rw.rule_tensors(rules)
    tables, mismatches = set(), 0
    for index, rule in enumerate(rules):
        batch = {k: v[index].expand(len(domain), *v.shape[1:]) for k, v in tensors.items()}
        out = rw.outcome_batch(batch, domain[:, :4], domain[:, 4:8], domain[:, 8])
        tables.add(out.to(torch.uint8).numpy().tobytes())
        for r in torch.randint(len(domain), (32,), generator=g).tolist():
            x = domain[r].tolist()
            mismatches += rw.outcome(rule, x[:4], x[4:8], x[8]) != int(out[r])
    floors = rw.floors()
    split = rw.split_rules()
    strata = {k: 0 for k in rw.STRATA}
    unavailable = {k: 0 for k in rw.STRATA}
    attempts = {k: [] for k in rw.STRATA}
    samples = 64 if args.stage == "audit" else 8
    for trial in range(samples):
        pair = [split["train"][int(i)] for i in torch.randint(len(split["train"]), (2,), generator=g)]
        for stratum in rw.STRATA:
            try:
                task = rw.sample_task(g, torch.tensor([0, 1]), pair, stratum, cap=200)
                strata[stratum] += 1
                attempts[stratum].append(task.attempts)
                assert rw.min_presses(task.rules, task.scene.attrs[0].tolist(), task.lamps.tolist(), task.goal, 2) == {"already": 0, "reach1": 1, "reach2": 2, "unreachable": None}[stratum]
            except rw.StratumUnavailable:
                unavailable[stratum] += 1
    rgb, entity = rw.render(rw.sample_scenes(g, torch.tensor([[0, 1], [48, 60]])), torch.tensor([[0, 1], [1, 0]]))
    result = dict(
        evaluation_scope="S0 audit: generator, independent truth, split, floors, strata; no model",
        gate=len(tables) == 280 and mismatches == 0 and digest(rw.split_groups()) == rw.manifest()["split_sha256"],
        metrics=dict(
            grammar=dict(rules=len(rules), distinct_functions=len(tables), scalar_mismatches=mismatches),
            split={k: len(v) for k, v in split.items()},
            floors={f"{f}_{k}": v for f, d in floors.items() for k, v in d.items()},
            strata=dict(**{f"{k}_available": v for k, v in strata.items()}, **{f"{k}_unavailable": v for k, v in unavailable.items()},
                        **{f"{k}_mean_attempts": float(np.mean(v)) if v else None for k, v in attempts.items()}),
        ),
        floors=floors,
        limitations=["Strata availability depends on the drawn rule pair; unavailable strata are counted, never retried silently."],
    )
    log_rows(path, [dict(split="validation", **result["metrics"]["grammar"])])
    finish(path, result, complete=True, images=(rgb, rgb))
    return path


# ---------------------------------------------------------------- S1 perception


def perception_batch(generator, kinds, count, device):
    pick = torch.tensor(kinds)[torch.randint(len(kinds), (count, 2), generator=generator)]
    scenes = rw.sample_scenes(generator, pick)
    lamps = torch.randint(2, (count, 2), generator=generator)
    rgb, entity = rw.render(ev._scenes_to(scenes, device), lamps.to(device))
    return scenes, lamps, rgb, entity


def train_perception(args, s):
    seed_everything(args.seed)
    model = nn.ModuleDict(dict(perception=SlotPerception(s["width"], s["slots"], s["iterations"], decoder_width=s["decoder_width"]))).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    settings = dict(stage="perception", seed=args.seed, updates=args.updates, lr=args.lr, device=args.device,
                    size=args.size, sizes=s, purpose="development", precision="fp32",
                    max_reserved_gib=args.max_reserved_gib,
                    objective="RGB MSE + 0.5*(matched mask CE + kind CE + attribute CE + lamp BCE); labels training-only")
    runner = Run(args.resume or args.output, settings=settings, data=rw.manifest(), recipe=__file__, model=model,
                 optimizer=optimizer, device=args.device, resume=args.resume is not None)
    started = time.perf_counter()
    try:
        while runner.step < args.updates and not stop(runner, args, started):
            training_mode(model)
            scenes, lamps, rgb, entity = perception_batch(runner.sampler, rw.KIND_SPLIT["train"], s["perception_batch"], args.device)
            percept = model["perception"](rgb)
            loss, metrics = perception_loss(percept, rgb, entity, scenes.attrs.to(args.device), lamps.to(args.device))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            runner.step += 1
            runner.log(dict(step=runner.step, split="train", **metrics))
            enforce_ceiling(args, runner)
            if runner.step % s["validate_every"] == 0 or runner.step == args.updates:
                runner.log(validate_perception(model["perception"], runner.step, args, s))
                runner.save()
                print(f"perception: saved step {runner.step}", flush=True)
        runner.save()
        with evaluation_mode(model):
            metrics = ev.perception_metrics(model["perception"], torch.Generator().manual_seed(args.seed + 17), "validation", s["validation_scenes"], args.device)
            _, _, rgb, _ = perception_batch(torch.Generator().manual_seed(args.seed + 18), rw.KIND_SPLIT["validation"], 8, args.device)
            recon = model["perception"](rgb).recon
        elapsed = time.perf_counter() - started
        result = dict(
            evaluation_scope="Development: validation textures, fresh scenes; C1 screen only; not the sealed test population",
            gate=None,
            metrics=dict(validation=flat("", {k: v for k, v in metrics.items() if not isinstance(v, list)}),
                         attributes={f"attribute_{i}": v for i, v in enumerate(metrics["attribute_accuracy"])},
                         resources=dict(**resources(args.device), updates=runner.step, seconds=elapsed,
                                        updates_per_second=runner.step / max(elapsed, 1e-9), **parameters(model))),
            screen_C1=min(metrics["attribute_accuracy"]) >= ev.THRESHOLDS["C1_attribute"] and metrics["lamp_accuracy"] >= ev.THRESHOLDS["C1_lamp"]
            and metrics["machine_pointer_accuracy"] >= ev.THRESHOLDS["C1_machine_pointer"] and metrics["object_pointer_accuracy"] >= ev.THRESHOLDS["C1_object_pointer"],
            stop_reason=stop_reason(runner, args, started),
            limitations=["Synthetic rendered scenes; training-only mask/attribute/lamp supervision is disclosed."],
        )
        finish(runner, result, complete=runner.step >= args.updates, images=(rgb.cpu(), recon.cpu()))
    except Exception as error:
        runner.status("failed", "incomplete", str(error))
        raise
    return runner.path


@torch.no_grad()
def validate_perception(perception, step, args, s):
    was = perception.training
    perception.eval()
    g = torch.Generator().manual_seed(args.seed + 5)
    scenes, lamps, rgb, entity = perception_batch(g, rw.KIND_SPLIT["validation"], min(32, s["validation_scenes"]), args.device)
    loss, metrics = perception_loss(perception(rgb), rgb, entity, scenes.attrs.to(args.device), lamps.to(args.device))
    perception.train(was)
    return dict(step=step, split="validation", **metrics)


def stop(runner, args, started):
    if args.stop_after is not None and runner.step >= args.stop_after:
        return True
    return args.max_minutes is not None and time.perf_counter() - started >= 60 * args.max_minutes


# ---------------------------------------------------------------- S2 core / S2s symbolic


def train_core(args, s, *, symbolic=False):
    seed_everything(args.seed)
    model = RuleModel(s, symbolic=symbolic).to(args.device)
    # The symbolic diagnostic's machine token is lamp+role only (no appearance):
    # appearance-key InfoNCE is unidentifiable there, so it is disabled, and the
    # supplied-symbol embeddings stay fixed (encode_episodes runs under no_grad).
    key_weight = 0.0 if symbolic else 0.2
    if symbolic:
        model.perception.requires_grad_(False)
        model.core.key_head.requires_grad_(False)  # unused without appearance
        perceive = ev.symbolic_perceiver(model.perception)
    else:
        load_component(model.perception, Path(args.perception) / "last.pt", "perception")
        model.perception.requires_grad_(False)
        perceive = ev.pixel_perceiver(model.perception)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    floors = rw.floors()
    settings = dict(stage="symbolic" if symbolic else "core", seed=args.seed, updates=args.updates, lr=args.lr,
                    device=args.device, size=args.size, sizes=s, purpose="development", precision="fp32",
                    max_reserved_gib=args.max_reserved_gib,
                    perception=None if symbolic else str(args.perception),
                    perception_sha256=None if symbolic else state_hash(model.perception),
                    objective="unweighted outcome BCE + next-token (k=1,2) + 0.5 rollout BCE"
                    + ("" if symbolic else " + 0.2 key InfoNCE"),
                    key_weight=key_weight,
                    diagnostic=("supplied render symbols and exact slot masks; fixed symbol embeddings; no appearance, "
                                "so no key objective and no retrieval claim; not an R1 pixel gate") if symbolic else None)
    data = rw.manifest()
    runner = Run(args.resume or args.output, settings=settings, data=data, recipe=__file__, model=model,
                 optimizer=optimizer, device=args.device, resume=args.resume is not None)
    split = rw.split_rules()
    validation = rw.sample_episodes(torch.Generator().manual_seed(args.seed + 7), split["validation"], rw.KIND_SPLIT["validation"],
                                    episodes=s["validation_episodes"], support=(max(s["support"]),), queries=s["queries"], p_empty=0.0)
    started = time.perf_counter()
    try:
        if runner.step == 0 and not args.resume:
            with torch.no_grad():  # fixed target scale from training frames (declared buffer)
                warm = rw.sample_episodes(torch.Generator().manual_seed(args.seed + 3), split["train"], rw.KIND_SPLIT["train"],
                                          episodes=max(2, s["episodes"]), support=s["support"], queries=s["queries"], p_empty=0.0)
                tokens = ev.encode_episodes(perceive, warm, args.device)
                model.variance.copy_(torch.cat((tokens.support.m_post, tokens.query.m_post)).var(0).clamp_min(1e-4))
        while runner.step < args.updates and not stop(runner, args, started):
            training_mode(model)
            batch = rw.sample_episodes(runner.sampler, split["train"], rw.KIND_SPLIT["train"], episodes=s["episodes"],
                                       support=s["support"], queries=s["queries"])
            tokens = ev.encode_episodes(perceive, batch, args.device)
            loss, metrics = episode_loss(model.core, tokens, model.variance, key_weight=key_weight)
            optimizer.zero_grad()
            loss.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in trainable):
                raise ValueError("Nonfinite gradient")
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            runner.step += 1
            runner.log(dict(step=runner.step, split="train", **metrics))
            enforce_ceiling(args, runner)
            if runner.step % s["validate_every"] == 0 or runner.step == args.updates:
                with evaluation_mode(model):
                    tokens = ev.encode_episodes(perceive, validation, args.device)
                    v_loss, v_metrics = episode_loss(model.core, tokens, model.variance, key_weight=key_weight)
                    m = ev.episode_metrics(model.core, perceive, validation, args.device, floors)
                runner.log(dict(step=runner.step, split="validation", **v_metrics,
                                **flat("nu_", m["nu"]), **flat("calibration_", m["calibration"])))
                runner.save()
                print(f"{settings['stage']}: saved step {runner.step}", flush=True)
        runner.save()
        with evaluation_mode(model):
            m = ev.episode_metrics(model.core, perceive, validation, args.device, floors)
        elapsed = time.perf_counter() - started
        result = dict(
            evaluation_scope=("Diagnostic: supplied symbols and exact masks" if symbolic else "Development: supplied support routing on validation kinds/rules; lives are the integrated test")
            + "; no retrieval, no planning",
            gate=None,
            metrics=dict(nu=flat("", m["nu"]), calibration=m["calibration"],
                         resources=dict(**resources(args.device), updates=runner.step, seconds=elapsed,
                                        updates_per_second=runner.step / max(elapsed, 1e-9), **parameters(model))),
            stop_reason=stop_reason(runner, args, started),
            limitations=["Episode metrics supply the concept routing; they do not test autonomous binding or planning."],
        )
        finish(runner, result, complete=runner.step >= args.updates)
    except Exception as error:
        runner.status("failed", "incomplete", str(error))
        raise
    return runner.path


# ---------------------------------------------------------------- S3 evaluate / gates


def evaluate(args, s):
    source = Path(args.core)
    record = json.loads((source / "run.json").read_text())["identity"]["settings"]
    if record["stage"] != "core":
        raise ValueError("Evaluation needs a pixel core run (symbolic runs are diagnostics)")
    sizes = record["sizes"]
    model = RuleModel(sizes).to(args.device)
    state = torch.load(source / "last.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"], strict=True)
    model.requires_grad_(False)
    perception_run = Path(record["perception"])
    perception_settings = json.loads((perception_run / "run.json").read_text())["identity"]["settings"]
    protocol = dict(population=args.population, size=args.size, model_size=record["size"], life_supports=list(s["life_supports"]),
                    lives_per_support=s["lives_per_support"], life_queries=s["life_queries"],
                    life_goals=s["life_goals"], distract=s["distract"], counter=s["counter"],
                    family_schedule=[list(ev.FAMILY_SCHEDULE[i % len(ev.FAMILY_SCHEDULE)]) for i in range(s["lives_per_support"])])
    settings = dict(stage="evaluate", seed=args.seed, device=args.device, population=args.population,
                    core=str(source), core_sha256=state_hash(model), purpose="development" if args.population != "test" else "formal-candidate",
                    sizes=sizes, protocol=protocol, agent=vars(ev.AgentSettings(seed=args.seed)))
    path = record_directory(args.output, settings, rw.manifest(), model)
    floors = rw.floors()
    provenance = dict(
        core_run=str(source), core_seed=record["seed"], core_updates=state["step"],
        core_sha256=state_hash(model.core),
        perception_run=str(perception_run), perception_seed=perception_settings["seed"],
        perception_sha256=state_hash(model.perception), protocol=protocol, evaluation_seed=args.seed,
        source_sha256=json.loads((path / "run.json").read_text())["source"]["sha256"],
        floors_sha256=digest(floors),
    )
    started = time.perf_counter()
    lives = []
    try:
        with evaluation_mode(model):
            before = state_hash(model)
            perception = ev.perception_metrics(model.perception, torch.Generator().manual_seed(args.seed + 1), args.population, s["validation_scenes"], args.device)
            g = torch.Generator().manual_seed(args.seed)
            for n in s["life_supports"]:
                for i in range(s["lives_per_support"]):
                    spec = ev.sample_life(g, args.population, n_support=n, queries=s["life_queries"], goals=s["life_goals"],
                                          distract=s["distract"], counter=s["counter"],
                                          families=ev.FAMILY_SCHEDULE[i % len(ev.FAMILY_SCHEDULE)])
                    life = ev.run_life(model.perception, model.core, spec, path / "lives" / f"n{n}_{i}",
                                       settings=ev.AgentSettings(seed=args.seed + i), device=args.device)
                    lives.append(life)
                    enforce_ceiling(args)
                    log_rows(path, [dict(step=len(lives), split="validation", n_support=n, **{f"check_{k}": float(v) for k, v in life["checks"].items() if v is not None})])
            unchanged = state_hash(model) == before
        with (path / "lives.jsonl").open("w") as f:
            for life in lives:
                f.write(json.dumps(life, sort_keys=True, default=str) + "\n")
        summary = ev.summarize(lives, floors, perception, provenance)
        summary["weights_unchanged_across_evaluation"] = unchanged
        elapsed = time.perf_counter() - started
        atomic_json(path / "summary.json", json.loads(json.dumps(summary, default=str)))
        result = dict(
            evaluation_scope=f"{args.population} population; single training seed point-estimate screen; formal gates need the gates stage",
            gate=None,
            metrics=dict(screen=summary["screen"], nu_full=summary["nu_full"], nu_empty=summary["nu_empty"],
                         swap=summary["swap"], calibration=summary["calibration"], bindings=summary["bindings"],
                         tasks_agent=_task_flat(summary["tasks"]["agent"]), tasks_empty=_task_flat(summary["tasks"]["empty_memory"]),
                         tasks_random=_task_flat(summary["tasks"]["random"]), claims=summary["claims"],
                         checks={k: f"{v['true']} true / {v['false']} false / {v['not_applicable']} n/a" for k, v in summary["checks"].items()},
                         resources=dict(**resources(args.device), seconds=elapsed, lives=len(lives),
                                        blob_bytes_mean=float(np.mean([l["memory"]["blob_bytes"] for l in lives])),
                                        store_bytes_mean=float(np.mean([l["memory"]["store_bytes"] for l in lives])))),
            limitations=["Synthetic, deterministic, fully observed RuleWorld-64; no natural-data or cross-domain claim.",
                         "Binding decisions are retained across evidence repair (no automatic re-verification)."],
        )
        finish(path, json.loads(json.dumps(result, default=str)), complete=True)
    except Exception as error:
        write_status(path, "failed", "incomplete", str(error))
        raise
    return path


def _task_flat(t):
    out = {}
    for k, v in t.items():
        if isinstance(v, dict):
            out[f"{k}_rate"], out[f"{k}_count"] = v["rate"], v["count"]
        else:
            out[k] = v
    return out


def gates(args, s):
    evaluations = [str(Path(e).resolve()) for e in args.evaluations]
    settings = dict(stage="gates", seed=args.seed, device="cpu", evaluations=evaluations, purpose="formal-candidate")
    path = record_directory(args.output, settings, rw.manifest())
    try:
        if len(set(evaluations)) != len(evaluations):
            raise ValueError("Duplicate evaluation paths are not independent training seeds")
        summaries = [json.loads((Path(e) / "summary.json").read_text()) for e in evaluations]
        result = ev.aggregate_gates(summaries)
        finish(path, dict(
            evaluation_scope=f"{len(summaries)} evaluation(s); status {result['status']}",
            gate=result["status"] == ev.PASS if result["status"] != ev.INELIGIBLE else None,
            metrics=json.loads(json.dumps(result, default=str)),
        ), complete=True)
    except Exception as error:
        write_status(path, "failed", "incomplete", str(error))
        raise
    return path


# ---------------------------------------------------------------- check


def check(args):
    """Tiny CPU pass through every stage; software verification only."""
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    base = dict(vars(args))
    s = SIZES["check"]

    def sub(**changes):
        a = argparse.Namespace(**{**base, **changes})
        return a

    audit(sub(stage="check-audit", output=root / "audit"), s)
    perception = train_perception(sub(output=root / "perception", updates=3, stop_after=None, resume=None, size="check"), s)
    core = train_core(sub(output=root / "core", perception=perception, updates=3, stop_after=None, resume=None, size="check"), s)
    train_core(sub(output=root / "symbolic", updates=3, stop_after=None, resume=None, size="check"), s, symbolic=True)
    evaluation = evaluate(sub(output=root / "evaluate", core=core, population="validation"), s)
    gates(sub(output=root / "gates", evaluations=[evaluation]), s)
    print(json.dumps(dict(output=str(root), stages=sorted(p.name for p in root.iterdir())), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", required=True, choices=("check", "audit", "perception", "core", "symbolic", "evaluate", "gates"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=1101)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--size", choices=tuple(SIZES), default="full")
    parser.add_argument("--updates", type=int, default=100000)
    parser.add_argument("--max-minutes", type=float)
    parser.add_argument("--max-reserved-gib", type=float, default=6.0)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--perception", type=Path)
    parser.add_argument("--core", type=Path)
    parser.add_argument("--population", choices=("validation", "test"), default="validation")
    parser.add_argument("--evaluations", type=Path, nargs="*")
    parser.add_argument("--check", action="store_true", help=argparse.SUPPRESS)
    args = resume_arguments(parser, parser.parse_args())
    if args.stage == "check":
        args.device = "cpu"
        args.size = "check"
        if args.output is None:
            parser.error("--output required")
        return check(args)
    if args.output is None and args.resume is None:
        parser.error("--output required (a new directory)")
    s = SIZES[args.size]
    if args.stage == "audit":
        audit(args, s)
    elif args.stage == "perception":
        train_perception(args, s)
    elif args.stage in ("core", "symbolic"):
        if args.stage == "core" and args.perception is None and args.resume is None:
            parser.error("--perception RUN required")
        train_core(args, s, symbolic=args.stage == "symbolic")
    elif args.stage == "evaluate":
        if args.core is None:
            parser.error("--core RUN required")
        evaluate(args, s)
    else:
        if not args.evaluations:
            parser.error("--evaluations required")
        gates(args, s)


if __name__ == "__main__":
    main()
