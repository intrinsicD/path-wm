"""R1 RuleWorld-64 reference integration: perceive, induce, remember, plan, correct.

Stages (see docs/integrated-architecture-plan.md):
  audit       S0 generator/split/floor/strata checks (CPU, no model)
  perception  S1 multiscale encoder + slots + decoder + training-only label heads
  core        S2 shared latent G/T core on frozen perception (prepare-once frames)
  symbolic    S2s structured-input diagnostic: same core on supplied render symbols;
              with --oracle-curriculum R: ORACLE application curriculum (R relation-only
              updates, then all training rules; rule IDs supplied, diagnostic only)
  evaluate    S3 frozen-weight pixel lives with controls; point-estimate screen
  gates       formal one-sided t lower bounds over >=2 evaluations (3 seeds planned)
  check       tiny CPU run of every stage above, for software verification only

Only frames, action records/receipts, goals and correction messages reach the
agent. Hidden rules/kinds/labels are generator, loss and evaluator knowledge.
"""

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from pathwm.data import rule_world as rw
from pathwm.evaluation import rule_world as ev
from pathwm.evaluation.report import write_report
from pathwm.io import (
    Run,
    atomic_json,
    digest,
    environment,
    file_hash,
    evaluation_mode,
    load_component,
    resume_arguments,
    seed_everything,
    source_record,
    state_hash,
    training_mode,
)
from pathwm.models.latent_core import LatentCore, cross_entropy, episode_loss, key_head
from pathwm.models.slots import SlotPerception, SymbolicSlots, match_slots, perception_loss, pointer
from pathwm.models.slots import cross_entropy as slot_cross_entropy

IDENTITY_TEMPERATURE, IDENTITY_WEIGHT = 0.1, 0.2  # the S2 key InfoNCE temperature and weight
# Frozen-perception key repair (predeclared): train-kind monitor stream and screen.
MONITOR_SEED, MONITOR_PAIRS = 3502, 64
KEY_SCREEN = dict(positive_q01=0.93, negative_q99=0.60, within_q99=0.60)

SIZES = dict(
    # full = E4 core form (docs/core-design.md): core width 128, 2 untied blocks, 2 inner rounds;
    # perception unchanged (slots 64, encoder/decoder 32). r1 = the former form, now a downscaled run.
    full=dict(width=64, core_width=128, core_blocks=2, heads=4, slots=7, iterations=3, decoder_width=32, loops=2,
              code_tokens=4, key_width=32,
              perception_batch=32, episodes=16, support=(8, 16, 32, 64, 128), queries=32,
              validation_scenes=256, validation_episodes=48, pool_episodes=64, validate_every=250,
              life_supports=(8, 32, 128), lives_per_support=4, life_queries=64, life_goals=16,
              distract=32, counter=24),
    r1=dict(width=64, core_width=64, core_blocks=1, heads=4, slots=7, iterations=3, decoder_width=32, loops=2,
              code_tokens=4, key_width=32,
              perception_batch=32, episodes=16, support=(8, 16, 32, 64, 128), queries=32,
              validation_scenes=256, validation_episodes=48, pool_episodes=64, validate_every=250,
              life_supports=(8, 32, 128), lives_per_support=4, life_queries=64, life_goals=16,
              distract=32, counter=24),
    check=dict(width=16, core_width=16, core_blocks=1, heads=2, slots=7, iterations=2, decoder_width=8, loops=2, code_tokens=2, key_width=8,
               perception_batch=4, episodes=2, support=(4, 8), queries=4,
               validation_scenes=8, validation_episodes=4, pool_episodes=4, validate_every=2,
               life_supports=(8,), lives_per_support=1, life_queries=4, life_goals=4,
               distract=8, counter=8),
)


class RuleModel(nn.Module):
    """Perception (pixel or symbolic diagnostic) + shared core + fixed target variance."""

    def __init__(self, s, symbolic=False):
        super().__init__()
        core = s.get("core_width", s["width"])
        if not symbolic and core != s["width"]:
            raise ValueError("Pixel perception (slot width %d) needs the planned 64->core projection before a "
                             "core of width %d; use --size r1 until it exists" % (s["width"], core))
        self.perception = (
            SymbolicSlots(core)  # supplied symbols are embedded directly in core width
            if symbolic
            else SlotPerception(s["width"], s["slots"], s["iterations"], decoder_width=s["decoder_width"])
        )
        self.core = LatentCore(core, s["heads"], s["loops"], s["code_tokens"], s["key_width"],
                               blocks=s.get("core_blocks", 1))
        self.register_buffer("variance", torch.ones(core))


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


def enforce_ceiling(args, runner=None, guard=None):
    """Cheap stage/update-boundary check of the declared reserved-memory ceiling.

    `guard` runs before the preserving save (frozen-perception hash check)."""
    if torch.device(args.device).type != "cuda":
        return
    reserved = torch.cuda.max_memory_reserved(args.device) / 2**30
    if reserved > args.max_reserved_gib:
        if runner is not None:
            if guard is not None:
                guard()
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


def finish(runner_or_path, result, *, complete, images=None, status=None):
    """Write the result first, then the report; report failure stays visible."""
    runner = runner_or_path if isinstance(runner_or_path, Run) else None
    path = runner.path if runner else Path(runner_or_path)
    atomic_json(path / "result.json", result)
    status = status or ("completed" if complete else "paused")
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


def perception_scenes(generator, kinds, count, *, randomize=0.0, augmentation=None):
    """Base scenes/lamps always come from `generator` with unchanged draws.

    With `randomize>0`, each machine body texture is replaced with that probability
    by a procedural texture drawn ONLY from the separate `augmentation` generator, so
    randomized and unchanged arms see identical base scenes, lamps and labels.
    """
    pick = torch.tensor(kinds)[torch.randint(len(kinds), (count, 2), generator=generator)]
    scenes = rw.sample_scenes(generator, pick)
    lamps = torch.randint(2, (count, 2), generator=generator)
    textures, stats = None, None
    if randomize:
        base = rw.kind_textures(scenes.kind)
        drawn, stats = rw.TextureSampler().sample(augmentation, count)
        replace = torch.rand(count, 2, generator=augmentation) < randomize
        textures = base.where(replace, drawn)
        stats = dict(stats, replaced=int(replace.sum()))
    return scenes, lamps, textures, stats


def perception_batch(generator, kinds, count, device, *, randomize=0.0, augmentation=None):
    scenes, lamps, textures, stats = perception_scenes(generator, kinds, count, randomize=randomize, augmentation=augmentation)
    rgb, entity = rw.render(ev._scenes_to(scenes, device), lamps.to(device), textures)
    return scenes, lamps, rgb, entity, stats


def paired_perception_batch(generator, kinds, count, device, *, randomize, augmentation, pairing):
    """View A (the unchanged default batch) followed by its paired view B (`rw.paired_view`).

    Returns the concatenated 2*count batch plus loss-only (textures, source) pairing.
    """
    scenes, lamps, textures, stats = perception_scenes(generator, kinds, count, randomize=randomize, augmentation=augmentation)
    textures = rw.kind_textures(scenes.kind) if textures is None else textures
    scenes_b, lamps_b, textures_b, source = rw.paired_view(pairing, scenes, lamps, textures)
    scenes = rw.Scenes.cat([scenes, scenes_b])
    lamps = torch.cat((lamps, lamps_b))
    textures = rw.Textures(*(torch.cat((a, b)) for a, b in zip(vars(textures).values(), vars(textures_b).values())))
    rgb, entity = rw.render(ev._scenes_to(scenes, device), lamps.to(device), textures)
    return scenes, lamps, rgb, entity, stats, (textures, source)


def pairing_stream(seed, step):
    """Deterministic per-update view-B stream, separate from base and texture streams."""
    return torch.Generator().manual_seed(seed * 1_000_003 + 104_729 + step)


def texture_matches(textures, n):
    """[n,n] exact body-texture equality between view-A machines (rows) and view-B machines."""
    colors = textures.colors.reshape(-1, 2, 3)
    pattern, period = textures.pattern.flatten(), textures.period.flatten()
    return ((colors[:n, None] == colors[None, n:]).flatten(2).all(-1)
            & (pattern[:n, None] == pattern[None, n:]) & (period[:n, None] == period[None, n:]))


def identity_loss(key, percept, scenes, textures, source, *, detached, temperature=IDENTITY_TEMPERATURE,
                  margin=None):
    """InfoNCE from view-A machine keys to view-B machine keys (S2 form, tau 0.1).

    Positive = the place showing the same sampled texture; other exactly equal textures
    are masked as false negatives. Machine tokens use the runtime pointer. `detached`
    trains the key on sg(slot): no identity gradient reaches perception.

    `margin=(positive, negative, weight)` optionally adds squared hinges on absolute
    cosine: same-machine pairs below `positive`, and different-texture pairs above
    `negative` (all cross-view pairs plus the two machines of each frame, equal textures
    masked). None keeps the original objective exactly.
    """
    device = percept.slots.device
    rows = torch.arange(len(scenes), device=device)
    slots = torch.stack([percept.slots[rows, pointer(percept.alpha, scenes.machine_xy[:, m].to(device))]
                         for m in (0, 1)], 1).flatten(0, 1)  # flat index = scene*2 + side
    if detached:
        slots = slots.detach()
    keys = F.normalize(key(slots), dim=-1)
    n = len(keys) // 2
    target = torch.argsort(source.flatten()).to(device)  # A machine i -> the B place showing it
    false_negative = texture_matches(textures, n).to(device)
    false_negative[torch.arange(n, device=device), target] = False
    similarity = (keys[:n] @ keys[n:].T / temperature).masked_fill(false_negative, -1e4)
    loss = cross_entropy(similarity, target)
    with torch.no_grad():
        metrics = dict(identity_nce=float(loss), identity_accuracy=float((similarity.argmax(-1) == target).float().mean()),
                       identity_false_negatives=int(false_negative.sum()))
    if margin is not None:
        positive, negative, weight = margin
        same, different, within = pair_cosines(keys, textures, target, false_negative)
        negatives = torch.cat((different, within))
        # A legal batch can hold only equal textures: no negatives, a zero (not NaN) term.
        hinge = (F.relu(positive - same).square().mean()
                 + F.relu(negatives - negative).square().sum() / max(len(negatives), 1))
        loss = loss + weight * hinge
        metrics.update(identity_margin=float(hinge.detach()),
                       margin_positive_violation=float((same < positive).float().mean()),
                       margin_negative_violation=float((negatives > negative).float().sum() / max(len(negatives), 1)),
                       margin_negative_pairs=len(negatives))
    return loss, metrics


def pair_cosines(keys, textures, target, false_negative):
    """Normalized machine keys [2n,D] (view A then B, flat scene*2+side) -> cosines of
    same-texture pairs, different-texture cross-view pairs and same-frame pairs.
    Equal textures are never negatives (cross-view mask; within-frame texture test)."""
    n = len(keys) // 2
    cosine = keys[:n] @ keys[n:].T
    rows = torch.arange(n, device=keys.device)
    same = cosine[rows, target]
    negative = torch.ones_like(cosine, dtype=torch.bool)
    negative[rows, target] = False
    different = cosine[negative & ~false_negative]
    frames = keys.reshape(-1, 2, keys.shape[-1])
    colors = textures.colors.reshape(-1, 2, 2, 3)
    distinct = ~((colors[:, 0] == colors[:, 1]).flatten(1).all(-1)
                 & (textures.pattern.reshape(-1, 2)[:, 0] == textures.pattern.reshape(-1, 2)[:, 1])
                 & (textures.period.reshape(-1, 2)[:, 0] == textures.period.reshape(-1, 2)[:, 1]))
    within = (frames[:, 0] * frames[:, 1]).sum(-1)[distinct.to(keys.device)]
    return same, different, within


def augmentation_stream(seed, step):
    """Deterministic per-update texture stream: exact resume, no base-sampler draws."""
    return torch.Generator().manual_seed(seed * 1_000_003 + 7_919 + step)


def warm_start(module, path):
    """Load perception weights from a parent run (new optimizer, new run; not a resume)."""
    path = Path(path)
    checkpoint = path / "last.pt" if path.is_dir() else path
    load_component(module, checkpoint, "perception")
    return dict(init_perception=str(checkpoint), init_perception_file_sha256=file_hash(checkpoint),
                init_perception_state_sha256=state_hash(module))


@torch.no_grad()
def key_monitor(model, device, randomize, margin=None):
    """Fixed TRAIN-kind paired-view cosines of the key on frozen perception.

    Two populations from one fixed stream (seed MONITOR_SEED, MONITOR_PAIRS scene pairs,
    rw.paired_view with swaps/inverted lamps): procedural bodies at the recorded
    randomization, and kind-table bodies. Never validation kinds; never used to select.
    With `margin=(positive, negative, ...)` the configured hinge violation rates are added.
    """
    out = {}
    for name, rate in (("procedural", randomize), ("kind_table", 0.0)):
        scenes, _, rgb, _, _, (textures, source) = paired_perception_batch(
            torch.Generator().manual_seed(MONITOR_SEED), rw.KIND_SPLIT["train"], MONITOR_PAIRS, device,
            randomize=rate, augmentation=torch.Generator().manual_seed(MONITOR_SEED + 1),
            pairing=torch.Generator().manual_seed(MONITOR_SEED + 2))
        percept = model["perception"](rgb)
        rows = torch.arange(len(scenes), device=device)
        slots = torch.stack([percept.slots[rows, pointer(percept.alpha, scenes.machine_xy[:, m].to(device))]
                             for m in (0, 1)], 1).flatten(0, 1)
        keys = F.normalize(model["key"](slots), dim=-1)
        n = len(keys) // 2
        target = torch.argsort(source.flatten()).to(device)
        false_negative = texture_matches(textures, n).to(device)
        false_negative[torch.arange(n, device=device), target] = False
        same, different, within = (v.cpu() for v in pair_cosines(keys, textures, target, false_negative))
        q = lambda v, p: float(torch.quantile(v, p))
        out[name] = dict(positive_min=float(same.min()), positive_q01=q(same, .01),
                         negative_q99=q(different, .99), negative_max=float(different.max()),
                         within_q99=q(within, .99), within_max=float(within.max()),
                         positive_count=len(same), negative_count=len(different), within_count=len(within))
        if margin is not None:
            rate = lambda hit: float(hit.float().sum() / max(len(hit), 1))
            out[name].update(positive_violation=rate(same < margin[0]), negative_violation=rate(different > margin[1]),
                             within_violation=rate(within > margin[1]))
    return out


def monitor_row(monitor):
    return {f"{name}_{k}": float(v) for name, values in monitor.items() for k, v in values.items()}


def key_screen(monitor, perception_unchanged):
    passed = perception_unchanged and all(
        m["positive_q01"] >= KEY_SCREEN["positive_q01"] and m["negative_q99"] <= KEY_SCREEN["negative_q99"]
        and m["within_q99"] <= KEY_SCREEN["within_q99"] for m in monitor.values())
    return dict(passed=bool(passed), thresholds=KEY_SCREEN, perception_unchanged=bool(perception_unchanged))


def parent_texture_randomization(path):
    """The warm-start parent's recorded S1 texture randomization (read, never assumed)."""
    path = Path(path)
    run = path if path.is_dir() else path.parent
    settings = json.loads((run / "run.json").read_text())["identity"]["settings"]
    return float(settings.get("texture_randomization", 0.0))


# Decoder-reconstruction mode: retrain only the actual slot decoder of a frozen perception
# run, either slot-only (as trained) or with its optional pyramid connections enabled.
DECODER_EVAL_SEEDS = dict(train=3602, validation=3603)  # fixed kind-table evaluation scenes
DECODER_EVAL_CHUNK = 64  # generation/forward chunk; the last chunk holds any remainder
DECODER_MASK_WEIGHT = 0.5  # the existing perception_loss weight of the matched mask CE
FROZEN_PARTS = ("encoder", "slot_attention", "heads")


def frozen_part_hashes(perception):
    return {name: state_hash(getattr(perception, name)) for name in FROZEN_PARTS}


def batch_hash(*tensors):
    """sha256 over the exact bytes (dtype and shape included) of the given tensors."""
    h = hashlib.sha256()
    for t in tensors:
        t = t.detach().cpu().contiguous()
        h.update(f"{t.dtype}{tuple(t.shape)}".encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def finite_tree(value, where="result"):
    """Raise unless every number in a nested result is finite (None marks an undefined value)."""
    if isinstance(value, dict):
        for k, v in value.items():
            finite_tree(v, f"{where}.{k}")
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            finite_tree(v, f"{where}[{i}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise FloatingPointError(f"Non-finite value at {where}")


def decoder_objective(percept, rgb, entity, mask_weight=DECODER_MASK_WEIGHT):
    """The decoder's trained objective: RGB MSE + 0.5 * matched mask CE (existing match_slots/cross_entropy)."""
    assign = match_slots(percept.alpha, entity)
    target = torch.argsort(assign, 1).gather(1, entity.flatten(1)).reshape(entity.shape)
    reconstruction = F.mse_loss(percept.recon, rgb)
    mask = slot_cross_entropy(percept.alpha, target)
    loss = reconstruction + mask_weight * mask
    if not all(torch.isfinite(v) for v in (loss, reconstruction, mask)):
        raise FloatingPointError("Non-finite decoder objective; refusing to optimize")
    return loss, dict(loss=float(loss), reconstruction=float(reconstruction), mask=float(mask))


def decoder_code_loss(perception, reconstruction, target_fine):
    """Raw fine-code MSE; frozen encoder differentiates pixels, target never learns."""
    predicted = perception.pyramid(reconstruction).scales[0].values
    if predicted.shape != target_fine.shape:
        raise ValueError("Reconstruction and target fine-code shape must match")
    loss = F.mse_loss(predicted, target_fine.detach())
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite decoder code feedback")
    return loss


def replaced_values(pyramid, values):
    """The same pyramid metadata with replaced feature values (diagnostic interventions only)."""
    return replace(pyramid, scales=tuple(replace(sc, values=v) for sc, v in zip(pyramid.scales, values)))


@torch.no_grad()
def decoder_eval(perception, device, scenes, interventions=()):
    """Fixed evaluation scenes per population (kind-table textures): per-image records and means.

    Per image: MSE; PSNR = 10 log10(1/MSE); machine-body MSE over pixels of entities 1-2
    (None if the image has none); gradient error = 0.5 * (mean squared difference of the
    horizontal adjacent-pixel forward differences, recon vs target, + the same vertically).
    Scenes are drawn in chunks of DECODER_EVAL_CHUNK from Generator(seed); a final smaller
    chunk holds any remainder, and every chunk is hashed. Slots always come from the true
    pyramid. Interventions (diagnostic, not operational inference) replace only the
    decoder's pyramid values: `zero` = zeros; `shuffled` = image i reads image i-1's pyramid
    (roll by one over the whole population order).
    """
    summary, records, images = {}, {}, None
    for population, seed in DECODER_EVAL_SEEDS.items():
        g, chunks, remaining = torch.Generator().manual_seed(seed), [], scenes
        while remaining > 0:
            count = min(DECODER_EVAL_CHUNK, remaining)
            remaining -= count
            _, _, rgb, entity, _ = perception_batch(g, rw.KIND_SPLIT[population], count, device)
            chunks.append((rgb, entity, perception.pyramid(rgb), batch_hash(rgb, entity)))
        variants = dict(true=None)
        for kind in interventions:
            per_scale = list(zip(*(tuple(sc.values for sc in c[2].scales) for c in chunks)))  # scale -> chunks
            if kind == "zero":
                new = [[torch.zeros_like(v) for v in values] for values in per_scale]
            elif kind == "shuffled":
                sizes = [len(c[0]) for c in chunks]
                new = [list(torch.roll(torch.cat(values), 1, 0).split(sizes)) for values in per_scale]
            else:
                raise ValueError(kind)
            variants[kind] = [replaced_values(c[2], [new[s][i] for s in range(len(new))]) for i, c in enumerate(chunks)]
        for variant, pyramids in variants.items():
            per = dict(mse=[], psnr=[], machine_body_mse=[], gradient_error=[])
            for i, (rgb, entity, pyramid, _) in enumerate(chunks):
                if pyramids is None:
                    recon = perception.from_pyramid(pyramid).recon
                else:
                    tokens = pyramid.as_tokens()
                    slots = perception.slot_attention(tokens.values, tokens.valid)
                    colors, alpha = perception.decoder(slots, pyramids[i])
                    recon = (alpha.softmax(1)[:, :, None] * colors).sum(1)
                error = (recon - rgb).square()
                mse = error.mean((1, 2, 3))
                machine = ((entity == 1) | (entity == 2))[:, None].to(error.dtype)
                pixels = machine.sum((1, 2, 3))
                body = (error * machine).sum((1, 2, 3)) / (3 * pixels).clamp_min(1)
                dx = lambda x: x[..., :, 1:] - x[..., :, :-1]
                dy = lambda x: x[..., 1:, :] - x[..., :-1, :]
                gradient = 0.5 * ((dx(recon) - dx(rgb)).square().mean((1, 2, 3))
                                  + (dy(recon) - dy(rgb)).square().mean((1, 2, 3)))
                per["mse"] += mse.tolist()
                per["psnr"] += (10 * torch.log10(1 / mse.clamp_min(1e-12))).tolist()
                per["machine_body_mse"] += [float(v) if n > 0 else None for v, n in zip(body, pixels)]
                per["gradient_error"] += gradient.tolist()
                if variant == "true" and population == "validation" and images is None:
                    images = (rgb[:8].cpu(), recon[:8].cpu())
            present = [v for v in per["machine_body_mse"] if v is not None]
            means = dict(mse=sum(per["mse"]) / scenes, psnr=sum(per["psnr"]) / scenes,
                         machine_body_mse=sum(present) / len(present) if present else None,
                         machine_body_images=len(present), gradient_error=sum(per["gradient_error"]) / scenes,
                         images=len(per["mse"]))
            finite_tree(per, f"{variant}.{population}")
            finite_tree(means, f"{variant}.{population}")
            summary.setdefault(variant, {})[population] = means
            records.setdefault(variant, {})[population] = per
        summary["inputs"] = summary.get("inputs", {})
        summary["inputs"][population] = dict(seed=seed, images=scenes, chunk_sizes=[len(c[0]) for c in chunks],
                                             chunk_sha256=[c[3] for c in chunks],
                                             sha256=hashlib.sha256("".join(c[3] for c in chunks).encode()).hexdigest())
    return summary, records, images



@torch.no_grad()
def decoder_mask_eval(perception, reference, device, scenes):
    """Independent fixed-population mask metrics; raw slots compared at machine centers.

    Matched pixel accuracy and CE use match_slots per image. Pointer agreement is
    against the frozen parent's raw slot ID at each of the two machine centers.
    Generator(seed) is local; evaluation does not consume the training sampler.
    """
    result = {}
    for population, seed in DECODER_EVAL_SEEDS.items():
        g = torch.Generator().manual_seed(seed)
        records = dict(pixel_accuracy=[], mask_ce=[], pointer_agreement=[])
        hashes = []
        remaining = scenes
        while remaining:
            count = min(DECODER_EVAL_CHUNK, remaining)
            remaining -= count
            scene, _, rgb, entity, _ = perception_batch(g, rw.KIND_SPLIT[population], count, device)
            pyramid = perception.pyramid(rgb)
            predicted = perception.from_pyramid(pyramid)
            baseline = reference.from_pyramid(pyramid)
            assignment = match_slots(predicted.alpha, entity)
            target = torch.argsort(assignment, 1).gather(1, entity.flatten(1)).reshape(entity.shape)
            records["pixel_accuracy"] += (predicted.alpha.argmax(1) == target).float().mean((1,2)).tolist()
            records["mask_ce"] += (-predicted.alpha.log_softmax(1).gather(1, target[:,None])[:,0].mean((1,2))).tolist()
            agreement = torch.stack([pointer(predicted.alpha, scene.machine_xy[:,m]) ==
                                     pointer(baseline.alpha, scene.machine_xy[:,m]) for m in range(2)],1)
            records["pointer_agreement"] += agreement.float().mean(1).tolist()
            hashes.append(batch_hash(rgb, entity))
        result[population] = dict(per_image=records, mean={k:sum(v)/len(v) for k,v in records.items()}, chunk_sha256=hashes)
    finite_tree(result)
    return result


def eval_row(split, step, means):
    return dict(step=step, split=split, **{f"{p}_{k}": v for p, m in means.items() for k, v in m.items()})


def write_evaluation(runner, name, step, summary, records):
    path = runner.path / name
    atomic_json(path, dict(step=step, definitions=decoder_eval.__doc__, summary=summary, per_image=records))
    return dict(path=name, sha256=file_hash(path), step=step)


def train_decoder_reconstruction(args, s):
    """Retrain only the actual BroadcastDecoder of a frozen perception checkpoint.

    Arm `slots`: the decoder as trained (slot-only). Arm `pyramid`: the same decoder with
    its zero-initialized pyramid connections enabled. Both start from the same parent
    decoder weights and see identical training batches (hashed per row); encoder, slot
    attention and heads are frozen and hash-guarded. Objective: RGB MSE + 0.5 * matched
    mask CE. The frozen heads' kind/attribute/lamp terms are logged as `semantic_*`
    diagnostics only: they are not optimized, but they read the matching, which depends on
    the decoder's alpha, so they may differ between arms and over training.
    """
    seed_everything(args.seed)
    model = nn.ModuleDict(dict(perception=SlotPerception(s["width"], s["slots"], s["iterations"],
                                                          decoder_width=s["decoder_width"])))
    perception = model["perception"]
    parent = warm_start(perception, args.init_perception)
    for name in FROZEN_PARTS:
        getattr(perception, name).requires_grad_(False)
    from copy import deepcopy
    mask_weight = getattr(args, "decoder_mask_weight", DECODER_MASK_WEIGHT)
    code_weight = getattr(args, "decoder_code_weight", 0.0)
    eval_every = getattr(args, "decoder_eval_every", 0)
    reference = deepcopy(perception).to(args.device).eval().requires_grad_(False) if eval_every else None
    parent_decoder = state_hash(perception.decoder)
    if args.decoder_reconstruction == "pyramid":
        perception.decoder.enable_pyramid_connections()
    if getattr(args, "decoder_subpixel", False):
        perception.decoder.enable_fine_subpixels()
    frozen = frozen_part_hashes(perception)
    model = model.to(args.device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    added = sum(p.numel() for n, p in perception.decoder.named_parameters() if n.startswith(("coarse.", "fine.", "fine_subpixel.")))
    interventions = ("zero", "shuffled") if args.decoder_reconstruction == "pyramid" else ()
    sampler = rw.TextureSampler()
    settings = dict(stage="perception", seed=args.seed, updates=args.updates, lr=args.lr, device=args.device,
                    size=args.size, sizes=s, purpose="development", precision="fp32",
                    max_reserved_gib=args.max_reserved_gib, decoder_reconstruction=args.decoder_reconstruction,
                    decoder_eval_scenes=args.decoder_eval_scenes,
                    decoder_mask_weight=mask_weight, decoder_eval_every=eval_every,
                    decoder_code_weight=code_weight,
                    decoder_subpixel=getattr(args, "decoder_subpixel", False),
                    objective=f"RGB MSE + {mask_weight} * matched mask CE + {code_weight} * raw fine-code MSE on "
                              "the decoder only; frozen kind/attribute/lamp terms logged as semantic_* diagnostics, "
                              "not optimized, assignment-dependent (may differ between arms)",
                    optimizer=dict(name="AdamW", lr=args.lr, parameters="decoder only", gradient_clip_norm=1.0),
                    batch=dict(scenes=s["perception_batch"], kinds="train", sampler="Run.sampler",
                               hash="batch_sha256 = sha256(rgb, entity, attrs, lamps) per training row"),
                    init_perception=str(args.init_perception),
                    warm_start=dict(parent, optimizer="fresh AdamW over decoder parameters only"),
                    parent_decoder_state_sha256=parent_decoder, frozen_part_state_sha256=frozen,
                    frozen_mode="requires_grad False; eval mode via training_mode; hash-checked before saves",
                    decoder_parameters=dict(total=sum(p.numel() for p in perception.decoder.parameters()),
                                            added_pyramid_connections=added, trainable=sum(p.numel() for p in trainable)),
                    texture_randomization=args.texture_randomization,
                    texture_sampler=dict(near_lamp=sampler.near_lamp, radius=sampler.radius, exclusion=sampler.exclusion,
                                         cap=sampler.cap, heldout_kinds=list(sampler.heldout),
                                         stream="Generator(seed*1000003 + 7919 + step); training scenes only"),
                    decoder_evaluation=dict(
                        seeds=DECODER_EVAL_SEEDS, scenes=args.decoder_eval_scenes, chunk=DECODER_EVAL_CHUNK,
                        textures="kind table (no randomization)", when="step 0 (parent decoder) and final",
                        definitions=decoder_eval.__doc__, interventions=list(interventions),
                        records="evaluation_initial.json / evaluation_final.json (per-image values, chunk hashes)"))
    runner = Run(args.resume or args.output, settings=settings, data=rw.manifest(), recipe=__file__, model=model,
                 optimizer=optimizer, device=args.device, resume=args.resume is not None)

    def guard():
        if frozen_part_hashes(perception) != frozen:
            raise RuntimeError("Frozen perception parts changed; refusing to save or report this run")

    started = time.perf_counter()
    try:
        guard()
        if runner.step == 0 and not any(r.get("split") == "decoder_eval_initial" for r in runner.rows):
            with evaluation_mode(model):
                summary, records, _ = decoder_eval(perception, args.device, args.decoder_eval_scenes)
            write_evaluation(runner, "evaluation_initial.json", 0, summary, records)
            runner.log(eval_row("decoder_eval_initial", 0, summary["true"]))
            if reference is not None:
                with evaluation_mode(model):
                    atomic_json(runner.path / "masks_initial.json", decoder_mask_eval(perception, reference, args.device, args.decoder_eval_scenes))
            enforce_ceiling(args, runner, guard=guard)
        while runner.step < args.updates and not stop(runner, args, started):
            training_mode(model)
            augmentation = augmentation_stream(args.seed, runner.step)
            scenes, lamps, rgb, entity, stats = perception_batch(
                runner.sampler, rw.KIND_SPLIT["train"], s["perception_batch"], args.device,
                randomize=args.texture_randomization, augmentation=augmentation)
            attrs, lamps = scenes.attrs.to(args.device), lamps.to(args.device)
            with torch.no_grad():
                pyramid = perception.pyramid(rgb)
            percept = perception.from_pyramid(pyramid)
            loss, metrics = decoder_objective(percept, rgb, entity, mask_weight)
            # Do not run the extra encoder for the zero-weight control. Preserve
            # gradients through its input, while its parameters/buffers stay frozen.
            if code_weight:
                code_loss = decoder_code_loss(perception, percept.recon, pyramid.scales[0].values)
                loss = loss + code_weight * code_loss
                metrics.update(loss=float(loss), code_mse=float(code_loss),
                               weighted_code_loss=float(code_weight * code_loss))
            with torch.no_grad():
                _, semantic = perception_loss(percept, rgb, entity, attrs, lamps)
            optimizer.zero_grad()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            if not torch.isfinite(norm):
                raise FloatingPointError("Non-finite decoder gradient; refusing to step")
            optimizer.step()
            runner.step += 1
            runner.log(dict(step=runner.step, split="train", **metrics, gradient_norm=float(norm),
                            batch_sha256=batch_hash(rgb, entity, attrs, lamps),
                            **{f"semantic_{k}": v for k, v in semantic.items()
                               if k not in ("loss", "reconstruction", "mask")}))
            enforce_ceiling(args, runner, guard=guard)
            if runner.step % s["validate_every"] == 0 or runner.step == args.updates:
                guard()
                runner.save()
            if eval_every and runner.step % eval_every == 0 and runner.step < args.updates:
                with evaluation_mode(model):
                    summary, records, _ = decoder_eval(perception, args.device, args.decoder_eval_scenes)
                    masks = decoder_mask_eval(perception, reference, args.device, args.decoder_eval_scenes)
                write_evaluation(runner, f"evaluation_step_{runner.step:06d}.json", runner.step, summary, records)
                atomic_json(runner.path / f"masks_step_{runner.step:06d}.json", masks)
                runner.log(eval_row("decoder_eval_periodic", runner.step, summary["true"]))
                enforce_ceiling(args, runner, guard=guard)
        guard()
        runner.save()
        with evaluation_mode(model):
            summary, records, images = decoder_eval(perception, args.device, args.decoder_eval_scenes, interventions)
        enforce_ceiling(args, runner, guard=guard)
        if reference is not None:
            with evaluation_mode(model):
                atomic_json(runner.path / "masks_final.json", decoder_mask_eval(perception, reference, args.device, args.decoder_eval_scenes))
        # A paused run keeps this evaluation in its files only, so an exact resume reproduces the rows.
        final_file = write_evaluation(runner, "evaluation_final.json", runner.step, summary, records)
        if runner.step >= args.updates:
            runner.log(eval_row("decoder_eval_final", runner.step, summary["true"]))
            runner.save()
        initial_path = runner.path / "evaluation_initial.json"
        initial = json.loads(initial_path.read_text())
        elapsed = time.perf_counter() - started
        result = dict(
            evaluation_scope="Decoder-only reconstruction retraining on a frozen perception checkpoint; fixed kind-table "
                             "scenes (train kinds seed 3602, validation kinds seed 3603). Arm-level metrics only; the "
                             "predeclared pair gate is evaluated in the aggregate.",
            gate=None, arm=args.decoder_reconstruction, evaluated_step=runner.step,
            metrics=dict(initial=initial["summary"]["true"], final=summary["true"],
                         interventions={k: summary[k] for k in interventions} or None,
                         inputs=summary["inputs"],
                         parameters=dict(model=parameters(model), decoder=settings["decoder_parameters"]),
                         resources=dict(**resources(args.device), updates=runner.step, seconds=elapsed)),
            evaluation_files=dict(initial=dict(path=initial_path.name, sha256=file_hash(initial_path), step=0),
                                  final=final_file),
            frozen_parts_unchanged=frozen_part_hashes(perception) == frozen,
            parent_checkpoint=parent, stop_reason=stop_reason(runner, args, started),
            limitations=["Reconstruction diagnostic only: no identity, persistent-memory or runtime claim.",
                         "A pyramid-connected decoder needs the frame's pyramid (retained frame, recomputed); "
                         "persistent memory still stores the 64-value slot.",
                         "Interventions replace decoder feature values only and are not operational inference.",
                         "semantic_* training terms are frozen-head diagnostics that depend on the alpha-derived "
                         "matching; they are not optimized and are not claimed identical across arms.",
                         "The encoder is the existing FeatureHierarchy (lossy merges), not the full invertible "
                         "filter-bank design."])
        finite_tree(result)
        finish(runner, result, complete=runner.step >= args.updates, images=images)
    except Exception as error:
        runner.status("failed", "incomplete", str(error))
        raise
    return runner.path


def train_perception(args, s):
    seed_everything(args.seed)
    model = nn.ModuleDict(dict(perception=SlotPerception(s["width"], s["slots"], s["iterations"], decoder_width=s["decoder_width"])))
    if args.identity:
        model["key"] = key_head(s["width"], s["key_width"])  # exported for S2 (`--init-key`)
    parent = warm_start(model["perception"], args.init_perception) if args.init_perception else {}
    key_source = None
    if args.init_key:  # explicit: the key exported by the same --init-perception S1 run
        checkpoint = Path(args.init_perception)
        checkpoint = checkpoint / "last.pt" if checkpoint.is_dir() else checkpoint
        load_component(model["key"], checkpoint, "key")
        key_source = dict(checkpoint=str(checkpoint), file_sha256=file_hash(checkpoint),
                          state_sha256=state_hash(model["key"]))
    frozen = None
    if args.freeze_perception:
        model["perception"].requires_grad_(False)
        frozen = state_hash(model["perception"])
    model = model.to(args.device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    margin = None
    if args.identity_margin_weight is not None:
        margin = (args.identity_margin_positive, args.identity_margin_negative, args.identity_margin_weight)
    settings = dict(stage="perception", seed=args.seed, updates=args.updates, lr=args.lr, device=args.device,
                    size=args.size, sizes=s, purpose="development", precision="fp32",
                    max_reserved_gib=args.max_reserved_gib,
                    objective="RGB MSE + 0.5*(matched mask CE + kind CE + attribute CE + lamp BCE); labels training-only")
    # Options appear in settings only when used, so default runs keep their identity.
    # Keys named like CLI flags are restored by resume_arguments.
    if args.texture_randomization:
        sampler = rw.TextureSampler()
        settings["texture_randomization"] = args.texture_randomization
        settings["texture_sampler"] = dict(
            near_lamp=sampler.near_lamp, radius=sampler.radius, exclusion=sampler.exclusion,
            cap=sampler.cap, heldout_kinds=list(sampler.heldout),
            stream="Generator(seed*1000003 + 7919 + step); training scenes only; validation unchanged",
        )
    if args.identity:
        settings["identity"] = args.identity
        settings["identity_weight"] = args.identity_weight
        settings["identity_objective"] = dict(
            temperature=IDENTITY_TEMPERATURE, key="pathwm.models.latent_core.key_head(width, key_width)",
            views="A = default batch; B = rw.paired_view: new layouts/objects, exact permuted body textures, inverted lamps",
            pairing_stream="Generator(seed*1000003 + 104729 + step)",
            loss="perception_loss(A+B) + identity_weight * InfoNCE(key(A machines) -> key(B machines))",
            gradient="perception and key" if args.identity == "joint" else "key only (reads sg(slot))",
            labels="pairing indices/textures are loss-only; the forward pass sees pixels only",
        )
    if parent:
        settings["init_perception"] = str(args.init_perception)
        settings["warm_start"] = dict(parent, optimizer="fresh AdamW; not a resume of the parent run")
    if key_source:
        settings["init_key"] = True
        settings["init_key_source"] = key_source
    if margin:
        settings["identity_margin_positive"], settings["identity_margin_negative"] = margin[:2]
        settings["identity_margin_weight"] = margin[2]
        settings["identity_objective"]["margin"] = (
            "+ weight * [mean relu(positive - cos(same))^2 + mean relu(cos(different) - negative)^2]; "
            "different = cross-view non-matching pairs + both machines of each frame; equal textures masked")
    if frozen:
        settings["freeze_perception"] = True
        settings["frozen_perception_state_sha256"] = frozen
        settings["objective"] = ("identity_weight * identity loss (InfoNCE" + (" + absolute margins" if margin else "")
                                 + ") on the key only; perception frozen, no perception loss")
        settings["identity_objective"]["loss"] = "identity_weight * identity loss only (perception frozen; no perception loss)"
        settings["identity_objective"]["gradient"] = "key only; perception frozen (no gradient, eval mode, hash-guarded)"
        settings["key_monitor"] = dict(
            seed=MONITOR_SEED, pairs=MONITOR_PAIRS, kinds="train", populations=["procedural", "kind_table"],
            view="rw.paired_view (new layouts, permuted textures incl. side swaps, inverted lamps)",
            statistics="positive q01/min, different q99/max, within-frame q99/max, counts",
            use="monitor and predeclared screen only; no validation kinds; no checkpoint selection",
            screen=KEY_SCREEN,
            violation_margins=None if margin is None else dict(positive=margin[0], negative=margin[1]))
    runner = Run(args.resume or args.output, settings=settings, data=rw.manifest(), recipe=__file__, model=model,
                 optimizer=optimizer, device=args.device, resume=args.resume is not None)
    started = time.perf_counter()
    try:
        if frozen:
            guard_frozen(model, frozen)
            if runner.step == 0 and not any(r.get("split") == "monitor_initial" for r in runner.rows):
                with evaluation_mode(model):  # the parent key before any update
                    runner.log(dict(step=0, split="monitor_initial",
                                    **monitor_row(key_monitor(model, args.device, args.texture_randomization, margin))))
                enforce_ceiling(args, runner, guard=lambda: guard_frozen(model, frozen))
        while runner.step < args.updates and not stop(runner, args, started):
            training_mode(model)
            augmentation = augmentation_stream(args.seed, runner.step)
            if args.identity:
                scenes, lamps, rgb, entity, stats, (textures, source) = paired_perception_batch(
                    runner.sampler, rw.KIND_SPLIT["train"], s["perception_batch"], args.device,
                    randomize=args.texture_randomization, augmentation=augmentation,
                    pairing=pairing_stream(args.seed, runner.step),
                )
            else:
                scenes, lamps, rgb, entity, stats = perception_batch(
                    runner.sampler, rw.KIND_SPLIT["train"], s["perception_batch"], args.device,
                    randomize=args.texture_randomization, augmentation=augmentation,
                )
            if frozen:
                with torch.no_grad():
                    percept = model["perception"](rgb)
                identity, metrics = identity_loss(model["key"], percept, scenes, textures, source,
                                                  detached=True, margin=margin)
                loss = args.identity_weight * identity
                metrics["loss"] = float(loss.detach())
            else:
                percept = model["perception"](rgb)
                loss, metrics = perception_loss(percept, rgb, entity, scenes.attrs.to(args.device), lamps.to(args.device))
            if args.identity and not frozen:
                identity, identity_metrics = identity_loss(model["key"], percept, scenes, textures, source,
                                                           detached=args.identity == "detached", margin=margin)
                metrics.update(identity_metrics, perception_loss=metrics["loss"])
                loss = loss + args.identity_weight * identity
                metrics["loss"] = float(loss.detach())
            if stats:
                metrics.update(texture_replaced=stats["replaced"], texture_near_lamp=stats["near_lamp"],
                               texture_heldout_rejections=stats["heldout_rejections"], texture_draws=stats["draws"])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            runner.step += 1
            runner.log(dict(step=runner.step, split="train", **metrics))
            enforce_ceiling(args, runner, guard=(lambda: guard_frozen(model, frozen)) if frozen else None)
            if runner.step % s["validate_every"] == 0 or runner.step == args.updates:
                if frozen:
                    guard_frozen(model, frozen)
                    with evaluation_mode(model):
                        runner.log(dict(step=runner.step, split="monitor",
                                        **monitor_row(key_monitor(model, args.device, args.texture_randomization, margin))))
                    enforce_ceiling(args, runner, guard=lambda: guard_frozen(model, frozen))
                else:
                    runner.log(validate_perception(model["perception"], runner.step, args, s))
                runner.save()
                print(f"perception: saved step {runner.step}", flush=True)
        if frozen:
            guard_frozen(model, frozen)
        runner.save()
        if frozen:
            return finish_frozen_key(runner, model, args, s, settings, started, key_source, margin, frozen)
        with evaluation_mode(model):
            metrics = ev.perception_metrics(model["perception"], torch.Generator().manual_seed(args.seed + 17), "validation", s["validation_scenes"], args.device)
            _, _, rgb, _, _ = perception_batch(torch.Generator().manual_seed(args.seed + 18), rw.KIND_SPLIT["validation"], 8, args.device)
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
        if args.identity:
            result["key_export"] = dict(component="key", architecture=settings["identity_objective"]["key"],
                                        width=s["width"], key_width=s["key_width"], mode=args.identity,
                                        state_sha256=state_hash(model["key"]))
            result["limitations"].append("Identity pairs are disclosed generator knowledge (same sampled texture); "
                                         "no autonomous discovery, C3 or natural-image claim.")
        finish(runner, result, complete=runner.step >= args.updates, images=(rgb.cpu(), recon.cpu()))
    except Exception as error:
        runner.status("failed", "incomplete", str(error))
        raise
    return runner.path


def guard_frozen(model, expected):
    if state_hash(model["perception"]) != expected:
        raise RuntimeError("Frozen perception changed; refusing to save or report this run")


def finish_frozen_key(runner, model, args, s, settings, started, key_source, margin, frozen):
    """Frozen-key result: train-kind monitor, predeclared screen, report; no validation kinds."""
    with evaluation_mode(model):
        monitor = key_monitor(model, args.device, args.texture_randomization, margin)
        _, _, rgb, _, _ = perception_batch(torch.Generator().manual_seed(MONITOR_SEED), rw.KIND_SPLIT["train"], 8,
                                           args.device, randomize=args.texture_randomization,
                                           augmentation=torch.Generator().manual_seed(MONITOR_SEED + 1))
        recon = model["perception"](rgb).recon
    enforce_ceiling(args, runner, guard=lambda: guard_frozen(model, frozen))  # monitors can be the peak
    unchanged = state_hash(model["perception"]) == frozen
    initial = next((r for r in runner.rows if r.get("split") == "monitor_initial"), None)
    elapsed = time.perf_counter() - started
    screen = key_screen(monitor, unchanged)
    result = dict(
        evaluation_scope="Training-kind monitor only (seed %d, %d pairs); frozen J perception; no validation kinds"
                         % (MONITOR_SEED, MONITOR_PAIRS),
        gate=screen["passed"], screen=screen,
        metrics=dict(monitor=monitor, initial_monitor=initial,
                     resources=dict(**resources(args.device), updates=runner.step, seconds=elapsed,
                                    updates_per_second=runner.step / max(elapsed, 1e-9), **parameters(model))),
        perception_unchanged=unchanged, init_key=key_source, stop_reason=stop_reason(runner, args, started),
        key_export=dict(component="key", architecture=settings["identity_objective"]["key"], width=s["width"],
                        key_width=s["key_width"], mode=args.identity, state_sha256=state_hash(model["key"])),
        limitations=["Screen uses training kinds only; held-out identity is decided by later calibration/evaluation.",
                     "Identity pairs are disclosed generator knowledge (same sampled texture); no natural-image claim.",
                     "Perception, decoder and heads are the frozen parent weights; only the key head was trained.",
                     "The absolute-margin term is a candidate repair supported by the diagnosis, not proven necessary."],
    )
    finish(runner, result, complete=runner.step >= args.updates, images=(rgb.cpu(), recon.cpu()))
    return runner.path


@torch.no_grad()
def validate_perception(perception, step, args, s):
    was = perception.training
    perception.eval()
    g = torch.Generator().manual_seed(args.seed + 5)
    scenes, lamps, rgb, entity, _ = perception_batch(g, rw.KIND_SPLIT["validation"], min(32, s["validation_scenes"]), args.device)
    loss, metrics = perception_loss(perception(rgb), rgb, entity, scenes.attrs.to(args.device), lamps.to(args.device))
    perception.train(was)
    return dict(step=step, split="validation", **metrics)


def stop(runner, args, started):
    if args.stop_after is not None and runner.step >= args.stop_after:
        return True
    return args.max_minutes is not None and time.perf_counter() - started >= 60 * args.max_minutes


# ---------------------------------------------------------------- S2 core / S2s symbolic

# §23 learnability ladder, fixed before any ladder run (relation ν on the pool population).
LADDER_SCREEN = dict(nu_full=0.8, over_empty=0.5, over_permuted=0.5)
# Control arms and secondary pools every 4th validation (1000 updates at full size) and at the end;
# the main pool's full arm at every validation. Evaluation never changes training (RNG/buffers kept).
CONTROL_EVERY_VALIDATIONS = 4
SMALL_POOL_SUPPORT = 8


def ladder_screen(ladder, nu, family="relation"):
    """L1 needs only ν_full; induction rungs also need support dependence over empty/permuted arms.

    ν is read for the trained family; relation runs keep their historical key `relation_nu`.
    """
    value = {arm: None if nu[arm] is None else nu[arm].get(family) for arm in nu}
    criteria = dict(nu_full=value["full"] is not None and value["full"] >= LADDER_SCREEN["nu_full"])
    if ladder > 1:
        for arm in ("empty", "permuted"):
            ok = None not in (value["full"], value[arm])
            criteria[f"over_{arm}"] = ok and value["full"] - value[arm] >= LADDER_SCREEN[f"over_{arm}"]
    key = "relation_nu" if family == "relation" else "family_nu"
    return dict(passed=all(criteria.values()), criteria=criteria, family=family, thresholds=LADDER_SCREEN, **{key: value})


def core_overrides(args):
    """Explicit E4 capacity overrides, recorded so that resume restores them (absent when unused)."""
    names = ("core_width", "core_blocks", "core_loops", "episodes")
    return {n: getattr(args, n) for n in names if getattr(args, n, None) is not None}


def train_core(args, s, *, symbolic=False):
    seed_everything(args.seed)
    model = RuleModel(s, symbolic=symbolic).to(args.device)
    # The symbolic diagnostic's machine token is lamp+role only (no appearance):
    # appearance-key InfoNCE is unidentifiable there, so it is disabled, and the
    # supplied-symbol embeddings stay fixed (encode_episodes runs under no_grad).
    key_weight = 0.0 if symbolic else 0.2
    auxiliary = getattr(args, "auxiliary_weight", 1.0)
    reader = getattr(args, "reader", "code")
    support_sizes = tuple(getattr(args, "support_sizes", None) or s["support"])
    if symbolic:
        model.perception.requires_grad_(False)
        model.core.key_head.requires_grad_(False)  # unused without appearance
        perceive = ev.symbolic_perceiver(model.perception)
    else:
        load_component(model.perception, Path(args.perception) / "last.pt", "perception")
        model.perception.requires_grad_(False)
        perceive = ev.pixel_perceiver(model.perception)
    key_source = None
    if getattr(args, "init_key", False):  # explicit only: never inferred from the perception run
        record = json.loads((Path(args.perception) / "run.json").read_text())["identity"]["settings"]
        if not record.get("identity"):
            raise ValueError("--init-key needs a perception run trained with --identity")
        load_component(model.core.key_head, Path(args.perception) / "last.pt", "key")
        key_source = dict(run=str(args.perception), mode=record["identity"], state_sha256=state_hash(model.core.key_head))
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    floors = rw.floors()
    settings = dict(stage="symbolic" if symbolic else "core", seed=args.seed, updates=args.updates, lr=args.lr,
                    **core_overrides(args),
                    device=args.device, size=args.size, sizes=s, purpose="development", precision="fp32",
                    max_reserved_gib=args.max_reserved_gib,
                    perception=None if symbolic else str(args.perception),
                    perception_sha256=None if symbolic else state_hash(model.perception),
                    objective=("unweighted outcome BCE" + (" + next-token (k=1,2) + 0.5 rollout BCE" if auxiliary == 1.0
                               else f" + {auxiliary} * (next-token (k=1,2) + 0.5 rollout BCE)" if auxiliary else ""))
                    + ("" if symbolic else " + 0.2 key InfoNCE"),
                    key_weight=key_weight, auxiliary_weight=auxiliary, reader=reader,
                    support_sizes=getattr(args, "support_sizes", None), train_support=list(support_sizes),
                    diagnostic=("supplied render symbols and exact slot masks; fixed symbol embeddings; no appearance, "
                                "so no key objective and no retrieval claim; not an R1 pixel gate") if symbolic else None)
    if key_source:
        settings["init_key"] = True
        settings["init_key_source"] = key_source
    split = rw.split_rules()
    ladder = getattr(args, "train_rules", None)

    def encoded(batch):
        """Symbolic evaluation tokens without rendering (bit-identical, tested); pixel runs render."""
        return ev.symbolic_episode_tokens(model.perception, batch, args.device) if symbolic else None

    mixed = getattr(args, "train_families", None)  # M1: one core, several family ladders
    family = getattr(args, "train_family", None) or "relation"
    families = list(mixed) if mixed else [family]
    pool = tuple(r for f in families for r in rw.family_ladder(f, ladder)) if ladder else split["train"]
    pool_episodes = s["pool_episodes"] * len(families)  # 64 per family in the evaluation pools
    repeats = getattr(args, "rule_repeats", None)
    delta_weights = getattr(args, "delta_weights", None)
    uniform_after = getattr(args, "uniform_after", None)
    if delta_weights:  # C1-C3: geometric skew over δ, per relation rule
        repeats = [delta_weights[r.delta] for r in pool]
    # N5/C: skewed training draws by repeating pool rules; evaluation pools stay uniform.
    train_pool = tuple(r for r, k in zip(pool, repeats) for _ in range(k)) if repeats else pool
    holdout = getattr(args, "holdout_ladder_rules", None)
    held = tuple(pool[i] for i in holdout) if holdout else ()
    if held:  # transfer probe: these ladder rules are never trained and form the held-out pool
        train_pool = tuple(r for r in train_pool if r not in held)
        pool = tuple(r for r in pool if r not in held)
    if ladder:  # §23 learnability ladder: only the training-rule pool changes
        settings.update(train_rules=ladder, train_rule_keys=[r.key() for r in pool],
                        pool_evaluation=dict(seed=args.seed + 11, episodes=pool_episodes, support=max(s["support"]),
                                             arms=list(ev.CONTROL_ARMS)),
                        screen=LADDER_SCREEN, train_family=family, rule_repeats=getattr(args, "rule_repeats", None),
                        **(dict(holdout_ladder_rules=list(holdout), holdout_rule_keys=[r.key() for r in held])
                           if held else {}),
                        delta_weights=delta_weights, uniform_after=uniform_after,
                        heldout_evaluation=dict(seed=args.seed + 23, episodes=pool_episodes,
                                                support=max(s["support"]),
                                                rules=("held-out ladder rules" if held
                                                       else f"validation {'/'.join(families)} rules")))
        if mixed:
            settings.update(train_families=families)
    data = rw.manifest()
    runner = Run(args.resume or args.output, settings=settings, data=data, recipe=__file__, model=model,
                 optimizer=optimizer, device=args.device, resume=args.resume is not None)
    pool_population = rw.sample_episodes(torch.Generator().manual_seed(args.seed + 11), pool, rw.KIND_SPLIT["train"],
                                         episodes=pool_episodes, support=(max(s["support"]),),
                                         queries=s["queries"], p_empty=0.0) if ladder else None
    # Secondary small-support pool (overnight plan): a finding, never part of the screen.
    small_population = rw.sample_episodes(torch.Generator().manual_seed(args.seed + 17), pool, rw.KIND_SPLIT["train"],
                                          episodes=pool_episodes, support=(SMALL_POOL_SUPPORT,),
                                          queries=s["queries"], p_empty=0.0) if ladder else None
    heldout_rules = held or tuple(r for r in split["validation"] if r.family in families)
    heldout_population = rw.sample_episodes(torch.Generator().manual_seed(args.seed + 23), heldout_rules,
                                            rw.KIND_SPLIT["validation"], episodes=pool_episodes,
                                            support=(max(s["support"]),), queries=s["queries"],
                                            p_empty=0.0) if ladder else None
    validation = rw.sample_episodes(torch.Generator().manual_seed(args.seed + 7), split["validation"], rw.KIND_SPLIT["validation"],
                                    episodes=s["validation_episodes"], support=(max(s["support"]),), queries=s["queries"], p_empty=0.0)
    started = time.perf_counter()
    try:
        if runner.step == 0 and not args.resume:
            with torch.no_grad():  # fixed target scale from training frames (declared buffer)
                warm = rw.sample_episodes(torch.Generator().manual_seed(args.seed + 3), train_pool, rw.KIND_SPLIT["train"],
                                          episodes=max(2, s["episodes"]), support=s["support"], queries=s["queries"], p_empty=0.0)
                tokens = encoded(warm) if symbolic else ev.encode_episodes(perceive, warm, args.device)
                model.variance.copy_(torch.cat((tokens.support.m_post, tokens.query.m_post)).var(0).clamp_min(1e-4))
        while runner.step < args.updates and not stop(runner, args, started):
            training_mode(model)
            skewed = uniform_after is None or runner.step < uniform_after
            batch = rw.sample_episodes(runner.sampler, train_pool if skewed else pool, rw.KIND_SPLIT["train"],
                                       episodes=s["episodes"],
                                       support=support_sizes, queries=s["queries"])
            tokens = encoded(batch) if symbolic else ev.encode_episodes(perceive, batch, args.device)
            loss, metrics = episode_loss(model.core, tokens, model.variance, key_weight=key_weight,
                                         auxiliary_weight=auxiliary, reader=reader)
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
                    tokens = encoded(validation) if symbolic else ev.encode_episodes(perceive, validation, args.device)
                    v_loss, v_metrics = episode_loss(model.core, tokens, model.variance, key_weight=key_weight,
                                                     auxiliary_weight=auxiliary, reader=reader)
                    m = ev.episode_metrics(model.core, perceive, validation, args.device, floors, reader=reader, tokens=encoded(validation))
                    every = CONTROL_EVERY_VALIDATIONS * s["validate_every"]
                    complete = runner.step % every == 0 or runner.step == args.updates
                    if ladder:
                        controls = ev.control_metrics(model.core, perceive, pool_population, args.device, floors, tokens=encoded(pool_population),
                                                      seed=args.seed + 13, reader=reader,
                                                      arms=ev.CONTROL_ARMS if complete else ("full",))
                        arms = {k: v for arm, nu in controls["nu"].items() if nu for k, v in flat(f"nu_{arm}_", nu).items()}
                        runner.log(dict(step=runner.step, split="pool", **arms))
                    if ladder and complete:
                        small = ev.control_metrics(model.core, perceive, small_population, args.device, floors, tokens=encoded(small_population),
                                                   seed=args.seed + 19, reader=reader)
                        arms = {k: v for arm, nu in small["nu"].items() if nu for k, v in flat(f"nu_{arm}_", nu).items()}
                        runner.log(dict(step=runner.step, split="pool_small", **arms))
                        held = ev.control_metrics(model.core, perceive, heldout_population, args.device, floors, tokens=encoded(heldout_population),
                                                  seed=args.seed + 29, reader=reader)
                        arms = {k: v for arm, nu in held["nu"].items() if nu for k, v in flat(f"nu_{arm}_", nu).items()}
                        runner.log(dict(step=runner.step, split="pool_heldout", **arms))
                runner.log(dict(step=runner.step, split="validation", **v_metrics,
                                **flat("nu_", m["nu"]), **flat("calibration_", m["calibration"])))
                runner.save()
                print(f"{settings['stage']}: saved step {runner.step}", flush=True)
        runner.save()
        with evaluation_mode(model):
            m = ev.episode_metrics(model.core, perceive, validation, args.device, floors, reader=reader, tokens=encoded(validation))
            controls = ev.control_metrics(model.core, perceive, pool_population, args.device, floors, tokens=encoded(pool_population),
                                          seed=args.seed + 13, reader=reader) if ladder else None
            small = ev.control_metrics(model.core, perceive, small_population, args.device, floors, tokens=encoded(small_population),
                                       seed=args.seed + 19, reader=reader) if ladder else None
            held = ev.control_metrics(model.core, perceive, heldout_population, args.device, floors, tokens=encoded(heldout_population),
                                      seed=args.seed + 29, reader=reader) if ladder else None
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
        if ladder:
            result["metrics"]["pool"] = controls
            result["metrics"]["pool_small"] = dict(small, support=SMALL_POOL_SUPPORT, role="secondary finding, not screened")
            result["metrics"]["pool_heldout"] = dict(held, role="held-out validation relations; transfer finding, not screened")
            if mixed:  # every family must pass on its own
                per_family = {f: ladder_screen(ladder, controls["nu"], f) for f in families}
                result["screen"] = dict(passed=all(v["passed"] for v in per_family.values()), per_family=per_family,
                                        families=families, thresholds=LADDER_SCREEN)
            else:
                result["screen"] = ladder_screen(ladder, controls["nu"], family)
            result["limitations"].append("§23 development screen: one seed, training-rule pool only; held-out "
                                         "validation is secondary and no transfer claim follows from the pool.")
        finish(runner, result, complete=runner.step >= args.updates)
    except Exception as error:
        runner.status("failed", "incomplete", str(error))
        raise
    return runner.path


# ---------------------------------------------------------------- S2s oracle application curriculum
#
# ORACLE DIAGNOSTIC (never runtime inference, induction, held-out, a capability or a
# gate): the training-rule ID selects a learned code for the unchanged shared core, on
# supplied symbols. Fixed two-stage schedule, promoted from the validated development
# runs R44 (relation rules only) -> C192 (all 192 training rules):
# runs/reviews/integrated_architecture_20260923/oracle-recipe-promotion-plan.md.

# Checkpoint layout (standalone diagnostic; M2 decision): the recipe's RuleModel state
# `perception.*`, `core.*`, `variance` (unused here), `oracle.codebook.weight`
# [192, code_tokens*width] in `rw.split_rules()["train"]` order, `oracle.relation_codebook.weight`
# [44, ...] and `oracle.stage`. It is NOT the historical `probe.build` layout: no historical
# C192/E1a/E1b/I1 consumer loads it, and `--stage evaluate` refuses symbolic runs. The core
# alone loads with the ordinary `load_component(core, run / "last.pt", "core")` (future use).

ORACLE_SCOPE = ("ORACLE DIAGNOSTIC: training-rule IDs select learned codes in training and evaluation; supplied "
                "symbols; training rules only; one seed. Not runtime inference, induction, a held-out result, a "
                "capability or a formal gate.")
ORACLE_P_EMPTY = 0.05  # support is unused by oracle codes; kept for the identical batch stream
ORACLE_POOLS = dict(mixed=9101, all44=9144)  # generator seeds of the historical evaluation pools
ORACLE_EVAL = dict(full=dict(episodes=8, support=128, queries=64),
                   check=dict(episodes=1, support=16, queries=8))
PAIRWISE = ("relation", "open", "close", "toggle")  # families whose event needs the match [d = 0]


def boundary_gate(size):
    """Full size: R44's precondition, the pre-transfer all44 screen must pass. Check size: exempt."""
    return "all44 screen must pass" if size == "full" else "exempt: check size, workflow only"


def retain_json(path, value):
    """Write once; a deterministic replay may only reproduce identical content."""
    value = json.loads(json.dumps(value, allow_nan=False))
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"{path.name} exists with different content; refusing to overwrite evidence")
        return
    atomic_json(path, value)


def retain_npz(path, arrays):
    if path.exists():
        with np.load(path) as old:
            if sorted(old.files) != sorted(arrays) or not all(np.array_equal(old[k], arrays[k]) for k in arrays):
                raise ValueError(f"{path.name} exists with different content; refusing to overwrite evidence")
        return
    temporary = path.with_name(path.name + ".partial.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def atomic_copy(source, target):
    temporary = target.with_name(target.name + ".partial")
    temporary.write_bytes(source.read_bytes())
    temporary.replace(target)


def same_state(a, b):
    """Exact equality of checkpoint structures (tensors bitwise)."""
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same_state(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(same_state(x, y) for x, y in zip(a, b))
    return a == b


def verified_snapshot(path, relation_updates, *, require_record=True):
    """This run's pre-transfer stage-0 snapshot and its gate record, or refuse.

    Both the checkpoint and the record must name THIS run (its run.json identity digest),
    the expected boundary step and stage 0, agree with each other and match the recorded
    hash. A valid pair from another run is not this run's evidence. Reads only.
    """
    snapshot, record = path / "relation_stage.pt", path / "relation_stage.json"
    if not snapshot.exists() or (require_record and not record.exists()):
        raise ValueError("relation_stage.pt/.json missing: the stage boundary cannot be verified; start a new run")
    saved = json.loads(record.read_text()) if record.exists() else None
    if saved is not None and file_hash(snapshot) != saved.get("sha256"):
        raise ValueError("relation_stage.pt does not match its recorded hash; refusing to continue")
    own = (digest(json.loads((path / "run.json").read_text())["identity"]), relation_updates, 0)
    kept = torch.load(snapshot, map_location="cpu", weights_only=True)
    bound = (kept.get("identity_sha256"), kept.get("step"), int(kept["model"]["oracle.stage"])) == own
    if saved is not None:
        bound = bound and (saved.get("identity_sha256"), saved.get("step"), saved.get("stage")) == own
    if not bound:
        raise ValueError("relation_stage.pt/.json belong to another run or boundary; refusing to continue")
    return saved


class OracleCodes(nn.Module):
    """Training-only codebook: one learned code per TRAINING rule, looked up by rule key.

    Stage 0 trains a separate 44-row relation table (as R44, so the other rows get no
    update or weight decay); `transfer()` writes it into the 192-row table (as C192).
    Validation/test rules have no code; unknown keys raise.
    """

    def __init__(self, s, train_rules):
        super().__init__()
        self.shape = (s["code_tokens"], s.get("core_width", s["width"]))
        self.index = {r.key(): i for i, r in enumerate(train_rules)}
        self.relation = [i for i, r in enumerate(train_rules) if r.family == "relation"]
        self.codebook = nn.Embedding(len(train_rules), s["code_tokens"] * s.get("core_width", s["width"]))
        nn.init.normal_(self.codebook.weight, std=0.02)  # after the core: the historical RNG order
        self.relation_codebook = nn.Embedding.from_pretrained(self.codebook.weight[self.relation].detach().clone(),
                                                              freeze=False)
        self.register_buffer("stage", torch.zeros((), dtype=torch.long))  # 0 relation, 1 mixed (checkpointed)

    def forward(self, rules):
        unknown = [r.key() for r in rules if r.key() not in self.index]
        if unknown:
            raise ValueError(f"oracle codes exist only for training rules: {unknown[:3]}")
        rows = [self.index[r.key()] for r in rules]
        device = self.codebook.weight.device
        if int(self.stage) == 0:
            where = {row: i for i, row in enumerate(self.relation)}
            if all(row in where for row in rows):  # the exact R44 lookup
                codes = self.relation_codebook(torch.tensor([where[row] for row in rows], device=device))
                return codes.reshape(len(rows), *self.shape)
            # Evaluation of other families before the boundary: the codebook the transfer would produce.
            table = self.codebook.weight.index_put((torch.tensor(self.relation, device=device),),
                                                   self.relation_codebook.weight)
            return table[torch.tensor(rows, device=device)].reshape(len(rows), *self.shape)
        return self.codebook(torch.tensor(rows, device=device)).reshape(len(rows), *self.shape)

    @torch.no_grad()
    def transfer(self):
        rows = torch.tensor(self.relation, device=self.codebook.weight.device)
        self.codebook.weight[rows] = self.relation_codebook.weight
        if not torch.equal(self.codebook.weight[rows], self.relation_codebook.weight):
            raise ValueError("relation rows were not transferred exactly")
        self.stage.fill_(1)


def oracle_stage_optimizer(model, lr):
    """Fresh AdamW per stage over the trainable core plus that stage's table (historical order)."""
    table = model.oracle.relation_codebook if int(model.oracle.stage) == 0 else model.oracle.codebook
    params = [p for p in model.core.parameters() if p.requires_grad] + [table.weight]
    return params, torch.optim.AdamW(params, lr=lr, weight_decay=0.01)


def pairwise_d(batch):
    """Evaluation only: d = (a_j + delta - b_k) mod 4; the truth must follow the match [d = 0]."""
    q = batch.query
    rules = [batch.rules[int(e)] for e in q.episode]
    j, k, delta = (torch.tensor([getattr(r, n) for r in rules]) for n in ("j", "k", "delta"))
    aj = batch.scenes.attrs[q.scene, q.a].gather(1, j[:, None]).squeeze(1)
    bk = batch.scenes.attrs[q.scene, q.b].gather(1, k[:, None]).squeeze(1)
    d = (aj + delta - bk) % 4
    m, s = d == 0, q.pre.gather(1, q.machine[:, None]).squeeze(1).bool()
    family = rules[0].family
    truth = dict(relation=m, open=s | m, close=s & ~m, toggle=s ^ m)[family]
    if any(r.family != family for r in rules) or not torch.equal(truth.long(), q.outcome.long()):
        raise ValueError(f"{family}: truth does not follow [d = 0]; alignment error")
    return d


def within_auc(event, score, groups):
    """Mean within-episode AUROC (ties 0.5); episodes with one class are left out."""
    values = []
    for g in np.unique(groups):
        pos, neg = score[(groups == g) & (event == 1)], score[(groups == g) & (event == 0)]
        if len(pos) and len(neg):
            values.append(float(((pos[:, None] > neg) + 0.5 * (pos[:, None] == neg)).mean()))
    return float(np.mean(values)) if values else None


def oracle_pools(size, train):
    """Historical pools: all44 (one episode per relation rule) and the five-family pool 9101."""
    e = ORACLE_EVAL[size]
    g = torch.Generator().manual_seed(ORACLE_POOLS["all44"])
    all44 = [rw.sample_episodes(g, [rule], rw.KIND_SPLIT["train"], episodes=1, support=(e["support"],),
                                queries=e["queries"], p_empty=0.0) for rule in train if rule.family == "relation"]
    g = torch.Generator().manual_seed(ORACLE_POOLS["mixed"])
    mixed = [rw.sample_episodes(g, [r for r in train if r.family == f], rw.KIND_SPLIT["train"], episodes=e["episodes"],
                                support=(e["support"],), queries=e["queries"], p_empty=0.0) for f in rw.FAMILIES]
    return dict(all44=all44, mixed=mixed)


@torch.no_grad()
def oracle_evaluate(model, pools, device, floors):
    """nu with full / zero / swap codes per pool and family, the development screen
    (nu_full >= 0.7, > zero, > swap, every episode defined) and pairwise second-bit AUROC."""
    rows = []
    for name, batches in pools.items():
        keys = [r.key() for b in batches for r in b.rules]
        offset = 0
        for part, batch in enumerate(batches):
            own = [r.key() for r in batch.rules]
            if name == "all44":  # next episode (cyclic) with a different rule, across the pool
                idx = [offset + i for i in range(len(own))]
                swap = [next(keys[(i + t) % len(keys)] for t in range(1, len(keys)) if keys[(i + t) % len(keys)] != keys[i])
                        for i in idx]
            else:  # first other episode of this family with a different rule
                swap = []
                for i in range(len(own)):
                    others = [o for o in range(len(own)) if o != i]
                    swap.append(own[([o for o in others if own[o] != own[i]] or others or [i])[0]])
            offset += len(own)
            rules = {r.key(): r for b in batches for r in b.rules}
            tokens = ev.symbolic_episode_tokens(model.perception, batch, device)
            q = tokens.query
            full = model.oracle(batch.rules)
            s = batch.query.pre.gather(1, batch.query.machine[:, None]).squeeze(1)
            d = pairwise_d(batch) if batch.rules[0].family in PAIRWISE else torch.full_like(s, -1)
            for control, codes in dict(full=full, zero=torch.zeros_like(full),
                                       swap=model.oracle([rules[k] for k in swap])).items():
                p = torch.sigmoid(model.core.apply(q.m_pre, q.a, q.b, codes[q.episode])[0]).cpu()
                for i in range(len(p)):
                    e = int(batch.query.episode[i])
                    rows.append(dict(pool=name, control=control, family=batch.rules[e].family, rule=own[e],
                                     group=f"{name}/{part}/{e}", truth=int(batch.query.outcome[i]), s=int(s[i]),
                                     d=int(d[i]), p=float(p[i])))
    metrics = {}
    for name in pools:
        nu = {c: ev.nu_from_rows([r for r in rows if r["pool"] == name and r["control"] == c], floors)
              for c in ("full", "zero", "swap")}
        families = sorted({r["family"] for r in rows if r["pool"] == name}, key=rw.FAMILIES.index)
        screen = {}
        for f in families:
            full, zero, swap = (nu[c].get(f) for c in ("full", "zero", "swap"))
            screen[f] = ("incomplete" if None in (full, zero, swap)
                         else "pass" if full >= 0.7 and full > zero and full > swap else "fail")
        overall = ("fail" if "fail" in screen.values() else
                   "incomplete" if "incomplete" in screen.values() or nu["full"]["groups_without_both_classes"] else "pass")
        second = {}
        for f in (f for f in families if f in PAIRWISE):
            chosen = [r for r in rows if r["pool"] == name and r["control"] == "full" and r["family"] == f]
            d, s, p = (np.array([r[k] for r in chosen]) for k in ("d", "s", "p"))
            groups = np.array([r["group"] for r in chosen])
            eligible = {"open": s == 0, "close": s == 1}.get(f, np.ones_like(s, dtype=bool))
            score = np.where((f in ev.TRANSITION_FAMILIES) & (s == 1), 1 - p, p)  # scored event's probability
            even = eligible & (d % 2 == 0)
            second[f] = within_auc((d[even] == 0).astype(int), score[even], groups[even])
        metrics[name] = dict(screen=overall, families=screen, second_bit_auc=second,
                             **{f"nu_{c}": {f: nu[c].get(f) for f in families} for c in nu},
                             groups_without_both_classes=nu["full"]["groups_without_both_classes"])
    return rows, metrics


def oracle_row(step, stage, metrics):
    row = dict(step=step, split="evaluation", stage=stage)
    for name, m in metrics.items():
        row[f"screen_{name}"] = m["screen"]
        for c in ("full", "zero", "swap"):
            row.update({f"nu_{c}_{name}_{f}": v for f, v in m[f"nu_{c}"].items() if v is not None})
        row.update({f"second_bit_{name}_{f}": v for f, v in m["second_bit_auc"].items() if v is not None})
    return row


def train_oracle(args, s):
    """Symbolic stage, --oracle-curriculum R: R relation-only updates, then mixed to --updates."""
    relation_updates = args.oracle_curriculum
    seed_everything(args.seed)
    train = rw.split_rules()["train"]
    model = RuleModel(s, symbolic=True)
    model.oracle = OracleCodes(s, train)
    model = model.to(args.device)
    for frozen in (model.perception, model.core.key_head, model.core.evidence_mlp):
        frozen.requires_grad_(False)  # fixed symbols, unused key head, bypassed evidence MLP
    gate = boundary_gate(args.size)
    if args.resume:  # refuse final runs and unverifiable boundaries before anything is written
        path = Path(args.resume)
        status = json.loads((path / "status.json").read_text())["result"]
        if status in ("completed", "stopped"):
            raise ValueError(f"run is final ({status}); an oracle curriculum is not extended, start a new output")
        state = torch.load(path / "last.pt", map_location="cpu", weights_only=True)
        if int(state["model"]["oracle.stage"]) == 1:
            if verified_snapshot(path, relation_updates)["decision"] not in ("pass", "exempt"):
                raise ValueError("mixed stage without a passed or exempt relation gate")
        elif (path / "relation_stage.pt").exists() or (path / "relation_stage.json").exists():
            if state["step"] != relation_updates:  # a stage-0 snapshot exists only in the boundary crash window
                raise ValueError("relation_stage artefacts do not belong to this checkpoint; refusing to continue")
            verified_snapshot(path, relation_updates, require_record=False)
        model.oracle.stage.fill_(int(state["model"]["oracle.stage"]))  # selects the checkpoint's optimizer
    params, optimizer = oracle_stage_optimizer(model, args.lr)
    relation = [r for r in train if r.family == "relation"]
    e = ORACLE_EVAL[args.size]
    workflow = ("WORKFLOW CHECK ONLY (tiny sizes; boundary gate exempt): software path, no learning claim. "
                if args.size == "check" else "")
    settings = dict(stage="symbolic", oracle_curriculum=relation_updates, seed=args.seed, updates=args.updates,
                    **core_overrides(args),
                    lr=args.lr, device=args.device, size=args.size, sizes=s, purpose="development", precision="fp32",
                    max_reserved_gib=args.max_reserved_gib, scope=workflow + ORACLE_SCOPE, boundary_gate=gate,
                    stages=[dict(name="relation", updates=[1, relation_updates], rules=len(relation)),
                            dict(name="mixed", updates=[relation_updates + 1, args.updates], rules=len(train))],
                    boundary=("stage-0 state saved and copied atomically to relation_stage.pt; all44 gate on it; then "
                              "44 trained rows -> 192-row codebook by rule key, fresh AdamW, sampler re-seeded (seed+1009)"),
                    objective="unweighted outcome BCE of core.apply(m_pre, a, b, oracle code) on queries",
                    batch=dict(episodes=s["episodes"], support=s["support"], queries=s["queries"], p_empty=ORACLE_P_EMPTY),
                    frozen=["perception (supplied symbols)", "core.key_head (unused)", "core.evidence_mlp (bypassed)"],
                    evaluation=dict(pools=ORACLE_POOLS, **e, every=s["validate_every"]),
                    checkpoint_layout=("standalone RuleModel + oracle.* codes; not the historical probe.build layout; "
                                       "no historical C192 consumer loads it"),
                    provenance="promotes runs R44 (core_relation_probe) and C192 (core_curriculum); see plan artifact")
    runner = Run(args.resume or args.output, settings=settings, data=rw.manifest(), recipe=__file__, model=model,
                 optimizer=optimizer, device=args.device, resume=args.resume is not None)
    floors = rw.floors()
    pools = oracle_pools(args.size, train)
    session = 1 + len(list(runner.path.glob("result_end_*.json")))
    started = time.perf_counter()

    def evaluation(name):
        """Raw predictions and metrics, written once (a deterministic replay must match)."""
        with evaluation_mode(model):
            rows, metrics = oracle_evaluate(model, pools, args.device, floors)
        retain_npz(runner.path / f"predictions_{name}.npz", {k: np.array([r[k] for r in rows]) for k in rows[0]})
        retain_json(runner.path / f"metrics_{name}.json", metrics)
        return metrics

    def conclude(stop, status, extra=None):
        """Per-session end evidence (never overwritten), then result.json and the report."""
        name = f"end_step{runner.step:05d}_s{session:02d}"
        metrics = evaluation(name)
        result = dict(
            evaluation_scope=workflow + ORACLE_SCOPE, gate=None, stop_reason=stop, boundary_gate=gate,
            session=session, session_result=f"result_{name}.json",
            metrics=dict(stage=int(model.oracle.stage), end=metrics, **(extra or {}),
                         screens={n: m["screen"] for n, m in metrics.items()},
                         resources=dict(**resources(args.device), updates=runner.step,
                                        seconds_this_session=time.perf_counter() - started, **parameters(model))),
            limitations=["Oracle conditioning: the rule ID is supplied; this measures code APPLICATION by the shared "
                         "core, not induction from evidence, retrieval, perception or planning.",
                         "Development screen on training rules only (fresh scenes); validation/test rules have no codes.",
                         "One seed; d uses generator parameters for evaluation only.",
                         "Standalone diagnostic checkpoint: not the historical probe.build layout; no historical "
                         "C192/E1a/E1b/I1 consumer loads it."],
        )
        retain_json(runner.path / f"result_{name}.json", result)
        finish(runner, result, complete=status != "paused", status=status)

    def boundary():
        """Pre-transfer snapshot (atomic), gate on it, then the transfer; replay-safe."""
        if (runner.path / "relation_stage.pt").exists():  # bound to this run before anything is written
            verified_snapshot(runner.path, relation_updates, require_record=False)
        runner.save()  # the stage-0 state at step R
        state = torch.load(runner.path / "last.pt", map_location="cpu", weights_only=True)
        snapshot = runner.path / "relation_stage.pt"
        if snapshot.exists():  # replay after a crash: it must be exactly this state (and its recorded hash)
            kept = torch.load(snapshot, map_location="cpu", weights_only=True)
            if not all(same_state(kept[k], state[k]) for k in ("step", "model", "optimizer", "sampler")):
                raise ValueError("relation_stage.pt differs from the replayed stage-0 state; refusing to continue")
        else:
            atomic_copy(runner.path / "last.pt", snapshot)
        metrics = evaluation("relation_stage")
        screen = metrics["all44"]["screen"]
        decision = "exempt" if gate.startswith("exempt") else screen
        retain_json(runner.path / "relation_stage.json",
                    dict(step=runner.step, stage=0, sha256=file_hash(snapshot), identity_sha256=state["identity_sha256"],
                         gate=gate, all44_screen=screen, decision=decision))
        if decision not in ("pass", "exempt"):  # R44 did not pass: never train the mixed stage
            conclude("relation_stage_failed", "stopped", dict(relation_gate=dict(all44_screen=screen, gate=gate)))
            return False
        model.oracle.transfer()
        new_params, new_optimizer = oracle_stage_optimizer(model, args.lr)
        runner.optimizer = new_optimizer
        runner.sampler.manual_seed(args.seed + 1009)  # C192 was a fresh run: same mixed stream
        runner.log(oracle_row(runner.step, 1, evaluation("boundary")))
        runner.save()
        return new_params, new_optimizer

    try:
        if runner.step == 0 and not args.resume:
            runner.log(oracle_row(0, 0, evaluation("start")))
            runner.save()
        while runner.step < args.updates and not stop(runner, args, started):
            if runner.step == relation_updates and int(model.oracle.stage) == 0:
                crossed = boundary()
                if crossed is False:
                    return runner.path
                params, optimizer = crossed
            stage = int(model.oracle.stage)
            training_mode(model)
            batch = rw.sample_episodes(runner.sampler, relation if stage == 0 else train, rw.KIND_SPLIT["train"],
                                       episodes=s["episodes"], support=s["support"], queries=s["queries"],
                                       p_empty=ORACLE_P_EMPTY)
            q = ev.symbolic_episode_tokens(model.perception, batch, args.device).query
            logits, _ = model.core.apply(q.m_pre, q.a, q.b, model.oracle(batch.rules)[q.episode])
            loss = F.binary_cross_entropy_with_logits(logits, q.outcome)  # natural, unweighted
            optimizer.zero_grad()
            loss.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in params):
                raise ValueError("Nonfinite gradient")
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            runner.step += 1
            runner.log(dict(step=runner.step, split="train", stage=stage, loss=float(loss.detach()),
                            query_accuracy=float(((logits > 0).float() == q.outcome).float().mean())))
            enforce_ceiling(args, runner)
            if runner.step % s["validate_every"] == 0 or runner.step == args.updates:
                runner.log(oracle_row(runner.step, stage, evaluation(f"step{runner.step:05d}")))
                runner.save()
                print(f"oracle curriculum: saved step {runner.step} (stage {stage})", flush=True)
        runner.save()
        complete = runner.step >= args.updates
        conclude(stop_reason(runner, args, started), "completed" if complete else "paused")
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
    train_oracle(sub(output=root / "oracle", updates=4, oracle_curriculum=2, stop_after=None, resume=None, size="check"), s)
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
    parser.add_argument("--texture-randomization", type=float, default=0.0,
                        help="perception only: per-machine probability of a procedural training body texture")
    parser.add_argument("--init-perception", type=Path,
                        help="perception only: warm-start weights from a run/last.pt (new optimizer and run)")
    parser.add_argument("--identity", choices=("joint", "detached"),
                        help="perception only: paired-view appearance-key InfoNCE (joint trains perception too)")
    parser.add_argument("--identity-weight", type=float,
                        help=f"perception only, with --identity: loss weight (default {IDENTITY_WEIGHT})")
    parser.add_argument("--oracle-curriculum", type=int, metavar="R",
                        help="symbolic only: ORACLE diagnostic, R relation-only updates then all training rules "
                             "until --updates (historical: 8000 of 16000)")
    parser.add_argument("--train-rules", type=int, choices=rw.LADDER_SIZES,
                        help="symbolic only (§23): train on a nested relation pool of 1/4/16/44 training rules")
    parser.add_argument("--auxiliary-weight", type=float, default=1.0,
                        help="symbolic only (§23 L4-O): weight of the next-token and rollout terms (0 = outcome BCE only)")
    parser.add_argument("--reader", choices=("code", "evidence"), default="code",
                        help="symbolic only (§23 L4-D1): T reads an induced code or the support evidence directly")
    parser.add_argument("--support-sizes", type=int, nargs="+",
                        help="symbolic only (overnight N1): training support sizes (default: the size profile's)")
    parser.add_argument("--rule-repeats", type=int, nargs="+",
                        help="symbolic with --train-rules only (overnight N5): per-rule training draw multiplicity")
    parser.add_argument("--train-family", choices=rw.FAMILIES,
                        help="symbolic with --train-rules (F1): family of the ladder pool (default relation)")
    parser.add_argument("--core-width", type=int, help="override the size profile's core width (E4 capacity axis)")
    parser.add_argument("--core-blocks", type=int, help="override the number of untied core blocks per inner round")
    parser.add_argument("--core-loops", type=int, help="override the number of inner rounds")
    parser.add_argument("--episodes", type=int, help="override the training episodes per update (dilution test)")
    parser.add_argument("--holdout-ladder-rules", type=int, nargs="+",
                        help="symbolic single-family ladder: never train these ladder indices; they form the transfer pool")
    parser.add_argument("--train-families", nargs="+", choices=rw.FAMILIES,
                        help="symbolic with --train-rules (M1): one core on the union of these family ladders")
    parser.add_argument("--delta-weights", type=int, nargs=4,
                        help="symbolic with --train-rules (C1-C3): training draw multiplicity per relation δ=0..3")
    parser.add_argument("--uniform-after", type=int,
                        help="symbolic with a skew option (C2): uniform training rules from this update on")
    parser.add_argument("--init-key", action="store_true",
                        help="core: initialize core.key_head from the --perception run's exported identity key; "
                             "perception (with --identity): load the key exported by the --init-perception run")
    parser.add_argument("--freeze-perception", action="store_true",
                        help="perception only: train only the loaded key on frozen --init-perception weights "
                             "(requires --init-key and --identity detached; perception hash-guarded)")
    parser.add_argument("--decoder-reconstruction", choices=("slots", "pyramid"),
                        help="perception only: retrain only the actual slot decoder of the frozen --init-perception "
                             "run, slot-only or with its pyramid connections enabled")
    parser.add_argument("--decoder-subpixel", action="store_true",
                        help="optional full-resolution fine residual, requires pyramid decoder")
    parser.add_argument("--decoder-mask-weight", type=float, default=None,
                        help="decoder-only objective mask CE weight; default0.5, set0 for RGB-only")
    parser.add_argument("--decoder-code-weight", type=float, default=0.0,
                        help="decoder-only raw fine-code feedback weight; encoder stays frozen")
    parser.add_argument("--decoder-eval-every", type=int, default=0,
                        help="decoder-only fixed evaluation/mask preservation every N updates (0 disables)")
    parser.add_argument("--decoder-eval-scenes", type=int,
                        help="decoder reconstruction: fixed evaluation scenes per population "
                             "(default sizes validation_scenes; 256 on the actual config)")
    parser.add_argument("--identity-margin-positive", type=float,
                        help="perception identity: absolute cosine target for same-machine pairs")
    parser.add_argument("--identity-margin-negative", type=float,
                        help="perception identity: absolute cosine ceiling for different-machine pairs")
    parser.add_argument("--identity-margin-weight", type=float,
                        help="perception identity: weight of the squared absolute-margin hinges (> 0)")
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
    overrides = {k: v for k, v in (("core_width", args.core_width), ("core_blocks", args.core_blocks),
                                   ("loops", args.core_loops), ("episodes", args.episodes)) if v is not None}
    if overrides:  # recorded through the run's `sizes` setting and restored on resume
        if min(overrides.values()) < 1 or overrides.get("core_width", s["heads"]) % s["heads"]:
            parser.error("core overrides must be positive and the width divisible by the head count")
        s = dict(s, **overrides)
    if (args.texture_randomization or args.init_perception) and args.stage != "perception":
        parser.error("--texture-randomization/--init-perception apply to the perception stage only")
    if not 0.0 <= args.texture_randomization <= 1.0:
        parser.error("--texture-randomization must be a probability")
    if (args.identity or args.identity_weight is not None) and args.stage != "perception":
        parser.error("--identity/--identity-weight apply to the perception stage only")
    if args.identity_weight is not None and not args.identity:
        parser.error("--identity-weight requires --identity")
    if args.identity and args.identity_weight is None:
        args.identity_weight = IDENTITY_WEIGHT
    if args.init_key and args.stage not in ("core", "perception"):
        parser.error("--init-key applies to the core or perception stage only")
    if args.stage == "perception" and args.resume is None:
        if args.init_key and not (args.identity and args.init_perception):
            parser.error("perception --init-key requires --identity and --init-perception (the key's source run)")
        if args.freeze_perception:
            if not (args.init_key and args.init_perception and args.identity == "detached"):
                parser.error("--freeze-perception requires --init-perception, --init-key and --identity detached")
            recorded = parent_texture_randomization(args.init_perception)
            supplied = {a.split("=", 1)[0] for a in sys.argv[1:]}
            if "--texture-randomization" in supplied and args.texture_randomization != recorded:
                parser.error(f"--freeze-perception uses the parent's recorded texture randomization {recorded}")
            args.texture_randomization = recorded
    if args.freeze_perception and args.stage != "perception":
        parser.error("--freeze-perception applies to the perception stage only")
    if args.decoder_subpixel and args.decoder_reconstruction != "pyramid":
        parser.error("--decoder-subpixel requires --decoder-reconstruction pyramid")
    if not math.isfinite(args.decoder_code_weight) or args.decoder_code_weight < 0:
        parser.error("--decoder-code-weight must be finite and nonnegative")
    if args.decoder_code_weight and args.decoder_reconstruction != "pyramid":
        parser.error("--decoder-code-weight requires --decoder-reconstruction pyramid")
    if args.decoder_reconstruction:
        if args.decoder_mask_weight is None:
            args.decoder_mask_weight = DECODER_MASK_WEIGHT
        if not math.isfinite(args.decoder_mask_weight) or args.decoder_mask_weight < 0 or args.decoder_eval_every < 0:
            parser.error("decoder mask weight and evaluation interval must be nonnegative")
        if args.stage != "perception":
            parser.error("--decoder-reconstruction applies to the perception stage only")
        if args.resume is None:
            if not args.init_perception:
                parser.error("--decoder-reconstruction requires --init-perception (the frozen parent run)")
            if args.identity or args.init_key or args.freeze_perception or args.identity_margin_weight is not None:
                parser.error("--decoder-reconstruction excludes identity, key, freeze and margin options")
            recorded = parent_texture_randomization(args.init_perception)
            supplied = {a.split("=", 1)[0] for a in sys.argv[1:]}
            if "--texture-randomization" in supplied and args.texture_randomization != recorded:
                parser.error(f"--decoder-reconstruction uses the parent's recorded texture randomization {recorded}")
            args.texture_randomization = recorded
        if args.decoder_eval_scenes is None:
            args.decoder_eval_scenes = SIZES[args.size]["validation_scenes"]
        if args.decoder_eval_scenes < (2 if args.decoder_reconstruction == "pyramid" else 1):
            parser.error("--decoder-eval-scenes must be positive (at least 2 for the pyramid arm's shuffle)")
    elif args.decoder_eval_scenes is not None or args.decoder_mask_weight is not None or args.decoder_eval_every:
        parser.error("--decoder-eval-scenes applies to --decoder-reconstruction only")
    margin = (args.identity_margin_positive, args.identity_margin_negative, args.identity_margin_weight)
    if any(v is not None for v in margin):
        if any(v is None for v in margin):
            parser.error("--identity-margin-positive/-negative/-weight must be given together")
        if args.stage != "perception" or not args.identity:
            parser.error("identity margins apply to the perception stage with --identity")
        positive, negative, weight = margin
        if not (-1.0 <= negative < positive <= 1.0) or not (math.isfinite(weight) and weight > 0):
            parser.error("identity margins need -1 <= negative < positive <= 1 and a positive finite weight")
    if args.oracle_curriculum is not None and (args.stage != "symbolic" or not 0 < args.oracle_curriculum < args.updates):
        parser.error("--oracle-curriculum R applies to the symbolic stage with 0 < R < --updates")
    if args.auxiliary_weight != 1.0 and (args.stage != "symbolic" or args.oracle_curriculum is not None
                                         or not (math.isfinite(args.auxiliary_weight) and args.auxiliary_weight >= 0)):
        parser.error("--auxiliary-weight applies to the symbolic induction stage and must be finite and nonnegative")
    if args.reader != "code" and (args.stage != "symbolic" or args.oracle_curriculum is not None):
        parser.error("--reader evidence applies to the symbolic induction stage without --oracle-curriculum")
    if args.support_sizes is not None and (args.stage != "symbolic" or args.oracle_curriculum is not None
                                           or min(args.support_sizes) < 1):
        parser.error("--support-sizes applies to the symbolic induction stage and needs positive sizes")
    pool_size = (args.train_rules or 0) * len(args.train_families or [None])
    if args.rule_repeats is not None and (args.train_rules is None or len(args.rule_repeats) != pool_size
                                          or min(args.rule_repeats) < 1):
        parser.error("--rule-repeats needs --train-rules and one positive count per ladder rule")
    # On resume, the recorded default train_family ("relation") is restored next to train_families.
    if args.holdout_ladder_rules is not None and (
            args.train_rules is None or args.train_families is not None
            or len(set(args.holdout_ladder_rules)) != len(args.holdout_ladder_rules)
            or not all(0 <= i < args.train_rules for i in args.holdout_ladder_rules)
            or len(args.holdout_ladder_rules) >= args.train_rules):
        parser.error("--holdout-ladder-rules needs a single-family --train-rules ladder, distinct valid indices "
                     "and at least one trained rule")
    if args.train_families is not None and (args.train_family is not None and args.resume is None
                                            or args.train_rules is None
                                            or len(set(args.train_families)) != len(args.train_families)
                                            or args.delta_weights is not None):
        parser.error("--train-families needs --train-rules, distinct families, no --train-family/--delta-weights")
    if args.train_families and "category" in args.train_families and args.train_rules not in (1, 4, 16):
        parser.error("category ladders have 1, 4 or 16 rules")
    if args.train_family is not None and args.train_rules is None:
        parser.error("--train-family needs --train-rules")
    if args.train_family == "category" and (args.train_rules not in (1, 4, 16) or args.delta_weights is not None):
        parser.error("category ladders have 1, 4 or 16 rules and no δ weights")
    if args.delta_weights is not None and (args.train_rules is None or args.rule_repeats is not None
                                           or min(args.delta_weights) < 1):
        parser.error("--delta-weights needs --train-rules, excludes --rule-repeats and needs positive weights")
    if args.uniform_after is not None and (args.delta_weights is None and args.rule_repeats is None
                                           or args.uniform_after < 1):
        parser.error("--uniform-after needs --delta-weights or --rule-repeats and a positive step")
    if args.train_rules is not None and (args.stage != "symbolic" or args.oracle_curriculum is not None):
        parser.error("--train-rules applies to the symbolic induction stage without --oracle-curriculum")
    if args.stage == "audit":
        audit(args, s)
    elif args.stage == "perception":
        (train_decoder_reconstruction if args.decoder_reconstruction else train_perception)(args, s)
    elif args.stage in ("core", "symbolic"):
        if args.stage == "core" and args.perception is None and args.resume is None:
            parser.error("--perception RUN required")
        if args.oracle_curriculum is not None:
            train_oracle(args, s)
        else:
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
