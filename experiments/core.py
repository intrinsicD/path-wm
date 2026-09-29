"""Shared core on Rule World, symbolic stage (docs/shared-core-plan.md, S2).

One press event per transition: the core binds the perceived scene, observes it,
predicts the press with the rule code Z induced from the episode's support
demonstrations, then observes the post frame. Perception is supplied symbolically
(diagnostic stage; `pathwm/models/symbolic.py`). Losses follow docs/core-design.md:
posterior reconstruction, prior prediction (one step and a two-step chain without
observation), KL with dynamics/representation weights 1/0.1 and one free nat per event.

Evaluation: ν on the prior's target-lamp prediction for fresh training-rule episodes
and for held-out validation rules, against copy (lamp unchanged) and against the
empty-support and swapped-support controls.

Stage `perception` (S4) trains slot perception from scratch on rendered frames of
training machine kinds and qualifies it on frames of validation kinds (never seen).

  python -m experiments.core --output runs/core/relation_l4 --family relation --rules 4
  python -m experiments.core --stage perception --output runs/core/perception
  python -m experiments.core --stage pixel --perception runs/core/s4_perception_1101/last.pt --output runs/core/pixel
  python -m experiments.core --resume runs/core/relation_l4
"""

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from pathwm.data import rule_world as rw
from pathwm.evaluation.report import write_report
from pathwm.evaluation.perception import THRESHOLDS as PERCEPTION_THRESHOLDS, perception_metrics, qualification
from pathwm.evaluation.rules import nu_from_rows
from pathwm.io import Run, atomic_json, file_hash, load_component, resume_arguments, seed_everything
from pathwm.models.core import CoreConfig, SharedCore
from pathwm.models.slots import SlotPerception, match_slots, perception_loss, pointer
from pathwm.models.symbolic import SymbolicDecoder, SymbolicEncoder, readout_loss, targets

# full = the decided configuration (E4); check = declared downscale for CPU tests only.
SIZES = dict(full=dict(width=128, heads=4, blocks=2, rounds=2),
             check=dict(width=32, heads=4, blocks=2, rounds=1))
# Perception (S4): slot width 64, decoder width 32, 7 slots (one per scene entity).
PERCEPTION_SIZES = dict(full=dict(width=64, iterations=3, decoder_width=32),
                        check=dict(width=16, iterations=1, decoder_width=8))
FREE_NATS, DYN, REP = 1.0, 1.0, 0.1
RUN_CONTROL = ("max_minutes", "log_every", "save_every")  # pacing only; not part of the run identity


class PerceptionModel(nn.Module):
    """Container so checkpoints hold `perception.*` (loaded later with load_component)."""

    def __init__(self, size):
        super().__init__()
        self.perception = SlotPerception(size["width"], 7, size["iterations"], decoder_width=size["decoder_width"])


@dataclass
class Events:
    """One batch of press events, aligned with the core's slot order (the bound frame).

    pre/post: evidence tokens [n,7,E] of the pre and post frames; action [n,A];
    machines [n,2]: core slots of the left/right machine; target [n]: core slot of the
    pressed machine; target_post [n]: its slot in the post frame's evidence;
    truth_pre/truth_post: readout labels per core slot (training signals only)."""
    pre: torch.Tensor
    post: torch.Tensor
    action: torch.Tensor
    machines: torch.Tensor
    target: torch.Tensor
    target_post: torch.Tensor
    truth_pre: dict
    truth_post: dict


def gather_truth(truth, entity_of_slot):
    return {k: v.gather(1, entity_of_slot.cpu().reshape(*entity_of_slot.shape, *[1] * (v.dim() - 2)).expand(
        *entity_of_slot.shape, *v.shape[2:])) for k, v in truth.items()}


def core_config(size, evidence, tied, continuous_only):
    return CoreConfig(width=size["width"], heads=size["heads"], blocks=size["blocks"], rounds=size["rounds"],
                      evidence_width=evidence, action_width=3 * evidence, tied=tied, continuous_only=continuous_only)


class SymbolicModel(nn.Module):
    """S2: supplied symbolic perception; core slot i is scene entity i."""

    def __init__(self, size, tied=True, continuous_only=False):
        super().__init__()
        w = size["width"]
        self.encoder = SymbolicEncoder(w)
        self.core = SharedCore(core_config(size, w, tied, continuous_only))
        self.decoder = SymbolicDecoder(w)

    def prepare(self, batch, device):
        return batch

    def events(self, batch, t, anchor=None):
        """Actions use lamp-free identity tokens, so `anchor` is not needed."""
        device = self.decoder.lamp.weight.device
        scn = batch.scenes.select(t.scene)
        enc = self.encoder
        ident = enc.identity(scn)
        i = torch.arange(len(t), device=device)
        m, a, b = t.machine.to(device) + 1, t.a.to(device) + 3, t.b.to(device) + 3
        machines = torch.tensor([1, 2], device=device).expand(len(t), -1)
        return Events(enc(scn, t.pre.to(device)), enc(scn, t.post.to(device)),
                      torch.cat((ident[i, m], ident[i, a], ident[i, b]), -1), machines, m, m,
                      targets(scn, t.pre), targets(scn, t.post))


class PixelModel(nn.Module):
    """S5: frozen retrained perception (S4); core slots follow the bound frame's slots."""

    LAMPS = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])

    def __init__(self, size, perception_size, perception_path, tied=True, continuous_only=False):
        super().__init__()
        p = perception_size
        self.perception = SlotPerception(p["width"], 7, p["iterations"], decoder_width=p["decoder_width"])
        load_component(self.perception, perception_path, "perception")
        self.perception.requires_grad_(False)
        self.core = SharedCore(core_config(size, p["width"], tied, continuous_only))
        self.decoder = SymbolicDecoder(size["width"])

    def train(self, mode=True):
        super().train(mode)
        self.perception.eval()  # frozen: never in training mode
        return self

    @torch.no_grad()
    def prepare(self, batch, device):
        """Render and perceive every scene once per lamp configuration (frame = scene*4 + l0*2 + l1)."""
        n = len(batch.scenes)
        index = torch.arange(n).repeat_interleave(4)
        lamps = self.LAMPS.repeat(n, 1)
        rgb, entity = rw.render(batch.scenes.select(index), lamps)
        slots, alpha, owner = [], [], []
        for start in range(0, len(rgb), 256):
            percept = self.perception(rgb[start:start + 256].to(device))
            slots.append(percept.slots)
            alpha.append(percept.alpha)
            owner.append(match_slots(percept.alpha, entity[start:start + 256].to(device)))
        return dict(batch=batch, slots=torch.cat(slots), alpha=torch.cat(alpha), entity_of_slot=torch.cat(owner))

    def events(self, prepared, t, anchor=None):
        """Actions point at slots of the anchor frame (the chain's first frame; default: pre)."""
        anchor = t if anchor is None else anchor
        batch, slots, alpha = prepared["batch"], prepared["slots"], prepared["alpha"]
        device = slots.device
        frame = lambda lamps: (t.scene * 4 + lamps[:, 0] * 2 + lamps[:, 1]).to(device)
        f_pre, f_post, f_anchor = frame(t.pre), frame(t.post), frame(anchor.pre)
        scn = batch.scenes.select(t.scene)
        point = lambda f, xy: pointer(alpha[f], xy.to(device))
        machines = torch.stack([point(f_anchor, scn.machine_xy[:, k]) for k in range(2)], -1)
        i = torch.arange(len(t), device=device)
        m = t.machine.to(device)
        a = point(f_anchor, scn.object_xy[torch.arange(len(t)), t.a])
        b = point(f_anchor, scn.object_xy[torch.arange(len(t)), t.b])
        target_post = point(f_post, scn.machine_xy[torch.arange(len(t)), t.machine])
        base = slots[f_anchor]
        owner = prepared["entity_of_slot"][f_anchor]
        return Events(slots[f_pre], slots[f_post], torch.cat((base[i, machines[i, m]], base[i, a], base[i, b]), -1),
                      machines, machines[i, m], target_post,
                      gather_truth(targets(scn, t.pre), owner), gather_truth(targets(scn, t.post), owner))


# ---------------------------------------------------------------- one batch of press events


def observe_frame(core, tokens, prior=None):
    valid = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
    prior = core.bind(tokens, valid) if prior is None else prior
    return core.observe(prior, tokens, valid)


def rule_codes(model, prepared, batch, device, control="full", reader="code"):
    """What `predict` reads per episode: (Z [E,K,D], None) from `induce`, or with
    reader="evidence" the uncompressed support events (tokens [E,N,D], valid [E,N]).
    Controls replace the support: empty, or swapped with the neighbouring episode's."""
    core, e = model.core, len(batch.rules)
    s = batch.support
    if control == "empty" or len(s) == 0:
        none = torch.zeros(e, 1, core.config.width, device=device)
        invalid = torch.zeros(e, 1, dtype=torch.bool, device=device)
        return (core.induce(none, invalid), None) if reader == "code" else (none, invalid)
    ev = model.events(prepared, s)
    i = torch.arange(len(s), device=device)
    tokens = core.event(ev.pre[i, ev.target], ev.action, ev.post[i, ev.target_post])
    episode = s.episode.to(device)
    counts = torch.bincount(episode, minlength=e)
    padded = tokens.new_zeros(e, int(counts.max()), tokens.shape[-1])
    valid = torch.zeros(e, int(counts.max()), dtype=torch.bool, device=device)
    rank = torch.cat([torch.arange(int(c), device=device) for c in counts])
    order = torch.argsort(episode, stable=True)
    padded[episode[order], rank] = tokens[order]
    valid[episode[order], rank] = True
    z, zv = (core.induce(padded, valid), None) if reader == "code" else (padded, valid)
    if control == "swapped":
        z, zv = z.roll(1, 0), None if zv is None else zv.roll(1, 0)
    return z, zv


def pick(context, episode):
    z, zv = context
    return z[episode], None if zv is None else zv[episode]


def press(model, prepared, t, context, device):
    """Observe the pre frame, predict the press reading the rule context, observe the post frame."""
    core = model.core
    ev = model.events(prepared, t)
    before = observe_frame(core, ev.pre)
    ones = torch.ones(len(t), device=device)
    z, zv = pick(context, t.episode.to(device))
    prior = core.predict(before, ev.action, dt=ones, z=z, z_valid=zv)
    after = observe_frame(core, ev.post, prior)
    return dict(ev=ev, before=before, prior=prior, after=after, i=torch.arange(len(t), device=device), t=t)


def lamp_logits(model, state, machines):
    """[n,2 machines,2] lamp logits read from the core slots of both machines."""
    tokens = model.core.tokens(state)
    rows = torch.arange(len(tokens), device=tokens.device)[:, None]
    return model.decoder(tokens[rows, machines])["lamp"]


def split_chain(c):
    first, second = c.step == 0, c.step == 1
    return (rw.Transitions(*(v[first] for v in vars(c).values())),
            rw.Transitions(*(v[second] for v in vars(c).values())))


def losses(model, batch, device, args):
    core, dec = model.core, model.decoder
    prepared = model.prepare(batch, device)
    z = rule_codes(model, prepared, batch, device, reader=args.reader)
    q = press(model, prepared, batch.query, z, device)
    ev, t = q["ev"], q["t"]
    tok = lambda s: core.tokens(s)[:, :7]
    recon = readout_loss(dec(tok(q["before"])), ev.truth_pre) + readout_loss(dec(tok(q["after"])), ev.truth_post)
    predicted = readout_loss(dec(tok(q["prior"])), ev.truth_post)
    dyn = core.kl(q["after"].detach(), q["prior"]).clamp_min(FREE_NATS).mean()
    rep = core.kl(q["after"], q["prior"].detach()).clamp_min(FREE_NATS).mean()
    # Two presses without an observation in between; the second press points at the
    # chain's first frame, so its action reveals nothing about the unobserved middle.
    s0, s1 = split_chain(batch.chain)
    e0, e1 = model.events(prepared, s0), model.events(prepared, s1, anchor=s0)
    ones = torch.ones(len(s0), device=device)
    z0, zv0 = pick(z, s0.episode.to(device))
    z1, zv1 = pick(z, s1.episode.to(device))
    p1 = core.predict(observe_frame(core, e0.pre), e0.action, dt=ones, z=z0, z_valid=zv0)
    p2 = core.predict(p1, e1.action, dt=ones, z=z1, z_valid=zv1)
    chain = F.cross_entropy(lamp_logits(model, p2, e1.machines).reshape(-1, 2), s1.post.to(device).reshape(-1))
    total = recon + predicted + chain + DYN * dyn + REP * rep
    parts = dict(recon=recon, predicted=predicted, chain=chain, kl_dyn=dyn, kl_rep=rep)
    if args.swap_weight:
        # Swap term against "predict ignores Z": the episode's own rule context must
        # explain the target lamp better than the neighbouring episode's context.
        i, m = q["i"], t.machine.to(device)
        target = t.outcome.to(device)
        own = F.cross_entropy(lamp_logits(model, q["prior"], ev.machines)[i, m], target, reduction="none")
        zs, zvs = z[0].roll(1, 0), None if z[1] is None else z[1].roll(1, 0)
        swapped_q = press(model, prepared, batch.query, (zs, zvs), device)
        other = F.cross_entropy(lamp_logits(model, swapped_q["prior"], ev.machines)[i, m], target, reduction="none")
        swap = F.softplus(own - other).mean()
        total = total + args.swap_weight * swap
        parts["swap"] = swap
    return total, {k: float(v.detach()) for k, v in parts.items()}


# ---------------------------------------------------------------- evaluation


@torch.no_grad()
def evaluate(model, batch, device, floors, reader="code"):
    """ν of the prior's target-lamp prediction per control, plus copy, posterior readout
    and lamp accuracy split by changed and unchanged machines (one step and two-step chain)."""
    model.eval()
    prepared = model.prepare(batch, device)
    out = {}
    for control in ("full", "empty", "swapped"):
        context = rule_codes(model, prepared, batch, device, control, reader)
        q = press(model, prepared, batch.query, context, device)
        t, i, ev = q["t"], q["i"], q["ev"]
        m = t.machine.to(device)
        lamps = lamp_logits(model, q["prior"], ev.machines)
        p = lamps.softmax(-1)[i, m, 1].cpu()
        truth, s = t.outcome, t.pre[torch.arange(len(t)), t.machine]
        rows = [dict(family=batch.rules[int(e)].family, group=int(e), truth=int(y), s=int(v), p=float(pp))
                for e, y, v, pp in zip(t.episode, truth, s, p)]
        nu = nu_from_rows(rows, floors)
        out[control] = dict(nu={k: v for k, v in nu.items() if k != "groups_without_both_classes"},
                            groups_without_both_classes=nu["groups_without_both_classes"],
                            accuracy=float(((p >= 0.5).long() == truth).float().mean()))
        if control == "full":
            copy = nu_from_rows([dict(r, p=float(r["s"])) for r in rows], floors)
            out["copy"] = dict(nu={k: v for k, v in copy.items() if k != "groups_without_both_classes"},
                               accuracy=float((s == truth).float().mean()))
            correct = (lamps.argmax(-1).cpu() == t.post)  # [n,2]
            changed = t.post != t.pre
            out["prior_lamp_changed"] = float(correct[changed].float().mean()) if changed.any() else float("nan")
            out["prior_lamp_unchanged"] = float(correct[~changed].float().mean())
            post = lamp_logits(model, q["after"], ev.machines).argmax(-1).cpu()
            out["posterior_lamp_accuracy"] = float((post == t.post).float().mean())
            s0, s1 = split_chain(batch.chain)
            e0, e1 = model.events(prepared, s0), model.events(prepared, s1, anchor=s0)
            ones = torch.ones(len(s0), device=device)
            z0, zv0 = pick(context, s0.episode.to(device))
            z1, zv1 = pick(context, s1.episode.to(device))
            p2 = model.core.predict(model.core.predict(observe_frame(model.core, e0.pre), e0.action, dt=ones,
                                                       z=z0, z_valid=zv0), e1.action, dt=ones, z=z1, z_valid=zv1)
            chain = lamp_logits(model, p2, e1.machines).argmax(-1).cpu()
            out["chain_lamp_accuracy"] = float((chain == s1.post).float().mean())
            out["chain_copy_accuracy"] = float((s0.pre == s1.post).float().mean())
    model.train()
    return out


def mean_nu(result):
    values = list(result["nu"].values())
    return sum(values) / len(values) if values else float("nan")


# ---------------------------------------------------------------- recipe


def pools(args):
    family = args.family
    train = rw.family_ladder(family, args.rules) if args.rules else tuple(
        r for r in rw.split_rules()["train"] if r.family == family)
    if args.rule_repeats:  # sampling weights per training rule, e.g. 7 1 1 1 (frequent entry rule)
        if len(args.rule_repeats) != len(train):
            raise ValueError("--rule-repeats needs one count per training rule")
        train = tuple(r for r, n in zip(train, args.rule_repeats) for _ in range(n))
    heldout = tuple(r for r in rw.split_rules()["validation"] if r.family == family)
    return train, heldout


def sample(g, rules, kinds, args, episodes):
    return rw.sample_episodes(g, rules, kinds, episodes=episodes, support=tuple(args.support),
                              queries=args.queries)


def finish(runner, result, complete):
    atomic_json(runner.path / "result.json", result)
    status = "completed" if complete else "paused"
    runner.status(status, "pending")
    try:
        write_report(runner.path)
    except Exception as error:
        runner.status(status, "failed", str(error))
        raise
    runner.status(status, "completed")


def train_perception(args):
    """S4: train slot perception from scratch; qualify on validation kinds."""
    seed_everything(args.seed)
    device = torch.device(args.device)
    model = PerceptionModel(PERCEPTION_SIZES[args.size]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.perception_lr, weight_decay=0.01)
    settings = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
                if k in ("stage", "size", "seed", "device", "updates", "batch", "perception_lr", "qualify_scenes")}
    data = dict(generator="rule_world", manifest=rw.manifest(), train_kinds=list(rw.KIND_SPLIT["train"]),
                qualification_kinds=list(rw.KIND_SPLIT["validation"]), thresholds=PERCEPTION_THRESHOLDS)
    runner = Run(args.output, settings=settings, data=data, recipe=__file__, model=model,
                 optimizer=optimizer, device=device, resume=args.resume is not None)
    started = time.monotonic()
    while runner.step < args.updates and (time.monotonic() - started) < 60 * args.max_minutes:
        scenes, lamps, rgb, entity = rw.sample_frames(runner.sampler, rw.KIND_SPLIT["train"], args.batch)
        percept = model.perception(rgb.to(device))
        loss, metrics = perception_loss(percept, rgb.to(device), entity.to(device), scenes.attrs, lamps)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = float(nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optimizer.step()
        runner.step += 1
        if runner.step % args.log_every == 0 or runner.step == args.updates:
            runner.log(dict(step=runner.step, split="train", grad_norm=norm, **metrics))
        if runner.step % args.save_every == 0:
            runner.save()
    runner.save()
    complete = runner.step >= args.updates
    model.eval()
    result = dict(step=runner.step, complete=complete,
                  parameters=sum(p.numel() for p in model.parameters()))
    for name, kinds, seed in (("train_kinds", rw.KIND_SPLIT["train"], 11),
                              ("validation_kinds", rw.KIND_SPLIT["validation"], 13)):
        g = torch.Generator().manual_seed(args.seed + seed)
        metrics = perception_metrics(model.perception, g, kinds, args.qualify_scenes, device)
        result[name] = dict(metrics=metrics, qualification=qualification(metrics))
        runner.log(dict(step=runner.step, split=f"evaluation_{name}",
                        **{k: v for k, v in metrics.items() if isinstance(v, float)},
                        attribute_min=min(metrics["attribute_accuracy"])))
    result["qualified"] = result["validation_kinds"]["qualification"]["passed"]
    if device.type == "cuda":
        result["torch_max_reserved_gib"] = torch.cuda.max_memory_reserved(device) / 2**30
    finish(runner, result, complete)
    return result


def train(args):
    seed_everything(args.seed)
    device = torch.device(args.device)
    size = SIZES[args.size]
    if args.stage == "pixel":
        if args.perception is None:
            raise ValueError("--stage pixel needs --perception (a qualified S4 checkpoint)")
        model = PixelModel(size, PERCEPTION_SIZES[args.size], args.perception, tied=not args.untied,
                           continuous_only=args.continuous_only).to(device)
    else:
        model = SymbolicModel(size, tied=not args.untied, continuous_only=args.continuous_only).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    train_rules, heldout_rules = pools(args)
    settings = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
                if k not in ("output", "resume", "check", *RUN_CONTROL)}
    data = dict(generator="rule_world", manifest=rw.manifest(), train_rules=[r.key() for r in train_rules],
                heldout_rules=[r.key() for r in heldout_rules],
                perception_sha256=None if args.stage != "pixel" else file_hash(args.perception))
    runner = Run(args.output, settings=settings, data=data, recipe=__file__, model=model,
                 optimizer=optimizer, device=device, resume=args.resume is not None)
    floors = rw.floors()
    kinds = rw.KIND_SPLIT
    started = time.monotonic()
    while runner.step < args.updates and (time.monotonic() - started) < 60 * args.max_minutes:
        batch = sample(runner.sampler, train_rules, kinds["train"], args, args.episodes)
        loss, parts = losses(model, batch, device, args)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = float(nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optimizer.step()
        runner.step += 1
        if runner.step % args.log_every == 0 or runner.step == args.updates:
            runner.log(dict(step=runner.step, split="train", loss=float(loss.detach()), grad_norm=norm, **parts))
        if runner.step % args.save_every == 0:
            runner.save()
    runner.save()
    complete = runner.step >= args.updates
    g = torch.Generator().manual_seed(args.seed + 7)  # fixed evaluation pools
    params = sum(p.numel() for p in model.core.blocks.parameters())
    result = dict(step=runner.step, complete=complete, core_block_parameters=params,
                  total_parameters=sum(p.numel() for p in model.parameters()),
                  train_rules=evaluate(model, sample(g, sorted(set(train_rules), key=lambda r: r.key()), kinds["train"], args, args.eval_episodes), device, floors, args.reader))
    if heldout_rules:
        result["heldout_rules"] = evaluate(model, sample(g, heldout_rules, kinds["validation"], args, args.eval_episodes),
                                           device, floors, args.reader)
    for pool in ("train_rules", "heldout_rules"):
        if pool in result:
            values = {f"{pool}/{k}_nu": mean_nu(v) for k, v in result[pool].items() if isinstance(v, dict) and "nu" in v}
            # Undefined ν (no group with both classes) stays visible in result.json only.
            runner.log(dict(step=runner.step, split=f"evaluation_{pool}",
                            **{k: v for k, v in values.items() if math.isfinite(v)}))
    if device.type == "cuda":
        result["torch_max_reserved_gib"] = torch.cuda.max_memory_reserved(device) / 2**30
    finish(runner, result, complete)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stage", choices=("symbolic", "perception", "pixel"), default="symbolic")
    parser.add_argument("--perception", type=Path, help="pixel stage: frozen S4 checkpoint")
    parser.add_argument("--size", choices=tuple(SIZES), default="full")
    parser.add_argument("--batch", type=int, default=32, help="perception stage: frames per update")
    parser.add_argument("--perception-lr", type=float, default=4e-4)
    parser.add_argument("--qualify-scenes", type=int, default=1024)
    parser.add_argument("--family", choices=rw.FAMILIES, default="relation")
    parser.add_argument("--rules", type=int, default=4, help="ladder size; 0 = every training rule of the family")
    parser.add_argument("--updates", type=int, default=6000)
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--queries", type=int, default=32)
    parser.add_argument("--support", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--eval-episodes", type=int, default=96)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--rule-repeats", type=int, nargs="+", default=None)
    parser.add_argument("--reader", choices=("code", "evidence"), default="code",
                        help="predict reads the induced Z or the uncompressed support events")
    parser.add_argument("--swap-weight", type=float, default=0.0)
    parser.add_argument("--untied", action="store_true", help="control: one block stack per operation (E2)")
    parser.add_argument("--continuous-only", action="store_true", help="control: no categorical code (E3)")
    parser.add_argument("--seed", type=int, default=1101)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-minutes", type=float, default=240.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--check", action="store_true", help=argparse.SUPPRESS)
    args = resume_arguments(parser, parser.parse_args(argv))
    if args.resume is not None:
        args.output = args.resume
    if args.output is None:
        parser.error("--output is required")
    return train_perception(args) if args.stage == "perception" else train(args)


if __name__ == "__main__":
    main()
