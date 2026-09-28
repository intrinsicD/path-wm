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

  python -m experiments.core --output runs/core/relation_l4 --family relation --rules 4
  python -m experiments.core --resume runs/core/relation_l4
"""

import argparse
import math
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from pathwm.data import rule_world as rw
from pathwm.evaluation.report import write_report
from pathwm.evaluation.rules import nu_from_rows
from pathwm.io import Run, atomic_json, resume_arguments, seed_everything
from pathwm.models.core import CoreConfig, SharedCore
from pathwm.models.symbolic import SymbolicDecoder, SymbolicEncoder, readout_loss, targets

# full = the decided configuration (E4); check = declared downscale for CPU tests only.
SIZES = dict(full=dict(width=128, heads=4, blocks=2, rounds=2),
             check=dict(width=32, heads=4, blocks=2, rounds=1))
FREE_NATS, DYN, REP = 1.0, 1.0, 0.1
RUN_CONTROL = ("max_minutes", "log_every", "save_every")  # pacing only; not part of the run identity


class SymbolicModel(nn.Module):
    def __init__(self, size, tied=True, continuous_only=False):
        super().__init__()
        w = size["width"]
        self.encoder = SymbolicEncoder(w)
        self.core = SharedCore(CoreConfig(width=w, heads=size["heads"], blocks=size["blocks"],
                                          rounds=size["rounds"], evidence_width=w, action_width=3 * w,
                                          tied=tied, continuous_only=continuous_only))
        self.decoder = SymbolicDecoder(w)


# ---------------------------------------------------------------- one batch of press events


def events(model, scenes, t, device):
    """Perceived tokens and lamp-free action tokens for Transitions `t`."""
    scn = scenes.select(t.scene)
    enc = model.encoder
    pre, post, ident = enc(scn, t.pre.to(device)), enc(scn, t.post.to(device)), enc.identity(scn)
    i = torch.arange(len(t), device=device)
    m, a, b = (t.machine.to(device) + 1), (t.a.to(device) + 3), (t.b.to(device) + 3)
    action = torch.cat((ident[i, m], ident[i, a], ident[i, b]), -1)
    return scn, pre, post, action, i, m


def observe_frame(core, tokens, prior=None):
    valid = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
    prior = core.bind(tokens, valid) if prior is None else prior
    return core.observe(prior, tokens, valid)


def rule_codes(model, batch, device, control="full", reader="code"):
    """What `predict` reads per episode: (Z [E,K,D], None) from `induce`, or with
    reader="evidence" the uncompressed support events (tokens [E,N,D], valid [E,N]).
    Controls replace the support: empty, or swapped with the neighbouring episode's."""
    core, e = model.core, len(batch.rules)
    s = batch.support
    if control == "empty" or len(s) == 0:
        none = torch.zeros(e, 1, core.config.width, device=device)
        invalid = torch.zeros(e, 1, dtype=torch.bool, device=device)
        return (core.induce(none, invalid), None) if reader == "code" else (none, invalid)
    _, pre, post, action, i, m = events(model, batch.scenes, s, device)
    tokens = core.event(pre[i, m], action, post[i, m])
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


def press(model, batch, t, context, device):
    """Observe the pre frame, predict the press reading the rule context, observe the post frame."""
    core = model.core
    scn, pre, post, action, i, m = events(model, batch.scenes, t, device)
    before = observe_frame(core, pre)
    ones = torch.ones(len(t), device=device)
    z, zv = pick(context, t.episode.to(device))
    prior = core.predict(before, action, dt=ones, z=z, z_valid=zv)
    after = observe_frame(core, post, prior)
    return dict(scn=scn, before=before, prior=prior, after=after, i=i, m=m, t=t)


def lamp_logits(model, state):
    return model.decoder(model.core.tokens(state)[:, 1:3])["lamp"]  # [n,2 machines,2]


def losses(model, batch, device, args):
    core, dec = model.core, model.decoder
    z = rule_codes(model, batch, device, reader=args.reader)
    q = press(model, batch, batch.query, z, device)
    scn, t = q["scn"], q["t"]
    tok = lambda s: core.tokens(s)[:, :7]
    recon = readout_loss(dec(tok(q["before"])), targets(scn, t.pre)) + \
        readout_loss(dec(tok(q["after"])), targets(scn, t.post))
    predicted = readout_loss(dec(tok(q["prior"])), targets(scn, t.post))
    dyn = core.kl(q["after"].detach(), q["prior"]).clamp_min(FREE_NATS).mean()
    rep = core.kl(q["after"], q["prior"].detach()).clamp_min(FREE_NATS).mean()
    # Two presses without an observation in between (chain rows: step 0, step 1).
    c = batch.chain
    first, second = c.step == 0, c.step == 1
    s0, s1 = rw.Transitions(*(v[first] for v in vars(c).values())), rw.Transitions(*(v[second] for v in vars(c).values()))
    _, pre0, _, a0, _, _ = events(model, batch.scenes, s0, device)
    _, _, _, a1, i1, m1 = events(model, batch.scenes, s1, device)
    ones = torch.ones(len(s0), device=device)
    z0, zv0 = pick(z, s0.episode.to(device))
    z1, zv1 = pick(z, s1.episode.to(device))
    p1 = core.predict(observe_frame(core, pre0), a0, dt=ones, z=z0, z_valid=zv0)
    p2 = core.predict(p1, a1, dt=ones, z=z1, z_valid=zv1)
    chain = F.cross_entropy(lamp_logits(model, p2).reshape(-1, 2), s1.post.to(device).reshape(-1))
    total = recon + predicted + chain + DYN * dyn + REP * rep
    parts = dict(recon=recon, predicted=predicted, chain=chain, kl_dyn=dyn, kl_rep=rep)
    if args.swap_weight:
        # Swap term against "predict ignores Z": the episode's own rule context must
        # explain the target lamp better than the neighbouring episode's context.
        i, m = q["i"], q["m"]
        target = t.outcome.to(device)
        own = F.cross_entropy(lamp_logits(model, q["prior"])[i, m - 1], target, reduction="none")
        zs, zvs = z[0].roll(1, 0), None if z[1] is None else z[1].roll(1, 0)
        swapped_q = press(model, batch, batch.query, (zs, zvs), device)
        other = F.cross_entropy(lamp_logits(model, swapped_q["prior"])[i, m - 1], target, reduction="none")
        swap = F.softplus(own - other).mean()
        total = total + args.swap_weight * swap
        parts["swap"] = swap
    return total, {k: float(v.detach()) for k, v in parts.items()}


# ---------------------------------------------------------------- evaluation


@torch.no_grad()
def evaluate(model, batch, device, floors, reader="code"):
    """ν of the prior's target-lamp prediction per control, plus copy and posterior readout."""
    model.eval()
    out = {}
    for control in ("full", "empty", "swapped"):
        q = press(model, batch, batch.query, rule_codes(model, batch, device, control, reader), device)
        t, i, m = q["t"], q["i"], q["m"]
        p = lamp_logits(model, q["prior"]).softmax(-1)[i, m - 1, 1].cpu()
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
            post = lamp_logits(model, q["after"]).argmax(-1).cpu()
            out["posterior_lamp_accuracy"] = float((post == t.post).float().mean())
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


def train(args):
    seed_everything(args.seed)
    device = torch.device(args.device)
    size = SIZES[args.size]
    model = SymbolicModel(size, tied=not args.untied, continuous_only=args.continuous_only).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    train_rules, heldout_rules = pools(args)
    settings = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
                if k not in ("output", "resume", "check", *RUN_CONTROL)}
    data = dict(generator="rule_world", manifest=rw.manifest(), train_rules=[r.key() for r in train_rules],
                heldout_rules=[r.key() for r in heldout_rules])
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
            runner.log(dict(step=runner.step, split=f"evaluation_{pool}", **{f"{pool}/{k}_nu": mean_nu(v) for k, v in result[pool].items()
                                                 if isinstance(v, dict) and "nu" in v}))
    if device.type == "cuda":
        result["torch_max_reserved_gib"] = torch.cuda.max_memory_reserved(device) / 2**30
    finish(runner, result, complete)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--size", choices=tuple(SIZES), default="full")
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
    return train(args)


if __name__ == "__main__":
    main()
