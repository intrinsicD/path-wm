"""Edit this recipe to compare native and recurrent output reads on paired tasks.

Example: python experiments/modality_readout.py --stage suite --output runs/readout
The suite owns core, frozen-output and joint runs; no general modality claims.
"""

import argparse
from pathlib import Path
import copy
import json
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from pathwm.data.modality_readout import KINDS, MODES, dataset, observations
from pathwm.models.agent import Thinker, ActionHead, ErrorMonitor
from pathwm.models.belief import BeliefAgent, BeliefCorrection, BeliefDynamics
from pathwm.models.hybrid_memory import HybridMemory
from pathwm.models.multiscale import (
    MultiScaleImageEncoder,
    MultiScaleAudioEncoder,
    MultiScaleTextEncoder,
)
from pathwm.models.modalities import ImageDecoder, AudioDecoder, TextDecoder
from pathwm.models.readout import RecurrentOutputAdapter, TemporalImageDecoder
from pathwm.io import (
    Run,
    seed_everything,
    atomic_json,
    atomic_torch,
    file_hash,
    state_hash,
    evaluation_mode,
    training_mode,
    digest,
)
from pathwm.evaluation.report import write_report
from pathwm.evaluation.modality_readout import evaluate_outputs, save_panels, aggregate

VARIANTS = ("native", "adapter1", "adapter2", "adapter4")


class Core(nn.Module):
    def __init__(self, width=24):
        super().__init__()
        encoders = {
            "text": MultiScaleTextEncoder(width, code_width=8, levels=3),
            "image": MultiScaleImageEncoder(width, code_width=8, levels=3),
            "audio": MultiScaleAudioEncoder(64, width, code_width=8, levels=3),
            "video": MultiScaleImageEncoder(width, video=True, code_width=8, levels=3),
        }
        self.agent = BeliefAgent(
            width=width,
            context_tokens=4,
            latent_groups=4,
            latent_codes=8,
            evidence_tokens=4,
            time_unit="seconds",
            encoders=encoders,
            decoders={},
            updater=BeliefCorrection(width, 4, 8, 2),
            dynamics=BeliefDynamics(width, 4, 8, 2),
            thinker=Thinker(width),
            memory=HybridMemory(width, recent=2, block=2, blocks=1, latent_codes=8),
            action_head=ActionHead(width),
            monitor=ErrorMonitor(width),
        )
        self.factor_head = nn.Linear(width * 12, 8)

    def forward(self, inputs, *, trace=None):
        batch = len(next(iter(inputs.values())).values)
        state = self.agent.initial_state(batch, session_id="readout-task")
        state = self.agent.observe(state, inputs, time=4.0, trace=trace)
        return self.agent.think(state, steps=2, trace=trace).tokens

    def factors(self, tokens):
        return self.factor_head(tokens.flatten(1)).split((3, 3, 2), dim=-1)


class Outputs(nn.Module):
    def __init__(self, variant, width=24):
        super().__init__()
        self.variant = variant
        self.decoders = nn.ModuleDict(
            {
                "text": TextDecoder(width),
                "image": ImageDecoder(width, 16),
                "audio": AudioDecoder(width, 192),
                "video": TemporalImageDecoder(width, 16),
            }
        )
        # Decoder initialization and caller RNG stay identical across variants.
        with torch.random.fork_rng():
            self.adapters = nn.ModuleDict(
                {
                    k: (
                        nn.Identity()
                        if variant == "native"
                        else RecurrentOutputAdapter(width, iterations=int(variant[-1]))
                    )
                    for k in KINDS
                }
            )

    def prepare(self, kind, tokens):
        return self.adapters[kind](tokens)

    def forward(self, kind, tokens, prefix=None):
        context = self.prepare(kind, tokens)
        if kind == "text":
            return self.decoders[kind](context, prefix)
        if kind == "video":
            return self.decoders[kind](context, torch.arange(4.0, device=tokens.device))
        return self.decoders[kind](context)

    def generate(self, tokens, max_tokens=28):
        return self.decoders["text"].generate(self.prepare("text", tokens), max_tokens)


class Model(nn.Module):
    def __init__(self, variant):
        super().__init__()
        self.core, self.outputs = Core(), Outputs(variant)


def target_batch(data, indices, device):
    return {k: v[indices].to(device) for k, v in data["targets"].items()}


def scales(data):
    return {
        k: float((data["targets"][k] - data["targets"][k].mean(0)).square().mean())
        for k in KINDS
        if k != "text"
    }


def oracle_states(populations):
    """Positive control only: explicit target facts, never an agent input path."""
    result = {}
    for split, data in populations.items():
        facts = data["targets"]["factors"]
        vector = torch.cat(
            [F.one_hot(facts[:, i], n) for i, n in enumerate((3, 3, 2))], 1
        ).float()
        vector = F.pad(vector, (0, 16))
        result[split] = {"all": vector[:, None].expand(-1, 12, -1).clone()}
    return result


def output_objective(model, tokens, target, kinds, normalizers):
    losses = {}
    for k in kinds:
        out = model.outputs(k, tokens, target["text"][:, :-1] if k == "text" else None)
        losses[k] = (
            F.cross_entropy(
                out.flatten(0, 1), target["text"][:, 1:].flatten(), ignore_index=0
            )
            / np.log(259)
            if k == "text"
            else F.mse_loss(out, target[k]) / normalizers[k]
        )
    # Sum retains the same per-branch gradient scale as isolated output fitting.
    return sum(losses.values()), {k: float(v.detach()) for k, v in losses.items()}


@torch.no_grad()
def cache_states(model, populations, seed, device):
    cache, scores = {}, []
    with evaluation_mode(model):
        for si, (split, data) in enumerate(populations.items()):
            cache[split] = {}
            for mi, mode in enumerate(MODES):
                # Cache actual sampled states, not target-derived latent vectors.
                torch.manual_seed(seed + 1000 + si * 100 + mi)
                tokens = model.core(observations(data, mode, device=device))
                cache[split][mode] = tokens.cpu()
                pred = torch.stack(
                    [p.argmax(-1) for p in model.core.factors(tokens)], 1
                ).cpu()
                correct = pred == data["targets"]["factors"]
                scores.append(
                    dict(
                        split=split,
                        input_mode=mode,
                        factor_accuracy=correct.float().mean(0).tolist(),
                        all_correct=float(correct.all(1).float().mean()),
                    )
                )
            if split in ("seen", "heldout"):
                for omitted in ("image", "audio", "video"):
                    torch.manual_seed(seed + 1000 + si * 100 + 5)
                    cache[split]["without_" + omitted] = model.core(
                        observations(data, "complementary", device=device, omit=omitted)
                    ).cpu()
    return cache, scores


def export_data(path, populations):
    arrays = {}
    for split, data in populations.items():
        for key, v in data["targets"].items():
            arrays[f"{split}.target.{key}"] = v.numpy()
        for group in ("inputs", "complementary"):
            for k, obs in data[group].items():
                for field in ("values", "times", "valid"):
                    v = getattr(obs, field)
                    if v is not None:
                        arrays[f"{split}.{group}.{k}.{field}"] = v.numpy()
    np.savez_compressed(path, **arrays)


def perform(args):
    seed_everything(args.seed)
    device = torch.device(args.device)
    populations = {
        s: dataset(s, args.seed) for s in ("train", "validation", "seen", "heldout")
    }
    data = populations["train"]
    model = Model(args.variant).to(device)
    initial_decoders = state_hash(model.outputs.decoders)
    core_source = None
    cache = None
    if args.stage != "core":
        if args.core is None:
            raise ValueError("Output training needs a factor-trained --core run")
        core_source = file_hash(args.core / "last.pt")
        checkpoint = torch.load(
            args.core / "last.pt", map_location=device, weights_only=True
        )
        model.core.load_state_dict(
            {
                k.removeprefix("core."): v
                for k, v in checkpoint["model"].items()
                if k.startswith("core.")
            }
        )
        cache = torch.load(args.core / "states.pt", weights_only=True)
        if args.stage == "oracle":
            cache = oracle_states(populations)
    model.requires_grad_(False)
    kinds = KINDS if args.stage == "joint" else (args.modality,)
    if args.stage in ("core", "joint"):
        model.core.requires_grad_(True)
        model.core.agent.action_head.requires_grad_(False)
        model.core.agent.monitor.requires_grad_(False)
    if args.stage == "joint":
        model.core.factor_head.requires_grad_(False)
    if args.stage != "core":
        for k in kinds:
            model.outputs.decoders[k].requires_grad_(True)
            model.outputs.adapters[k].requires_grad_(True)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=0.003
    )
    settings = dict(
        seed=args.seed,
        stage=args.stage,
        variant=args.variant,
        modality=args.modality,
        steps=args.steps,
        batch=24,
        width=24,
        outer_iterations=2,
        objective="factor-CE"
        if args.stage == "core"
        else "sum normalized per-output losses",
        initial_decoder_sha256=initial_decoders,
        core_source_sha256=core_source,
        context_source="explicit ground-truth factors (oracle control only)"
        if args.stage == "oracle"
        else "actual learned core states",
    )
    if args.stage != "core":
        settings["core"] = str(args.core.resolve())
    data_identity = dict(
        generator=file_hash("pathwm/data/modality_readout.py"),
        seed=args.seed,
        splits={
            k: dict(
                count=len(v["ids"]),
                ids=digest(v["ids"]),
                factors=digest(v["targets"]["factors"].tolist()),
            )
            for k, v in populations.items()
        },
        frozen_cache=None
        if args.stage in ("core", "oracle")
        else file_hash(args.core / "states.pt"),
    )
    run = Run(
        args.output,
        settings=settings,
        data=data_identity,
        recipe=__file__,
        model=model,
        optimizer=optimizer,
        device=device,
        resume=args.resume,
    )
    before_core = state_hash(model.core)
    frozen_before = {
        n: p.detach().clone()
        for n, p in model.named_parameters()
        if not p.requires_grad
    }
    normalizers = scales(data)
    end = (
        min(args.steps, args.stop_after) if args.stop_after is not None else args.steps
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    training_mode(model)
    try:
        while run.step < end:
            mode = "all" if args.stage == "oracle" else MODES[run.step % len(MODES)]
            indices = run.sample(len(data["ids"]), 24)
            wanted = target_batch(data, indices, device)
            optimizer.zero_grad(set_to_none=True)
            tokens = (
                cache["train"][mode][indices].to(device)
                if args.stage in ("frozen", "oracle")
                else model.core(observations(data, mode, indices, device))
            )
            if args.stage == "core":
                loss = (
                    sum(
                        F.cross_entropy(p, wanted["factors"][:, i])
                        for i, p in enumerate(model.core.factors(tokens))
                    )
                    / 3
                )
                metrics = {}
            else:
                loss, metrics = output_objective(
                    model, tokens, wanted, kinds, normalizers
                )
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training objective")
            loss.backward()
            norm = nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 5.0
            )
            if not torch.isfinite(norm):
                raise ValueError("Nonfinite training gradient")
            optimizer.step()
            run.step += 1
            run.log(
                dict(
                    step=run.step,
                    split="train",
                    loss=float(loss.detach()),
                    input_mode=mode,
                    gradient_norm=float(norm),
                    **metrics,
                )
            )
            if run.step % 64 == 0:
                run.save()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        run.save()
        for n, p in model.named_parameters():
            if n in frozen_before and not torch.equal(p, frozen_before[n]):
                raise AssertionError(f"Frozen parameter changed: {n}")
        if args.stage in ("frozen", "oracle"):
            assert state_hash(model.core) == before_core
        resource = dict(
            training_seconds_this_invocation=elapsed,
            parameters=sum(p.numel() for p in model.parameters()),
            trainable_parameters=sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
            adapter_parameters=sum(
                p.numel() for p in model.outputs.adapters.parameters()
            ),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else None,
            peak_reserved_bytes=torch.cuda.max_memory_reserved(device)
            if device.type == "cuda"
            else None,
            extra_attention_calls_per_output=0
            if args.variant == "native"
            else 2 * int(args.variant[-1]),
            core_sha256=state_hash(model.core),
            frozen_preserved=True,
            core_parameters=sum(p.numel() for p in model.core.parameters()),
            decoder_parameters={
                k: sum(p.numel() for p in model.outputs.decoders[k].parameters())
                for k in KINDS
            },
            per_adapter_parameters={
                k: sum(p.numel() for p in model.outputs.adapters[k].parameters())
                for k in KINDS
            },
        )
        if end < args.steps:
            atomic_json(
                run.path / "readout.json",
                dict(
                    stage=args.stage,
                    variant=args.variant,
                    modality=args.modality,
                    partial=True,
                    metrics=[],
                    resources=resource,
                ),
            )
            run.status("paused", "pending")
            write_report(run.path)
            return run.path
        # Result completion is separate from rendering; preserve failures.
        with evaluation_mode(model), torch.no_grad():
            if args.stage in ("core", "joint"):
                cache, core_scores = cache_states(model, populations, args.seed, device)
            else:
                core_scores = None
            if args.stage == "core":
                atomic_torch(run.path / "states.pt", cache)
                export_data(run.path / "data.npz", populations)
                metrics, arrays = [], {}
            else:
                metrics, arrays = evaluate_outputs(
                    model.outputs, cache, populations, kinds, normalizers, device
                )
                np.savez_compressed(run.path / "outputs.npz", **arrays)
                save_panels(run.path, arrays, metrics)
            core_gate = (
                None
                if core_scores is None
                else all(
                    min(r["factor_accuracy"]) >= 0.9
                    for r in core_scores
                    if r["split"] == "seen" and r["input_mode"] != "complementary"
                )
            )
            payload = dict(
                stage=args.stage,
                variant=args.variant,
                modality=args.modality,
                seed=args.seed,
                metrics=metrics,
                core_scores=core_scores,
                core_gate=core_gate,
                resources=resource,
                normalizers=normalizers,
                partial=False,
            )
            atomic_json(run.path / "readout.json", payload)
        run.status("complete", "pending")
        write_report(run.path)
        print(
            json.dumps(
                dict(
                    path=str(run.path),
                    stage=args.stage,
                    variant=args.variant,
                    modality=args.modality,
                    seconds=round(elapsed, 2),
                    core_gate=core_gate,
                )
            ),
            flush=True,
        )
        return run.path
    except BaseException as exc:
        prior = json.loads((run.path / "status.json").read_text())
        run.status(
            prior["result"] if prior["result"] in ("complete", "paused") else "failed",
            "failed",
            str(exc),
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("core", "frozen", "joint", "oracle", "suite"),
        default="suite",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7201)
    parser.add_argument("--variant", choices=VARIANTS, default="native")
    parser.add_argument("--modality", choices=KINDS, default="text")
    parser.add_argument("--core", type=Path)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--steps", type=int)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.steps is not None and args.steps < 1:
        parser.error("Steps must be positive")
    if args.stop_after is not None and args.stop_after < 1:
        parser.error("Stop-after must be positive")
    if args.stage != "suite":
        args.steps = (
            args.steps
            or {"core": 768, "frozen": 256, "joint": 384, "oracle": 256}[args.stage]
        )
        perform(args)
        return
    if args.resume or args.stop_after:
        parser.error("Resume a specific child run, not the suite")
    args.output.mkdir(parents=True, exist_ok=False)
    for seed in (args.seed, args.seed + 1):
        a = copy.copy(args)
        a.seed = seed
        a.stage = "core"
        a.steps = args.steps or 768
        a.output = args.output / f"seed{seed}" / "core"
        a.variant = "native"
        core = perform(a)
        for variant in VARIANTS:
            for kind in KINDS:
                a = copy.copy(args)
                a.seed = seed
                a.stage = "frozen"
                a.steps = args.steps or 256
                a.variant = variant
                a.modality = kind
                a.core = core
                a.output = args.output / f"seed{seed}" / f"frozen_{variant}_{kind}"
                perform(a)
            a = copy.copy(args)
            a.seed = seed
            a.stage = "joint"
            a.steps = args.steps or 384
            a.variant = variant
            a.core = core
            a.output = args.output / f"seed{seed}" / f"joint_{variant}"
            perform(a)
    aggregate(args.output)


if __name__ == "__main__":
    main()
