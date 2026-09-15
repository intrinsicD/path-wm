"""Edit this recipe to compare native and recurrent output reads on paired tasks.

Example: python experiments/modality_readout.py --stage suite --output runs/readout
The suite owns core, frozen-output and joint runs; no general modality claims.
"""

import argparse
from dataclasses import replace
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
from pathwm.evaluation.modality_readout import (
    evaluate_outputs,
    save_panels,
    aggregate,
    capture_readout_stages,
    fit_factor_probe,
    predict_factor_probe,
    save_stage_panel,
    summarize_repair,
)

VARIANTS = ("native", "adapter1", "adapter2", "adapter4")


class Core(nn.Module):
    def __init__(self, width=24, *, belief_readout="sampled"):
        super().__init__()
        if belief_readout not in ("sampled", "probabilities"):
            raise ValueError("Unknown working belief readout")
        self.belief_readout = belief_readout
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

    def forward(
        self,
        inputs,
        *,
        trace=None,
        return_state=False,
        posterior_features=None,
        temperature=1.0,
    ):
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("Posterior temperature must be finite and positive")
        batch = len(next(iter(inputs.values())).values)
        state = self.agent.initial_state(batch, session_id="readout-task")

        def correction_logits(module, arguments, logits):
            if posterior_features is not None:
                posterior_features["raw_logits"] = logits
            return logits if temperature == 1.0 else logits / temperature

        hook = None
        try:
            if posterior_features is not None or temperature != 1.0:
                hook = self.agent.updater.head.register_forward_hook(correction_logits)
            state = self.agent.observe(state, inputs, time=4.0, trace=trace)
        finally:
            if hook is not None:
                hook.remove()
        workspace = state
        if self.belief_readout == "probabilities":
            # Experimental working context only. The returned/stored categorical
            # state and its sampling RNG remain exactly native; no soft IDs.
            world = (
                state.h
                + self.agent.readout(state.logits.softmax(-1).flatten(1))[:, None]
            )
            workspace = replace(
                state,
                tokens=torch.cat(
                    (world, state.tokens[:, self.agent.groups["world"] :]), 1
                ),
            )
        tokens = self.agent.think(workspace, steps=2, trace=trace).tokens
        return (tokens, state) if return_state else tokens

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
    def __init__(self, variant, *, posterior_aux=False):
        super().__init__()
        self.core, self.outputs = Core(), Outputs(variant)
        # Diagnostic supervision only; no inference read or RNG/init change.
        with torch.random.fork_rng():
            self.posterior_head = nn.Linear(32, 8) if posterior_aux else None


def load_initial(model, directory, device):
    checkpoint = torch.load(
        directory / "last.pt", map_location=device, weights_only=True
    )
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    allowed = {"posterior_head.weight", "posterior_head.bias"}
    if unexpected or not set(missing) <= allowed:
        raise ValueError(
            f"Incompatible initialization: missing={missing}, unexpected={unexpected}"
        )
    model.core.belief_readout = source_readout(directory)
    return file_hash(directory / "last.pt")


def source_readout(directory):
    """Old run records predate the optional continuous working readout."""
    settings = json.loads((directory / "run.json").read_text())["identity"]["settings"]
    mode = settings.get("belief_readout", "sampled")
    if mode not in ("sampled", "probabilities"):
        raise ValueError("Unknown saved working belief readout")
    return mode


def target_batch(data, indices, device):
    return {k: v[indices].to(device) for k, v in data["targets"].items()}


def load_encoders(model, directory, device):
    """Warm-start only perceptual features; leave the fresh updater/readout intact."""
    checkpoint = torch.load(
        directory / "last.pt", map_location=device, weights_only=True
    )
    prefix = "core.agent.encoders."
    model.core.agent.encoders.load_state_dict(
        {
            k.removeprefix(prefix): v
            for k, v in checkpoint["model"].items()
            if k.startswith(prefix)
        }
    )
    model.core.agent.encoders.requires_grad_(False).eval()
    return file_hash(directory / "last.pt")


def factor_objective(predictions, targets, task):
    terms = [F.cross_entropy(p, targets[:, i]) for i, p in enumerate(predictions)]
    if task not in ("all", "direction"):
        raise ValueError("Unknown factor objective")
    return (sum(terms) / 3 if task == "all" else terms[2] / 3), terms


def training_input_mode(step, selection):
    if selection not in ("rotating", *MODES):
        raise ValueError("Unknown training input selection")
    return MODES[step % len(MODES)] if selection == "rotating" else selection


def factor_task_screen(scores, task, selection):
    """Require every registered cell; transfer cells never change a local screen."""
    training_input_mode(0, selection)
    required = MODES if selection == "rotating" else (selection,)
    selected = [
        r for r in scores if r["split"] == "seen" and r["input_mode"] in required
    ]
    if len(selected) != len(required) or {r["input_mode"] for r in selected} != set(
        required
    ):
        return False
    return all(
        (r["factor_accuracy"][2] if task == "direction" else min(r["factor_accuracy"]))
        >= 0.9
        for r in selected
    )


def factor_gradient_audit(terms, parameters):
    """Unweighted per-task derivatives; no .grad, RNG or optimizer mutation."""
    gradients = []
    for term in terms:
        values = torch.autograd.grad(
            term, parameters, retain_graph=True, allow_unused=True
        )
        gradients.append(
            torch.cat(
                [
                    (torch.zeros_like(p) if v is None else v).detach().flatten()
                    for p, v in zip(parameters, values)
                ]
            )
        )
    names = ("color", "place", "direction")
    norms = [g.norm() for g in gradients]
    result = {f"task_gradient_norm_{n}": float(v) for n, v in zip(names, norms)}
    for i in range(3):
        for j in range(i + 1, 3):
            result[f"task_gradient_cosine_{names[i]}_{names[j]}"] = float(
                (gradients[i] @ gradients[j]) / (norms[i] * norms[j]).clamp_min(1e-20)
            )
    return result


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


def diagnose(args):
    """Frozen stage readers using the same data, model and standalone reports."""
    seed_everything(args.seed)
    device = torch.device(args.device)
    source_weights = torch.load(
        args.core / "last.pt", map_location="cpu", weights_only=True
    )["model"]
    model = Model("native", posterior_aux="posterior_head.weight" in source_weights).to(
        device
    )
    source_hash = load_initial(model, args.core, device)
    model.requires_grad_(False)
    optimizer = torch.optim.Adam(model.core.factor_head.parameters(), lr=0.003)
    populations = {
        s: dataset(s, args.seed) for s in ("train", "validation", "seen", "heldout")
    }
    run = Run(
        args.output,
        settings=dict(
            seed=args.seed,
            purpose="diagnostic",
            source_checkpoint_sha256=source_hash,
            core=str(args.core.resolve()),
            method="Frozen stage ridge; train-only scaling and validation-only choice",
            encoder_token_cap_per_scale=64,
        ),
        data=dict(
            generator=file_hash("pathwm/data/modality_readout.py"), seed=args.seed
        ),
        recipe=__file__,
        model=model,
        optimizer=optimizer,
        device=device,
    )
    before = state_hash(model)
    cache, arrays, rows, choices = {}, {}, [], []
    try:
        with evaluation_mode(model), torch.no_grad():
            for si, (split, data) in enumerate(populations.items()):
                cache[split] = {}
                arrays[f"{split}.factors"] = data["targets"]["factors"].numpy()
                for mi, mode in enumerate(MODES):
                    torch.manual_seed(args.seed + 1000 + si * 100 + mi)
                    _, stages = capture_readout_stages(
                        model.core, observations(data, mode, device=device)
                    )
                    cache[split][mode] = {k: v.numpy() for k, v in stages.items()}
                    for k, v in stages.items():
                        arrays[f"{split}.{mode}.{k}"] = v.numpy()
        for mode in MODES:
            for stage in cache["train"][mode]:
                train, validation = (
                    cache["train"][mode][stage],
                    cache["validation"][mode][stage],
                )
                reader, selection = fit_factor_probe(
                    train,
                    arrays["train.factors"],
                    validation,
                    arrays["validation.factors"],
                )
                permutation = np.random.default_rng(args.seed + 45).permutation(
                    len(train)
                )
                shuffled, _ = fit_factor_probe(
                    train,
                    arrays["train.factors"][permutation],
                    validation,
                    arrays["validation.factors"],
                    alphas=(reader["alpha"],),
                )
                choices.append(
                    dict(
                        input_mode=mode,
                        stage=stage,
                        features=train.shape[1],
                        nonconstant_features=int((train.std(0) > 1e-4).sum()),
                        selected_alpha=reader["alpha"],
                        validation=selection,
                    )
                )
                for label, probe in (("ridge", reader), ("shuffled", shuffled)):
                    for k, v in probe.items():
                        arrays[f"reader.{mode}.{stage}.{label}.{k}"] = v
                    for split in ("seen", "heldout"):
                        pred = predict_factor_probe(probe, cache[split][mode][stage])
                        correct = pred == arrays[f"{split}.factors"]
                        arrays[f"prediction.{split}.{mode}.{stage}.{label}"] = pred
                        row = dict(
                            split=split,
                            input_mode=mode,
                            stage=stage,
                            probe=label,
                            factor_accuracy=correct.mean(0).tolist(),
                            all_correct=float(correct.all(1).mean()),
                        )
                        rows.append(row)
                        run.log(
                            dict(
                                step=1,
                                split="diagnostic",
                                population=split,
                                input_mode=mode,
                                stage=stage,
                                probe=label,
                                color=row["factor_accuracy"][0],
                                location=row["factor_accuracy"][1],
                                direction=row["factor_accuracy"][2],
                                all_correct=row["all_correct"],
                            )
                        )
        # Positive readout control with explicit facts, separate from core evidence.
        oracle = oracle_states(populations)
        positive, _ = fit_factor_probe(
            oracle["train"]["all"].flatten(1).numpy(),
            arrays["train.factors"],
            oracle["validation"]["all"].flatten(1).numpy(),
            arrays["validation.factors"],
        )
        oracle_scores = {}
        for split in ("seen", "heldout"):
            pred = predict_factor_probe(
                positive, oracle[split]["all"].flatten(1).numpy()
            )
            oracle_scores[split] = float(
                (pred == arrays[f"{split}.factors"]).all(1).mean()
            )
        if state_hash(model) != before:
            raise AssertionError("Frozen diagnostic mutated source model")
        np.savez_compressed(run.path / "stage_probes.npz", **arrays)
        payload = dict(
            seed=args.seed,
            source_checkpoint_sha256=source_hash,
            model_unchanged=True,
            rows=rows,
            choices=choices,
            oracle_control=oracle_scores,
            scope="Linear accessibility under unequal feature dimensions; not a bound on nonlinear recovery or a proof of information loss.",
        )
        atomic_json(run.path / "stage_probe.json", payload)
        run.step = 1
        run.save()
        run.status("complete", "pending")
        save_stage_panel(run.path, rows)
        write_report(run.path)
        print(
            json.dumps(
                dict(
                    path=str(run.path),
                    oracle_control=oracle_scores,
                    model_unchanged=True,
                )
            ),
            flush=True,
        )
    except BaseException as exc:
        prior = json.loads((run.path / "status.json").read_text())
        run.status(
            prior["result"] if prior["result"] == "complete" else "failed",
            "failed",
            str(exc),
        )
        raise


def perform(args):
    seed_everything(args.seed)
    device = torch.device(args.device)
    populations = {
        s: dataset(s, args.seed) for s in ("train", "validation", "seen", "heldout")
    }
    data = populations["train"]
    model = Model(
        args.variant, posterior_aux=args.initial is not None and args.stage == "core"
    ).to(device)
    model.core.belief_readout = args.belief_readout
    initial_decoders = state_hash(model.outputs.decoders)
    initial_source = (
        None if args.initial is None else load_initial(model, args.initial, device)
    )
    encoder_source = (
        None
        if args.encoder_source is None
        else load_encoders(model, args.encoder_source, device)
    )
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
        model.core.belief_readout = source_readout(args.core)
        cache = torch.load(args.core / "states.pt", weights_only=True)
        if args.stage == "oracle":
            cache = oracle_states(populations)
    model.requires_grad_(False)
    kinds = KINDS if args.stage == "joint" else (args.modality,)
    if args.stage in ("core", "joint"):
        model.core.requires_grad_(True)
        model.core.agent.action_head.requires_grad_(False)
        model.core.agent.monitor.requires_grad_(False)
    if encoder_source:
        model.core.agent.encoders.requires_grad_(False)
    if args.posterior_aux:
        model.posterior_head.requires_grad_(True)
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
        initialization_checkpoint_sha256=initial_source,
        encoder_source_sha256=encoder_source,
        factor_task=args.factor_task,
        input_mode=args.input_mode,
        belief_readout=model.core.belief_readout,
        gradient_audit_every=args.gradient_audit_every,
        factor_coefficients=[1 / 3, 1 / 3, 1 / 3]
        if args.factor_task == "all"
        else [0, 0, 1 / 3],
        initialization_optimizer="fresh Adam"
        if initial_source
        else (
            "pretrained frozen encoders; fresh updater/readout/Adam"
            if encoder_source
            else "fresh model/Adam"
        ),
        posterior_aux_weight=args.posterior_aux,
        posterior_aux_source=args.posterior_source,
        temperature_warmup=args.temperature_warmup,
        temperature_warmup_updates=512,
        evaluation_temperature=1.0,
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
    before_encoders = state_hash(model.core.agent.encoders)
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
            mode = (
                "all"
                if args.stage == "oracle"
                else training_input_mode(run.step, args.input_mode)
            )
            indices = run.sample(len(data["ids"]), 24)
            wanted = target_batch(data, indices, device)
            optimizer.zero_grad(set_to_none=True)
            state = None
            posterior_features = {} if args.posterior_source == "raw" else None
            temperature = 1.0 + (args.temperature_warmup - 1.0) * max(
                0.0, 1.0 - run.step / 512.0
            )
            if args.stage in ("frozen", "oracle"):
                tokens = cache["train"][mode][indices].to(device)
            else:
                tokens, state = model.core(
                    observations(data, mode, indices, device),
                    return_state=True,
                    posterior_features=posterior_features,
                    temperature=temperature,
                )
            if args.stage == "core":
                loss, factor_terms = factor_objective(
                    model.core.factors(tokens), wanted["factors"], args.factor_task
                )
                metrics = dict(factor_loss=float(loss.detach()))
                if (
                    args.encoder_source
                    or args.gradient_audit_every
                    or args.factor_task != "all"
                ):
                    metrics.update(
                        {
                            f"factor_ce_{k}": float(v.detach())
                            for k, v in zip(
                                ("color", "place", "direction"), factor_terms
                            )
                        }
                    )
                if (
                    args.gradient_audit_every
                    and run.step % args.gradient_audit_every == 0
                ):
                    metrics.update(
                        factor_gradient_audit(
                            factor_terms, list(model.core.agent.updater.parameters())
                        )
                    )
                if args.temperature_warmup != 1.0:
                    metrics["training_temperature"] = temperature
                if args.posterior_aux:
                    aux_features = (
                        state.logits.softmax(-1).flatten(1)
                        if posterior_features is None
                        else F.layer_norm(posterior_features["raw_logits"], (32,))
                    )
                    auxiliary = model.posterior_head(aux_features).split((3, 3, 2), -1)
                    aux_loss = (
                        sum(
                            F.cross_entropy(p, wanted["factors"][:, i])
                            for i, p in enumerate(auxiliary)
                        )
                        / 3
                    )
                    metrics["posterior_aux_loss"] = float(aux_loss.detach())
                    loss = loss + args.posterior_aux * aux_loss
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
        if encoder_source:
            assert state_hash(model.core.agent.encoders) == before_encoders
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
            encoder_sha256=state_hash(model.core.agent.encoders),
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
                factor_task=args.factor_task,
                task_input_modes=list(MODES)
                if args.input_mode == "rotating"
                else [args.input_mode],
                belief_readout=model.core.belief_readout,
                task_gate=None
                if core_scores is None
                else factor_task_screen(core_scores, args.factor_task, args.input_mode),
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
        choices=(
            "core",
            "frozen",
            "joint",
            "oracle",
            "suite",
            "diagnose",
            "repair-report",
        ),
        default="suite",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7201)
    parser.add_argument("--variant", choices=VARIANTS, default="native")
    parser.add_argument("--modality", choices=KINDS, default="text")
    parser.add_argument("--core", type=Path)
    parser.add_argument(
        "--belief-readout",
        choices=("sampled", "probabilities"),
        default="sampled",
        help="Experimental working-context source; persistent categorical state stays hard",
    )
    parser.add_argument(
        "--encoder-source",
        type=Path,
        help="Load and freeze ONLY these core encoders; fresh remaining core/Adam",
    )
    parser.add_argument("--factor-task", choices=("all", "direction"), default="all")
    parser.add_argument(
        "--input-mode",
        choices=("rotating", *MODES),
        default="rotating",
        help="Core training inputs; other evaluated modes are transfer diagnostics",
    )
    parser.add_argument(
        "--gradient-audit-every",
        type=int,
        default=0,
        help="Observational shared-updater task gradients;0 disables",
    )
    parser.add_argument(
        "--reference", type=Path, help="Original study root for repair-report only"
    )
    parser.add_argument(
        "--posterior-source", choices=("probabilities", "raw"), default="probabilities"
    )
    parser.add_argument(
        "--temperature-warmup",
        type=float,
        default=1.0,
        help="Training temperature at start; annealed to1 over512 updates, evaluation always1",
    )
    parser.add_argument(
        "--initial",
        type=Path,
        help="Start from saved model weights with fresh optimizer; not resume",
    )
    parser.add_argument(
        "--posterior-aux",
        type=float,
        default=0.0,
        help="Training-only posterior factor CE weight for core continuation",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--steps", type=int)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.input_mode != "rotating" and args.stage != "core":
        parser.error("Input selection is supported for core training only")
    if args.belief_readout != "sampled" and (
        args.stage != "core" or args.encoder_source is None or args.initial
    ):
        parser.error(
            "Continuous working control requires fresh core with frozen --encoder-source"
        )
    if args.gradient_audit_every < 0:
        parser.error("Gradient audit interval must be nonnegative")
    if args.encoder_source or args.factor_task != "all" or args.gradient_audit_every:
        if (
            args.stage != "core"
            or args.initial
            or args.posterior_aux
            or args.temperature_warmup != 1.0
        ):
            parser.error(
                "Fresh factor comparison requires core stage without continuation, auxiliary or temperature curriculum"
            )
    if args.stage == "repair-report":
        if args.reference is None:
            parser.error("Repair report needs --reference")
        print(summarize_repair(args.output, args.reference))
        return
    if args.posterior_aux < 0 or not np.isfinite(args.posterior_aux):
        parser.error("Posterior auxiliary weight must be finite and nonnegative")
    if args.posterior_aux and (args.stage != "core" or args.initial is None):
        parser.error("Posterior auxiliary requires core continuation with --initial")
    if args.posterior_source == "raw" and not args.posterior_aux:
        parser.error(
            "Raw posterior auxiliary source requires positive auxiliary weight"
        )
    if not np.isfinite(args.temperature_warmup) or args.temperature_warmup < 1.0:
        parser.error("Temperature warmup must be finite and >=1")
    if args.temperature_warmup != 1.0 and (
        args.stage != "core" or args.initial is None
    ):
        parser.error("Temperature warmup requires core continuation with --initial")
    if args.initial and args.stage not in ("core", "oracle"):
        parser.error("Weight initialization supported for core/oracle only")
    if args.stage == "diagnose":
        if args.core is None or args.resume or args.stop_after or args.steps:
            parser.error("Diagnosis needs --core and does not train or resume")
        diagnose(args)
        return
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
