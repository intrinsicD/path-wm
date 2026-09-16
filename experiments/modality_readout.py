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
    LayerReadout,
)
from pathwm.models.modalities import ImageDecoder, AudioDecoder, TextDecoder
from pathwm.models.readout import RecurrentOutputAdapter, TemporalImageDecoder
from pathwm.models.tasks import (
    Actor,
    OutputControl,
    TaskRequest,
    TaskSession,
    TaskInterpreter,
    MetadataEncoder,
)
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
from pathwm.evaluation.modality_suite import (
    PROTOCOL as CAPABILITY_PROTOCOL,
    build_suite,
    measure_factors,
)

VARIANTS = ("native", "adapter1", "adapter2", "adapter4")


class Core(nn.Module):
    def __init__(self, width=24, *, belief_readout="sampled"):
        super().__init__()
        if belief_readout not in ("sampled", "probabilities"):
            raise ValueError("Unknown working belief readout")
        self.belief_readout = belief_readout
        self.request_readout = "none"
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
        requests=None,
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
        goal = None
        if requests is not None and self.request_readout != "none":
            if len(requests) != batch:
                raise ValueError("One request per batch member is required")
            caller = Actor("user", "caller")
            sessions = [
                TaskSession(
                    TaskRequest(
                        "readout-request",
                        question if self.request_readout == "instruction" else ".",
                        caller,
                        (OutputControl("text", "required", caller),),
                    )
                )
                for question in requests
            ]
            goal = self.agent.task_tokens(workspace, sessions, trace=trace)
        tokens = self.agent.think(workspace, steps=2, goal=goal, trace=trace).tokens
        return (tokens, state) if return_state else tokens

    def factors(self, tokens):
        return self.factor_head(tokens.flatten(1)).split((3, 3, 2), dim=-1)


class Outputs(nn.Module):
    def __init__(
        self, variant, width=24, *, video_conditioning="context", video_palette=0
    ):
        super().__init__()
        self.variant = variant
        self.decoders = nn.ModuleDict(
            {
                "text": TextDecoder(width),
                "image": ImageDecoder(width, 16),
                "audio": AudioDecoder(width, 192),
                "video": TemporalImageDecoder(
                    width,
                    16,
                    time_conditioning=video_conditioning,
                    palette_size=video_palette,
                ),
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
    def __init__(
        self,
        variant,
        *,
        posterior_aux=False,
        video_conditioning="context",
        video_palette=0,
    ):
        super().__init__()
        self.core, self.outputs = (
            Core(),
            Outputs(
                variant,
                video_conditioning=video_conditioning,
                video_palette=video_palette,
            ),
        )
        # Diagnostic supervision only; no inference read or RNG/init change.
        with torch.random.fork_rng():
            self.posterior_head = nn.Linear(32, 8) if posterior_aux else None


def configure_encoder_readout(model, mode):
    """Attach a tiny optional readout without reinitializing existing weights."""
    if mode not in ("native", "layers"):
        raise ValueError("Unknown encoder readout")
    for encoder in model.core.agent.encoders.values():
        if mode == "native":
            encoder.pyramid.layer_readout = None
        elif encoder.pyramid.layer_readout is None:
            parameter = next(encoder.parameters())
            encoder.pyramid.layer_readout = LayerReadout(
                len(encoder.pyramid.stages)
            ).to(device=parameter.device, dtype=parameter.dtype)
    return mode


def source_encoder_readout(directory):
    manifest = directory / "run.json"
    if not manifest.exists():
        # Legacy encoder-only checkpoints need no manifest. Strict state loading
        # still rejects adapter weights without their architecture metadata.
        return "native"
    return json.loads(manifest.read_text())["identity"]["settings"].get(
        "encoder_readout", "native"
    )


def configure_request_readout(model, mode):
    """Reuse the agent task path, preserving source weights and caller RNG."""
    if mode not in ("none", "constant", "instruction"):
        raise ValueError("Unknown request readout")
    agent = model.core.agent
    if mode == "none":
        agent.task_interpreter = agent.metadata_encoder = None
    elif agent.task_interpreter is None:
        device = next(model.parameters()).device
        with torch.random.fork_rng():
            agent.task_interpreter = TaskInterpreter(agent.width).to(device)
            agent.metadata_encoder = MetadataEncoder(agent.width).to(device)
    model.core.request_readout = mode


def load_initial(model, directory, device):
    checkpoint_path = directory / "last.pt" if directory.is_dir() else directory
    configure_encoder_readout(model, source_encoder_readout(checkpoint_path.parent))
    settings = json.loads((checkpoint_path.parent / "run.json").read_text())[
        "identity"
    ]["settings"]
    configure_request_readout(model, settings.get("request_readout", "none"))
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    allowed = {"posterior_head.weight", "posterior_head.bias"}
    if unexpected or not set(missing) <= allowed:
        raise ValueError(
            f"Incompatible initialization: missing={missing}, unexpected={unexpected}"
        )
    model.core.belief_readout = source_readout(checkpoint_path.parent)
    model.outputs.decoders["video"].time_conditioning = settings.get(
        "video_conditioning", "context"
    )
    return file_hash(checkpoint_path)


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
    configure_encoder_readout(model, source_encoder_readout(directory))
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
    capability = args.stage == "capabilities"
    run = Run(
        args.output,
        settings=dict(
            seed=args.seed,
            purpose="diagnostic",
            source_checkpoint_sha256=source_hash,
            core=str(args.core.resolve()),
            method="Frozen stage ridge; train-only scaling and validation-only choice",
            encoder_token_cap_per_scale=64,
            **(dict(capability_protocol=CAPABILITY_PROTOCOL) if capability else {}),
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
    started = time.perf_counter()
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
        if capability:
            # Separate full Cartesian population, never used for probe fitting or selection.
            populations["intervention"] = dataset("intervention", args.seed)
            measured, predictions, draw_seeds = measure_factors(
                model.core, populations, seed=args.seed, device=device
            )
            np.savez_compressed(run.path / "capability_predictions.npz", **predictions)
            export_data(run.path / "capability_inputs.npz", populations)
            source = dict(
                checkpoint=str((args.core / "last.pt").resolve()),
                checkpoint_sha256=source_hash,
                settings=json.loads((args.core / "run.json").read_text())["identity"][
                    "settings"
                ],
            )
            suite = build_suite(measured, rows, source=source, seed=args.seed)
            suite["draw_seeds"] = draw_seeds
            suite["model_unchanged"] = state_hash(model) == before
            suite["resources"] = dict(
                seconds=time.perf_counter() - started,
                parameters=sum(p.numel() for p in model.parameters()),
                neural_updates=0,
                ridge_readers=len(choices) * 2 + 1,
                ridge_solutions=sum(len(c["validation"]) + 1 for c in choices) + 5,
                peak_memory="not measured",
            )
            if (
                not suite["model_unchanged"]
                or file_hash(args.core / "last.pt") != source_hash
            ):
                raise AssertionError(
                    "Capability evaluation mutated frozen model/source"
                )
            atomic_json(run.path / "capability_suite.json", suite)
            atomic_json(
                run.path / "result.json",
                dict(
                    evaluation_scope=CAPABILITY_PROTOCOL["scope"],
                    metrics=suite["coverage"],
                    protocol_sha256=suite["protocol_sha256"],
                    broad_capability="not established",
                ),
            )
            examples = []
            for mode in MODES:
                failures = measured[f"heldout.{mode}"]["failures"]
                i = failures[0]["index"] if failures else 0
                d = populations["heldout"]
                entry = dict(
                    mode=mode,
                    id=d["ids"][i],
                    expected=d["targets"]["factors"][i].tolist(),
                    predicted_by_draw=predictions[f"heldout.{mode}.predictions"][
                        :, i
                    ].tolist(),
                )
                for kind, obs in observations(d, mode, [i]).items():
                    if kind == "text":
                        from pathwm.models.modalities import bytes_text

                        entry[kind] = bytes_text(obs.values[0])
                    else:
                        values = obs.values[0]
                        entry[kind] = (
                            values[0] if kind == "image" else values
                        ).tolist()
                    entry[kind + "_times"] = obs.times[0].tolist()
                    entry[kind + "_valid"] = (
                        None if obs.valid is None else obs.valid[0].tolist()
                    )
                examples.append(entry)
            atomic_json(run.path / "capability_examples.json", examples)
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


def restore_readout(directory, seed, device):
    """Restore the complete saved architecture, including optional diagnostic heads."""
    settings = json.loads((directory / "run.json").read_text())["identity"]["settings"]
    checkpoint = torch.load(
        directory / "last.pt", map_location="cpu", weights_only=True
    )
    seed_everything(seed)
    model = Model(
        settings["variant"],
        posterior_aux="posterior_head.weight" in checkpoint["model"],
        video_conditioning=settings.get("video_conditioning", "context"),
        video_palette=settings.get("video_palette", 0),
    )
    source_hash = load_initial(model, directory, device)
    return model.to(device), settings, source_hash


def order_requests(record, *, novel=False):
    """Same evidence, different visible requests; no new clips or hidden task IDs."""
    if record["case"] != "VID.order":
        raise ValueError("Request contrasts currently require VID.order")
    questions = (
        {
            "first": ("Welche Farbe kam zuerst?", "Gib nur die erste Farbe an."),
            "sequence": ("Welche Farben der Reihe?", "Gib beide Farben der Reihe."),
        }
        if novel
        else {
            "first": (record["question"], "Erste Farbe?"),
            "sequence": (
                "Nenne beide Farben in ihrer zeitlichen Reihenfolge.",
                "Beide Farben",
            ),
        }
    )
    return [
        record
        | dict(
            id=f"{record['id']}/{novel}/{form}/{i}",
            pair=f"{record['pair']}/{novel}/{form}/{i}",
            question=question,
            format=form,
            wording=i,
            novel=novel,
            choices=record["choices"]
            if form == "first"
            else [
                " ".join((record["choices"][j], record["choices"][1 - j]))
                for j in range(2)
            ],
        )
        for form, values in questions.items()
        for i, question in enumerate(values)
    ]


def grounded_records(data, case, *, request_contrasts=False, request_profile="narrow"):
    """An explicit supervised calibration population; never train on scored cohorts."""
    if not any(
        c["id"] == case and c["domain"] == "controlled contrast" for c in data.cases
    ):
        raise ValueError("Grounded training requires a controlled contrast case")
    rows = [
        r for r in data.records if r["case"] == case and r["split"] == "calibration"
    ]
    if not rows:
        raise ValueError("No calibration training examples")
    if request_profile not in ("narrow", "balanced"):
        raise ValueError("Unknown request profile")
    if request_profile == "balanced":
        if not request_contrasts:
            raise ValueError("Balanced requests need contrasts")
        from pathwm.data.request_meaning import request_corpus, apply_request

        requests = [r for r in request_corpus() if r["split"] == "calibration"]
        return [apply_request(r, q) for r in rows for q in requests]
    return [v for r in rows for v in order_requests(r)] if request_contrasts else rows


def grounded_batch(data, rows, indices, device):
    from torch.nn.utils.rnn import pad_sequence
    from pathwm.models.modalities import Observation, bytes_batch

    if any(r["split"] != "calibration" for r in rows):
        raise ValueError("Only calibration examples may enter grounded training")
    selected = [rows[int(i)] for i in indices]
    items = [data.inputs(r, device=device) for r in selected]
    if any(item.keys() != items[0].keys() for item in items):
        raise ValueError("Batch one task with the same evidence modalities")
    inputs = {}
    for kind in items[0]:
        observations_ = [item[kind] for item in items]
        inputs[kind] = Observation(
            pad_sequence([o.values[0] for o in observations_], batch_first=True),
            pad_sequence([o.times[0] for o in observations_], batch_first=True),
            pad_sequence(
                [
                    torch.ones_like(o.times[0], dtype=torch.bool)
                    if o.valid is None
                    else o.valid[0]
                    for o in observations_
                ],
                batch_first=True,
            ),
        )
    targets, _ = bytes_batch(
        [r["choices"][r["answer"]] for r in selected], device=device
    )
    return inputs, targets


def configure_grounded_training(model, scope):
    if scope not in ("decoder", "core", "interpreter"):
        raise ValueError("Grounded scope must be decoder, core or interpreter")
    model.requires_grad_(False)
    if scope == "interpreter":
        if (
            model.core.request_readout != "instruction"
            or model.core.agent.task_interpreter is None
        ):
            raise ValueError("Interpreter fitting requires an instruction source")
        model.core.agent.task_interpreter.requires_grad_(True)
        training_mode(model)
        return
    if scope == "core":
        model.core.requires_grad_(True)
        for module in (
            model.core.agent.encoders,
            model.core.factor_head,
            model.core.agent.action_head,
            model.core.agent.monitor,
        ):
            module.requires_grad_(False)
    model.outputs.decoders["text"].requires_grad_(True)
    model.outputs.adapters["text"].requires_grad_(True)
    training_mode(model)


def grounded_objective(model, tokens, targets):
    out = model.outputs("text", tokens, targets[:, :-1])
    return F.cross_entropy(
        out.flatten(0, 1), targets[:, 1:].flatten(), ignore_index=0
    ) / np.log(259)


def retention_kl(student, teacher, labels):
    """Next-byte KL: include first EOS, never padding or finished continuations."""
    eos = labels == 2
    valid = (labels != 0) & ((eos.cumsum(1) - eos.long()) == 0)
    divergence = F.kl_div(
        student.log_softmax(-1), teacher.softmax(-1), reduction="none"
    ).sum(-1)
    return divergence[valid].mean() / np.log(student.shape[-1])


def retention_targets(teacher, inputs, targets):
    """Replay source outputs without advancing the student's random stream."""
    with evaluation_mode(teacher), torch.no_grad():
        tokens = teacher.core(inputs)
        prefixes = teacher.outputs.generate(tokens, 28)
        return dict(
            text_target=teacher.outputs("text", tokens, targets["text"][:, :-1]),
            text_greedy=teacher.outputs("text", tokens, prefixes[:, :-1]),
            prefixes=prefixes,
            **{k: teacher.outputs(k, tokens) for k in KINDS if k != "text"},
        )


def retention_objective(model, tokens, targets, reference, normalizers):
    losses = {}
    for name, sequence in (
        ("text_target", targets["text"]),
        ("text_greedy", reference["prefixes"]),
    ):
        logits = model.outputs("text", tokens, sequence[:, :-1])
        losses[name] = retention_kl(logits, reference[name], sequence[:, 1:])
    for kind in KINDS:
        if kind != "text":
            losses[kind] = (
                F.mse_loss(model.outputs(kind, tokens), reference[kind])
                / normalizers[kind]
            )
    total = 0.5 * (losses["text_target"] + losses["text_greedy"])
    total = total + sum(losses[k] for k in KINDS if k != "text")
    return total, {k: float(v.detach()) for k, v in losses.items()}


def objective_gradient_norm(loss, parameters):
    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=True, allow_unused=True
    )
    return float(sum(g.square().sum() for g in gradients if g is not None).sqrt())


def summarize_request_answers(results):
    metrics = []
    for novel in dict.fromkeys(r["novel"] for r in results):
        for condition in dict.fromkeys(
            r["condition"] for r in results if r["novel"] == novel
        ):
            group = [
                r
                for r in results
                if r["novel"] == novel and r["condition"] == condition
            ]
            for form in ("first", "sequence", "both"):
                by_draw = []
                for draw in range(3):
                    selected = [r for r in group if r["draw"] == draw]
                    if form == "both":
                        paired = {}
                        for r in selected:
                            paired.setdefault((r["clip"], r["wording"]), []).append(
                                r["exact"]
                            )
                        if not paired or any(len(v) != 2 for v in paired.values()):
                            raise ValueError("Incomplete opposite-request answer pair")
                        values = [all(v) for v in paired.values()]
                    else:
                        values = [r["exact"] for r in selected if r["format"] == form]
                    by_draw.append(float(np.mean(values)))
                metrics.append(
                    dict(
                        novel=novel,
                        condition=condition,
                        format=form,
                        exact_by_draw=by_draw,
                        exact_min=min(by_draw),
                    )
                )
    return metrics


def evaluate_request_contrasts(model, data, path, device):
    """Free answers and routing ablations; labels never enter model inputs."""
    from pathwm.models.modalities import bytes_text

    rows = [r for r in data.records if r["case"] == "VID.order"]
    pair_rng = {
        p: 19401 + i * 10 for i, p in enumerate(dict.fromkeys(r["pair"] for r in rows))
    }
    results = []
    start = time.perf_counter()
    with evaluation_mode(model), torch.no_grad():
        for base in (r for r in rows if r["split"] == "test"):
            for novel in (False, True):
                for row in order_requests(base, novel=novel):
                    conditions = ["full"]
                    if novel and row["wording"] == 0:
                        conditions += [
                            "omitted",
                            "last_frame",
                            "observation_only",
                            "request_only",
                            "scrambled_request_only",
                        ]
                    for draw in range(3):
                        for condition in conditions:
                            torch.manual_seed(pair_rng[base["pair"]] + draw)
                            request = row["question"]
                            observed = row
                            if condition in ("request_only", "scrambled_request_only"):
                                observed = row | dict(question=".")
                            if condition == "observation_only":
                                request = "."
                            elif condition == "scrambled_request_only":
                                request = request[::-1]
                            inputs = data.inputs(
                                observed,
                                omit=("video",) if condition == "omitted" else (),
                                device=device,
                            )
                            if condition == "last_frame":
                                video = inputs["video"]
                                inputs["video"] = replace(
                                    video,
                                    values=video.values[:, -1:],
                                    times=video.times[:, -1:],
                                )
                            tokens = model.core(inputs, requests=[request])
                            ids = model.outputs.generate(tokens, 16)[0]
                            generated = bytes_text(ids)
                            expected = row["choices"][row["answer"]]
                            ended = bool((ids == 2).any())
                            results.append(
                                dict(
                                    clip=base["id"],
                                    pair=base["pair"],
                                    novel=novel,
                                    format=row["format"],
                                    wording=row["wording"],
                                    draw=draw,
                                    condition=condition,
                                    question=row["question"],
                                    generated=generated,
                                    expected=expected,
                                    ended=ended,
                                    exact=ended and generated == expected,
                                    first_word=(generated.split() or [""])[0]
                                    == expected.split()[0],
                                )
                            )
    metrics = summarize_request_answers(results)
    result = dict(
        metrics=metrics,
        examples=results,
        seconds=time.perf_counter() - start,
        scope="Full answers: familiar and two novel wordings per format. Routing/omission controls: only the equal-length novel wording pair. Repeated development clips; no general-language claim.",
    )
    atomic_json(path / "request_contrasts.json", result)
    return result


def evaluate_request_meanings(model, data, path, device):
    """New prefix templates, actual generated answers and necessity controls."""
    from pathwm.data.request_meaning import request_corpus, apply_request
    from pathwm.models.modalities import bytes_text

    results, working = [], []
    requests = [r for r in request_corpus() if r["split"] == "test"]
    records = [
        r for r in data.records if r["case"] == "VID.order" and r["split"] == "test"
    ]
    pair_rng = {
        p: 19701 + i * 10
        for i, p in enumerate(dict.fromkeys(r["pair"] for r in records))
    }
    start = time.perf_counter()
    with evaluation_mode(model), torch.no_grad():
        for base in records:
            for request in requests:
                row = apply_request(base, request)
                conditions = (
                    ["full"] if request["style"] == "direct" else ["order_stress"]
                )
                if request["family"] == "test/0":
                    conditions += ["omitted", "last_frame", "constant_request"]
                for draw in range(3):
                    for condition in conditions:
                        torch.manual_seed(pair_rng[base["pair"]] + draw)
                        observed = (
                            row
                            if condition != "constant_request"
                            else row | dict(question=".")
                        )
                        inputs = data.inputs(
                            observed,
                            omit=("video",) if condition == "omitted" else (),
                            device=device,
                        )
                        if condition == "last_frame":
                            video = inputs["video"]
                            inputs["video"] = replace(
                                video,
                                values=video.values[:, -1:],
                                times=video.times[:, -1:],
                            )
                        tokens = model.core(
                            inputs,
                            requests=[
                                "."
                                if condition == "constant_request"
                                else row["question"]
                            ],
                        )
                        working.append(tokens.detach().cpu())
                        expected = row["choices"][row["answer"]]
                        results.append(
                            dict(
                                clip=base["id"],
                                pair=base["pair"],
                                novel=True,
                                format=row["format"],
                                wording=request["pair"],
                                draw=draw,
                                condition=condition,
                                question=row["question"],
                                expected=expected,
                            )
                        )
        # Preserve every individually seeded core call; batch only deterministic decoding.
        for start_index in range(0, len(working), 64):
            tokens = torch.cat(working[start_index : start_index + 64]).to(device)
            generated_ids = model.outputs.generate(tokens, 16)
            for row, ids in zip(results[start_index : start_index + 64], generated_ids):
                generated = bytes_text(ids)
                ended = bool((ids == 2).any())
                row.update(
                    generated=generated,
                    ended=ended,
                    exact=ended and generated == row["expected"],
                    first_word=generated.split()[:1] == row["expected"].split()[:1],
                    format_correct=len(generated.split())
                    == (1 if row["format"] == "first" else 2),
                )
    result = dict(
        metrics=summarize_request_answers(results),
        examples=results,
        seconds=time.perf_counter() - start,
        scope="Held-out prefix composition and separate phrase-order stress; source controls on first held-out prefix only.",
    )
    atomic_json(path / "request_meanings.json", result)
    return result


def request_evaluate(args):
    from pathwm.data.understanding import UnderstandingData

    model, settings, source_hash = restore_readout(args.core, args.seed, args.device)
    model.requires_grad_(False).eval()
    data = UnderstandingData(args.understanding_suite)
    run = Run(
        args.output,
        settings=dict(
            seed=args.seed,
            purpose="diagnostic",
            source=str(args.core.resolve()),
            source_sha256=source_hash,
        ),
        data=dict(fixtures=data.identity),
        recipe=__file__,
        model=model,
        optimizer=torch.optim.Adam([next(model.parameters())], lr=0.001),
        device=args.device,
    )
    before = state_hash(model)
    try:
        result = evaluate_request_meanings(model, data, run.path, args.device)
        if before != state_hash(model) or source_hash != file_hash(
            args.core / "last.pt"
        ):
            raise AssertionError("Evaluation changed source")
        atomic_json(
            run.path / "result.json",
            dict(
                evaluation_scope=result["scope"],
                metrics={
                    f"{r['condition']}/{r['format']}": r for r in result["metrics"]
                },
            ),
        )
        run.save()
        run.status("complete", "pending")
        write_report(run.path)
    except BaseException as exc:
        prior = json.loads((run.path / "status.json").read_text())
        run.status(
            prior["result"] if prior["result"] == "complete" else "failed",
            "failed",
            str(exc),
        )
        raise


def request_diagnose(args):
    """Audit balanced instruction meaning on the actual frozen request path."""
    from pathwm.data.understanding import UnderstandingData
    from pathwm.data.request_meaning import request_corpus
    from pathwm.evaluation.request_meaning import capture_request_stages, request_probe
    from pathwm.models.modalities import bytes_text

    device = torch.device(args.device)
    model, settings, source_hash = restore_readout(args.core, args.seed, device)
    if (
        model.core.request_readout != "instruction"
        or model.core.belief_readout != "sampled"
    ):
        raise ValueError("Use a sampled instruction-route source for request diagnosis")
    model.requires_grad_(False).eval()
    data = UnderstandingData(args.understanding_suite)
    contexts = [
        r
        for r in data.records
        if r["case"] == "VID.order" and r["split"] == "calibration"
    ][:2]
    if len(contexts) != 2 or {r["answer"] for r in contexts} != {0, 1}:
        raise ValueError("Need two opposite calibration contexts")
    corpus = request_corpus()
    run = Run(
        args.output,
        settings=dict(
            seed=args.seed,
            purpose="diagnostic",
            source=str(args.core.resolve()),
            source_sha256=source_hash,
            protocol="balanced-request-meaning-v1",
            encoder_bins=16,
            scope="Frozen controlled request accessibility, not general language",
        ),
        data=dict(fixtures=data.identity, requests=corpus, contexts=contexts),
        recipe=__file__,
        model=model,
        optimizer=torch.optim.Adam([next(model.parameters())], lr=0.001),
        device=device,
    )
    start = time.perf_counter()
    before = state_hash(model)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    rows, features, outputs = [], {}, []
    try:
        with evaluation_mode(model), torch.no_grad():
            for context, base in enumerate(contexts):
                torch.manual_seed(args.seed)
                _, state = model.core(
                    data.inputs(base | dict(question="."), device=device),
                    return_state=True,
                )
                physical = state.tokens.clone(), state.logits.clone()
                for request in corpus:
                    values, tokens = capture_request_stages(
                        model, state, request["question"]
                    )
                    row = request | dict(context=context)
                    rows.append(row)
                    for key, value in values.items():
                        features.setdefault(key, []).append(value[0])
                    ids = model.outputs.generate(tokens, 16)[0]
                    expected = base["choices"][base["answer"]]
                    if request["label"]:
                        expected += " " + base["choices"][1 - base["answer"]]
                    answer = bytes_text(ids)
                    outputs.append(
                        row
                        | dict(
                            generated=answer,
                            expected=expected,
                            ended=bool((ids == 2).any()),
                            exact=answer == expected and bool((ids == 2).any()),
                            first_word=answer.split()[:1] == expected.split()[:1],
                            format_correct=len(answer.split()) == 1 + request["label"],
                        )
                    )
                if not torch.equal(state.tokens, physical[0]) or not torch.equal(
                    state.logits, physical[1]
                ):
                    raise AssertionError("Request path mutated physical state")
        features = {k: torch.stack(v) for k, v in features.items()}
        families = list(dict.fromkeys(r["family"] for r in rows))
        features["length"] = torch.tensor([[len(r["question"].encode())] for r in rows])
        features["prefix"] = torch.tensor(
            [[float(r["family"] == f) for f in families] for r in rows]
        )
        features["byte_histogram"] = torch.stack(
            [
                torch.bincount(
                    torch.tensor(list(r["question"].encode())), minlength=256
                )
                for r in rows
            ]
        )
        direct = [i for i, r in enumerate(rows) if r["style"] == "direct"]
        stress = [i for i, r in enumerate(rows) if r["style"] == "order"]
        direct_rows = [rows[i] for i in direct]
        results, fitted = {}, {}
        for stage, x in features.items():
            for null in (
                (None, 9711, 9712, 9713)
                if stage in ("encoder", "instruction", "task", "working")
                else (None,)
            ):
                key = stage if null is None else f"{stage}/family_flip{null}"
                value, weights = request_probe(
                    x[direct], direct_rows, family_flip_seed=null
                )
                z = x[stress].double()
                pred = (
                    ((z - weights["mean"]) / weights["std"]) @ weights["weights"]
                    + weights["target_mean"]
                ).argmax(-1)
                value["order_stress"] = dict(
                    predictions=pred.tolist(),
                    accuracy=float(
                        (pred == torch.tensor([rows[i]["label"] for i in stress]))
                        .double()
                        .mean()
                    ),
                )
                results[key], fitted[key] = value, weights
                run.log(
                    dict(
                        step=len(results),
                        split="diagnostic",
                        stage=key,
                        accuracy=value["metrics"]["test"]["accuracy"],
                        paired=value["metrics"]["test"]["paired"],
                    )
                )
        if (
            state_hash(model) != before
            or file_hash(args.core / "last.pt") != source_hash
        ):
            raise AssertionError("Frozen source changed")
        resources = dict(
            seconds=time.perf_counter() - start,
            parameters=sum(p.numel() for p in model.parameters()),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else None,
        )
        atomic_torch(run.path / "request_features.pt", features)
        atomic_torch(run.path / "request_probes.pt", fitted)
        atomic_json(run.path / "request_examples.json", outputs)
        atomic_json(
            run.path / "request_diagnosis.json",
            dict(
                rows=rows,
                results=results,
                resources=resources,
                encoder_feasible=all(
                    results["encoder"]["metrics"]["test"][k] >= 0.8
                    for k in ("accuracy", "paired")
                ),
                frozen_preserved=True,
            ),
        )
        atomic_json(
            run.path / "result.json",
            dict(
                evaluation_scope="Controlled request composition; probes are separate from generated answers",
                metrics={k: v["metrics"] for k, v in results.items()},
                resources=resources,
                limits=[
                    "Byte-length and prefix are paired nuisance controls; vocabulary is deliberately shared.",
                    "Two fixed contexts are not independent natural-language samples. Order stress is separate.",
                    "Probe accessibility does not prove deployed use. Failures do not establish information loss.",
                ],
            ),
        )
        run.save()
        run.status("complete", "pending")
        write_report(run.path)
    except BaseException as exc:
        prior = json.loads((run.path / "status.json").read_text())
        run.status(
            prior["result"] if prior["result"] == "complete" else "failed",
            "failed",
            str(exc),
        )
        raise
    print(json.dumps(dict(path=str(run.path), resources=resources)), flush=True)


def grounded(args):
    """Small matched continuation through the existing core and output decoder."""
    from pathwm.data.understanding import UnderstandingData
    from pathwm.evaluation.understanding import choice_scores
    from pathwm.models.modalities import bytes_text

    data = UnderstandingData(args.understanding_suite)
    contrasts = getattr(args, "request_contrasts", False)
    profile = getattr(args, "request_profile", "narrow")
    interpreter_only = args.grounded_scope == "interpreter"
    if interpreter_only and (not contrasts or getattr(args, "retention_weight", 0)):
        raise ValueError(
            "Interpreter-only fitting needs contrasts and no retention loss"
        )
    rows = grounded_records(
        data, args.grounded_case, request_contrasts=contrasts, request_profile=profile
    )
    request_count = len(rows) // len(grounded_records(data, args.grounded_case))
    device = torch.device(args.device)
    model, source_settings, source_hash = restore_readout(args.core, args.seed, device)
    configure_request_readout(
        model,
        getattr(args, "request_readout", None)
        or source_settings.get("request_readout", "none"),
    )
    configure_grounded_training(model, args.grounded_scope)
    retention_weight = getattr(args, "retention_weight", 0.0)
    if not np.isfinite(retention_weight) or retention_weight < 0:
        raise ValueError("Retention weight must be finite and nonnegative")
    # Construct from the SOURCE before Run restores any resumed student weights.
    teacher = (
        copy.deepcopy(model).requires_grad_(False).eval() if retention_weight else None
    )
    teacher_hash = state_hash(teacher) if teacher is not None else None
    frozen = {
        n: p.detach().clone()
        for n, p in model.named_parameters()
        if not p.requires_grad
    }
    original_seed = source_settings.get("replay_seed", source_settings["seed"])
    populations = {
        s: dataset(s, original_seed) for s in ("train", "validation", "seen", "heldout")
    }
    normalizers = scales(populations["train"])
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=0.001)
    settings = dict(
        seed=args.seed,
        stage="grounded",
        variant=source_settings["variant"],
        steps=args.steps,
        batch=8,
        learning_rate=0.001,
        case=args.grounded_case,
        request_readout=model.core.request_readout,
        request_contrasts=contrasts,
        request_profile=profile,
        scope=args.grounded_scope,
        calibration_training_ids=[r["id"] for r in rows],
        replay_seed=original_seed,
        replay="none; interpreter-only"
        if interpreter_only
        else "alternate original multimodal replay and QA updates",
        objective="QA byte CE/log259; replay sum normalized per-output losses",
        initialization_checkpoint_sha256=source_hash,
        initialization_checkpoint=str(args.core.resolve()),
        belief_readout=model.core.belief_readout,
        encoder_readout=source_encoder_readout(args.core),
        video_conditioning=model.outputs.decoders["video"].time_conditioning,
        video_palette=model.outputs.decoders["video"].palette_size,
        diagnostic_calibration_overlap=args.grounded_case,
        retention_weight=retention_weight,
        retention_teacher_sha256=teacher_hash,
        retention="Replay-only source KL (target + greedy prefixes)/2 + normalized image/audio/video MSE; paired underlying noise"
        if teacher is not None
        else None,
    )
    run = Run(
        args.output,
        settings=settings,
        data=dict(
            fixtures=data.identity,
            replay_seed=original_seed,
            replay_ids=digest(populations["train"]["ids"]),
            grounded_records_sha256=digest(rows),
        ),
        recipe=__file__,
        model=model,
        optimizer=optimizer,
        device=device,
        resume=args.resume,
    )

    def symbolic():
        with evaluation_mode(model), torch.no_grad():
            cached, _ = cache_states(model, populations, original_seed, device)
            return evaluate_outputs(
                model.outputs, cached, populations, KINDS, normalizers, device
            )

    try:
        baseline_path = run.path / "symbolic_before.json"
        if not args.resume:
            atomic_json(run.path / "grounded_training_examples.json", rows)
            before, _ = symbolic()
            atomic_json(baseline_path, before)
        else:
            before = json.loads(baseline_path.read_text())
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        end = min(args.steps, args.stop_after or args.steps)
        training_mode(model)
        while run.step < end:
            optimizer.zero_grad(set_to_none=True)
            diagnostics = {}
            if interpreter_only or run.step % 2 == 0:
                if interpreter_only:
                    choices = request_count
                    clip_indices = run.sample(len(rows) // choices, 8)
                    request_rng = torch.Generator().manual_seed(
                        args.seed + 1777 + run.step
                    )
                    request_indices = torch.randint(
                        choices, (8,), generator=request_rng
                    ).numpy()
                    indices = clip_indices * choices + request_indices
                else:
                    indices = run.sample(len(rows), 8)
                inputs, target = grounded_batch(data, rows, indices, device)
                requests = [rows[int(i)]["question"] for i in indices]
                loss = grounded_objective(
                    model, model.core(inputs, requests=requests), target
                )
                objective = "qa"
            else:
                replay = populations["train"]
                indices = run.sample(len(replay["ids"]), 8)
                mode = MODES[(run.step // 2) % len(MODES)]
                inputs = observations(replay, mode, indices, device)
                target = target_batch(replay, indices, device)
                reference = (
                    retention_targets(teacher, inputs, target)
                    if teacher is not None
                    else None
                )
                tokens = model.core(inputs)
                loss, _ = output_objective(
                    model,
                    tokens,
                    target,
                    KINDS,
                    normalizers,
                )
                if reference is not None:
                    anchor, components = retention_objective(
                        model, tokens, target, reference, normalizers
                    )
                    diagnostics = dict(
                        replay_loss=float(loss.detach()),
                        anchor_loss=float(anchor.detach()),
                        **components,
                    )
                    if run.step == 1 or run.step % 128 == 127:
                        diagnostics["replay_gradient_norm"] = objective_gradient_norm(
                            loss, trainable
                        )
                        diagnostics["weighted_anchor_gradient_norm"] = (
                            objective_gradient_norm(
                                retention_weight * anchor, trainable
                            )
                        )
                    loss = loss + retention_weight * anchor
                objective = "replay"
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite grounded objective")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                trainable,
                5.0,
                error_if_nonfinite=True,
            )
            optimizer.step()
            run.step += 1
            run.log(
                dict(
                    step=run.step,
                    split="train",
                    loss=float(loss.detach()),
                    objective=objective,
                    **diagnostics,
                )
            )
            if run.step % 128 == 0:
                run.save()
                print(json.dumps(run.rows[-1]), flush=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        run.save()
        for name, parameter in model.named_parameters():
            if name in frozen and not torch.equal(parameter, frozen[name]):
                raise AssertionError(f"Frozen parameter changed: {name}")
        if file_hash(args.core / "last.pt") != source_hash:
            raise AssertionError("Source checkpoint changed")
        if teacher is not None and state_hash(teacher) != teacher_hash:
            raise AssertionError("Frozen retention teacher changed")
        resources = dict(
            training_seconds_this_invocation=elapsed,
            parameters=sum(p.numel() for p in model.parameters()),
            trainable_parameters=sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
            frozen_preserved=True,
            teacher_preserved=teacher is not None,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else None,
        )
        metrics, arrays, controls = [], {}, []
        if end == args.steps:
            metrics, arrays = symbolic()
            np.savez_compressed(run.path / "outputs.npz", **arrays)
            save_panels(run.path, arrays, metrics)
            if args.grounded_case.startswith("VID.") and not interpreter_only:
                with evaluation_mode(model), torch.no_grad():
                    for r in data.records:
                        if r["case"] != args.grounded_case or r["split"] != "test":
                            continue
                        for draw in range(3):
                            torch.manual_seed(9401 + draw)
                            inputs = data.inputs(r, device=device)
                            video = inputs["video"]
                            inputs["video"] = replace(
                                video,
                                values=video.values[:, -1:],
                                times=video.times[:, -1:],
                            )
                            tokens = model.core(inputs, requests=[r["question"]])
                            scores = choice_scores(model.outputs, tokens, r["choices"])
                            controls.append(
                                dict(
                                    id=r["id"],
                                    draw=draw,
                                    answer=r["answer"],
                                    predicted=int(scores.argmax()),
                                    scores=scores.cpu().tolist(),
                                )
                            )
                    generated = []
                    for r in [
                        r
                        for r in data.records
                        if r["case"] == args.grounded_case and r["split"] == "test"
                    ]:
                        torch.manual_seed(9401)
                        tokens = model.core(
                            data.inputs(r, device=device), requests=[r["question"]]
                        )
                        generated.append(
                            dict(
                                id=r["id"],
                                expected=r["choices"][r["answer"]],
                                generated=bytes_text(
                                    model.outputs.generate(tokens, 12)[0]
                                ),
                            )
                        )
                atomic_json(run.path / "grounded_examples.json", generated)
            if contrasts:
                evaluator = (
                    evaluate_request_meanings
                    if interpreter_only
                    else evaluate_request_contrasts
                )
                request_result = evaluator(model, data, run.path, device)
                atomic_json(
                    run.path / "result.json",
                    dict(
                        evaluation_scope=request_result["scope"],
                        metrics={
                            f"{'novel' if r['novel'] else 'familiar'}/{r['condition']}/{r['format']}": r
                            for r in request_result["metrics"]
                        },
                        limits=[
                            "Raw questions, outputs, EOS and first-word diagnostics are in request_meanings.json or request_contrasts.json.",
                            "Per-fit format success is not overall adoption; compare both sources and regression gates.",
                        ],
                    ),
                )
        atomic_json(
            run.path / "grounded.json",
            dict(
                settings=settings,
                resources=resources,
                symbolic_before=before,
                symbolic_after=metrics,
                last_frame_controls=controls,
                last_frame_accuracy=None
                if not controls
                else float(np.mean([r["answer"] == r["predicted"] for r in controls])),
                diagnostic_note="This task's calibration cohort trained the agent; its later calibration probes are not independent.",
            ),
        )
        atomic_json(
            run.path / "readout.json",
            dict(
                stage="grounded",
                variant=settings["variant"],
                modality="text",
                metrics=metrics,
                resources=resources,
                partial=end < args.steps,
            ),
        )
        run.status("complete" if end == args.steps else "paused", "pending")
        write_report(run.path)
    except BaseException as exc:
        prior = json.loads((run.path / "status.json").read_text())
        run.status(
            prior["result"] if prior["result"] in ("complete", "paused") else "failed",
            "failed",
            str(exc),
        )
        raise
    print(json.dumps(dict(path=str(run.path), resources=resources)), flush=True)
    return run.path


def select_understanding(data, cases):
    """An explicit subset is a different evaluation contract, never hidden coverage."""
    if not cases:
        return data
    selected = sorted(set(cases))
    if set(selected) - {c["id"] for c in data.cases}:
        raise ValueError("Unknown understanding case selection")
    subset = copy.copy(data)
    subset.records = [r for r in data.records if r["case"] in selected]
    subset.cases = [c for c in data.cases if c["id"] in selected]
    subset.identity = digest(dict(parent=data.identity, cases=selected))
    subset.manifest = data.manifest | dict(
        profile=data.manifest["profile"] + "/selected:" + ",".join(selected),
        records=subset.records,
        cases=subset.cases,
        limits=data.manifest["limits"]
        + ["Explicit task subset; not the complete regression battery."],
    )
    return subset


def understanding(args):
    """Run a prepared regression battery against an unchanged full checkpoint."""
    from pathwm.data.understanding import UnderstandingData
    from pathwm.evaluation.understanding import evaluate_understanding

    data = select_understanding(
        UnderstandingData(args.understanding_suite),
        getattr(args, "understanding_cases", None),
    )
    model, settings, source_hash = restore_readout(args.core, args.seed, args.device)
    model.requires_grad_(False)
    source = dict(
        path=str(args.core.resolve()),
        sha256=source_hash,
        settings=settings,
        output_training_scope=(
            f"Grounded QA: {settings.get('case')}; calibration cohort also trained the agent, so probes are not independent"
            if settings["stage"] == "grounded"
            else "joint multimodal symbolic outputs"
            if settings["stage"] == "joint"
            else "Check source settings: decoder may be untrained for this endpoint"
        ),
    )
    result = evaluate_understanding(
        model,
        data,
        args.output,
        source=source,
        seed=args.seed,
        device=args.device,
        recipe=__file__,
        reference=args.reference,
    )
    if file_hash(args.core / "last.pt") != source_hash:
        raise AssertionError("Source checkpoint changed during evaluation")
    print(
        json.dumps(dict(path=str(args.output), coverage=result["coverage"])), flush=True
    )
    return args.output


def exploration_cache(model, data, case, seed, device):
    """One frozen, versioned latent draw per example; labels never enter the core."""
    from pathwm.models.modalities import bytes_batch

    cache = {}
    with evaluation_mode(model), torch.no_grad():
        for split in ("calibration", "validation", "test"):
            rows = [
                r for r in data.records if r["case"] == case and r["split"] == split
            ]
            if not rows:
                raise ValueError(
                    "Exploration needs separate calibration/validation/test"
                )
            tokens = {}
            for condition in ("full", "omitted"):
                values = []
                for i, row in enumerate(rows):
                    torch.manual_seed(
                        seed
                        + 10000
                        * (1 + ("calibration", "validation", "test").index(split))
                        + i
                    )
                    inputs = data.inputs(
                        row,
                        omit=() if condition == "full" else row["required"],
                        device=device,
                    )
                    values.append(
                        model.core(inputs, requests=[row["question"]]).detach().cpu()
                    )
                tokens[condition] = torch.cat(values)
            target, _ = bytes_batch([r["choices"][r["answer"]] for r in rows])
            cache[split] = dict(tokens=tokens, targets=target, rows=rows)
    return cache


def exploration_scores(model, cache, split, device):
    from pathwm.evaluation.understanding import choice_scores

    item = cache[split]
    with evaluation_mode(model), torch.no_grad():
        tokens = item["tokens"]["full"].to(device)
        ce = float(grounded_objective(model, tokens, item["targets"].to(device)))
        predictions = {}
        for condition, values in item["tokens"].items():
            predictions[condition] = [
                int(
                    choice_scores(
                        model.outputs, t[None].to(device), row["choices"]
                    ).argmax()
                )
                for t, row in zip(values, item["rows"])
            ]
        labels = [r["answer"] for r in item["rows"]]
    return dict(
        ce=ce,
        predictions=predictions,
        labels=labels,
        accuracy=float(np.mean(np.array(predictions["full"]) == labels)),
        omitted_accuracy=float(np.mean(np.array(predictions["omitted"]) == labels)),
    )


def exploration_advance(
    base, cache, directory, branch, depth, *, seed, contract, device
):
    """One exact existing decoder-training continuation, restored on every call."""
    import shutil
    import hashlib

    start = time.perf_counter()
    if not 0 <= branch < contract["branches"] or not 1 <= depth <= contract["depth"]:
        raise ValueError("Search action outside declared horizon")
    path = Path(directory) / f"branch{branch}"
    checkpoint = path / "last.pt"
    if (depth == 1) == checkpoint.exists():
        raise ValueError("Search action does not match branch frontier")
    if (
        depth > 1
        and torch.load(checkpoint, map_location="cpu", weights_only=True)["step"]
        != (depth - 1) * contract["block_updates"]
    ):
        raise ValueError("Search continuation skipped or repeated a block")
    seed_everything(seed)
    model = copy.deepcopy(base)
    configure_grounded_training(model, "decoder")
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=contract["learning_rates"][branch],
    )
    settings = dict(
        seed=seed,
        stage="grounded",
        variant="native",
        case="VID.order",
        scope="decoder",
        steps=contract["block_updates"] * contract["depth"],
        search_contract=contract,
        branch=branch,
    )
    cache_identity = digest(
        {
            split: dict(
                rows=digest(item["rows"]),
                tensors=[
                    hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()
                    for t in (item["targets"], *item["tokens"].values())
                ],
            )
            for split, item in cache.items()
        }
    )
    run = Run(
        path,
        settings=settings,
        data=dict(
            cache_contract=contract["fixtures"], seed=seed, cache_sha256=cache_identity
        ),
        recipe=__file__,
        model=model,
        optimizer=optimizer,
        device=device,
        resume=depth > 1,
    )
    if run.step != (depth - 1) * contract["block_updates"]:
        raise ValueError("Search continuation skipped or repeated a block")
    parent = None if depth == 1 else file_hash(checkpoint)
    train = cache["calibration"]
    training_mode(model)
    while run.step < depth * contract["block_updates"]:
        indices = run.sample(len(train["rows"]), 8)
        optimizer.zero_grad(set_to_none=True)
        loss = grounded_objective(
            model,
            train["tokens"]["full"][indices].to(device),
            train["targets"][indices].to(device),
        )
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite exploration training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            5,
            error_if_nonfinite=True,
        )
        optimizer.step()
        run.step += 1
        run.log(dict(step=run.step, split="train", loss=float(loss.detach())))
    validation = exploration_scores(model, cache, "validation", device)
    run.log(
        dict(
            step=run.step,
            split="validation",
            loss=validation["ce"],
            accuracy=validation["accuracy"],
        )
    )
    run.save()
    snapshot = path / f"block{depth}.pt"
    shutil.copyfile(checkpoint, snapshot)
    node = dict(
        branch=branch,
        depth=depth,
        score=-validation["ce"],
        seconds=time.perf_counter() - start,
        parent=parent,
        checkpoint=str(snapshot.resolve()),
        sha256=file_hash(snapshot),
        validation=validation,
        model_sha256=state_hash(model),
    )
    atomic_json(path / f"block{depth}.json", node)
    run.status("complete" if run.step == settings["steps"] else "paused", "pending")
    atomic_json(
        path / "result.json",
        dict(
            evaluation_scope="Partial decoder-training branch; validation only, no capability claim",
            metrics=validation,
            limitations=["Search may stop before its maximum depth."],
        ),
    )
    write_report(path)
    node["seconds"] = time.perf_counter() - start
    atomic_json(path / f"block{depth}.json", node)
    return node


def exploration(args):
    """Small real branch search and offline policy selection; no new training engine."""
    import shutil
    from pathwm.data.understanding import UnderstandingData
    from pathwm.evaluation.search import run_search, select_policy
    from pathwm.io import environment

    if args.search_mode == "select":
        if not args.search_histories:
            raise ValueError("Policy selection needs completed development histories")
        histories = [
            json.loads((p / "tree.json").read_text()) for p in args.search_histories
        ]
        started = time.perf_counter()
        result = select_policy(histories)
        args.output.mkdir(parents=True, exist_ok=False)
        result["seconds"] = time.perf_counter() - started
        atomic_json(args.output / "selection.json", result)
        for name in ("run.json", "recipe.py", "environment.txt"):
            shutil.copyfile(args.search_histories[0] / name, args.output / name)
        (args.output / "metrics.jsonl").write_text(
            "".join(
                json.dumps(
                    dict(
                        step=i + 1,
                        split="replay",
                        loss=-row["mean_utility"],
                        policy=row["policy"],
                    )
                )
                + "\n"
                for i, row in enumerate(result["candidates"])
            )
        )
        atomic_json(
            args.output / "result.json",
            dict(
                evaluation_scope="Offline policy selection over two recorded development worlds",
                metrics=result,
                limitations=[
                    "Historical utility only; no new training or reliable-transfer evidence."
                ],
            ),
        )
        atomic_json(
            args.output / "status.json",
            dict(result="complete", report="pending", step=0, error=None),
        )
        write_report(args.output)
        print(
            json.dumps(
                dict(selected=result["selected"], mean_utility=result["mean_utility"])
            ),
            flush=True,
        )
        return args.output
    if args.core is None or args.understanding_suite is None:
        raise ValueError("Exploration needs --core and --understanding-suite")
    start = time.perf_counter()
    base, original, source_hash = restore_readout(args.core, args.seed, args.device)
    first_encoder = next(iter(base.core.agent.encoders.values()))
    if (
        original["variant"] != "native"
        or base.core.request_readout != "none"
        or base.core.belief_readout != "sampled"
        or first_encoder.pyramid.layer_readout is not None
        or base.outputs.decoders["video"].time_conditioning != "context"
        or base.outputs.decoders["video"].palette_size != 0
    ):
        raise ValueError(
            "First replay experiment requires native sampled/context readout without layers, palette or request adaptation"
        )
    base.requires_grad_(False).eval()
    data = UnderstandingData(args.understanding_suite)
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_json(
        args.output / "status.json",
        dict(result="running", report="pending", step=0, error=None),
    )
    if torch.device(args.device).type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    cache = exploration_cache(base, data, "VID.order", args.seed, args.device)
    atomic_torch(args.output / "cache.pt", cache)
    contract = dict(
        branches=4,
        depth=4,
        block_updates=64,
        learning_rates=[0.0003, 0.001, 0.003, 0.01],
        fixtures=data.identity,
        source_sha256=source_hash,
        environment=environment(args.device),
        recipe_sha256=file_hash(__file__),
        library_sha256=digest(
            {str(p): file_hash(p) for p in Path("pathwm").rglob("*.py")}
        ),
        objective="existing grounded byte CE/log259; calibration only; frozen single-draw cache",
    )
    if args.search_selection is not None:
        selection = json.loads(args.search_selection.read_text())
        if selection["contract"] != contract or str(args.seed) in selection["worlds"]:
            raise ValueError(
                "Policy selection contract/world differs or reuses a development world"
            )
        policy = selection["selected"]
    else:
        policy = args.search_policy
    nodes = []

    def execute(branch, depth):
        if (
            time.perf_counter() - start > 300
            or shutil.disk_usage(args.output).free < 500 * 2**20
        ):
            raise RuntimeError("Exploration time/disk budget exceeded")
        node = exploration_advance(
            base,
            cache,
            args.output,
            branch,
            depth,
            seed=args.seed,
            contract=contract,
            device=args.device,
        )
        nodes.append(node)
        atomic_json(
            args.output / "partial_tree.json",
            dict(world=str(args.seed), contract=contract, nodes=nodes),
        )
        print(
            json.dumps({k: node[k] for k in ("branch", "depth", "score", "seconds")}),
            flush=True,
        )
        return node

    try:
        if args.search_mode == "collect":
            for depth in range(1, 5):
                for branch in range(4):
                    execute(branch, depth)
            search = dict(
                policy="exhaustive_history", status="complete", steps=len(nodes)
            )
        else:
            search = run_search(policy, execute)
        chosen = max(nodes, key=lambda n: n["score"])
        state = torch.load(
            chosen["checkpoint"], map_location=args.device, weights_only=True
        )
        base.load_state_dict(state["model"], strict=True)
        selected = args.output / "selected"
        selected.mkdir()
        source = Path(chosen["checkpoint"]).parent
        for name in ("run.json", "recipe.py", "source_index.json", "environment.txt"):
            shutil.copyfile(source / name, selected / name)
        shutil.copytree(source / "source", selected / "source")
        shutil.copyfile(chosen["checkpoint"], selected / "last.pt")
        (selected / "metrics.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in state["rows"])
        )
        # Reserved test is read only AFTER this episode's policy has stopped and selected its best node.
        test = (
            None
            if args.search_mode == "collect"
            else exploration_scores(base, cache, "test", args.device)
        )
        result = dict(
            world=str(args.seed),
            contract=contract,
            nodes=nodes,
            search=search,
            policy_selection_sha256=None
            if args.search_selection is None
            else file_hash(args.search_selection),
            selected=chosen,
            test=test,
            cache_sha256=file_hash(args.output / "cache.pt"),
            resources=dict(
                seconds=time.perf_counter() - start,
                updates=sum(contract["block_updates"] for _ in nodes),
                peak_allocated_bytes=torch.cuda.max_memory_allocated()
                if torch.device(args.device).type == "cuda"
                else None,
            ),
        )
        if file_hash(args.core / "last.pt") != source_hash:
            raise AssertionError("Exploration changed the source checkpoint")
        atomic_json(args.output / "tree.json", result)
        atomic_json(
            selected / "result.json",
            dict(
                evaluation_scope="Validation-selected decoder in bounded search; no model adoption",
                metrics=dict(validation_ce=-chosen["score"], test=test),
                limitations=[
                    "One cached draw; development task, not general understanding."
                ],
            ),
        )
        atomic_json(
            selected / "status.json",
            dict(result="complete", report="pending", step=state["step"], error=None),
        )
        write_report(selected)
        # Reuse the existing standalone report machinery for the complete search ledger.
        for name in ("run.json", "recipe.py", "environment.txt"):
            shutil.copyfile(selected / name, args.output / name)
        (args.output / "metrics.jsonl").write_text(
            "".join(
                json.dumps(
                    dict(
                        step=i + 1,
                        split="validation",
                        loss=-n["score"],
                        branch=n["branch"],
                    )
                )
                + "\n"
                for i, n in enumerate(nodes)
            )
        )
        atomic_json(
            args.output / "result.json",
            dict(
                evaluation_scope="Recorded branch exploration; no unobserved outcomes simulated",
                metrics=dict(
                    policy=search["policy"],
                    blocks=len(nodes),
                    validation_ce=-chosen["score"],
                    test=test,
                ),
                limitations=[
                    "Two-world engineering pilot; no reliable-transfer claim.",
                    "Collection, policy development and final audits are additional cost.",
                ],
            ),
        )
        atomic_json(
            args.output / "status.json",
            dict(result="complete", report="pending", step=len(nodes), error=None),
        )
        write_report(args.output)
        result["resources"]["end_to_end_seconds"] = time.perf_counter() - start
        atomic_json(args.output / "tree.json", result)
    except BaseException as exc:
        prior = json.loads((args.output / "status.json").read_text())
        atomic_json(
            args.output / "status.json",
            prior
            | dict(
                result=prior["result"] if prior["result"] == "complete" else "failed",
                report="failed",
                error=str(exc),
            ),
        )
        raise
    return args.output


def perform(args):
    seed_everything(args.seed)
    device = torch.device(args.device)
    populations = {
        s: dataset(s, args.seed) for s in ("train", "validation", "seen", "heldout")
    }
    data = populations["train"]
    model = Model(
        args.variant,
        posterior_aux=args.initial is not None
        and args.stage == "core"
        and args.encoder_source is None,
        video_conditioning=getattr(args, "video_conditioning", None) or "context",
        video_palette=getattr(args, "video_palette", 0),
    ).to(device)
    model.core.belief_readout = args.belief_readout
    initial_decoders = state_hash(model.outputs.decoders)
    initial_source = (
        None if args.initial is None else load_initial(model, args.initial, device)
    )
    if getattr(args, "video_conditioning", None) is not None:
        model.outputs.decoders["video"].time_conditioning = args.video_conditioning
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
        configure_encoder_readout(model, source_encoder_readout(args.core))
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
    first_encoder = next(iter(model.core.agent.encoders.values()))
    encoder_readout = getattr(args, "encoder_readout", None) or (
        "layers" if first_encoder.pyramid.layer_readout is not None else "native"
    )
    configure_encoder_readout(model, encoder_readout)
    model.requires_grad_(False)
    kinds = KINDS if args.stage == "joint" else (args.modality,)
    if args.stage in ("core", "joint"):
        model.core.requires_grad_(True)
        model.core.agent.action_head.requires_grad_(False)
        model.core.agent.monitor.requires_grad_(False)
    if encoder_source:
        model.core.agent.encoders.requires_grad_(False)
        if encoder_readout == "layers":
            for encoder in model.core.agent.encoders.values():
                encoder.pyramid.layer_readout.requires_grad_(True)
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
        video_conditioning=model.outputs.decoders["video"].time_conditioning,
        video_palette=model.outputs.decoders["video"].palette_size,
        steps=args.steps,
        batch=24,
        width=24,
        outer_iterations=2,
        initialization_checkpoint_sha256=initial_source,
        initialization_checkpoint=None
        if args.initial is None
        else str(args.initial.resolve()),
        encoder_source_sha256=encoder_source,
        factor_task=args.factor_task,
        input_mode=args.input_mode,
        belief_readout=model.core.belief_readout,
        encoder_readout=encoder_readout,
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
        if encoder_source and encoder_readout == "native":
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
            encoder_readout=encoder_readout,
            encoder_readout_parameters=sum(
                p.numel()
                for n, p in model.core.agent.encoders.named_parameters()
                if ".layer_readout." in n
            ),
            encoder_readout_gates={
                k: e.pyramid.layer_readout.gates.detach().cpu().tolist()
                for k, e in model.core.agent.encoders.items()
                if e.pyramid.layer_readout is not None
            },
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
    except BaseException as exc:
        prior = json.loads((run.path / "status.json").read_text())
        run.status(
            prior["result"] if prior["result"] in ("complete", "paused") else "failed",
            "failed",
            str(exc),
        )
        raise

    # A failed child evaluation must not mark the completed training report failed.
    if (
        getattr(args, "understanding_suite", None) is not None
        and args.stage in ("core", "joint")
        and run.step == args.steps
    ):
        evaluation_args = copy.copy(args)
        evaluation_args.core = run.path
        evaluation_args.output = run.path / "understanding"
        understanding(evaluation_args)
    return run.path


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
            "capabilities",
            "understanding-prepare",
            "understanding",
            "grounded",
            "repair-report",
            "exploration",
            "request-diagnose",
            "request-evaluate",
        ),
        default="suite",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7201)
    parser.add_argument("--variant", choices=VARIANTS, default="native")
    parser.add_argument(
        "--encoder-readout",
        choices=("native", "layers"),
        help="Optional same-scale depth readout; omitted restores source architecture",
    )
    parser.add_argument("--modality", choices=KINDS, default="text")
    parser.add_argument("--core", type=Path)
    parser.add_argument(
        "--search-mode", choices=("collect", "select", "execute"), default="execute"
    )
    parser.add_argument(
        "--search-policy",
        choices=("round_robin12", "round_robin8", "greedy", "plateau1", "plateau2"),
        default="round_robin12",
    )
    parser.add_argument("--search-histories", type=Path, nargs="+")
    parser.add_argument("--search-selection", type=Path)
    parser.add_argument(
        "--understanding-cases",
        nargs="+",
        help="Explicit evaluation subset with a separate comparison contract",
    )
    parser.add_argument(
        "--grounded-case",
        default="VID.order",
        help="Controlled calibration task to supervise",
    )
    parser.add_argument(
        "--grounded-scope",
        choices=("decoder", "core", "interpreter"),
        default="decoder",
    )
    parser.add_argument(
        "--request-profile", choices=("narrow", "balanced"), default="narrow"
    )
    parser.add_argument(
        "--request-readout",
        choices=("none", "constant", "instruction"),
        help="Grounded task route; omitted restores the source mode",
    )
    parser.add_argument(
        "--request-contrasts",
        action="store_true",
        help="Grounded VID.order training with paired short/sequence requests",
    )
    parser.add_argument(
        "--retention-weight",
        type=float,
        default=0.0,
        help="Grounded replay-only frozen-source output distillation;0 disables",
    )
    parser.add_argument(
        "--understanding-suite",
        type=Path,
        help="Prepared fixtures; optionally evaluate after completed core/joint training",
    )
    parser.add_argument(
        "--real-root", type=Path, default=Path("data/tau_urban_av_2021")
    )
    parser.add_argument("--profile", choices=("quick", "full"), default="quick")
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
        "--reference",
        type=Path,
        help="Repair study root or matching understanding baseline run",
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
        help="Run directory or checkpoint file: load weights/readout metadata with fresh optimizer; not resume",
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
    parser.add_argument(
        "--video-conditioning",
        choices=("context", "query"),
        help="Place requested time in context or queries; default restores source metadata or context",
    )
    parser.add_argument(
        "--video-palette",
        type=int,
        choices=(0, 4),
        default=0,
        help="Optional learned untimed palette; requires query timing",
    )
    args = parser.parse_args()
    if args.stage in ("request-diagnose", "request-evaluate"):
        if (
            args.core is None
            or args.understanding_suite is None
            or args.steps
            or args.resume
            or args.stop_after
        ):
            parser.error(
                "Request diagnosis needs source and fixtures without training/resume"
            )
        (request_diagnose if args.stage == "request-diagnose" else request_evaluate)(
            args
        )
        return
    if args.stage == "exploration":
        if args.resume or args.stop_after or args.steps:
            parser.error(
                "Exploration uses the fixed block protocol and fresh output directories"
            )
        exploration(args)
        return
    if args.stage == "grounded":
        if args.core is None or args.understanding_suite is None:
            parser.error("Grounded training needs --core and --understanding-suite")
        args.steps = args.steps or 768
        if args.steps < 1 or (args.stop_after is not None and args.stop_after < 1):
            parser.error("Training lengths must be positive")
        grounded(args)
        return
    if args.stage == "understanding-prepare":
        from pathwm.data.understanding import prepare_understanding

        print(prepare_understanding(args.output, args.real_root, args.profile))
        return
    if args.stage == "understanding":
        if (
            args.core is None
            or args.understanding_suite is None
            or args.resume
            or args.stop_after
            or args.steps
        ):
            parser.error(
                "Understanding evaluation needs --core and --understanding-suite, without training/resume"
            )
        understanding(args)
        return
    if args.understanding_suite is not None and args.stage not in (
        "core",
        "joint",
        "suite",
    ):
        parser.error(
            "Automatic understanding evaluation is supported after core/joint training"
        )
    if args.encoder_readout is not None and args.stage != "core":
        parser.error(
            "Encoder readout selection is for core training; evaluation restores its source"
        )
    if args.video_palette and args.video_conditioning != "query":
        parser.error("Video palette requires explicit --video-conditioning query")
    if args.input_mode != "rotating" and args.stage not in ("core", "frozen"):
        parser.error(
            "Input selection is supported for core or frozen-output training only"
        )
    if args.video_conditioning is not None and args.stage not in (
        "oracle",
        "frozen",
        "joint",
    ):
        parser.error("Video conditioning is supported for output training only")
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
            or (args.initial and args.encoder_source is None)
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
    if args.stage in ("diagnose", "capabilities"):
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
