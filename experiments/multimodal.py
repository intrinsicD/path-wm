"""One editable multimodal world-model recipe, initially a CPU development run.

Synthetic synchronized images/video, impact waveforms and short byte descriptions
exercise every adapter. --dataset pusht uses actual observations/actions and omits
audio/text. Defaults are software-development budgets, not capability benchmarks.
"""

import argparse
import copy
import json
from pathlib import Path
import wave

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from pathwm.models.modalities import (
    Observation,
    ImageEncoder,
    VectorEncoder,
    TextEncoder,
    ImageDecoder,
    AudioDecoder,
    TextDecoder,
    bytes_batch,
    bytes_text,
)
from pathwm.models.agent_state import EpisodicMemory
from pathwm.models.agent import (
    MultimodalAgent,
    ObservationUpdate,
    LatentDynamics,
    Thinker,
    ActionHead,
    ErrorMonitor,
)
from pathwm.data.sequences import PushTSequences
from pathwm.evaluation.agent import plan
from pathwm.evaluation.report import write_report, render_report
from pathwm.io import (
    Run,
    atomic_json,
    digest,
    evaluation_mode,
    seed_everything,
    training_mode,
    trainable_parameters,
    resume_arguments,
)
from pathwm.training.improvement import replay_probabilities, try_improvement


def build_model(width=32, image_size=16, audio_samples=32):
    # Replace a constructor here to compare components; no registration is needed.
    return MultimodalAgent(
        width=width,
        encoders={
            "image": ImageEncoder(width),
            "video": ImageEncoder(width),
            "audio": VectorEncoder(audio_samples, width),
            "text": TextEncoder(width),
        },
        decoders={
            "image": ImageDecoder(width, image_size),
            "audio": AudioDecoder(width, audio_samples),
            "text": TextDecoder(width),
        },
        updater=ObservationUpdate(width),
        dynamics=LatentDynamics(width),
        thinker=Thinker(width),
        memory=EpisodicMemory(capacity=16, retrieve_count=2),
        action_head=ActionHead(width),
        monitor=ErrorMonitor(width),
    )


class SyntheticEpisodes:
    """Deterministic, artificial data with split-specific seeds; no external assets.

    A colored soft ball has inertia, controlled acceleration and reflecting walls.
    Impact sound and left/right/hit descriptions share its timestamps. Each index
    is a separate generated episode, so train/validation never share trajectories.
    """

    def __init__(
        self,
        split="train",
        count=32,
        image_size=16,
        audio_samples=32,
        history=2,
        horizon=2,
    ):
        if (
            split not in ("train", "validation", "test")
            or min(count, history, horizon) < 1
        ):
            raise ValueError("Invalid synthetic split or population")
        seed = dict(train=101, validation=202, test=303)[split]
        generator = np.random.default_rng(seed)
        n, length = count, history + horizon
        pos = generator.uniform(0.12, 0.88, (n, 2)).astype("float32")
        velocity = generator.uniform(-0.12, 0.12, (n, 2)).astype("float32")
        color = generator.uniform(0.35, 1, (n, 3)).astype("float32")
        actions = generator.uniform(-1, 1, (n, length - 1, 2)).astype("float32")
        y, x = np.meshgrid(
            np.linspace(0, 1, image_size), np.linspace(0, 1, image_size), indexing="ij"
        )
        images, audio, descriptions = [], [], []
        hit = np.zeros(n, dtype=bool)
        for t in range(length):
            distance = (x[None] - pos[:, 0, None, None]) ** 2 + (
                y[None] - pos[:, 1, None, None]
            ) ** 2
            ball = np.exp(-distance / 0.008).astype("float32")
            images.append(ball[:, None] * color[:, :, None, None])
            phase = np.arange(audio_samples, dtype="float32") / 8000
            sound = 0.4 * np.sin(2 * np.pi * (400 + pos[:, :1] * 400) * phase)
            audio.append((sound * hit[:, None]).astype("float32"))
            descriptions.append(
                [
                    ("left" if p[0] < 0.5 else "right") + (" hit" if h else " move")
                    for p, h in zip(pos, hit)
                ]
            )
            if t < length - 1:
                velocity = 0.9 * velocity + 0.06 * actions[:, t]
                pos = pos + velocity
                contact = (pos < 0.08) | (pos > 0.92)
                hit = contact.any(1)
                velocity = np.where(contact, -velocity, velocity)
                pos = np.where(
                    pos < 0.08, 0.16 - pos, np.where(pos > 0.92, 1.84 - pos, pos)
                )
        text, _ = bytes_batch(
            [descriptions[t][i] for i in range(n) for t in range(length)]
        )
        self.arrays = dict(
            images=torch.from_numpy(np.stack(images, 1)),
            actions=torch.from_numpy(actions),
            audio=torch.from_numpy(np.stack(audio, 1)),
            text=text.reshape(n, length, -1),
        )
        self.history, self.horizon = history, horizon
        self.identity = dict(
            kind="synthetic-controlled-ball-v1",
            split=split,
            seed=seed,
            count=count,
            image_size=image_size,
            audio_samples=audio_samples,
            sample_rate=8000,
            history=history,
            horizon=horizon,
            time_unit="one generated transition",
            content_sha256=digest(
                {k: digest(v.tolist()) for k, v in self.arrays.items()}
            ),
        )

    def __len__(self):
        return len(self.arrays["images"])

    def batch(self, indices, device="cpu"):
        ids = torch.as_tensor(indices, dtype=torch.long)
        if ids.ndim != 1 or len(ids) < 1 or (ids < 0).any() or (ids >= len(self)).any():
            raise ValueError("Batch indices must be nonempty and in bounds")
        return {
            **{k: v[ids].to(device) for k, v in self.arrays.items()},
            "sources": [
                f"synthetic/{self.identity['split']}/episode-{int(i)}" for i in ids
            ],
        }


class RealEpisodes:
    def __init__(self, root, split, count, image_size, history, horizon):
        self.data = PushTSequences(
            root, split, history=history, horizon=horizon, limit=count
        )
        self.image_size = image_size
        self.history, self.horizon = history, horizon
        self.identity = dict(
            self.data.identity,
            resize=image_size,
            modalities=["image", "video"],
            action_transform="2 * normalized_xy - 1",
            time_unit="one recorded transition; no physical timestamp calibration",
        )

    def __len__(self):
        return len(self.data)

    def batch(self, indices, device="cpu"):
        raw = self.data.batch(indices, device)
        images = torch.cat((raw["history"], raw["future"]), 1)
        b, t = images.shape[:2]
        images = F.interpolate(
            images.flatten(0, 1),
            (self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).reshape(b, t, 3, self.image_size, self.image_size)
        actions = torch.cat((raw["history_actions"], raw["actions"]), 1) * 2 - 1
        if not torch.isfinite(actions).all() or (actions.abs() > 1.00001).any():
            raise ValueError(
                "Prepared action coordinates do not match normalized [0,1] contract"
            )
        sources = []
        for index in indices:
            episode, start = self.data.windows[int(index)]
            group = int(self.data.episodes[episode]["group_id"])
            sources.append(
                f"pusht/{self.data.identity['split']}/group-{group}/start-{start}"
            )
        return {"images": images, "actions": actions, "sources": sources}


def observations(batch, t):
    images = batch["images"]
    times = images.new_full((len(images), 1), t)
    start = max(0, t - 1)
    video_times = torch.arange(start, t + 1, device=images.device, dtype=images.dtype)[
        None
    ].expand(len(images), -1)
    result = {
        "image": Observation(images[:, t : t + 1], times),
        "video": Observation(images[:, start : t + 1], video_times),
    }
    if "audio" in batch:
        result["audio"] = Observation(batch["audio"][:, t : t + 1], times)
    if "text" in batch:
        text = batch["text"][:, t]
        result["text"] = Observation(text, times.expand_as(text), text != 0)
    return result


def observe_history(model, batch, history, *, trace=None, dropout=0.0):
    state = model.initial_state(len(batch["images"]))
    for t in range(history):
        obs = observations(batch, t)
        if dropout:
            obs = {
                k: v
                for k, v in obs.items()
                if k == "image" or torch.rand(()) >= dropout
            }
        state = model.observe(
            state,
            obs,
            time=t,
            previous_action=None if t == 0 else batch["actions"][:, t - 1],
            trace=trace,
        )
        source = json.dumps(batch["sources"], separators=(",", ":")) + f"/frame-{t}"
        state = model.remember(state, source=source)
    return model.think(state, steps=1, trace=trace)


def output_losses(model, state, batch, t):
    prefix = batch["text"][:, t, :-1] if "text" in batch else None
    outputs = model.decode(
        state,
        text_prefix=prefix,
        modalities=[
            "image",
            *(["audio"] if "audio" in batch else []),
            *(["text"] if "text" in batch else []),
        ],
    )
    losses = {
        "image_mse": (outputs["image"] - batch["images"][:, t]).square().mean((1, 2, 3))
    }
    if "audio" in batch:
        losses["audio_mse"] = (outputs["audio"] - batch["audio"][:, t]).square().mean(1)
    if "text" in batch:
        target = batch["text"][:, t, 1:]
        ce = F.cross_entropy(
            outputs["text"].transpose(1, 2), target, ignore_index=0, reduction="none"
        )
        losses["text_ce"] = ce.sum(1) / (target != 0).sum(1).clamp_min(1)
    return losses


class LearningState(nn.Module):
    """Checkpoint the deployed model, its training-only EMA copy and replay scores."""

    def __init__(self, model, population):
        super().__init__()
        self.agent = model
        self.target = copy.deepcopy(model).requires_grad_(False).eval()
        self.register_buffer("replay_errors", torch.ones(population))
        self.register_buffer("proposals", torch.tensor(0))
        self.register_buffer("accepted", torch.tensor(0))

    @torch.no_grad()
    def update_target(self, decay):
        for target, current in zip(self.target.parameters(), self.agent.parameters()):
            target.lerp_(current, 1 - decay)
        for target, current in zip(self.target.buffers(), self.agent.buffers()):
            target.copy_(current)


def objective(learner, batch, history=2, horizon=2, dropout=0.0):
    model = learner.agent
    state = observe_history(model, batch, history, dropout=dropout)
    reconstruction = output_losses(model, state, batch, history - 1)
    with torch.no_grad():
        target = observe_history(learner.target, batch, history)
    totals, errors, latent_losses = {}, [], []
    future = state
    for h in range(horizon):
        t = history + h
        future = model.imagine(future, batch["actions"][:, t - 1], dt=1.0)
        losses = output_losses(model, future, batch, t)
        for k, v in losses.items():
            totals[k] = totals.get(k, 0) + v / horizon
        errors.append(losses["image_mse"])
        with torch.no_grad():
            target = learner.target.observe(
                target,
                observations(batch, t),
                time=t,
                previous_action=batch["actions"][:, t - 1],
            )
            target = learner.target.think(target, steps=1)
        residual = (future.tokens - target.tokens.detach()) * (-future.log_scale).exp()
        latent_losses.append(
            (
                0.5 * residual.square() + future.log_scale + 0.5 * np.log(2 * np.pi)
            ).mean()
        )
    image_error = torch.stack(errors).mean(0)
    action = batch["actions"][:, history - 1].clamp(-0.9999, 0.9999)
    action_mean, action_scale = model.action_head(state.tokens)
    raw = torch.atanh(action)
    action_nll = (
        0.5 * ((raw - action_mean) * (-action_scale).exp()).square()
        + action_scale
        + 0.5 * np.log(2 * np.pi)
        + torch.log1p(-action.square())
    ).mean()
    # Explicit small weights; data-space objectives anchor the evolving latent target.
    weighted = {
        f"future_{k}": v.mean() * (0.1 if k == "text_ce" else 1.0)
        for k, v in totals.items()
    }
    weighted.update(
        {
            f"reconstruct_{k}": 0.25 * v.mean() * (0.1 if k == "text_ce" else 1.0)
            for k, v in reconstruction.items()
        }
    )
    weighted["latent_nll"] = 0.01 * torch.stack(latent_losses).mean()
    weighted["action_nll"] = 0.05 * action_nll
    weighted["self_error_mse"] = (
        0.1 * (model.monitor(state.tokens) - image_error.detach()).square().mean()
    )
    std = state.tokens.flatten(0, 1).std(0, unbiased=False)
    weighted["variance_floor"] = 0.01 * F.relu(0.1 - std).mean()
    diagnostics = {k: v.detach().mean() for k, v in totals.items()}
    diagnostics.update(
        latent_error_mse=(future.tokens - target.tokens).detach().square().mean(),
        latent_predicted_variance=future.log_scale.detach().mul(2).exp().mean(),
        self_error_mse=(model.monitor(state.tokens) - image_error.detach())
        .detach()
        .square()
        .mean(),
    )
    return weighted, image_error.detach(), diagnostics


def update(learner, optimizer, data, indices, settings):
    training_mode(learner)
    batch = data.batch(indices, settings["device"])
    optimizer.zero_grad(set_to_none=True)
    losses, errors, _ = objective(
        learner, batch, settings["history"], settings["horizon"], dropout=0.25
    )
    loss = sum(losses.values())
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite multimodal objective")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(
        trainable_parameters(learner), 1.0, error_if_nonfinite=True
    )
    optimizer.step()
    learner.update_target(settings["ema_decay"])
    with torch.no_grad():
        # Duplicate sampled indices are deliberately averaged, independently of order.
        ids = torch.as_tensor(indices, device=errors.device)
        for index in ids.unique():
            learner.replay_errors[index] = errors[ids == index].mean()
    return {
        **{k: float(v.detach()) for k, v in losses.items()},
        "loss": float(loss.detach()),
        "grad_norm": float(norm),
    }


@torch.no_grad()
def evaluate(learner, data, settings):
    sums, seen, pooled = {}, 0, []
    with evaluation_mode(learner):
        for start in range(0, len(data), settings["batch_size"]):
            ids = np.arange(start, min(start + settings["batch_size"], len(data)))
            batch = data.batch(ids, settings["device"])
            losses, _, raw = objective(
                learner, batch, settings["history"], settings["horizon"]
            )
            state = observe_history(learner.agent, batch, settings["history"])
            pooled.append(state.tokens.mean(1))
            current = batch["images"][:, settings["history"] - 1]
            target = batch["images"][
                :, settings["history"] : settings["history"] + settings["horizon"]
            ]
            raw["copy_image_mse"] = (current[:, None] - target).square().mean()
            raw["loss"] = sum(losses.values())
            if "audio" in batch:
                raw["silence_audio_mse"] = (
                    batch["audio"][:, settings["history"] :].square().mean()
                )
            for k, v in raw.items():
                sums[k] = sums.get(k, 0.0) + float(v) * len(ids)
            seen += len(ids)
    values = torch.cat(pooled)
    centered = values - values.mean(0)
    singular = torch.linalg.svdvals(centered)
    proportions = singular.square() / singular.square().sum().clamp_min(1e-12)
    rank = (
        (-(proportions * proportions.clamp_min(1e-12).log()).sum()).exp()
        if singular.square().sum() > 1e-12
        else 0.0
    )
    return {
        **{k: v / seen for k, v in sums.items()},
        "evaluated_windows": seen,
        "latent_between_example_std": float(values.std(0, unbiased=False).mean()),
        "latent_effective_rank": float(rank),
    }


def make_data(settings, split):
    count = (
        settings["train_windows"]
        if split == "train"
        else settings["validation_windows"]
    )
    if settings["dataset"] == "synthetic":
        return SyntheticEpisodes(
            split,
            count,
            settings["image_size"],
            settings["audio_samples"],
            settings["history"],
            settings["horizon"],
        )
    return RealEpisodes(
        settings["data_root"],
        split,
        count,
        settings["image_size"],
        settings["history"],
        settings["horizon"],
    )


def check(settings):
    seed_everything(settings["seed"])
    data = make_data(settings, "train")
    learner = LearningState(
        build_model(
            settings["width"], settings["image_size"], settings["audio_samples"]
        ),
        len(data),
    ).to(settings["device"])
    batch = data.batch(
        np.arange(min(settings["batch_size"], len(data))), settings["device"]
    )
    losses, _, _ = objective(learner, batch, settings["history"], settings["horizon"])
    sum(losses.values()).backward()
    return {
        "data": data.identity,
        "parameters": sum(p.numel() for p in learner.agent.parameters()),
        "losses": {k: float(v.detach()) for k, v in losses.items()},
        "gradients": {
            name: any(
                p.grad is not None and bool(p.grad.abs().sum())
                for p in module.parameters()
            )
            for name, module in learner.agent.named_children()
        },
        "target_has_gradients": any(
            p.grad is not None for p in learner.target.parameters()
        ),
    }


@torch.no_grad()
def save_examples(run, learner, data, settings):
    with evaluation_mode(learner):
        model = learner.agent
        batch = data.batch(np.arange(min(3, len(data))), settings["device"])
        trace = {}
        state = observe_history(model, batch, settings["history"], trace=trace)
        imagined, future = [], state
        for h in range(settings["horizon"]):
            step_trace = {}
            future = model.imagine(
                future,
                batch["actions"][:, settings["history"] - 1 + h],
                trace=step_trace,
            )
            imagined.append(future)
            trace.update({f"future.{h}.{k}": v for k, v in step_trace.items()})
        video = model.decode_video(imagined)
        audio = torch.stack(
            [model.decode(s, modalities=["audio"])["audio"] for s in imagined], 1
        )
        text = model.generate_text(imagined[-1], max_tokens=24)
        proposed = model.propose_action(state)
        # Compare proposal, zero and negative proposal under an explicit goal image.
        candidates = torch.stack((proposed, torch.zeros_like(proposed), -proposed), 1)
        candidates = candidates[:, :, None].expand(-1, -1, settings["horizon"], -1)
        goal = batch["images"][:, -1]
        decision = plan(
            model,
            state,
            candidates,
            lambda s: (model.decode(s, modalities=["image"])["image"] - goal)
            .square()
            .mean((1, 2, 3)),
            lower=-1.0,
            upper=1.0,
            trace=trace,
        )
        baseline_image = model.decode(state, modalities=["image"])["image"]
        ablations = {}
        for role in model.layout:
            changed = model.decode(
                model.intervene(state, role, 0.0), modalities=["image"]
            )["image"]
            ablations[role] = float((changed - baseline_image).square().mean())
        torch.save(
            {
                "state": state.to_dict(),
                "future": [s.to_dict() for s in imagined],
                "trace": trace,
            },
            run.path / "inspection.pt",
        )
        np.savez_compressed(
            run.path / "inspection.npz",
            attention=trace["observe.attention"][0].mean(0).numpy(),
            token_activity=state.tokens[0].abs().cpu().numpy(),
            video=video.cpu().numpy(),
            audio=audio.cpu().numpy(),
        )
        np.savez_compressed(
            run.path / "multimodal_outputs.npz",
            video=video.cpu().numpy(),
            audio=audio.cpu().numpy(),
            text_tokens=text.cpu().numpy(),
            actions=decision.actions.cpu().numpy(),
        )
        frames = [
            Image.fromarray(
                (frame.permute(1, 2, 0).cpu().clamp(0, 1).numpy() * 255).astype("uint8")
            )
            for frame in video[0]
        ]
        frames[0].save(
            run.path / "imagined.gif",
            save_all=True,
            append_images=frames[1:],
            duration=250,
            loop=0,
        )
        with wave.open(str(run.path / "imagined.wav"), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(8000)
            # Clips are contiguous decoder waveform chunks, not claimed synchronized speech.
            stream.writeframes(
                (audio[0].flatten().cpu().clamp(-1, 1).numpy() * 32767)
                .astype("<i2")
                .tobytes()
            )
        atomic_json(
            run.path / "inspection.json",
            {
                "step": run.step,
                "latent_groups": model.groups,
                "generated_text": [bytes_text(row) for row in text],
                "plan_scores": decision.scores.cpu().tolist(),
                "selected_candidates": decision.indices.cpu().tolist(),
                "memory_sources": list(state.memory.sources),
                "time": state.time.cpu().tolist(),
                "future_times": [s.time.cpu().tolist() for s in imagined],
                "self_error_estimate": model.monitor(state.tokens).cpu().tolist(),
                "group_activity_rms": {
                    name: float(state.tokens[:, sl].square().mean().sqrt())
                    for name, sl in model.layout.items()
                },
                "zero_group_image_change_mse": ablations,
                "attention_inputs": trace["observe.input_modalities"],
                "audio_sample_rate": 8000,
                "audio_samples_per_state": settings["audio_samples"],
                "trace_keys": sorted(trace),
                "caveats": [
                    "Development outputs, not trained semantic or physical capabilities.",
                    "The demonstration planner is given the true final image as its explicit goal.",
                    "The goal is used only by scoring; future observations never enter imagined inference.",
                    "Uncertainty is conditional latent spread, not a calibrated confidence score.",
                    "Audio/text outputs on PushT are untrained because those modalities are absent.",
                    "Token roles, attention weights and PCA colors are not causal explanations.",
                ],
            },
        )
        features = {
            name: state.tokens[:, sl].transpose(1, 2).unsqueeze(2)
            for name, sl in model.layout.items()
        }
        write_report(
            run.path, {"rgb": batch["images"][:, -1]}, {"rgb": video[:, -1]}, features
        )


def train(settings, output, *, resume=False, stop_after=None):
    seed_everything(settings["seed"])
    training, validation = (
        make_data(settings, "train"),
        make_data(settings, "validation"),
    )
    learner = LearningState(
        build_model(
            settings["width"], settings["image_size"], settings["audio_samples"]
        ),
        len(training),
    ).to(settings["device"])
    optimizer = torch.optim.AdamW(
        trainable_parameters(learner), lr=settings["learning_rate"]
    )
    run = Run(
        output,
        settings=settings,
        data={"train": training.identity, "validation": validation.identity},
        recipe=__file__,
        model=learner,
        optimizer=optimizer,
        device=settings["device"],
        resume=resume,
    )

    def sample():
        probabilities = replay_probabilities(learner.replay_errors.cpu())
        return torch.multinomial(
            probabilities,
            settings["batch_size"],
            replacement=True,
            generator=run.sampler,
        ).numpy()

    try:
        if not run.rows:
            run.log(
                dict(
                    step=0,
                    split="validation",
                    **evaluate(learner, validation, settings),
                )
            )
            run.save()
        end = (
            min(settings["steps"], run.step + stop_after)
            if stop_after is not None
            else settings["steps"]
        )
        for step in range(run.step + 1, end + 1):
            metrics = update(learner, optimizer, training, sample(), settings)
            run.step = step
            run.log(
                dict(
                    step=step,
                    split="train",
                    main_examples=step * settings["batch_size"],
                    **metrics,
                )
            )
            if settings["improve_every"] and step % settings["improve_every"] == 0:
                metric_names = [
                    "image_mse",
                    *(
                        ["audio_mse", "text_ce"]
                        if settings["dataset"] == "synthetic"
                        else []
                    ),
                ]

                def admission_metrics():
                    measured = evaluate(learner, validation, settings)
                    return {k: measured[k] for k in metric_names}

                admission = try_improvement(
                    learner,
                    optimizer,
                    lambda: update(learner, optimizer, training, sample(), settings),
                    admission_metrics,
                    primary="image_mse",
                    min_improvement=1e-5,
                    tolerances={
                        k: (
                            0.01
                            if k == "text_ce"
                            else 0.001
                            if k == "audio_mse"
                            else 0.0
                        )
                        for k in metric_names
                    },
                    generators=(run.sampler,),
                )
                learner.proposals.add_(1)
                learner.accepted.add_(int(admission["accepted"]))
                run.log(dict(step=step, split="proposal", **admission))
            if step % settings["evaluate_every"] == 0 or step == end:
                val = evaluate(learner, validation, settings)
                run.log(
                    dict(
                        step=step,
                        split="validation",
                        attempted_extra_updates=int(learner.proposals),
                        accepted_extra_updates=int(learner.accepted),
                        **val,
                    )
                )
                print(
                    f"step {step}/{settings['steps']} validation image MSE {val['image_mse']:.6f}; copy {val['copy_image_mse']:.6f}",
                    flush=True,
                )
            run.save()
        result = "completed" if run.step == settings["steps"] else "paused"
        run.status(result, "pending")
    except BaseException as exc:
        run.status("failed", "pending", f"{type(exc).__name__}: {exc}")
        try:
            render_report(run.path)
        except Exception as report_error:
            run.status("failed", "failed", f"{exc}; report: {report_error}")
        raise
    try:
        save_examples(run, learner, validation, settings)
    except BaseException as exc:
        run.status(result, "failed", f"{type(exc).__name__}: {exc}")
        raise
    return run.path


def export_diagrams(
    output, *, depth=2, width=32, image_size=16, audio_samples=32, seed=42
):
    """Draw current modules and an executed example; no training or downloads."""
    from pathwm.evaluation.diagrams import CallFlow, architecture, write_diagrams
    from pathwm.io import source_record

    def plan_to_image(model, state, candidates, goal):
        return plan(
            model,
            state,
            candidates,
            lambda future: (model.decode(future, modalities=["image"])["image"] - goal)
            .square()
            .mean((1, 2, 3)),
            lower=-1.0,
            upper=1.0,
        )

    seed_everything(seed)
    model = build_model(width, image_size, audio_samples).eval()
    data = SyntheticEpisodes(
        count=1,
        history=2,
        horizon=2,
        image_size=image_size,
        audio_samples=audio_samples,
    )
    batch = data.batch([0])
    flow = CallFlow()
    with torch.no_grad(), evaluation_mode(model):
        inputs = {
            name: flow.input(name + " input", observation)
            for name, observation in observations(batch, 1).items()
        }
        initial = flow.call(model.initial_state, 1)
        observed = flow.call(model.observe, initial, inputs, time=1.0)
        remembered = flow.call(
            model.remember, observed, source="synthetic/train/episode-0/frame-1"
        )
        thought = flow.call(model.think, remembered, steps=1)
        action = flow.call(model.propose_action, thought)
        future = flow.call(model.imagine, thought, action, dt=1.0)
        later = flow.call(model.imagine, future, action, dt=1.0)
        outputs = flow.call(model.decode, future, modalities=["image", "audio"])
        for name, value in outputs.items():
            flow.output(name + " output", value)
        flow.output("text output", flow.call(model.generate_text, future, max_tokens=8))
        flow.output("video output", flow.call(model.decode_video, [future, later]))
        candidates = flow.input("Candidate action sequences", torch.zeros(1, 2, 2, 2))
        goal = flow.input("Goal image", batch["images"][:, -1])
        decision = flow.call(plan_to_image, model, thought, candidates, goal)
        flow.output("Selected action sequence", decision.actions)
    root = Path(__file__).resolve().parents[1]
    provenance = dict(
        seed=seed,
        torch_version=torch.__version__,
        device="cpu",
        depth=depth,
        model=dict(width=width, image_size=image_size, audio_samples=audio_samples),
        data=data.identity,
        initialization="fresh model; no trained-capability claim",
        calls="one observation, memory write, one thought, two imagined steps and one bounded planning call",
        sources={
            str(Path(p).relative_to(root)): sha
            for p, sha in source_record(__file__, model)["files"].items()
        },
    )
    return write_diagrams(
        output,
        {"architecture": architecture(model, depth), "data_flow": flow.graph()},
        provenance,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--diagram",
        type=Path,
        nargs="?",
        const=Path("docs/diagrams"),
        help="Export architecture/data-flow diagrams to this directory (CPU, no training)",
    )
    parser.add_argument("--diagram-depth", type=int, default=2)
    parser.add_argument("--output", default="runs/multimodal_first")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument(
        "--dataset", choices=["synthetic", "pusht"], default="synthetic"
    )
    parser.add_argument("--data-root", default="data/pusht_world_model/cchi_v1")
    for name, default in [
        ("steps", 8),
        ("batch-size", 2),
        ("width", 32),
        ("image-size", 16),
        ("audio-samples", 32),
        ("history", 2),
        ("horizon", 2),
        ("train-windows", 32),
        ("validation-windows", 8),
        ("evaluate-every", 4),
        ("seed", 42),
        ("improve-every", 4),
    ]:
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--ema-decay", type=float, default=0.99)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.diagram is not None and (
        args.check
        or args.resume
        or args.stop_after is not None
        or args.device != "cpu"
        or args.dataset != "synthetic"
    ):
        parser.error(
            "--diagram uses a fresh synthetic CPU example; run it separately from training/check/resume"
        )
    if args.diagram_depth < 0:
        parser.error("Diagram depth must be nonnegative")
    args = resume_arguments(parser, args)
    counts = [
        args.steps,
        args.batch_size,
        args.width,
        args.image_size,
        args.audio_samples,
        args.history,
        args.horizon,
        args.train_windows,
        args.validation_windows,
        args.evaluate_every,
    ]
    if (
        min(counts) < 1
        or args.width % 4
        or args.image_size % 4
        or args.improve_every < 0
    ):
        parser.error(
            "Counts must be positive; width and image size must divide by 4; improve-every may be zero"
        )
    if (
        not 0 <= args.ema_decay < 1
        or not np.isfinite(args.learning_rate)
        or args.learning_rate <= 0
    ):
        parser.error("Learning rate must be positive and EMA decay in [0,1)")
    if args.stop_after is not None and args.stop_after < 1:
        parser.error("Stop-after must be positive")
    if args.diagram is not None:
        print(
            export_diagrams(
                args.diagram,
                depth=args.diagram_depth,
                width=args.width,
                image_size=args.image_size,
                audio_samples=args.audio_samples,
                seed=args.seed,
            )
        )
        return
    settings = {
        k: v
        for k, v in vars(args).items()
        if k
        not in ("check", "output", "resume", "stop_after", "diagram", "diagram_depth")
    }
    settings.update(
        purpose="development",
        precision="fp32",
        time_unit="one dataset transition",
        objective="data reconstruction + future outputs + EMA latent Gaussian NLL + action NLL + error supervision",
        proposal_budget="one extra update per improve-every interval, rolled back on rejection",
    )
    if args.check:
        print(json.dumps(check(settings), indent=2))
    else:
        print(
            train(
                settings,
                args.resume or args.output,
                resume=args.resume is not None,
                stop_after=args.stop_after,
            )
            / "report.html"
        )


if __name__ == "__main__":
    main()
