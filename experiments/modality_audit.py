"""Isolated small codec fits and persistent integration for every current modality.

No independent holdout or general speech/language/video capability claim. Native
decoder quality and untrained state-context compatibility are reported separately.
"""

import argparse
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
import json
import subprocess
import wave

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image

from pathwm.models.multiscale import (
    MultiScaleImageEncoder,
    MultiScaleAudioEncoder,
    MultiScaleTextEncoder,
)
from pathwm.models.modalities import (
    ImageDecoder,
    AudioDecoder,
    TextDecoder,
    Observation,
    bytes_batch,
    bytes_text,
)
from pathwm.models.agent import Thinker, ActionHead, ErrorMonitor
from pathwm.models.belief import BeliefAgent, BeliefCorrection, BeliefDynamics
from pathwm.models.belief_state import Packet
from pathwm.models.hybrid_memory import HybridMemory
from pathwm.world_state.modules import (
    CandidateEncoder,
    Candidate,
    AssociationBinder,
    ReplaceUpdater,
    ContextEncoder,
)
from pathwm.world_state.session import WorldSession
from pathwm.world_state.retrieval import Query
from pathwm.world_state.inspection import WorldTrace, inspect_store
from pathwm.io import (
    Run,
    seed_everything,
    atomic_json,
    atomic_torch,
    file_hash,
    evaluation_mode,
    resume_arguments,
)
from pathwm.evaluation.report import write_report


KINDS = ("text", "audio", "video", "image")


class Codec(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        common = dict(code_width=8, levels=3)
        if kind == "text":
            self.encoder, self.decoder = (
                MultiScaleTextEncoder(16, **common),
                TextDecoder(16),
            )
        elif kind == "audio":
            self.encoder, self.decoder = (
                MultiScaleAudioEncoder(256, 16, **common),
                AudioDecoder(16, 256),
            )
        else:
            self.encoder = MultiScaleImageEncoder(16, video=kind == "video", **common)
            self.decoder = ImageDecoder(16, 16)

    def decode(self, tokens, obs, *, zero=False, shuffle=False):
        values = torch.zeros_like(tokens.values) if zero else tokens.values
        if shuffle:
            values = values.roll(1, 0)  # fixed wrong-example context at evaluation only
        if self.kind == "text":
            return self.decoder(values, obs.values[:, :-1], valid=tokens.valid)
        if self.kind == "video":
            # Same decoder at each target time, no future feature admitted. This
            # reconstructs observed frames, not a forecast of unobserved frames.
            return torch.stack(
                [
                    self.decoder(
                        values, valid=tokens.valid & (tokens.times <= time[:, None])
                    )
                    for time in obs.times.unbind(1)
                ],
                1,
            )
        return self.decoder(values, valid=tokens.valid)

    def forward(self, obs, *, trace=None):
        tokens = self.encoder(obs, trace=trace).as_tokens()
        return self.decode(tokens, obs), tokens


class Study(nn.Module):
    def __init__(self):
        super().__init__()
        self.codecs = nn.ModuleDict({kind: Codec(kind) for kind in KINDS})
        self.projections = nn.ModuleDict(
            {kind: CandidateEncoder(16, 4, 16) for kind in KINDS}
        )
        self.context = ContextEncoder(
            16, {"state": nn.Identity()}, {"state": ("belief", "state-v1")}
        )
        self.updater = ReplaceUpdater(16)
        self.agent = BeliefAgent(
            width=16,
            context_tokens=4,
            latent_groups=4,
            latent_codes=4,
            evidence_tokens=4,
            time_unit="seconds",
            encoders={kind: codec.encoder for kind, codec in self.codecs.items()},
            decoders={
                kind: self.codecs[kind].decoder for kind in ("text", "audio", "image")
            },
            updater=BeliefCorrection(16, 4, 4, 2),
            dynamics=BeliefDynamics(16, 4, 4, 2),
            thinker=Thinker(16),
            memory=HybridMemory(16, recent=2, block=2, blocks=1, latent_codes=4),
            action_head=ActionHead(16),
            monitor=ErrorMonitor(16),
        )
        self.requires_grad_(False)
        self.codecs.requires_grad_(True)

    def session(self, snapshot=None):
        binder = AssociationBinder()
        binder.scorer.eval()
        modules = dict(
            agent=self.agent,
            binder=binder,
            updater=self.updater,
            context_encoder=self.context,
        )
        return (
            WorldSession(**modules)
            if snapshot is None
            else WorldSession.restore(snapshot, **modules)
        )


def examples():
    text, mask = bytes_batch(["rot", "blau", "gruen", "gelb"])
    time = torch.arange(256).float() / 256
    audio = torch.stack(
        [
            0.6 * torch.sin(2 * torch.pi * f * time + i * 0.2)
            for i, f in enumerate([3, 5, 7, 11])
        ]
    )
    video = torch.full((4, 4, 3, 16, 16), 0.05)
    colors = torch.tensor(
        [[0.9, 0.15, 0.1], [0.1, 0.2, 0.9], [0.1, 0.8, 0.2], [0.8, 0.7, 0.1]]
    )
    for i in range(4):
        for t in range(4):
            y, x = 2 + 2 * i, 1 + 3 * (t if i % 2 == 0 else 3 - t)
            video[i, t, :, y : y + 4, x : x + 4] = colors[i, :, None, None]
    return {
        "text": Observation(text, torch.ones_like(text, dtype=torch.float64), mask),
        "audio": Observation(audio[:, None], torch.ones(4, 1, dtype=torch.float64)),
        "video": Observation(
            video, torch.arange(1, 5, dtype=torch.float64)[None].expand(4, -1)
        ),
        "image": Observation(
            video[:, :1].clone(), torch.ones(4, 1, dtype=torch.float64)
        ),
    }


def target(kind, obs):
    return (
        obs.values[:, 1:]
        if kind == "text"
        else obs.values
        if kind == "video"
        else obs.values[:, 0]
    )


def objective(kind, output, obs):
    wanted = target(kind, obs)
    return (
        F.cross_entropy(output.flatten(0, 1), wanted.flatten(), ignore_index=0)
        if kind == "text"
        else F.mse_loss(output, wanted)
    )


def loss_value(kind, output, obs):
    return float(objective(kind, output, obs).detach())


def save_wave(path, values, rate=8000):
    samples = np.rint(np.clip(values, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(samples.tobytes())


def real_inputs(source):
    # Explicit one-source transport check; raw clip remains immutable. Audio is
    # one second at 8 kHz, padded into 256-sample chunks; the final partial chunk
    # is omitted here rather than claiming its padding is observed signal.
    args = ["ffmpeg", "-v", "error", "-i", str(source), "-t", "1"]
    raw = subprocess.run(
        args
        + [
            "-an",
            "-vf",
            "fps=4,scale=16:16",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
    ).stdout
    frames = (
        torch.from_numpy(
            np.frombuffer(raw, dtype=np.uint8).copy().reshape(-1, 16, 16, 3)
        )
        .permute(0, 3, 1, 2)
        .float()
        / 255
    )
    raw = subprocess.run(
        args + ["-vn", "-ac", "1", "-ar", "8000", "-f", "f32le", "pipe:1"],
        check=True,
        capture_output=True,
    ).stdout
    samples = torch.from_numpy(np.frombuffer(raw, dtype="<f4").copy())
    chunks = samples[: len(samples) // 256 * 256].reshape(1, -1, 256)
    text, mask = bytes_batch(
        ["Tür öffnen. Grüße aus Köln!", "Ton und Video sind Beobachtungen."]
    )
    return {
        "video": Observation(
            frames[None], torch.arange(len(frames), dtype=torch.float64)[None] / 4
        ),
        "audio": Observation(
            chunks,
            torch.arange(1, chunks.shape[1] + 1, dtype=torch.float64)[None]
            * 256
            / 8000,
        ),
        "text": Observation(text, torch.ones_like(text, dtype=torch.float64), mask),
    }


@torch.no_grad()
def evaluate(model, data, runner, trace, source):
    metrics, arrays = {}, {}
    # Reproducible evaluation random draws independent of pause/resume or debug.
    with torch.random.fork_rng():
        torch.manual_seed(61302)
        for kind, obs in data.items():
            codec = model.codecs[kind]
            output, tokens = codec(obs, trace=trace)
            score = dict(
                loss=loss_value(kind, output, obs),
                zero_context_loss=loss_value(
                    kind, codec.decode(tokens, obs, zero=True), obs
                ),
                shuffled_context_loss=loss_value(
                    kind, codec.decode(tokens, obs, shuffle=True), obs
                ),
                feature_tokens=tokens.values.shape[1],
            )
            arrays[kind + "_target"] = target(kind, obs).numpy()
            arrays[kind + "_output"] = output.numpy()
            if kind == "text":
                generated = codec.decoder.generate(
                    tokens.values, max_tokens=12, valid=tokens.valid
                )
                null = codec.decoder.generate(
                    torch.zeros_like(tokens.values), max_tokens=12, valid=tokens.valid
                )
                score.update(
                    generated=[bytes_text(row) for row in generated],
                    target=[bytes_text(row) for row in obs.values],
                    zero_context_generated=[bytes_text(row) for row in null],
                )
                score["free_exact_match"] = (
                    sum(a == b for a, b in zip(score["generated"], score["target"])) / 4
                )
                wanted = target(kind, obs)
                counts = torch.bincount(wanted[wanted != 0], minlength=259).float()
                probs = counts / counts.sum()
                score["unigram_nll"] = float(-probs[wanted[wanted != 0]].log().mean())
            else:
                wanted = target(kind, obs)
                score["constant_reference_loss"] = float(
                    F.mse_loss(wanted.mean(0, keepdim=True).expand_as(wanted), wanted)
                )
                if kind == "audio":
                    score["silence_loss"] = float(wanted.square().mean())
                    save_wave(
                        runner.path / "audio_target.wav", wanted.flatten().numpy()
                    )
                    save_wave(
                        runner.path / "audio_output.wav", output.flatten().numpy()
                    )
                if kind == "video":
                    reversed_obs = replace(obs, values=obs.values.flip(1))
                    reversed_output, _ = codec(reversed_obs)
                    score["reversed_input_output_change"] = float(
                        F.mse_loss(output, reversed_output)
                    )
                    score["copy_first_frame_loss"] = float(
                        F.mse_loss(wanted[:, :1].expand_as(wanted), wanted)
                    )
                    for name, values in [
                        ("video_target", wanted),
                        ("video_output", output),
                    ]:
                        frames = [
                            Image.fromarray(
                                np.rint(
                                    x.permute(1, 2, 0).numpy().clip(0, 1) * 255
                                ).astype("uint8")
                            )
                            for x in values[0]
                        ]
                        frames[0].save(
                            runner.path / (name + ".gif"),
                            save_all=True,
                            append_images=frames[1:],
                            duration=250,
                            loop=0,
                        )
            # Independent single-modality session: no other input can supply content.
            one = Observation(
                obs.values[:1],
                obs.times[:1],
                None if obs.valid is None else obs.valid[:1],
            )
            features = codec.encoder(one).as_tokens()
            keys, values = model.projections[kind].pool(features)
            candidate = Candidate(
                "window",
                kind,
                kind,
                keys[0, 0],
                values[0, 0],
                kind,
                "audit-v1",
                provenance=one.provenance,
            )
            session = model.session()
            record = session.observe(
                kind,
                occurred_at=float(one.times.max()),
                available_at=float(one.times.max()),
                candidates=(candidate,),
                packets=(Packet(kind, kind, one),),
                trace=trace,
            )
            entity = record["bindings"][0]["entity_id"]
            snapshot = session.snapshot()
            state, context, encoded = session.think(
                Query(entity_ids=(entity,)), trace=trace
            )
            restored = model.session(snapshot)
            replay, _, _ = restored.think(Query(entity_ids=(entity,)))
            score["exact_session_replay"] = torch.equal(state.tokens, replay.tokens)
            score["source_preserved"] = session.store.evidence()[0].modality == kind
            # Frozen, untrained adapters/core: report compatibility separately.
            runtime = (
                codec.decoder(state.tokens, one.values[:, :-1])
                if kind == "text"
                else codec.decoder(state.tokens)
            )
            runtime_target = one.values[:, -1] if kind == "video" else target(kind, one)
            score["untrained_state_path_loss"] = (
                float(
                    F.cross_entropy(
                        runtime.flatten(0, 1), runtime_target.flatten(), ignore_index=0
                    )
                )
                if kind == "text"
                else float(F.mse_loss(runtime, runtime_target))
            )
            score["state_path_trained"] = False
            atomic_torch(runner.path / (kind + "_session.pt"), snapshot)
            metrics[kind] = score
        real = real_inputs(source)
        transport = {}
        for kind, obs in real.items():
            pyramid = model.codecs[kind].encoder(obs, trace=trace)
            pooled = model.projections[kind].pool(pyramid)
            transport[kind] = dict(
                input_shape=list(obs.values.shape),
                feature_shapes=[list(s.values.shape) for s in pyramid.scales],
                finite=all(
                    torch.isfinite(s.values).all().item() for s in pyramid.scales
                ),
                candidate_shape=list(pooled[0].shape),
            )
        # Co-present raw packets form one episode event, not a guessed cross-modal identity.
        session = model.session()
        packets = tuple(
            Packet(
                kind,
                kind,
                Observation(
                    obs.values[:1],
                    obs.times[:1],
                    None if obs.valid is None else obs.valid[:1],
                ),
            )
            for kind, obs in real.items()
        )
        session.observe(
            "real-av-text", occurred_at=1, available_at=1, packets=packets, trace=trace
        )
        state, _, _ = session.think(Query(), trace=trace)
        transport["mixed_event"] = dict(
            observation_count=state.observation_count,
            sources=list(state.memory.recent[0].sources),
            finite=bool(torch.isfinite(state.tokens).all()),
        )
        atomic_json(runner.path / "world_state.json", inspect_store(session.store))
        trace.export(runner.path)
    np.savez_compressed(runner.path / "modalities.npz", **arrays)
    return metrics, transport


def run(args):
    seed_everything(args.seed)
    torch.set_num_threads(2)
    model = Study()
    data = examples()
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=0.003
    )
    settings = dict(
        seed=args.seed,
        steps=args.steps,
        per_modality_updates=args.steps,
        learning_rate=0.003,
        purpose="development",
        modalities=list(KINDS),
        source=str(args.source.resolve()),
        scope="four fixed examples per branch; isolated codec learning and separate untrained persistent adapter",
    )
    runner = Run(
        args.resume or args.output,
        settings=settings,
        data={"kind": "modality-audit-v1", "source_sha256": file_hash(args.source)},
        recipe=__file__,
        model=model,
        optimizer=optimizer,
        device="cpu",
        resume=args.resume is not None,
    )
    trace = WorldTrace(max_records=500, tensor_values=8192)
    try:
        while runner.step < args.steps * len(KINDS):
            if args.stop_after is not None and runner.step >= args.stop_after:
                break
            kind = KINDS[runner.step % len(KINDS)]
            model.train()
            optimizer.zero_grad(set_to_none=True)
            codec = model.codecs[kind]
            with (
                trace.capture(codec, ["encoder", "decoder"], gradients=True)
                if runner.step < 4
                else nullcontext()
            ):
                output, _ = codec(data[kind])
                loss = objective(kind, output, data[kind])
                loss.backward()
            if not all(
                p.grad is None or torch.isfinite(p.grad).all()
                for p in model.parameters()
            ):
                raise ValueError("Nonfinite gradient")
            optimizer.step()
            runner.step += 1
            runner.log(
                dict(
                    step=runner.step,
                    split="train",
                    modality=kind,
                    loss=float(loss.detach()),
                )
            )
        runner.save()
        with evaluation_mode(model):
            metrics, transport = evaluate(model, data, runner, trace, args.source)
        for kind, score in metrics.items():
            rows = [r for r in runner.rows if r["modality"] == kind]
            if not rows:
                continue
            score["initial_loss"] = rows[0]["loss"]
            score["loss_ratio"] = score["loss"] / rows[0]["loss"]
            score["learning_gate"] = score["loss_ratio"] <= 0.8
        atomic_json(
            runner.path / "result.json",
            dict(
                metrics=metrics,
                real_transport=transport,
                gate=all(
                    x.get("learning_gate", False) and x["exact_session_replay"]
                    for x in metrics.values()
                ),
                scope=settings["scope"],
                source_sha256=file_hash(args.source),
            ),
        )
        atomic_json(
            runner.path / "modality_audit.json",
            dict(
                metrics=metrics,
                real_transport=transport,
                audio_rate=8000,
                audio_samples=256,
                video_fps=4,
                real_source=str(args.source),
                limitations=[
                    "Development fitting only; no independent holdout",
                    "Untrained state adapters are not semantically compatible by tensor shape alone",
                    "Whole-window candidates are not object discovery",
                    "Audio is 32 ms/chunk, not speech synthesis",
                    "Video is observed-frame reconstruction, not future prediction",
                    "Real AV is a one-second 16x16/8kHz transport check, not semantic evaluation",
                ],
            ),
        )
        complete = runner.step == args.steps * len(KINDS)
        runner.status("completed" if complete else "paused", "pending")
        write_report(runner.path)
        print(
            json.dumps(
                {"output": str(runner.path), "step": runner.step, "metrics": metrics},
                indent=2,
            )
        )
    except Exception as error:
        # Preserve completed training when only final evaluation/reporting failed.
        runner.status(
            "completed" if runner.step == args.steps * len(KINDS) else "failed",
            "failed",
            str(error),
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("runs/modality_foundation_v1/development")
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--seed", type=int, default=61301)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/memory_media_v1/episodes/0LDP7/video.mp4"),
    )
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--check", action="store_true")
    args = resume_arguments(parser, parser.parse_args())
    if args.check:
        args.stop_after = 4
    if args.steps < 1 or (args.stop_after is not None and args.stop_after < 1):
        parser.error("positive updates required")
    run(args)


if __name__ == "__main__":
    main()
