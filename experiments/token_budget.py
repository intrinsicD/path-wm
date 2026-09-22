"""Profile bounded belief-state processing; no training or quality claim.

python -m experiments.token_budget --output runs/token_budget_baseline --device cuda
The packet reference preserves the original complete-event scheduling for comparison.
"""

import argparse
import gc
from pathlib import Path
from statistics import median
import time

import torch

from pathwm.models.agent import Thinker, ActionHead, ErrorMonitor
from pathwm.models.belief import BeliefAgent, BeliefCorrection, BeliefDynamics
from pathwm.models.belief_state import Packet
from pathwm.models.hybrid_memory import HybridMemory
from pathwm.models.modalities import Observation, ImageDecoder
from pathwm.models.multiscale import (
    MultiScaleImageEncoder,
    MultiScaleAudioEncoder,
    MultiScaleTextEncoder,
)
from pathwm.evaluation.workload import Workload
from pathwm.evaluation.report import write_report
from pathwm.io import Run, atomic_json, seed_everything


def build_model():
    return BeliefAgent(
        width=32,
        context_tokens=16,
        latent_groups=8,
        latent_codes=8,
        evidence_tokens=8,
        encoders=dict(
            image=MultiScaleImageEncoder(32),
            video=MultiScaleImageEncoder(32, video=True),
            audio=MultiScaleAudioEncoder(32, 32),
            text=MultiScaleTextEncoder(32),
        ),
        decoders=dict(image=ImageDecoder(32, 16)),
        updater=BeliefCorrection(32, 8, 8, 2),
        dynamics=BeliefDynamics(32, 8, 8, 2),
        thinker=Thinker(32),
        memory=HybridMemory(32, recent=32, block=8, blocks=16),
        action_head=ActionHead(32),
        monitor=ErrorMonitor(32),
    )


def examples(device, *, large=False):
    def image(size, frames=1):
        return Observation(
            torch.rand(1, frames, 3, size, size, device=device),
            torch.linspace(0, 1, frames, device=device)[None],
        )

    cases = {f"image_{size}": {"image": image(size)} for size in (32, 64, 128)}
    cases["video_4x32"] = {"video": image(32, 4)}
    if large:
        cases["image_256"] = {"image": image(256)}
        cases["video_4x64"] = {"video": image(64, 4)}
    cases["multimodal"] = dict(
        image=image(32),
        video=image(32, 4),
        audio=Observation(
            torch.rand(1, 1, 32, device=device), torch.ones(1, 1, device=device)
        ),
        text=Observation(
            torch.randint(1, 259, (1, 16), device=device),
            torch.ones(1, 16, device=device),
        ),
    )
    return cases


def packet_reference(model, state, observations):
    ordinal = state.ordinal + 1
    pending = model.begin_event(
        state, event_id=f"event-{ordinal}", ordinal=ordinal, time=1.0
    )
    for name, observation in sorted(observations.items()):
        pending = model.add_packet(
            pending, Packet(f"{ordinal}/{name}", name, observation)
        )
    return model.commit_event(pending)


def forward(model, state, observations, reference, loops):
    state = (
        packet_reference(model, state, observations)
        if reference
        else model.observe(state, observations, time=1.0)
    )
    state = model.think(state, steps=loops)
    loss = (
        state.logits.square().mean()
        + state.tokens.square().mean()
        + state.evidence.square().mean()
    )
    return state, loss


def measure(model, state, obs, reference, args):
    def synchronize():
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()

    rows = []
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()  # once before warmup, not between steady-state repeats
    for i in range(args.warmups + args.repeats):
        model.zero_grad(set_to_none=True)
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(args.seed + 1)
        synchronize()
        begin = time.perf_counter()
        output, loss = forward(model, state, obs, reference, args.loops)
        synchronize()
        after_forward = time.perf_counter()
        loss.backward()
        synchronize()
        after_backward = time.perf_counter()
        peak_allocated = (
            torch.cuda.max_memory_allocated()
            if args.device.startswith("cuda")
            else None
        )
        peak_reserved = (
            torch.cuda.max_memory_reserved() if args.device.startswith("cuda") else None
        )
        del output, loss
        synchronize()
        begin_inference = time.perf_counter()
        with torch.no_grad():
            output, loss = forward(model, state, obs, reference, args.loops)
        synchronize()
        inference_ms = 1000 * (time.perf_counter() - begin_inference)
        if i >= args.warmups:
            rows.append(
                dict(
                    forward_ms=1000 * (after_forward - begin),
                    backward_ms=1000 * (after_backward - after_forward),
                    inference_ms=inference_ms,
                    peak_allocated_bytes=peak_allocated,
                    peak_reserved_bytes=peak_reserved,
                    loss=float(loss.detach()),
                )
            )
        if (
            args.device.startswith("cuda")
            and torch.cuda.max_memory_allocated() > 6 * 1024**3
        ):
            raise RuntimeError("Declared 6 GiB allocation ceiling exceeded")
        del output, loss
    model.zero_grad(set_to_none=True)
    torch.manual_seed(args.seed + 1)
    with torch.no_grad(), Workload(model) as work:
        out, _ = forward(model, state, obs, reference, args.loops)
        work.record_state(out)
    forwards = sorted(r["forward_ms"] for r in rows)
    return dict(
        raw=rows,
        forward_ms=median(r["forward_ms"] for r in rows),
        forward_range_ms=[min(forwards), max(forwards)],
        backward_ms=median(r["backward_ms"] for r in rows),
        inference_ms=median(r["inference_ms"] for r in rows),
        peak_allocated_bytes=max(r["peak_allocated_bytes"] or 0 for r in rows),
        peak_reserved_bytes=max(r["peak_reserved_bytes"] or 0 for r in rows),
        workload=work.summary(),
    )


def profile_operators(model, state, obs, reference, args):
    # After ALL latency measurements: profiler initialization can affect later
    # timings even outside its explicit recording context.
    handles, regions = [], []

    def enter(name):
        def hook(*unused):
            region = torch.profiler.record_function("region/" + name)
            region.__enter__()
            regions.append(region)

        return hook

    def leave(*unused):
        regions.pop().__exit__(None, None, None)

    modules = {"encoder/" + name: module for name, module in model.encoders.items()}
    modules.update(
        {
            name: getattr(model, name)
            for name in (
                "updater",
                "dynamics",
                "thinker",
                "evidence_encoder",
                "observation_resampler",
            )
            if getattr(model, name, None) is not None
        }
    )
    for name, module in modules.items():
        handles.append(module.register_forward_pre_hook(enter(name)))
        handles.append(module.register_forward_hook(leave, always_call=True))
    activities = [torch.profiler.ProfilerActivity.CPU]
    if args.device.startswith("cuda"):
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    try:
        with torch.no_grad(), torch.profiler.profile(activities=activities) as profile:
            forward(model, state, obs, reference, args.loops)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
    finally:
        for handle in handles:
            handle.remove()
    return [
        dict(
            name=e.key,
            device_type=str(e.device_type),
            calls=e.count,
            cpu_total_us=e.cpu_time_total,
            device_total_us=e.device_time_total,
        )
        for e in profile.key_averages()
        if e.key.startswith("region/") or "scaled_dot_product" in e.key
    ]


def encoder_access(model, mode, window):
    """Same weights, explicit layout/access arm; no persistent feature cache."""
    for name, encoder in model.encoders.items():
        for merge in encoder.pyramid.merges:
            merge.packed = mode != "dense_encoder"
        for i, stage in enumerate(encoder.pyramid.stages):
            stage.window = (
                (2 if name == "video" else 1, window, window)
                if mode == "local_encoder"
                and name in ("image", "video")
                and i < len(encoder.pyramid.stages) - 1
                else None
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--loops", type=int, default=4)
    parser.add_argument("--resampler", type=int, default=0)
    parser.add_argument(
        "--encoder-window",
        type=int,
        default=0,
        help="Compare dense / packed merges / packed fine windows; 0 runs the event/resampler comparison",
    )
    args = parser.parse_args()
    if args.encoder_window < 0 or (args.encoder_window and args.resampler):
        parser.error("Compare local encoders separately from learned resampling")
    if args.repeats < 1 or args.warmups < 0 or args.loops < 0 or args.resampler < 0:
        parser.error(
            "Positive repetitions, nonnegative warmups/loops/resampler required"
        )
    seed_everything(args.seed)
    model = build_model().to(args.device).eval()
    cases = examples(args.device, large=bool(args.encoder_window))
    with torch.no_grad():
        state = model.initial_state(1, session_id="token-budget")
    if args.resampler:
        from pathwm.models.resampler import LatentResampler

        # Initialized after the shared baseline, registered for reproducible snapshots.
        model.observation_resampler = LatentResampler(32, tokens=args.resampler).to(
            args.device
        )
    sampler = getattr(model, "observation_resampler", None)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-3
    )  # snapshot only; no updates
    run = Run(
        args.output,
        settings={
            **vars(args),
            "output": str(args.output),
            "purpose": "resource-profile",
        },
        data={"source": "seeded synthetic shape probes; not a quality dataset"},
        recipe=__file__,
        model=model,
        optimizer=optimizer,
        device=args.device,
    )
    results = {}
    started = time.monotonic()
    try:
        for index, (name, obs) in enumerate(cases.items()):
            results[name] = {}
            # Alternate arm order to reduce a systematic warm/clock bias.
            arms = [("packet_reference", True), ("complete_event", False)]
            if args.encoder_window:
                arms = [
                    ("dense_encoder", False),
                    ("packed_merges", False),
                    ("local_encoder", False),
                ]
            if index % 2:
                arms.reverse()
            if sampler is not None:
                arms.append(("resampled", False))
            for label, reference in arms:
                model.observation_resampler = sampler if label == "resampled" else None
                if args.encoder_window:
                    encoder_access(model, label, args.encoder_window)
                result = measure(model, state, obs, reference, args)
                results[name][label] = result
                run.log(
                    dict(
                        step=index,
                        split=f"profile_{name}_{label}",
                        **{
                            k: v
                            for k, v in result.items()
                            if k not in ("workload", "raw", "operators")
                        },
                    )
                )
                print(
                    name,
                    label,
                    {k: round(v, 3) for k, v in result.items() if isinstance(v, float)},
                    flush=True,
                )
                if time.monotonic() - started > 600:
                    raise RuntimeError("Declared ten-minute benchmark budget exceeded")
        for name, obs in cases.items():
            for label, result in results[name].items():
                model.observation_resampler = sampler if label == "resampled" else None
                if args.encoder_window:
                    encoder_access(model, label, args.encoder_window)
                result["operators"] = profile_operators(
                    model, state, obs, label == "packet_reference", args
                )
        model.observation_resampler = sampler
        if args.encoder_window:
            encoder_access(model, "dense_encoder", args.encoder_window)
        atomic_json(run.path / "workload.json", results)
        metrics = {
            name + "/" + arm: {
                k: v
                for k, v in result.items()
                if k not in ("workload", "raw", "operators")
            }
            for name, arms in results.items()
            for arm, result in arms.items()
        }
        if args.encoder_window:
            target = results["image_256"]

            def total(r):
                return r["forward_ms"] + r["backward_ms"]

            improvement = 1 - total(target["local_encoder"]) / total(
                target["dense_encoder"]
            )
        else:
            improvement = (
                1
                - results["multimodal"]["complete_event"]["forward_ms"]
                / results["multimodal"]["packet_reference"]["forward_ms"]
            )
        atomic_json(
            run.path / "result.json",
            dict(
                evaluation_scope="Untrained synthetic shape/resource probes; not reconstruction, prediction or information-retention evidence.",
                metrics=metrics,
                timing_reduction=improvement,
                timing_screen="image256 forward+backward local/dense"
                if args.encoder_window
                else "multimodal forward complete/packet",
                timing_screen_passed=improvement >= 0.2,
                resampler_parameters=0
                if sampler is None
                else sum(p.numel() for p in sampler.parameters()),
                protocol="Deterministic FP32 eval-mode differentiable forward + backward; fixed empty initial memory; allocator warmups; hooks/profiler in separate passes. No optimizer updates. Raw repeats and backend/region observations in workload.json.",
                limitations="Attention FLOPs are partial estimates; score elements are not materialized memory. Memory peaks are whole-pass, reserved memory diagnostic only. Small synthetic cases can be dispatch-bound. No streaming long-memory, trained quality, whole-pipeline VRAM or high-resolution retention claim.",
            ),
        )
        run.save()
        run.status("complete", "pending")
    except BaseException as error:
        atomic_json(run.path / "partial_workload.json", results)
        run.status("failed", "pending", str(error))
        raise
    print(write_report(run.path))


if __name__ == "__main__":
    main()
