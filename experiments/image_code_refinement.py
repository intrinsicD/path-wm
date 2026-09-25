"""Slow CUDA debugging reconstruction from actual native saved fine codes.

No model weights are trained. Child processes receive only the checkpoint and code;
source pixels remain in the parent scorer. This does not qualify generated codes.
The registered full native decoder is required (pyramid + fine subpixel connections).

    python -m experiments.image_code_refinement --checkpoint RUN/last.pt --output NEW_RUN --scenes 256 --starts decoder
    python -m experiments.image_code_refinement --checkpoint RUN/last.pt --code CODE.pt --decoded OUTPUT.pt
"""

import argparse, hashlib, json, math, subprocess, sys, time, shutil
from pathlib import Path
import torch
import torch.utils.serialization.config
from pathwm.io import (
    atomic_json,
    atomic_torch,
    load_component,
    state_hash,
    file_hash,
    source_record,
    environment,
    seed_everything,
)
from pathwm.models.slots import SlotPerception
from pathwm.models.image_code import (
    make_image_code,
    save_image_code,
    load_image_code,
    decode_image_code,
)


def model(path):
    p = SlotPerception().eval()
    p.decoder.enable_pyramid_connections()
    p.decoder.enable_fine_subpixels()
    load_component(p, path, "perception")
    return p.cuda().eval().requires_grad_(False)


def sha(*xs):
    h = hashlib.sha256()
    for x in xs:
        h.update(f"{x.dtype}{tuple(x.shape)}".encode())
        h.update(x.cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def child(a):
    # Load optimizer runtime before auditing model-data reads. No model/image inputs.
    warm = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([warm])
    warm.grad = torch.zeros_like(warm)
    opt.step()
    reads = []

    def audit(event, args):
        if (
            event == "open"
            and isinstance(args[0], (str, bytes))
            and args[1] is not None
            and "r" in args[1]
        ):
            reads.append(str(args[0]))

    sys.addaudithook(audit)
    p = model(a.checkpoint)
    code = load_image_code(a.code)
    before = state_hash(p)
    rng = torch.get_rng_state().clone()
    cuda = torch.cuda.get_rng_state().clone()
    with torch.no_grad():
        initial = decode_image_code(p, code).recon
    x = (
        initial.clone() if a.start == "decoder" else torch.full_like(initial, 0.5)
    ).requires_grad_()
    target = code["fine"]["values"].cuda()
    optimizer = torch.optim.Adam([x], lr=0.03)
    rows = []
    for step in range(a.steps):
        per = (p.pyramid(x).scales[0].values - target).square().mean((1, 2))
        loss = per.sum()
        grad = torch.autograd.grad(loss, x)[0]
        if not torch.isfinite(grad).all():
            raise FloatingPointError("Nonfinite pixel gradient")
        x.grad = grad
        optimizer.param_groups[0]["lr"] = (
            0.03 * (1 + math.cos(math.pi * step / a.steps)) / 2
        )
        optimizer.step()
        with torch.no_grad():
            x.clamp_(0, 1)
        if step % 100 == 0:
            rows.append(dict(step=step, feature_mse=per.detach().cpu().tolist()))
    with torch.no_grad():
        feature = (
            (p.pyramid(x).scales[0].values - target)
            .square()
            .mean((1, 2))
            .cpu()
            .tolist()
        )
    allowed = {a.checkpoint.resolve(), a.code.resolve()}
    checks = dict(
        encoder_binding=state_hash(p.encoder) == code["encoder_sha256"],
        weights=before == state_hash(p),
        parameter_grads=all(v.grad is None for v in p.parameters()),
        rng=torch.equal(rng, torch.get_rng_state())
        and torch.equal(cuda, torch.cuda.get_rng_state()),
        steps=int(optimizer.state[x]["step"]) == a.steps,
        finite=bool(torch.isfinite(x).all()),
        bounded=bool(((x >= 0) & (x <= 1)).all()),
        data_reads=all(Path(f).resolve() in allowed for f in reads),
        memory=torch.cuda.max_memory_reserved() / 2**30 <= 2,
    )
    atomic_torch(a.decoded, dict(rgb=x.detach().cpu(), initial=initial.cpu()))
    atomic_json(
        str(a.decoded) + ".json",
        dict(
            checks=checks,
            feature_mse=feature,
            rows=rows,
            read_paths=reads,
            reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
        ),
    )
    assert all(checks.values()), checks


def errors(x, y, entity):
    e = (x - y).square()
    body = ((entity == 1) | (entity == 2))[:, None]
    n = body.sum((1, 2, 3))
    dx = lambda z: z[:, :, :, 1:] - z[:, :, :, :-1]
    dy = lambda z: z[:, :, 1:, :] - z[:, :, :-1, :]
    return dict(
        mse=e.mean((1, 2, 3)).tolist(),
        machine_body_mse=((e * body).sum((1, 2, 3)) / (3 * n).clamp_min(1)).tolist(),
        gradient_error=(
            0.5
            * (
                (dx(x) - dx(y)).square().mean((1, 2, 3))
                + (dy(x) - dy(y)).square().mean((1, 2, 3))
            )
        ).tolist(),
    )


def parent(a):
    from pathwm.data import rule_world as rw
    from pathwm.evaluation.report import write_report

    started = time.monotonic()
    out = a.output
    out.mkdir(exist_ok=False)
    p = model(a.checkpoint)
    source = source_record(__file__, p)
    old = SlotPerception().eval()
    load_component(
        old, "runs/real_visual_joint_repair_3501_u6000_v1/last.pt", "perception"
    )
    assert state_hash(p.encoder) == state_hash(old.encoder)
    del old
    atomic_json(
        out / "run.json",
        dict(
            schema="pathwm-run-v1",
            identity=dict(
                settings=dict(
                    stage="native-code-refinement",
                    steps=a.steps,
                    starts=a.starts,
                    scenes=a.scenes,
                    chunk=16,
                    checkpoint=str(a.checkpoint),
                    checkpoint_sha256=file_hash(a.checkpoint),
                    encoder_sha256=state_hash(p.encoder),
                    lr=0.03,
                    schedule="cosine",
                    budget_seconds=600,
                    max_gib=2,
                ),
                environment=environment("cuda"),
            ),
            source=source,
        ),
    )
    shutil.copyfile(__file__, out / "recipe.py")
    atomic_json(
        out / "status.json",
        dict(result="running", report="pending", step=0, error=None),
    )
    original = json.loads(
        Path(
            "runs/native_pyramid_decoder_3601_pyramid_v1/evaluation_final.json"
        ).read_text()
    )
    checks = {}
    metrics = {}
    rows = []
    examples = None
    try:
        for pop, seed in [("train", 3602), ("validation", 3603)]:
            g = torch.Generator().manual_seed(seed)
            metrics[pop] = {
                start: {
                    k: []
                    for k in (
                        "mse",
                        "machine_body_mse",
                        "gradient_error",
                        "feature_mse",
                    )
                }
                for start in a.starts
            }
            for chunk in range(a.scenes // 64):
                picked = torch.tensor(rw.KIND_SPLIT[pop])[
                    torch.randint(len(rw.KIND_SPLIT[pop]), (64, 2), generator=g)
                ]
                scene = rw.sample_scenes(g, picked)
                lamps = torch.randint(2, (64, 2), generator=g)
                scene = rw.Scenes(*(v.cuda() for v in vars(scene).values()))
                rgb, entity = rw.render(scene, lamps.cuda())
                checks[f"{pop}_{chunk}_inputs"] = (
                    sha(rgb, entity)
                    == original["summary"]["inputs"][pop]["chunk_sha256"][chunk]
                )
                with torch.no_grad():
                    code = make_image_code(
                        p,
                        p.pyramid(rgb),
                        provenance=[
                            dict(
                                kind="observed",
                                id=f"{pop}/{chunk * 64 + i}",
                                available_at=0.0,
                            )
                            for i in range(64)
                        ],
                    )
                for offset in range(0, 64, 16):
                    part = dict(
                        code,
                        fine={
                            k: (v[offset : offset + 16] if torch.is_tensor(v) else v)
                            for k, v in code["fine"].items()
                        },
                        condition=code["condition"][offset : offset + 16],
                        provenance=code["provenance"][offset : offset + 16],
                    )
                    path = out / f"{pop}-{chunk}-{offset}.code.pt"
                    save_image_code(path, part)
                    for start in a.starts:
                        if time.monotonic() - started > 600:
                            raise TimeoutError("Refinement600secondbudget")
                        decoded = out / f"{pop}-{chunk}-{offset}-{start}.pt"
                        subprocess.run(
                            [
                                sys.executable,
                                "-m",
                                "experiments.image_code_refinement",
                                "--checkpoint",
                                str(a.checkpoint),
                                "--code",
                                str(path),
                                "--decoded",
                                str(decoded),
                                "--steps",
                                str(a.steps),
                                "--start",
                                start,
                            ],
                            check=True,
                            timeout=max(1, 600 - (time.monotonic() - started)),
                        )
                        result = torch.load(decoded, weights_only=True)
                        receipt = json.loads(Path(str(decoded) + ".json").read_text())
                        checks[f"{pop}_{chunk}_{offset}_{start}_software"] = all(
                            receipt["checks"].values()
                        )
                        raw = errors(
                            result["rgb"],
                            rgb[offset : offset + 16].cpu(),
                            entity[offset : offset + 16].cpu(),
                        )
                        raw["feature_mse"] = receipt["feature_mse"]
                        raw.update(
                            {
                                "decoder_" + k: v
                                for k, v in errors(
                                    result["initial"],
                                    rgb[offset : offset + 16].cpu(),
                                    entity[offset : offset + 16].cpu(),
                                ).items()
                            }
                        )
                        for k, v in raw.items():
                            metrics[pop][start].setdefault(k, []).extend(v)
                        rows.append(
                            dict(
                                step=a.steps,
                                split=f"{pop}_{start}",
                                **{k: sum(v) / len(v) for k, v in raw.items()},
                            )
                        )
                        if (
                            examples is None
                            and pop == "validation"
                            and start == "decoder"
                        ):
                            examples = (
                                rgb[offset : offset + 8].cpu(),
                                result["rgb"][:8],
                            )
        checks["source_unchanged"] = all(
            file_hash(f) == h for f, h in source["files"].items()
        )
        checks["time"] = time.monotonic() - started <= 600
        quality = {
            f"{pop}_{start}": max(v["mse"]) <= 1e-4
            and sum(v["machine_body_mse"]) / len(v["machine_body_mse"]) <= 1e-4
            and sum(v["gradient_error"]) / len(v["gradient_error"]) <= 1e-4
            for pop, m in metrics.items()
            for start, v in m.items()
        }
        result = dict(
            valid=all(checks.values()),
            gate=all(checks.values())
            and all(quality[f"{pop}_decoder"] for pop in metrics),
            checks=checks,
            quality=quality,
            metrics=metrics,
            seconds=time.monotonic() - started,
            limitations=[
                "Observed-code witnesses only, not global encoder injectivity or natural images.",
                "Slow pixel optimization, not faithful fast neural decoding.",
                "No generation or off-manifold code claim.",
                "GPU jobs overlap; elapsed time is only a budget check.",
                "Existing renderer; structural QA only.",
            ],
        )
        atomic_json(out / "result.json", result)
        (out / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        atomic_json(
            out / "status.json",
            dict(result="completed", report="pending", step=a.steps, error=None),
        )
    except Exception as error:
        atomic_json(
            out / "status.json",
            dict(result="failed", report="incomplete", step=0, error=str(error)),
        )
        raise
    print(write_report(out, batch=dict(rgb=examples[0]), outputs=dict(rgb=examples[1])))
    print(
        json.dumps(
            dict(
                valid=result["valid"],
                gate=result["gate"],
                quality=quality,
                seconds=result["seconds"],
            )
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--code", type=Path)
    parser.add_argument("--decoded", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--start", default="decoder", choices=("decoder", "gray"))
    parser.add_argument(
        "--starts", nargs="+", choices=("decoder", "gray"), default=["decoder", "gray"]
    )
    parser.add_argument("--scenes", type=int, default=64)
    a = parser.parse_args()
    if a.steps < 1 or bool(a.code) != bool(a.decoded) or bool(a.code) == bool(a.output):
        parser.error(
            "Positive steps and either --output or --code plus --decoded required"
        )
    if not a.code and (a.scenes not in (64, 256) or "decoder" not in a.starts):
        parser.error(
            "Verification requires64 or256 scenes per population and the decoder primary"
        )
    seed_everything(812)
    child(a) if a.code else parent(a)
