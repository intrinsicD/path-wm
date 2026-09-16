"""Shared spatial image VAE + optional causal temporal posterior refinement.

python -m experiments.video_vae --output runs/shared_video_vae_v1/seed7301/frame
Add --temporal for the matched causal arm. Fixed development sources, no download.
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from pathwm.io import (
    Run,
    atomic_json,
    file_hash,
    seed_everything,
    evaluation_mode,
    state_hash,
)
from pathwm.models.modalities import Observation
from pathwm.models.spatial_vae import SpatialVAE, vae_loss
from pathwm.models.video_vae import VideoVAE, CausalLatentMixer
from pathwm.evaluation.report import write_report


SOURCES = dict(
    train=["0EJAG", "0GFE8", "0JQ26", "0LDP7"],
    validation=["12VVC"],
    evaluation=["1KKYX"],
)


def data(root, image_examples):
    """Whole source videos stay in one split. Explicit low-resolution development."""
    rows, identity = {}, {}
    for split, names in SOURCES.items():
        values = []
        records = []
        for name in names:
            path = Path(root) / name / "video.mp4"
            command = [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(path),
                "-t",
                "4",
                "-an",
                "-vf",
                "fps=4,scale=48:48",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "pipe:1",
            ]
            raw = subprocess.run(command, check=True, capture_output=True).stdout
            frames = np.frombuffer(raw, dtype=np.uint8).copy().reshape(-1, 48, 48, 3)
            if len(frames) != 16:
                raise ValueError(f"Expected 16 frames, got {len(frames)} from {path}")
            x = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255
            values.append(x.reshape(4, 4, 3, 48, 48))
            import hashlib

            records.append(
                dict(
                    path=str(path.resolve()),
                    sha256=file_hash(path),
                    decoded_sha256=hashlib.sha256(raw).hexdigest(),
                    command=command,
                )
            )
        rows[split] = torch.cat(values)
        identity[split] = records
    with np.load(image_examples, allow_pickle=False) as arrays:
        rows["retention"] = torch.from_numpy(arrays["input"][:8].copy()).float()
    identity["retention"] = dict(
        path=str(Path(image_examples).resolve()),
        sha256=file_hash(image_examples),
        count=len(rows["retention"]),
        role="previously inspected image reconstruction examples; evaluation only",
    )
    return rows, identity


def observation(x):
    return Observation(
        x,
        torch.arange(x.shape[1], device=x.device, dtype=torch.float64)[None].expand(
            len(x), -1
        )
        / 4,
    )


def metrics(pred, target):
    p = pred.clamp(0, 1)
    return dict(
        raw_mse=float((pred - target).square().mean()),
        mse=float((p - target).square().mean()),
        color_mse=float((p.mean((-1, -2)) - target.mean((-1, -2))).square().mean()),
        edge_mse=float(
            (
                (p.diff(dim=-1) - target.diff(dim=-1)).square().mean()
                + (p.diff(dim=-2) - target.diff(dim=-2)).square().mean()
            )
            / 2
        ),
        frame_difference_mse=float(
            (p.diff(dim=1) - target.diff(dim=1)).square().mean()
        ),
    )


@torch.no_grad()
def evaluate(model, sets):
    scores, arrays = {}, {}
    with evaluation_mode(model):
        for split in ("validation", "evaluation"):
            x = sets[split]
            y, p = model(observation(x), sample=False)
            frame, fp = model.image(x.flatten(0, 1), sample=False)
            frame = frame.reshape_as(x)
            g = torch.Generator(device=x.device).manual_seed(7501)
            sampled = model.decode(
                p.sample(g),
                p.original_size,
                torch.ones(x.shape[:2], device=x.device, dtype=torch.bool),
            )
            zero = model.decode(
                torch.zeros_like(p.mu),
                p.original_size,
                torch.ones(x.shape[:2], device=x.device, dtype=torch.bool),
            )
            # Final-frame target unchanged; history replaced with another clip's.
            wrong = x.clone()
            wrong[:, :-1] = x.roll(1, 0)[:, :-1]
            wrong_y = model(observation(wrong), sample=False)[0]
            scores[split] = dict(
                video=metrics(y, x),
                frame=metrics(frame, x),
                sampled=metrics(sampled, x),
                zero=metrics(zero, x),
                kl_bits_per_frame=float(p.kl_per_image().mean() / np.log(2)),
                frame_kl_bits=float(fp.kl_per_image().mean() / np.log(2)),
                wrong_past_final_mse=float(
                    (wrong_y[:, -1].clamp(0, 1) - x[:, -1]).square().mean()
                ),
                correct_past_final_mse=float(
                    (y[:, -1].clamp(0, 1) - x[:, -1]).square().mean()
                ),
                history_output_change=float((wrong_y[:, -1] - y[:, -1]).abs().mean()),
            )
            if split == "evaluation":
                arrays.update(
                    target=x.cpu().numpy(),
                    mean=y.cpu().numpy(),
                    frame=frame.cpu().numpy(),
                    sampled=sampled.cpu().numpy(),
                    zero=zero.cpu().numpy(),
                    wrong_past=wrong_y.cpu().numpy(),
                    mu=p.mu.cpu().numpy(),
                    logvar=p.logvar.cpu().numpy(),
                )
        x = sets["retention"]
        y, p = model.image(x, sample=False)
        scores["image_retention"] = dict(
            raw_mse=float((y - x).square().mean()),
            mse=float((y.clamp(0, 1) - x).square().mean()),
        )
        arrays.update(retention_target=x.cpu().numpy(), retention_mean=y.cpu().numpy())
    return scores, arrays


def panel(arrays, path):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    fig = Figure(figsize=(12, 8), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(6, 8)
    for row, (key, label) in enumerate(
        [
            ("target", "Source"),
            ("frame", "Image only"),
            ("mean", "Video mean"),
            ("sampled", "Video sample"),
            ("zero", "Zero latent"),
            ("error", "Abs. error"),
        ]
    ):
        for col in range(8):
            clip, t = divmod(col, 4)
            x = (
                np.abs(arrays["mean"][clip, t] - arrays["target"][clip, t])
                if key == "error"
                else arrays[key][clip, t]
            )
            ax = axes[row, col]
            ax.imshow(np.clip(x.transpose(1, 2, 0), 0, 1))
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(label)
            if row == 0:
                ax.set_title(f"Clip{clip + 1} / t{t}")
    fig.suptitle(
        "Real-video development reconstruction; first two clips, every frame\n48×48 input resampling; error displayed on fixed [0,1] scale; no forecast"
    )
    fig.savefig(path, dpi=110)


def inpaint_input(x, *, current_only=False, clean=False):
    """Mask RGB before encoding. Return a fresh input, never alter the target."""
    values = x.clone()
    region = torch.zeros_like(x[:, -1, :1])
    h, w = x.shape[-2:]
    if min(h, w) < 16:
        raise ValueError("The registered mask needs at least16x16 pixels")
    region[..., h // 2 - 8 : h // 2 + 8, w // 2 - 8 : w // 2 + 8] = 1
    if not clean:
        values[:, -1] = values[:, -1] * (1 - region) + 0.5 * region
    if current_only:
        values[:, :-1] = 0
    return values, region


def region_metrics(y, target, region):
    error = (y.clamp(0, 1) - target).square()
    mask = region.expand_as(error)
    return dict(
        mse=float(error.mean()),
        masked_mse=float((error * mask).sum() / mask.sum()),
        outside_mse=float((error * (1 - mask)).sum() / (1 - mask).sum()),
        raw_mse=float((y - target).square().mean()),
    )


@torch.no_grad()
def evaluate_inpaint(model, sets, current_only):
    scores, arrays = {}, {}
    with evaluation_mode(model):
        for split in ("validation", "evaluation"):
            x = sets[split]
            masked, region = inpaint_input(x, current_only=current_only)
            clean, _ = inpaint_input(x, current_only=current_only, clean=True)
            wrong = masked.clone()
            wrong[:, :-1] = masked.roll(1, 0)[:, :-1]
            zero = masked.clone()
            zero[:, :-1] = 0
            predictions = {
                "masked": model(observation(masked), sample=False)[0][:, -1],
                "clean": model(observation(clean), sample=False)[0][:, -1],
                "wrong_history": model(observation(wrong), sample=False)[0][:, -1],
                "zero_history": model(observation(zero), sample=False)[0][:, -1],
                "frozen_masked": model.image(masked[:, -1], sample=False)[0],
                "frozen_clean": model.image(x[:, -1], sample=False)[0],
            }
            scores[split] = {
                name: region_metrics(y, x[:, -1], region)
                for name, y in predictions.items()
            }
            if split == "evaluation":
                arrays = {name: y.cpu().numpy() for name, y in predictions.items()}
                arrays.update(
                    target=x[:, -1].cpu().numpy(),
                    input=masked[:, -1].cpu().numpy(),
                    region=region.cpu().numpy(),
                )
        retained = model.image(sets["retention"], sample=False)[0]
        scores["image_retention_mse"] = float(
            (retained.clamp(0, 1) - sets["retention"]).square().mean()
        )
    return scores, arrays


def train_inpaint(args):
    """Conditional repair of frozen-codec latents; only temporal weights train."""
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < 304 * 1024**2:
        raise RuntimeError("Need room for artifacts above300MiB reserve")
    seed_everything(args.seed)
    image = SpatialVAE.load(args.source).requires_grad_(False).eval()
    frozen_hash = state_hash(image)
    model = VideoVAE(
        image,
        temporal=CausalLatentMixer(
            image.config["latent_channels"],
            spatial_kernel=args.spatial_kernel,
            spatial_iterations=args.spatial_iterations,
        ),
    )
    sets, identity = data(args.data, args.source.parent / "examples.npz")
    cfg = dict(
        seed=args.seed,
        steps=args.steps,
        task="inpaint",
        batch_size=2,
        lr=0.001,
        spatial_kernel=args.spatial_kernel,
        spatial_iterations=args.spatial_iterations,
        current_only=args.current_only,
        wall_seconds=60,
        source=str(args.source.resolve()),
        source_sha256=file_hash(args.source),
        frozen_image_state_sha256=frozen_hash,
        architecture=image.config,
        objective="0.5*(4x masked/1x outside weighted RGB MSE + clean RGB MSE); mean latent; no KL",
        mask="center16x16 RGB set0.5 before frame-independent encoding",
        zero_history="Black RGB past frames; original timestamps and validity",
        checkpoint="Trainable temporal module only; frozen image requires recorded source hash",
        example_labels={
            "input": "Unmasked current-frame target",
            "rgb": "Repaired masked frame",
            "sampled": "Frozen image codec on masked input",
        },
    )
    optimizer = torch.optim.AdamW(
        model.temporal.parameters(), lr=cfg["lr"], weight_decay=1e-4
    )
    seed_everything(args.seed)
    run = Run(
        output,
        settings=cfg,
        data=identity,
        recipe=__file__,
        model=model.temporal,
        optimizer=optimizer,
        device="cpu",
        resume=args.resume,
    )
    # Dense convolution MACs for ONE video forward of batch2x4frames. Equal
    # tensor shapes imply equal MACs for correct/blank history; time is separate.
    macs = []
    handles = []

    def count_macs(module, inputs, out):
        kernel = int(np.prod(module.kernel_size))
        macs.append(out.numel() * module.in_channels // module.groups * kernel)

    for module in model.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Conv3d)):
            handles.append(module.register_forward_hook(count_macs))
    with torch.no_grad():
        model(
            observation(
                inpaint_input(sets["train"][:2], current_only=args.current_only)[0]
            ),
            sample=False,
        )
    for handle in handles:
        handle.remove()
    if not args.resume:
        initial, _ = evaluate_inpaint(model, sets, args.current_only)
        atomic_json(output / "initial.json", initial)
    start = perf_counter()
    prior = sum(r.get("elapsed_seconds", 0) for r in run.rows if r["split"] == "timing")
    end = min(args.steps, run.step + args.stop_after) if args.stop_after else args.steps
    try:
        for step in range(run.step + 1, end + 1):
            if prior + perf_counter() - start > cfg["wall_seconds"]:
                break
            if step % 32 == 1 and shutil.disk_usage(output.parent).free < 302 * 1024**2:
                raise RuntimeError("Preserve300MiB disk reserve")
            x = sets["train"][run.sample(len(sets["train"]), cfg["batch_size"])]
            masked, region = inpaint_input(x, current_only=args.current_only)
            clean, _ = inpaint_input(x, current_only=args.current_only, clean=True)
            optimizer.zero_grad(set_to_none=True)
            y = model(observation(masked), sample=False)[0][:, -1]
            yc = model(observation(clean), sample=False)[0][:, -1]
            weight = (1 + 3 * region).expand_as(y)
            damaged_loss = ((y - x[:, -1]).square() * weight).sum() / weight.sum()
            clean_loss = (yc - x[:, -1]).square().mean()
            loss = 0.5 * (damaged_loss + clean_loss)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(
                model.temporal.parameters(), 1, error_if_nonfinite=True
            )
            optimizer.step()
            run.step = step
            run.log(
                dict(
                    step=step,
                    split="train",
                    loss=float(loss.detach()),
                    masked_objective=float(damaged_loss.detach()),
                    clean_objective=float(clean_loss.detach()),
                    gradient=float(grad),
                )
            )
            if step % 64 == 0 or step == end:
                scores, _ = evaluate_inpaint(
                    model,
                    {
                        "validation": sets["validation"],
                        "evaluation": sets["validation"],
                        "retention": sets["retention"],
                    },
                    args.current_only,
                )
                run.log(
                    dict(
                        step=step,
                        split="validation",
                        loss=scores["validation"]["masked"]["masked_mse"],
                    )
                )
                run.save()
        run.log(
            dict(step=run.step, split="timing", elapsed_seconds=perf_counter() - start)
        )
        run.save()
        assert state_hash(image) == frozen_hash
        assert all(p.grad is None for p in image.parameters())
        scores, arrays = evaluate_inpaint(model, sets, args.current_only)
        np.savez_compressed(output / "evaluation.npz", **arrays)
        state = (
            "completed"
            if run.step == args.steps
            else "paused"
            if run.step == end
            else "budget-stopped"
        )
        atomic_json(
            output / "result.json",
            dict(
                completed=state == "completed",
                step=run.step,
                scores=scores,
                metrics={
                    f"{kind}_{metric}": value
                    for kind, row in scores["evaluation"].items()
                    for metric, value in row.items()
                },
                temporal_parameters=sum(p.numel() for p in model.temporal.parameters()),
                parameters=sum(p.numel() for p in model.parameters()),
                conv_macs_per_batch=sum(macs),
                compute_scope="Conv2d/Conv3d multiply-accumulates, one batch2x4 video forward; excludes normalization, activation and backward; dense shapes identical for history/current-only",
                training_seconds=sum(
                    r.get("elapsed_seconds", 0)
                    for r in run.rows
                    if r["split"] == "timing"
                ),
                frozen_image_state_sha256=state_hash(image),
                peak_gpu_memory_mib=None,
                evaluation_scope="Fixed central RGB occlusion on previously inspected real-video development sources; not motion semantics or generation. Frozen image encoder and decoder; source checkpoint required.",
            ),
        )
        run.status(state, "pending")
        write_report(
            output,
            batch={"rgb": torch.from_numpy(arrays["target"][:2])},
            outputs={
                "rgb": torch.from_numpy(arrays["masked"][:2]),
                "sampled": torch.from_numpy(arrays["frozen_masked"][:2]),
            },
        )
    except BaseException as error:
        run.save()
        result_state = json.loads((output / "status.json").read_text())["result"]
        run.status(
            result_state if (output / "result.json").exists() else "failed",
            "failed",
            str(error),
        )
        raise
    print(
        f"{output}: {state}, masked MSE={scores['evaluation']['masked']['masked_mse']:.6f}",
        flush=True,
    )
    return scores


def train(args):
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < 300 * 1024**2:
        raise RuntimeError("Preserve 300MiB disk reserve")
    seed_everything(args.seed)
    model = VideoVAE(SpatialVAE.load(args.source), temporal=args.temporal)
    sets, identity = data(args.data, args.source.parent / "examples.npz")
    cfg = dict(
        seed=args.seed,
        steps=args.steps,
        temporal=args.temporal,
        source=str(args.source.resolve()),
        source_sha256=file_hash(args.source),
        batch_size=2,
        lr=3e-4,
        beta=0.01,
        variance=0.5,
        wall_seconds=120,
        objective="0.5*(video Gaussian VAE + frame Gaussian VAE); summed RGB and KL per original pixel",
        architecture=model.image.config,
        purpose="shared image/video codec development",
        example_labels={
            "input": "Source video frames",
            "rgb": "Shared codec video mean",
            "sampled": "Sampled video reconstruction",
        },
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    seed_everything(args.seed)  # matched noise after different constructors
    run = Run(
        output,
        settings=cfg,
        data=identity,
        recipe=__file__,
        model=model,
        optimizer=optimizer,
        device="cpu",
        resume=args.resume,
    )
    if not args.resume:
        initial, _ = evaluate(model, sets)
        atomic_json(output / "initial.json", initial)
    start = perf_counter()
    prior = sum(r.get("elapsed_seconds", 0) for r in run.rows if r["split"] == "timing")
    end = min(args.steps, run.step + args.stop_after) if args.stop_after else args.steps
    try:
        for step in range(run.step + 1, end + 1):
            if prior + perf_counter() - start > cfg["wall_seconds"]:
                break
            x = sets["train"][run.sample(len(sets["train"]), cfg["batch_size"])]
            optimizer.zero_grad(set_to_none=True)
            y, p = model(observation(x))
            yi, pi = model.image(x.flatten(0, 1))
            lv, tv = vae_loss(
                y.flatten(0, 1), x.flatten(0, 1), p, cfg["beta"], cfg["variance"]
            )
            li, ti = vae_loss(yi, x.flatten(0, 1), pi, cfg["beta"], cfg["variance"])
            loss = 0.5 * (lv + li)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1, error_if_nonfinite=True
            )
            optimizer.step()
            run.step = step
            run.log(
                dict(
                    step=step,
                    split="train",
                    loss=float(loss.detach()),
                    video_distortion=float(tv["distortion"].detach()),
                    image_distortion=float(ti["distortion"].detach()),
                    video_rate=float(tv["rate"].detach()),
                    image_rate=float(ti["rate"].detach()),
                    gradient=float(grad),
                )
            )
            if step % 32 == 0 or step == end:
                score, _ = evaluate(model, sets)
                run.log(
                    dict(
                        step=step,
                        split="validation",
                        loss=score["validation"]["video"]["raw_mse"],
                        retention_mse=score["image_retention"]["mse"],
                    )
                )
                run.save()
                print(
                    f"step {step}: {score['validation']['video']['mse']:.6f}",
                    flush=True,
                )
        run.log(
            dict(step=run.step, split="timing", elapsed_seconds=perf_counter() - start)
        )
        run.save()
        score, arrays = evaluate(model, sets)
        np.savez_compressed(output / "evaluation.npz", **arrays)
        # Export the updated image codec; video weights + optimizer are in last.pt.
        model.image.save(output / "image_weights.pt")
        flat = {
            f"{split}_{name}": value
            for split, items in score.items()
            for name, value in items.items()
        }
        state = (
            "completed"
            if run.step == args.steps
            else "paused"
            if run.step == end
            else "budget-stopped"
        )
        atomic_json(
            output / "result.json",
            dict(
                completed=state == "completed",
                step=run.step,
                metrics=flat,
                scores=score,
                parameters=sum(p.numel() for p in model.parameters()),
                temporal_parameters=sum(p.numel() for p in model.temporal.parameters()),
                training_seconds=sum(
                    r.get("elapsed_seconds", 0)
                    for r in run.rows
                    if r["split"] == "timing"
                ),
                peak_gpu_memory_mib=None,
                evaluation_scope="Previously inspected real-video development sources; split by clip source. Reconstruction of observed frames, not future prediction, general video generation or agent integration. Source image weights are actually shared.",
            ),
        )
        run.status(state, "pending")
        panel(arrays, output / "comparison.png")
        write_report(
            output,
            batch={"rgb": sets["evaluation"][:2].flatten(0, 1)},
            outputs={
                "rgb": torch.from_numpy(arrays["mean"][:2]).flatten(0, 1),
                "sampled": torch.from_numpy(arrays["sampled"][:2]).flatten(0, 1),
            },
        )
    except BaseException as error:
        run.save()
        status = json.loads((output / "status.json").read_text())
        run.status(
            status["result"] if (output / "result.json").exists() else "failed",
            "failed",
            str(error),
        )
        raise
    return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(
            "runs/spatial_vae_repair_v1/formal/seed57301/color/training/weights.pt"
        ),
    )
    parser.add_argument(
        "--data", type=Path, default=Path("data/memory_media_v1/episodes")
    )
    parser.add_argument("--seed", type=int, default=7301)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--temporal", action="store_true")
    parser.add_argument(
        "--task", choices=("reconstruct", "inpaint"), default="reconstruct"
    )
    parser.add_argument("--spatial-kernel", type=int, default=3)
    parser.add_argument("--spatial-iterations", type=int, default=0)
    parser.add_argument("--current-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args()
    if args.steps < 1 or (args.stop_after is not None and args.stop_after < 1):
        parser.error("positive step budgets required")
    if args.task == "reconstruct" and (
        args.spatial_kernel != 3 or args.spatial_iterations or args.current_only
    ):
        parser.error("Spatial context comparison flags require --task inpaint")
    (train_inpaint if args.task == "inpaint" else train)(args)
