"""Bounded, frozen stage readouts and compute accounting for spatial VAEs."""

from time import perf_counter
import torch
from torch import nn
from torch.nn import functional as F

from pathwm.io import evaluation_mode
from pathwm.models.blocks import Attention
from pathwm.models.photo_probe import RidgeReader


def probe_readout(x, y, held_x, held_y, *, ridge=1e-3, epsilon=1e-8):
    """Train-only normalization; row-shuffle checks example/cell association.

    A score measures heldout accessibility to this finite reader family, not
    mutual information. Hyperparameters are fixed, never selected on heldout data.
    """
    mean = y.double().mean(0)
    variance = float((y.double() - mean).square().mean())
    degenerate = variance <= epsilon

    def metric(pred):
        error = float((pred - held_y.double()).square().mean())
        return dict(mse=error, normalized_mse=None if degenerate else error / variance)

    result = dict(
        target_train_variance=variance,
        epsilon=epsilon,
        degenerate=degenerate,
        train_mean=metric(mean),
        training_points=len(x),
        heldout_points=len(held_x),
    )
    permutation = torch.randperm(
        len(held_x), generator=torch.Generator().manual_seed(57131)
    )
    for kernel in ("linear", "rbf"):
        reader = RidgeReader.fit(x, y, ridge=ridge, kernel=kernel, bandwidth=1.0)
        result[kernel] = metric(RidgeReader.predict(reader, held_x))
        result["shuffled_" + kernel] = metric(
            RidgeReader.predict(reader, held_x[permutation])
        )
    return result


@torch.no_grad()
def collect_stage_samples(model, rgb, device="cpu", cells_per_image=4, seed=57121):
    """Read only bounded cells, preserving an image-group split chosen by caller."""
    generator = torch.Generator().manual_seed(seed)
    result, geometry = {}, {}
    with evaluation_mode(model):
        for image in rgb.split(1):
            p, trace = model.inspect(image.to(device))
            factor = 1
            for i, stage in enumerate(model.encoder.stages):
                factor *= stage.factor
                name = f"stage_{i}"
                before, after = [
                    trace[name + "." + key]
                    for key in ("before_compression", "after_compression")
                ]
                h, w = after.shape[-2:]
                cells = torch.randperm(h * w, generator=generator)[
                    : min(cells_per_image, h * w)
                ]
                patch = F.unfold(
                    trace["input.padded"], kernel_size=factor, stride=factor
                )
                if patch.shape[-1] != h * w or before.shape[-2:] != after.shape[-2:]:
                    raise ValueError("Probe cells and common RGB patches must align")
                values = dict(
                    before=before.flatten(2).transpose(1, 2)[0, cells].cpu(),
                    after=after.flatten(2).transpose(1, 2)[0, cells].cpu(),
                    rgb=patch.transpose(1, 2)[0, cells].cpu(),
                )
                for key, value in values.items():
                    result.setdefault(name, {}).setdefault(key, []).append(value)
                geometry[name] = dict(
                    before_compression=list(before.shape),
                    after_compression=list(after.shape),
                    scalar_count_before=before[0].numel(),
                    scalar_count_after=after[0].numel(),
                    original_size=list(p.original_size),
                    padded_size=list(p.padded_size),
                    rgb_patch_width=3 * factor**2,
                    factor=factor,
                    boundary="entangled stride convolution"
                    if stage.entangled
                    else "explicit channel compression",
                )
    return {
        name: {k: torch.cat(v) for k, v in values.items()}
        for name, values in result.items()
    }, geometry


def stage_probes(model, training, validation, test, device="cpu"):
    start = perf_counter()
    populations = {}
    for name, images in [
        ("train", training),
        ("validation", validation),
        ("test", test),
    ]:
        populations[name], geometry = collect_stage_samples(model, images, device)
    results = {}
    for stage, source in populations["train"].items():
        results[stage] = {}
        for split in ("validation", "test"):
            held = populations[split][stage]
            results[stage][split] = {
                label: probe_readout(source[x], source[y], held[x], held[y])
                for label, x, y in [
                    ("feature_recovery", "after", "before"),
                    ("identity_control", "before", "before"),
                    ("rgb_before", "before", "rgb"),
                    ("rgb_after", "after", "rgb"),
                ]
            }
    return dict(
        stages=results,
        geometry=geometry,
        seconds=perf_counter() - start,
        settings=dict(
            cells_per_image=4,
            seed=57121,
            ridge=1e-3,
            rbf_bandwidth=1.0,
            epsilon=1e-8,
            shuffled_control="Permute heldout (image,cell) rows, preserving channel vectors",
        ),
        scope="Frozen bounded linear/RBF accessibility; no certified information loss or semantic retention",
    )


@torch.no_grad()
def profile_codec(model, x, repeats=5):
    """Forward Conv/Linear and attention matmul MAC estimate; nonlinear ops omitted."""
    macs, active = [], set()

    def count(module, args, out):
        active.update(id(p) for p in module.parameters(recurse=False))
        if isinstance(module, nn.Conv2d):
            macs.append(out.numel() * module.weight[0].numel())
        elif isinstance(module, nn.Linear):
            macs.append(out.numel() * module.in_features)
        elif isinstance(module, Attention):
            q, k, _ = args
            macs.append(2 * q.shape[0] * q.shape[1] * k.shape[1] * module.width)

    with evaluation_mode(model):
        hooks = [m.register_forward_hook(count) for m in model.modules()]
        try:
            model(x, sample=False)
        finally:
            for h in hooks:
                h.remove()
        if x.is_cuda:
            torch.cuda.synchronize(x.device)
        start = perf_counter()
        for _ in range(repeats):
            model(x, sample=False)
        if x.is_cuda:
            torch.cuda.synchronize(x.device)
        seconds = (perf_counter() - start) / repeats
    return dict(
        parameters=sum(p.numel() for p in model.parameters()),
        active_parameters=sum(p.numel() for p in model.parameters() if id(p) in active),
        forward_macs=sum(macs),
        approximate_flops=2 * sum(macs),
        batch_shape=list(x.shape),
        inference_seconds_per_batch=seconds,
        convention="One multiply-add=one MAC; FLOPs=2MAC. Conv/Linear/QK/AV only; norms, activation, softmax, shuffle and backward excluded.",
    )


def phase_statistics(x, period=4, border=8):
    """Position-mod-period variation, excluding per-image channel DC offsets."""
    if x.ndim != 4 or min(x.shape[-2:]) <= 2 * border + period:
        raise ValueError("Phase statistic needs a nonempty interior BCHW grid")
    x = x[..., border:-border, border:-border] if border else x
    h, w = (x.shape[-2] // period) * period, (x.shape[-1] // period) * period
    x = x[..., :h, :w].float()
    phases = x.reshape(*x.shape[:2], h // period, period, w // period, period).mean(
        (2, 4)
    )
    phases = phases - phases.mean((-2, -1), keepdim=True)
    # Cell-wise random phase permutation: preserve each local cell's values,
    # disrupt a shared phase. Descriptive null, not a significance test.
    cells = (
        x.reshape(*x.shape[:2], h // period, period, w // period, period)
        .permute(0, 1, 2, 4, 3, 5)
        .flatten(-2)
    )
    generator = torch.Generator(device=x.device).manual_seed(57211)
    order = torch.rand(cells.shape, device=x.device, generator=generator).argsort(-1)
    null = cells.gather(-1, order).mean((2, 3))
    null = null - null.mean(-1, keepdim=True)
    return dict(
        rms=float(phases.square().mean().sqrt()),
        coherent_rms=float(phases.mean(0).square().mean().sqrt()),
        cell_permutation_rms=float(null.square().mean().sqrt()),
        mean_phase_map=phases.mean(0).tolist(),
        period=period,
        border=border,
    )


def color_grid_metrics(prediction, target):
    from pathwm.models.spatial_vae import rgb_opponents

    if prediction.shape != target.shape:
        raise ValueError("Color diagnostics require aligned RGB arrays")
    p, t = prediction.detach().cpu().float(), target.detach().cpu().float()
    pc, tc = rgb_opponents(p), rgb_opponents(t)
    error = p - t
    interior = error[..., 8:-8, 8:-8]
    border_sum = error.square().sum() - interior.square().sum()
    scores = dict(
        raw_mse=float(error.square().mean()),
        clipped_mse=float((p.clamp(0, 1) - t).square().mean()),
        clipped_fraction=float(((p < 0) | (p > 1)).float().mean()),
        global_rgb_mse=float(error.mean((2, 3)).square().mean()),
        global_chroma_mse=float((pc.mean((2, 3)) - tc.mean((2, 3))).square().mean()),
        pixel_chroma_mse=float((pc - tc).square().mean()),
        chroma_gain=float((pc * tc).sum() / tc.square().sum().clamp_min(1e-12)),
        rgb_bias=error.mean((0, 2, 3)).tolist(),
        interior_mse=float(interior.square().mean()),
        border_mse=float(border_sum / (error.numel() - interior.numel())),
    )
    for period in [2, 4]:
        residual = phase_statistics(error, period)
        scores[f"period{period}"] = dict(
            residual_rms=residual["rms"],
            coherent_rms=residual["coherent_rms"],
            null_rms=residual["cell_permutation_rms"],
            mean_phase_map=residual["mean_phase_map"],
            target_rms=phase_statistics(t, period)["rms"],
        )
    return scores


@torch.no_grad()
def color_features(model, rgb, device="cpu"):
    """Global image-level mean feature readouts, same RGB target at every boundary."""
    features = {}

    def add(key, value):
        features.setdefault(key, []).append(value.mean((2, 3)).detach().cpu())

    with evaluation_mode(model):
        for x in rgb.split(8):
            x = x.to(device)
            p, trace = model.inspect(x)
            add("input", x)
            for key, value in trace.items():
                if (
                    key == "stem.output"
                    or key.endswith("compression")
                    or key in ["posterior.mu", "posterior.input"]
                ):
                    add(key, value)
            decoded = model.decoder.input(p.mu)
            add("decoder.input", decoded)
            for i, stage in enumerate(model.decoder.stages):
                decoded = stage(decoded)
                add(f"decoder.stage_{i}", decoded)
            add(
                "decoder.rgb",
                model.decoder.output(decoded)[..., : x.shape[-2], : x.shape[-1]],
            )
    return {key: torch.cat(values) for key, values in features.items()}


def color_readouts(model, training, validation, test, device="cpu"):
    sets = {
        name: color_features(model, images, device)
        for name, images in [
            ("train", training),
            ("validation", validation),
            ("test", test),
        ]
    }
    return {
        key: dict(
            width=x.shape[-1],
            **{
                split: probe_readout(
                    x, sets["train"]["input"], features[key], features["input"]
                )
                for split, features in sets.items()
                if split != "train"
            },
        )
        for key, x in sets["train"].items()
    }


@torch.no_grad()
def constant_decoder_diagnostic(model, device="cpu"):
    colors = torch.cartesian_prod(*[torch.tensor([0.15, 0.5, 0.85])] * 3)
    images = colors[:, :, None, None].expand(-1, -1, 64, 64).to(device)
    with evaluation_mode(model):
        posterior = model.encode(images)
        # Different constants at realistic learned magnitudes plus zero; no noise or encoder skip.
        fields = torch.cat(
            [
                torch.zeros_like(posterior.mu[:1, :, :1, :1]),
                posterior.mu.mean((2, 3), keepdim=True),
            ],
            0,
        )
        fields = fields.expand(-1, -1, 32, 32).contiguous()
        trace = {}

        def hook(name):
            def capture(module, args, out):
                # Central half avoids the finite local-convolution boundary support.
                border = min(out.shape[-2:]) // 4
                values = phase_statistics(out.cpu(), period=4, border=border)
                trace[name] = dict(
                    shape=list(out.shape),
                    rms=values["rms"],
                    null_rms=values["cell_permutation_rms"],
                )

            return capture

        selected = {
            "decoder.input": model.decoder.input,
            "decoder.output": model.decoder.output,
        }
        for i, stage in enumerate(model.decoder.stages):
            for key in ["expansion", "processing", "upsample", "post_process"]:
                selected[f"decoder.stage_{i}.{key}"] = getattr(stage, key)
        handles = [
            module.register_forward_hook(hook(key)) for key, module in selected.items()
        ]
        try:
            decoded = model.decode(fields, (128, 128)).cpu()
        finally:
            for handle in handles:
                handle.remove()
        rec = model.decode(posterior.mu, (64, 64)).cpu()
    return dict(
        trace=trace,
        output_rms=phase_statistics(decoded, 4, 16)["rms"],
        constants=fields[:, :, 0, 0].cpu().tolist(),
        palette_metrics=color_grid_metrics(rec, images.cpu()),
        scope="Constant latent fields, deterministic decoder, central interiors; no claim of the unique training cause",
    ), dict(palette_input=images.cpu(), palette_output=rec, constant_output=decoded)


@torch.no_grad()
def local_color_readouts(model, training, validation, test, device="cpu"):
    sets = {}
    with evaluation_mode(model):
        for split, images, seed in [
            ("train", training, 57215),
            ("validation", validation, 57216),
            ("test", test, 57217),
        ]:
            matrices = {}
            rng = torch.Generator().manual_seed(seed)
            noise = torch.Generator(device=device).manual_seed(seed + 100)
            for x in images.split(1):
                p, trace = model.inspect(x.to(device))
                factor = model.encoder.factor
                targets = F.avg_pool2d(trace["input.padded"], factor, factor)
                count = p.mu.shape[-2] * p.mu.shape[-1]
                ids = torch.randperm(count, generator=rng)[:4]
                for key, value in [
                    ("target", targets),
                    ("before_posterior", trace["posterior.input"]),
                    ("mu", p.mu),
                    ("sampled_z", p.sample(noise)),
                ]:
                    matrices.setdefault(key, []).append(
                        value.flatten(2).transpose(1, 2)[0, ids].cpu()
                    )
            sets[split] = {k: torch.cat(v) for k, v in matrices.items()}
    source = sets["train"]
    return {
        key: dict(
            width=source[key].shape[1],
            **{
                split: probe_readout(
                    source[key], source["target"], rows[key], rows["target"]
                )
                for split, rows in sets.items()
                if split != "train"
            },
        )
        for key in ["before_posterior", "mu", "sampled_z"]
    }
