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
