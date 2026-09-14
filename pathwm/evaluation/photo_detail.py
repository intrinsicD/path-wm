"""Inspect the frozen photographic input/state/readout path without repairing it."""

import torch
from torch.nn import functional as F
from pathwm.data.photo_recall import photo_history
from pathwm.models.modalities import Observation
from pathwm.models.memory_output import FrozenFeatureNormalization


@torch.no_grad()
def stage_values(model, rgb, *, native=True, start=0):
    image_encoder = model.agent.encoders["image"]
    base = (
        image_encoder.base
        if isinstance(image_encoder, FrozenFeatureNormalization)
        else image_encoder
    )
    obs = Observation(rgb[:, None], rgb.new_zeros(len(rgb), 1))
    stem = base.stem(obs).values
    encoded = image_encoder(obs).as_tokens().values
    history = photo_history(rgb)
    h = model.observe_history(history)
    recall = model.query(h["final"], history[:, -1], "reset")
    grid = F.avg_pool2d(rgb, 4)
    stages = dict(
        raw_grid=grid,
        stem=stem,
        encoder=encoded,
        observed_all=h["initial"].tokens,
        stored_all=h["stored"].tokens,
        stored_working=model.working(h["stored"]),
        recall_all=recall.tokens,
        recall_working=model.working(recall),
    )
    features = model.teacher(rgb)
    head = model.agent.decoders["image"].head
    controls = dict(
        grid=grid,
        target=rgb,
        codec=head(features),
        codec_no_detail=head(
            dict(features, detail=torch.zeros_like(features["detail"]))
        ),
    )
    if native:
        controls["native"] = model.agent.decoders["image"](
            model.output_normalization(model.working(recall)),
            seed=13,
            sample_ids=[i // 2 for i in range(start, start + len(rgb))],
        )
    audit = dict(
        stored_snapshot_exact=torch.equal(
            h["stored"].tokens, h["final"].memory.values[:, -1]
        ),
        bank_snapshots=h["final"].memory.values.shape[1],
        retrieval_count=model.agent.memory.retrieve_count,
        token_shapes={k: list(v.shape[1:]) for k, v in stages.items()},
    )
    if not audit["stored_snapshot_exact"]:
        raise RuntimeError("Memory write changed the state")
    return (
        {k: v.detach().flatten(1).clone() for k, v in stages.items()},
        {k: v.detach().clone() for k, v in controls.items()},
        audit,
    )


@torch.no_grad()
def patch_audit(layer):
    w = layer.weight.detach().cpu().double().flatten(1)
    _, s, vh = torch.linalg.svd(w, full_matrices=True)
    rank = int(torch.linalg.matrix_rank(w))
    p = layer.kernel_size[0]
    average = torch.kron(
        torch.eye(3, dtype=w.dtype), torch.ones(p * p, p * p, dtype=w.dtype) / (p * p)
    )
    direction = vh[-1]
    direction = direction / direction.abs().max()
    return dict(
        input_dimensions=w.shape[1],
        output_dimensions=w.shape[0],
        rank_float64=rank,
        rank_float32_tolerance=int(
            (s > s[0] * max(w.shape) * torch.finfo(torch.float32).eps).sum()
        ),
        null_dimensions=w.shape[1] - rank,
        singular_values=s.tolist(),
        top3_weight_energy_fraction=float(s[:3].square().sum() / s.square().sum()),
        zero_mean_detail_weight_energy_fraction=float(
            (w @ (torch.eye(w.shape[1], dtype=w.dtype) - average)).square().sum()
            / w.square().sum()
        ),
        null_direction=direction.tolist(),
        null_max_response=float((w @ direction).abs().max()),
        scope="Algebraic non-injectivity of this patch layer, plus conditioning of these weights. Small nonzero directions are not declared absent.",
    )


def grid_score(image, target):
    image, target = image.double(), target.double()
    if (
        image.shape != target.shape
        or image.ndim != 4
        or not torch.isfinite(image).all()
    ):
        raise ValueError("Aligned finite BCHW images required")
    mse = (image - target).square().mean((1, 2, 3))
    spatial = (
        (
            (image - image.mean((-2, -1), keepdim=True))
            - (target - target.mean((-2, -1), keepdim=True))
        )
        .square()
        .mean()
    )
    edge = (
        (image.diff(dim=-1) - target.diff(dim=-1)).square().mean()
        + (image.diff(dim=-2) - target.diff(dim=-2)).square().mean()
    ) / 2
    return dict(
        rgb_mse=float(mse.mean()),
        mean_psnr=float((-10 * mse.clamp_min(1e-12).log10()).mean()),
        edge_mse=float(edge),
        spatial_mse=float(spatial),
        count=len(image),
    )
