"""Read-only diagnostics of a world model's internals.

What: representation spectrum and Gaussianity, encoder attention and patch PCA,
predictor attention, AdaLN gates and action sensitivity, module parameter and
gradient norms, rollout error by horizon, and a linear probe from latents to
physical state. How: forward hooks and autograd on a model in eval mode; every
entry point restores mode, attention implementation and gradients so a saved
checkpoint is never changed (verify with `state_digest`). Why: a falling loss is
not evidence of a healthy representation or of a predictor that uses actions.
"""
import hashlib
import math

import numpy as np
import torch
from scipy import stats

from world_model.objective import loss_from_embeddings

MODULE_GROUPS = ('encoder', 'projector', 'predictor', 'action_encoder', 'pred_proj')


def _array(z):
    z = z.detach().cpu().double().numpy() if torch.is_tensor(z) else np.asarray(z, dtype=np.float64)
    if z.ndim != 2 or len(z) < 2:
        raise ValueError('Expected a [samples, dimensions] matrix with at least two samples')
    return z


def state_digest(model):
    """SHA256 over parameters and buffers; detects any weight or BatchNorm change."""
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def covariance_spectrum(z):
    """Eigenvalues of the embedding covariance with collapse summaries.

    effective_rank: exp(entropy) of normalized eigenvalues (Roy & Vetterli 2007).
    rankme: the same on normalized singular values (Garrido et al. 2023).
    participation_ratio: (sum e)^2 / sum e^2. An isotropic Gaussian scores D.
    """
    z = _array(z)
    centered = z - z.mean(0, keepdims=True)
    eigenvalues = np.clip(np.linalg.eigvalsh(centered.T @ centered / (len(z) - 1))[::-1], 0, None)
    total = eigenvalues.sum()
    if total <= 0:
        raise ValueError('Embeddings have zero variance')

    def entropy_rank(values):
        p = values / values.sum()
        p = p[p > 0]
        return float(math.exp(-(p * np.log(p)).sum()))

    per_dim_std = centered.std(0, ddof=1)
    return dict(eigenvalues=eigenvalues.tolist(), dimension=int(z.shape[1]), samples=int(len(z)),
                effective_rank=entropy_rank(eigenvalues), rankme=entropy_rank(np.sqrt(eigenvalues)),
                participation_ratio=float(total ** 2 / (eigenvalues ** 2).sum()),
                top_eigenvalue_fraction=float(eigenvalues[0] / total),
                per_dim_std_min=float(per_dim_std.min()), per_dim_std_median=float(np.median(per_dim_std)),
                per_dim_std_max=float(per_dim_std.max()))


def attention_entropy(probs):
    """Normalized Shannon entropy in [0, 1] per head; probs are [..., heads, queries, keys]."""
    entropy = -(torch.special.xlogy(probs, probs)).sum(-1) / math.log(probs.shape[-1])
    return entropy.mean(dim=tuple(i for i in range(entropy.ndim) if i != entropy.ndim - 2))


def gaussianity(z, max_samples=500, seed=0):
    """Per-dimension Shapiro–Wilk, skewness and kurtosis, plus pooled Q–Q quantiles.

    SIGReg targets an isotropic Gaussian, so W below 0.95 (LeJEPA's reading) or
    large excess kurtosis marks dimensions the regularizer has not shaped.
    """
    z = _array(z)
    if len(z) > max_samples:
        z = z[np.random.default_rng(seed).choice(len(z), max_samples, replace=False)]
    standardized = (z - z.mean(0)) / np.maximum(z.std(0, ddof=1), 1e-12)
    w = np.array([stats.shapiro(column).statistic for column in standardized.T])
    probabilities = np.arange(1, 100) / 100
    pooled = standardized.reshape(-1)
    return dict(shapiro_w_mean=float(w.mean()), shapiro_w_min=float(w.min()),
                shapiro_fraction_below_0_95=float((w < 0.95).mean()),
                skewness_mean_abs=float(np.abs(stats.skew(standardized, axis=0)).mean()),
                excess_kurtosis_mean=float(stats.kurtosis(standardized, axis=0).mean()),
                samples=int(len(z)),
                qq=dict(theoretical=stats.norm.ppf(probabilities).tolist(),
                        sample=np.quantile(pooled, probabilities).tolist()))


def linear_probe(z_train, y_train, z_val, y_val, targets, ridge=1e-3):
    """Closed-form ridge readout from latents to labelled targets; held-out R² per target."""
    z_train, z_val = _array(z_train), _array(z_val)
    y_train, y_val = np.asarray(y_train, dtype=np.float64), np.asarray(y_val, dtype=np.float64)
    if y_train.shape[1] != len(targets) or y_val.shape[1] != len(targets):
        raise ValueError('Target names must match label columns')
    mean, scale = z_train.mean(0), np.maximum(z_train.std(0, ddof=1), 1e-12)
    design = lambda z: np.hstack([(z - mean) / scale, np.ones((len(z), 1))])
    x = design(z_train)
    penalty = ridge * np.eye(x.shape[1])
    penalty[-1, -1] = 0.0
    weights = np.linalg.solve(x.T @ x + penalty, x.T @ y_train)
    residual = ((design(z_val) @ weights - y_val) ** 2).sum(0)
    total = ((y_val - y_val.mean(0)) ** 2).sum(0)
    r2 = 1 - residual / np.maximum(total, 1e-12)
    return dict(r2={name: float(v) for name, v in zip(targets, r2)}, r2_mean=float(r2.mean()), ridge=ridge,
                train_samples=int(len(z_train)), val_samples=int(len(z_val)))


def _causal_attention(qkv, heads):
    q, k, _ = qkv.chunk(3, dim=-1)
    q = q.reshape(*q.shape[:2], heads, -1).transpose(1, 2)
    k = k.reshape(*k.shape[:2], heads, -1).transpose(1, 2)
    scores = q @ k.transpose(-1, -2) * q.shape[-1] ** -0.5
    mask = torch.ones(q.shape[-2], k.shape[-2], dtype=torch.bool, device=q.device).triu(1)
    return scores.masked_fill(mask, float('-inf')).softmax(-1)


@torch.no_grad()
def predictor_internals(model, states, actions):
    """Causal attention, AdaLN gate magnitudes and embedding norms of the predictor."""
    was_training = model.training
    model.eval()
    blocks = model.predictor.transformer.layers
    gates, qkvs, handles = [], [], []
    try:
        for block in blocks:
            handles.append(block.adaLN_modulation.register_forward_hook(lambda m, i, o: gates.append(o.detach())))
            handles.append(block.attn.to_qkv.register_forward_hook(lambda m, i, o: qkvs.append(o.detach())))
        model.predict(states, actions)
        action_embedding = model.action_encoder(actions)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    if len(gates) != len(blocks) or len(qkvs) != len(blocks):
        raise RuntimeError('Predictor hooks did not observe every block exactly once')
    attention = torch.stack([_causal_attention(qkv, block.attn.heads) for qkv, block in zip(qkvs, blocks)])
    entropy = torch.stack([attention_entropy(layer) for layer in attention])  # [layers, heads]
    return dict(attention=attention.mean(1), attention_entropy=entropy,
                gate_msa=torch.stack([g.chunk(6, -1)[2].abs().mean() for g in gates]),
                gate_mlp=torch.stack([g.chunk(6, -1)[5].abs().mean() for g in gates]),
                action_embedding_norm=float(action_embedding.norm(dim=-1).mean()),
                state_norm=float(states.norm(dim=-1).mean()))


@torch.no_grad()
def sensitivity(model, states, actions, directions=8, seed=0, epsilon=1e-3):
    """Mean ||J v|| / ||v|| of the next-state prediction along random input directions.

    Central finite differences (double backward through fused attention is not
    available everywhere). Compared between action input and state input: a
    predictor that ignores actions has action sensitivity near zero regardless
    of latent scale.
    """
    was_training = model.training
    model.eval()
    generator = torch.Generator(device=states.device).manual_seed(seed)
    result = {}
    try:
        for name, inputs, function in (('action', actions, lambda a: model.predict(states, a)),
                                       ('state', states, lambda s: model.predict(s, actions))):
            step = epsilon * max(float(inputs.float().pow(2).mean().sqrt()), 1.0)
            ratios = []
            for _ in range(directions):
                v = torch.randn(inputs.shape, generator=generator, device=inputs.device, dtype=inputs.dtype)
                v = v / v.norm()
                difference = function(inputs + step * v) - function(inputs - step * v)
                ratios.append(float(difference.float().norm() / (2 * step)))
            result[name] = float(np.mean(ratios))
    finally:
        model.train(was_training)
    result['action_over_state'] = result['action'] / result['state'] if result['state'] > 0 else float('nan')
    return result


def parameter_norms(model):
    return {group: float(torch.sqrt(sum(p.detach().float().square().sum() for p in getattr(model, group).parameters())))
            for group in MODULE_GROUPS}


def gradient_norms(model, pixels, actions, regularizer, weight=0.09):
    """Per-module gradient norms of the full objective on one batch, without any update.

    Runs in eval mode so BatchNorm buffers and dropout masks are untouched; the
    measured gradient therefore differs slightly from a training-mode step.
    """
    was_training = model.training
    model.eval()
    for p in model.parameters():
        p.grad = None
    try:
        with torch.enable_grad():
            terms = loss_from_embeddings(model, model.encode(pixels), actions, regularizer, weight)
            terms['loss'].backward()
        result = {}
        for group in MODULE_GROUPS:
            grads = [p.grad for p in getattr(model, group).parameters() if p.grad is not None]
            result[group] = float(torch.sqrt(sum(g.float().square().sum() for g in grads))) if grads else 0.0
        result['total'] = float(torch.sqrt(sum(p.grad.float().square().sum() for p in model.parameters()
                                               if p.grad is not None)))
        result.update({k: float(v.detach()) for k, v in terms.items()})
    finally:
        for p in model.parameters():
            p.grad = None
        model.train(was_training)
    return result


@torch.no_grad()
def rollout_horizon(model, pixels, actions):
    """Autoregressive and teacher-forced latent error per horizon, with the copy baseline."""
    was_training = model.training
    model.eval()
    try:
        z = model.encode(pixels)
        future = model.rollout(z[:, :1], actions[:, :-1])
        history = model.predictor.pos_embedding.shape[1]
        # Teacher forcing: predict step h from the true states inside the history window.
        one_step = [model.predict(z[:, max(0, h - history):h], actions[:, max(0, h - history):h])[:, -1]
                    for h in range(1, z.shape[1])]
    finally:
        model.train(was_training)
    horizons = list(range(1, z.shape[1]))
    mse = lambda a, b: float((a.float() - b.float()).square().mean())
    return dict(horizon=horizons,
                prediction_mse=[mse(future[:, h - 1], z[:, h]) for h in horizons],
                one_step_mse=[mse(one_step[h - 1], z[:, h]) for h in horizons],
                copy_mse=[mse(z[:, 0], z[:, h]) for h in horizons],
                previous_mse=[mse(z[:, h - 1], z[:, h]) for h in horizons])


@torch.no_grad()
def encoder_maps(model, pixels):
    """Last-layer CLS-to-patch attention per head, patch-token PCA as RGB, entropy per layer."""
    encoder = model.encoder
    was_training = model.training
    implementation = encoder.config._attn_implementation
    model.eval()
    try:
        if hasattr(encoder, 'set_attn_implementation'):
            encoder.set_attn_implementation('eager')
        else:
            encoder.config._attn_implementation = 'eager'
        output = encoder(pixels.float(), interpolate_pos_encoding=True, output_attentions=True)
    finally:
        if hasattr(encoder, 'set_attn_implementation'):
            encoder.set_attn_implementation(implementation)
        else:
            encoder.config._attn_implementation = implementation
        model.train(was_training)
    patches = output.last_hidden_state.shape[1] - 1
    grid = int(round(math.sqrt(patches)))
    if grid * grid != patches:
        raise ValueError('Patch tokens do not form a square grid')
    cls = output.attentions[-1][:, :, 0, 1:]
    cls = cls / cls.sum(-1, keepdim=True)
    tokens = output.last_hidden_state[:, 1:].float()
    flat = tokens.reshape(-1, tokens.shape[-1])
    _, _, v = torch.linalg.svd(flat - flat.mean(0), full_matrices=False)
    components = (flat - flat.mean(0)) @ v[:3].T
    low, high = components.min(0).values, components.max(0).values
    rgb = ((components - low) / torch.clamp(high - low, min=1e-12)).reshape(len(pixels), grid, grid, 3)
    entropy = torch.stack([attention_entropy(layer).mean() for layer in output.attentions])
    return dict(cls_attention=cls.reshape(len(pixels), cls.shape[1], grid, grid), patch_pca=rgb,
                attention_entropy=entropy, grid=grid)
