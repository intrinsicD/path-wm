"""Read-only diagnostics for the fixed four-head encoder attention and PCA."""
import math

import numpy as np
import torch


def entropy(probabilities):
    """Normalized Shannon entropy per query/head, over the final key axis."""
    if probabilities.shape[-1] < 2:
        raise ValueError('normalized entropy requires at least two keys')
    return -(probabilities * probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()).sum(-1) / math.log(probabilities.shape[-1])


def attention_details(module, query, key, value):
    """Explicit matmul reference, independent of the SDPA forward implementation."""
    def split(tensor):
        return tensor.reshape(tensor.shape[0], tensor.shape[1], 4, 16).transpose(1, 2)

    q = split(module.query_projection(query))
    k = split(module.key_projection(key))
    v = split(module.value_projection(value))
    probabilities = ((q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])).softmax(-1)

    def project(mixed):
        return module.output_projection(mixed.transpose(1, 2).reshape(query.shape[0], query.shape[1], 64))

    output = project(probabilities @ v)
    uniform = project(v.mean(-2, keepdim=True).expand(-1, -1, query.shape[1], -1))
    return dict(probabilities=probabilities, entropy=entropy(probabilities),
                output=output, uniform_output=uniform)


def variance_parts(tokens):
    """Population mean squared vector variation, image means plus spatial residuals."""
    tokens = np.asarray(tokens, dtype=np.float64)
    image_means = tokens.mean(1, keepdims=True)
    grand_mean = tokens.mean((0, 1), keepdims=True)
    total = np.square(tokens - grand_mean).sum(-1).mean()
    between = np.square(image_means - grand_mean).sum(-1).mean()
    within = np.square(tokens - image_means).sum(-1).mean()
    return dict(total_variance=float(total), image_mean_variance=float(between),
                within_image_variance=float(within),
                image_mean_fraction=float(between / total) if total > 0 else None)


def fit_pca(training, image_centered=False):
    x = np.asarray(training, dtype=np.float64)
    if image_centered:
        x = x - x.mean(1, keepdims=True)
    x = x.reshape(-1, x.shape[-1])
    mean = x.mean(0)
    x = x - mean
    values, vectors = np.linalg.eigh(x.T @ x / (len(x) - 1))
    basis = vectors[:, -3:][:, ::-1]
    # Deterministic signs within this fit; unrelated fits still have unrelated colors.
    signs = np.sign(basis[np.argmax(np.abs(basis), axis=0), np.arange(3)])
    basis = basis * signs
    low, high = np.percentile(x @ basis, [2, 98], axis=0)
    return dict(mean=mean, basis=basis, low=low, high=high,
                top3_variance_fraction=float(values[-3:].sum() / values.sum()),
                image_centered=image_centered)


def pca_colors(tokens, fit):
    x = np.asarray(tokens, dtype=np.float64)
    if fit['image_centered']:
        x = x - x.mean(1, keepdims=True)
    projected = (x - fit['mean']) @ fit['basis']
    scaled = (projected - fit['low']) / np.maximum(fit['high'] - fit['low'], 1e-9)
    return np.clip(scaled, 0, 1), float(np.mean((scaled < 0) | (scaled > 1)))
