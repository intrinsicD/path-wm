"""Closed-form diagnostic readers; no claim of exhaustive latent decodability."""

import torch


class RidgeReader:
    @staticmethod
    def kernel(x, y, kind, bandwidth):
        dot = x @ y.T / x.shape[1]
        if kind == "linear":
            return dot
        if kind != "rbf" or bandwidth <= 0:
            raise ValueError("Use linear or positive-bandwidth rbf kernel")
        distance = (
            x.square().mean(1)[:, None] + y.square().mean(1)[None] - 2 * dot
        ).clamp_min(0)
        return torch.exp(-distance / (2 * bandwidth**2))

    @staticmethod
    def factor(x, y, *, kernel="linear", bandwidth=1.0):
        x, y = x.detach().double(), y.detach().double()
        if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(x) < 2:
            raise ValueError("Aligned nonempty training matrices required")
        if not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise ValueError("Finite training matrices required")
        mean = x.mean(0)
        std = x.std(0, correction=0).clamp_min(1e-4)
        target_mean = y.mean(0)
        z = (x - mean) / std
        gram = RidgeReader.kernel(z, z, kernel, bandwidth)
        eigenvalues, vectors = torch.linalg.eigh(gram)
        factor = dict(
            mean=mean,
            std=std,
            training=z,
            target_mean=target_mean,
            kernel=kernel,
            bandwidth=bandwidth,
        )
        return factor, eigenvalues.clamp_min(0), vectors, vectors.T @ (y - target_mean)

    @staticmethod
    def solve(factor, *, ridge):
        if ridge <= 0:
            raise ValueError("Positive ridge required")
        state, values, vectors, rotated = factor
        return dict(
            state, ridge=ridge, alpha=vectors @ (rotated / (values[:, None] + ridge))
        )

    @staticmethod
    def fit(x, y, *, ridge, kernel="linear", bandwidth=1.0):
        return RidgeReader.solve(
            RidgeReader.factor(x, y, kernel=kernel, bandwidth=bandwidth), ridge=ridge
        )

    @staticmethod
    def predict(state, x):
        x = (x.detach().double() - state["mean"]) / state["std"]
        return (
            RidgeReader.kernel(
                x, state["training"], state["kernel"], state["bandwidth"]
            )
            @ state["alpha"]
            + state["target_mean"]
        )
