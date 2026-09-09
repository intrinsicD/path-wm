"""Distribution losses with explicit stop-gradient and Monte Carlo conventions."""

import math
import torch
from torch.nn import functional as F


def categorical_kl(q, p):
    """KL(q||p) per categorical group, before free-nat clipping."""
    qlog, plog = q.log_softmax(-1), p.log_softmax(-1)
    return (qlog.exp() * (qlog - plog)).sum(-1)


def split_kl(q, p, *, free_nats=1.0, valid=None):
    if free_nats < 0:
        raise ValueError("Free nats must be nonnegative")

    def average(values):
        if valid is None:
            return values.mean()
        return (values * valid).sum() / valid.sum().clamp_min(1)

    return (
        average(categorical_kl(q.detach(), p).sum(-1).clamp_min(free_nats)),
        average(categorical_kl(q, p.detach()).sum(-1).clamp_min(free_nats)),
    )


def partial_kl(full, partial):
    return categorical_kl(full.detach(), partial).sum(-1).mean()


def marginal_nll(log_likelihoods):
    """[samples,B] joint observable likelihood; log MC mean has downward bias.

    Gradients through straight-through latent draws are a biased surrogate. This
    function does not average predictions or density-reweight posterior samples.
    """
    if log_likelihoods.ndim != 2 or len(log_likelihoods) < 1:
        raise ValueError("Marginal likelihood expects [samples,B]")
    return math.log(len(log_likelihoods)) - torch.logsumexp(log_likelihoods, 0)


def gaussian_log_likelihood(prediction, target, sigma=0.1):
    if sigma <= 0:
        raise ValueError("Gaussian observation scale must be positive")
    return (
        (
            -0.5 * ((prediction - target) / sigma).square()
            - math.log(sigma)
            - 0.5 * math.log(2 * math.pi)
        )
        .flatten(1)
        .sum(1)
    )


def marking_loss(score, loss_without, loss_with, *, storage_cost=0.01):
    """Regress delayed utility net of explicit storage cost; targets are detached."""
    utility = (loss_without - loss_with).detach() - storage_cost
    return F.mse_loss(score, utility)
