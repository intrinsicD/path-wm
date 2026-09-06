"""Gradient geometry for diagnostic clones; no training recipe changes.

Geometry identities and initial implementation contributed by Claude through MCP,
2026-09-06 (public generic task); adapted to a small Gram matrix to bound memory.
Ratios are in units of whole batch gradients, NOT a per-example critical batch.
"""
from contextlib import contextmanager
import torch


def gradient_geometry(grads):
    if len(grads) < 2 or any(g.shape != grads[0].shape for g in grads):
        raise ValueError('Need at least two same-shaped gradients')
    n = len(grads)
    gram = torch.empty(n, n, dtype=torch.float64)
    for i, a in enumerate(grads):
        for j in range(i + 1):
            gram[i, j] = gram[j, i] = torch.dot(a.flatten().double(), grads[j].flatten().double())
    if not torch.isfinite(gram).all():
        raise ValueError('Non-finite gradients')
    mean_sq = float(gram.mean())
    trace = max(0., float((gram.diag().sum() - n * mean_sq) / (n - 1)))
    signal = float((gram.sum() - gram.diag().sum()) / (n * (n - 1)))
    cos = [float(gram[i,j] / (gram[i,i]*gram[j,j]).sqrt())
           for i in range(n) for j in range(i) if gram[i,i] > 0 and gram[j,j] > 0]
    return dict(replicates=n, mean_norm_squared=mean_sq, sample_covariance_trace=trace,
                signal_estimate=signal, noise_to_signal=trace/signal if signal > 0 else None,
                pairwise_cosine_mean=sum(cos)/len(cos) if cos else None)


def module_geometry(pred, reg):
    p, r = float(pred.norm()), float(reg.norm())
    return dict(prediction_norm=p, weighted_sigreg_norm=r, total_norm=float((pred+reg).norm()),
                cosine=float(torch.dot(pred, reg)/(p*r)) if p*r else None)


def adam_delta(parameter, gradient, state, group):
    """Analytical next AdamW parameter delta, with pre-clipped input gradient.

    No input is modified; defaults match the saved non-AMSGrad local optimizer.
    Caller supplies the NEXT schedule learning rate, not a new optimizer state.
    """
    if group.get('amsgrad') or group.get('maximize'):
        raise ValueError('Unsupported Adam variant')
    beta1, beta2 = group['betas']
    step = int(state['step']) + 1
    m = beta1*state['exp_avg'] + (1-beta1)*gradient
    v = beta2*state['exp_avg_sq'] + (1-beta2)*gradient.square()
    direction = (m/(1-beta1**step)) / ((v/(1-beta2**step)).sqrt()+group['eps'])
    return -group['lr']*(direction+group['weight_decay']*parameter)


@contextmanager
def preserved_state(model):
    """Restore buffers, modes and RNG even on failure; prohibit optimizer use."""
    buffers = {k:v.detach().clone() for k,v in model.named_buffers()}
    modes = [(m,m.training) for m in model.modules()]
    devices = sorted({p.device.index for p in model.parameters() if p.is_cuda})
    with torch.random.fork_rng(devices=devices):
        try:
            yield
        finally:
            with torch.no_grad():
                for k,v in model.named_buffers(): v.copy_(buffers[k])
            for module, mode in modes: module.training = mode
            model.zero_grad(set_to_none=True)
