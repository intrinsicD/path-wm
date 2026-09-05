"""Training primitives; encoder recomputation preserves full-batch statistics.

Only the deterministic ViT is chunked. Projectors, dynamics and SIGReg always
see the full batch. This is not microbatch accumulation of the objective.
"""
from contextlib import nullcontext
import torch
from torch import nn
from world_model.objective import loss_from_embeddings


def autocast_context(device, precision):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16) if precision == 'bf16' else nullcontext()


def backward_batch(model, pixels, actions, regularizer, weight=.09,
                   encoder_chunk=0, precision='float32'):
    if not encoder_chunk:
        with autocast_context(pixels.device, precision):
            terms = loss_from_embeddings(model, model.encode(pixels), actions, regularizer, weight)
        terms['loss'].backward()
        return {k: v.detach().float() for k,v in terms.items()}
    for m in model.encoder.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm) or (isinstance(m, nn.Dropout) and m.p):
            raise ValueError('Encoder recomputation requires deterministic, stateless layers')
    # ViT attention has an independent dropout field in its config.
    if getattr(model.encoder.config, 'attention_probs_dropout_prob', 0):
        raise ValueError('Attention dropout requires RNG replay; disable chunking')
    b,t = pixels.shape[:2]
    flat = pixels.flatten(0,1)
    with torch.no_grad(), autocast_context(pixels.device, precision):
        features = torch.cat([model.features(x) for x in flat.split(encoder_chunk)])
    features.requires_grad_()
    with autocast_context(pixels.device, precision):
        z = model.projector(features).reshape(b,t,-1)
        terms = loss_from_embeddings(model,z,actions,regularizer,weight)
    terms['loss'].backward()
    for x,g in zip(flat.split(encoder_chunk), features.grad.split(encoder_chunk)):
        with autocast_context(pixels.device, precision):
            recomputed = model.features(x)
        torch.autograd.backward(recomputed, g)
    return {k: v.detach().float() for k,v in terms.items()}
