"""CEM trajectory search with the baseline terminal latent-distance cost.

Adapted algorithm: stable-worldmodel@6f1e499e9cc0c898d326112f485c1062c3d20f24,
planning/solver/cem.py (MIT, third_party/swm/LICENSE). Initial observations and
goals are encoded once because they are identical across all CEM candidates.
"""
import torch


@torch.no_grad()
def cem(cost, horizon, action_dim, device, *, samples=300, iterations=30, elites=30,
        seed=42, initial_std=1.0):
    if not 1 < elites <= samples: raise ValueError('Need 1 < elites <= samples')
    generator=torch.Generator(device=device).manual_seed(seed)
    mean=torch.zeros(horizon,action_dim,device=device)
    std=torch.full_like(mean,initial_std)
    evaluations=0
    for _ in range(iterations):
        candidates=torch.randn(samples,horizon,action_dim,device=device,generator=generator)*std+mean
        candidates[0]=mean
        values=cost(candidates);evaluations+=samples
        if values.shape!=(samples,) or not torch.isfinite(values).all():
            raise ValueError('Invalid rollout costs')
        chosen=candidates[values.topk(elites,largest=False).indices]
        mean=chosen.mean(0);std=chosen.std(0,correction=0)
    # Verify the exact mean that will be executed, not an unscored aggregation.
    value=cost(mean[None]);evaluations+=1
    return mean,dict(terminal_cost=float(value[0]),candidate_evaluations=evaluations,
                     predictor_steps=evaluations*horizon)


def latent_cost(model, context, goal):
    """Single-environment initial latent [1,H,D] and goal latent [1,1,D].

    This first planner uses one current observation (H=1), as the reference
    closed-loop initialization does. Historical contexts require known actions.
    """
    if context.shape[1]!=1: raise ValueError('This planner expects one current observation')
    def cost(actions):
        rollout=model.rollout(context.expand(len(actions),-1,-1),actions)
        return (rollout[:,-1]-goal[0,-1]).square().sum(-1)
    return cost
