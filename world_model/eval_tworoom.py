"""TwoRoom source-state control with the pinned simulator and shared baseline CEM.

The default simulator geometry is verified against source pixels and recorded
transitions by scripts.prepare_tworoom_evaluation. Success is the upstream
post-action distance <16 pixels, including when the initial state is close.
"""
import time
import numpy as np
import torch
from third_party.swm.tworoom import TwoRoomEnv
from world_model.data import preprocess_pixels
from world_model.evaluation import initial_observations
from world_model.planning import cem, latent_cost


def _environment(state, target, seed):
    state, target = np.asarray(state, dtype=np.float32), np.asarray(target, dtype=np.float32)
    if state.shape != (2,) or target.shape != (2,) or not np.isfinite([state, target]).all():
        raise ValueError('TwoRoom requires finite two-dimensional agent and goal positions')
    env = TwoRoomEnv()
    env.reset(seed=seed)
    env._set_state(state); env._set_goal_state(target)
    return env


def _distance(env):
    return float(torch.linalg.vector_norm(env.agent_position-env.target_position))


def run_actions(state, target, actions, *, budget, seed=42):
    """Replay at most the supplied actions and budget; stop at first success."""
    actions = np.asarray(actions, dtype=np.float32)
    if budget < 1 or actions.ndim != 2 or actions.shape[1] != 2 or not len(actions) or not np.isfinite(actions).all():
        raise ValueError('A positive budget and nonempty finite (N,2) actions are required')
    env = _environment(state, target, seed)
    initial = _distance(env) < 16.
    used = 0; success = False
    try:
        for action in actions[:budget]:
            _, _, success, _, _ = env.step(action); used += 1
            if success: break
        return dict(success=bool(success), initial_success=initial, steps=used,
                    state_distance=_distance(env), final_state=env.agent_position.tolist())
    finally:
        env.close()


@torch.no_grad()
def evaluate_case(model, source_frame, goal_frame, state, target, stats, *, seed=42,
                  samples=300, iterations=30, elites=30, budget=50, frameskip=5):
    if budget < 1 or frameskip < 1: raise ValueError('Positive control and action-block budgets required')
    mean, std = np.asarray(stats['mean']), np.asarray(stats['std'])
    if mean.shape != (2,) or std.shape != (2,) or not np.isfinite([mean,std]).all() or (std <= 0).any():
        raise ValueError('Invalid saved action normalization')
    device = next(model.parameters()).device
    env = _environment(state, target, seed)
    initial, goal_pixels = initial_observations(source_frame, goal_frame)
    initial_success = _distance(env) < 16.
    success = False; used = 0; calls = 0; plans = []; executed = []
    started = time.monotonic()
    generator = torch.Generator(device=device).manual_seed(seed)
    try:
        goal = model.encode(preprocess_pixels(goal_pixels.to(device)))
        while used < budget and not success:
            pixels = initial.to(device) if used == 0 else torch.from_numpy(env.render().copy()).permute(2,0,1)[None,None].to(device)
            context = model.encode(preprocess_pixels(pixels))
            actions, details = cem(latent_cost(model, context, goal), 5, 2*frameskip, device,
                samples=samples, iterations=iterations, elites=elites, generator=generator)
            calls += details['predictor_steps']; plans.append(details)
            # Keep the same float32 inverse-transform arithmetic as PushT/upstream.
            raw = actions.reshape(-1,2).cpu().numpy().copy()
            raw *= std; raw += mean
            for action in raw[:budget-used]:
                _, _, success, _, _ = env.step(action)
                used += 1; executed.append(action.copy())
                if success: break
        return dict(success=bool(success), initial_success=initial_success, steps=used,
            state_distance=_distance(env), final_state=env.agent_position.tolist(),
            predictor_steps=calls, seconds=time.monotonic()-started, plans=plans,
            actions=np.asarray(executed, dtype=np.float32))
    finally:
        env.close()
