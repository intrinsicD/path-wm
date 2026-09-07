"""Held-out, matched controls for the paddle experiment.

Simulator labels occur only in diagnostics and the explicitly privileged policy.
The deployed planner receives a PlanningState, never an environment or labels.
All readout errors below are in physical units, with time measured in intervals.
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import os
import platform
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .types import ObservationLatent, PlanningState, action_one_hot


POSITION_NAMES = ["ball_x", "ball_y", "paddle_x"]
STATE_NAMES = ["ball_x", "ball_y", "ball_vx", "ball_vy", "paddle_x"]
COLLISIONS = {"side_wall", "ceiling", "paddle_hit"}
CONTROLLERS = ("learned", "reset", "random", "tracker", "privileged")
PRIVILEGED_PLANNER_IMPLEMENTATION = "exhaustive-5-state-advance-scalar-prefix-v2"


def _pin_numeric_mode():
    """Standalone evaluation must use the same FP32 baseline as training."""
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def _device(requested="auto"):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; select --device cpu or run with GPU access")
    return device


def _rgb(frames, device):
    return torch.as_tensor(np.asarray(frames).copy(), device=device).permute(0, 3, 1, 2).float() / 255


def _synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _action(ids, device):
    return action_one_hot(torch.as_tensor(ids, dtype=torch.long, device=device))


@torch.inference_mode()
def assimilate_history(system, frames, actions, device="cpu"):
    """Return I'_t after exactly one observation update at each real boundary."""
    if len(frames) != len(actions) + 1 or not len(frames):
        raise ValueError("History requires T+1 frames and T previous executable actions")
    memory = torch.zeros(1, 128, device=device)
    for t, frame in enumerate(frames):
        observation = system["E"](_rgb(frame[None], device))
        previous = torch.zeros(1, 3, device=device) if t == 0 else _action([actions[t - 1]], device)
        memory = system["U"](memory, observation, previous)
    return PlanningState(observation, memory)


@torch.inference_mode()
def matched_rollout(system, initial, actions):
    """Prediction/copy/reset share source S and executed actions at every horizon.

    Reset means assimilating the current S from zeros with the initial marker.
    Copy keeps S fixed while still advancing U with the matched candidate action.
    Targets are deliberately absent from this API.
    """
    if actions.ndim != 2:
        raise ValueError("Actions must have shape [batch, horizon]")
    fresh = system["U"](torch.zeros_like(initial.memory), initial.observation,
                        torch.zeros(len(actions), 3, device=initial.memory.device))
    states = {
        "prediction": PlanningState(initial.observation.clone(), initial.memory.clone()),
        "copy": PlanningState(initial.observation.clone(), initial.memory.clone()),
        "reset": PlanningState(initial.observation.clone(), fresh),
    }
    result = {name: [] for name in states}
    for k in range(actions.shape[1]):
        action = _action(actions[:, k], initial.memory.device)
        for name, state in states.items():
            predicted = initial.observation.clone() if name == "copy" else system["P"](state.observation, state.memory, action)
            memory = system["U"](state.memory, predicted, action)
            states[name] = PlanningState(predicted, memory)
            result[name].append(states[name])
    return result


def summarize_prediction_records(records):
    """Do not mix horizons, collision populations, or coordinate denominators."""
    result = {}
    for method in sorted({r["method"] for r in records}):
        result[method] = {}
        for horizon in sorted({r["horizon"] for r in records if r["method"] == method}):
            group = [r for r in records if r["method"] == method and r["horizon"] == horizon]
            result[method][str(horizon)] = {}
            for kind in ("all", "collision", "no_collision", *sorted(COLLISIONS)):
                selected = [r for r in group if kind == "all"
                            or kind == "collision" and bool(r["collision"])
                            or kind == "no_collision" and not bool(r["collision"])
                            or kind in COLLISIONS and kind in r.get("collision_types", [])]
                item = {"count": len(selected), "h_mae": None, "r_mae": None, "latent_error": None}
                if selected:
                    h = np.asarray([r["h_abs_error"] for r in selected])
                    r = np.asarray([r["r_abs_error"] for r in selected])
                    item.update(h_mae=h.mean(0).tolist(), r_mae=r.mean(0).tolist(),
                                h_p95=np.quantile(h, .95, axis=0).tolist(), h_max=h.max(0).tolist(),
                                r_p95=np.quantile(r, .95, axis=0).tolist(), r_max=r.max(0).tolist(),
                                latent_error=float(np.mean([x["latent_error"] for x in selected])),
                                terminal_count=sum(bool(x.get("terminal_target", False)) for x in selected),
                                terminal_false_negative_count=sum(bool(x.get("terminal_false_negative", False)) for x in selected))
                result[method][str(horizon)][kind] = item
    return result


@dataclass
class LinearVelocityProbe:
    weight: torch.Tensor
    bias: torch.Tensor

    def __call__(self, features):
        return features @ self.weight + self.bias


def fit_linear_velocity_probe(features, velocity, ridge=1e-2):
    """Diagnostic ridge linear head; fit inputs must come from training only.

    The dual solve avoids a 20,480-square covariance matrix. Identical frozen S
    necessarily produces identical output, regardless of history or pair labels.
    """
    if len(features) != len(velocity) or len(features) < 2 or ridge <= 0:
        raise ValueError("Probe needs aligned training rows, at least two rows, and positive ridge")
    x, y = features.float(), velocity.float()
    xmean, ymean = x.mean(0), y.mean(0)
    x, y = x - xmean, y - ymean
    if x.shape[0] <= x.shape[1]:
        gram = x @ x.T
        gram.diagonal().add_(ridge * max(1., float(gram.diagonal().mean())))
        weight = x.T @ torch.linalg.solve(gram, y)
    else:
        gram = x.T @ x
        gram.diagonal().add_(ridge * max(1., float(gram.diagonal().mean())))
        weight = torch.linalg.solve(gram, x.T @ y)
    return LinearVelocityProbe(weight, ymean - xmean @ weight)


@torch.inference_mode()
def _encode_episode(system, episode, device, frame_batch=128):
    frames = episode["frames"]
    chunks = [system["E"](_rgb(frames[i:i + frame_batch], device)) for i in range(0, len(frames), frame_batch)]
    observation = ObservationLatent.cat(chunks)
    memories = []
    memory = torch.zeros(1, 128, device=device)
    for t in range(len(frames)):
        previous = torch.zeros(1, 3, device=device) if t == 0 else _action([episode["actions"][t - 1]], device)
        memory = system["U"](memory, observation[t:t + 1], previous)
        memories.append(memory)
    return observation, torch.cat(memories)


def _events(episode):
    if "events" in episode:
        return episode["events"]
    return json.loads(str(episode["events_json"].item()))


def _collision_types(episode, start, stop):
    return sorted({e["type"] for e in _events(episode)
                   if start <= e["transition_index"] < stop and e["type"] in COLLISIONS})


def _statistics(system):
    statistics = system.get("statistics", {})
    if "v_fine" not in statistics or "v_coarse" not in statistics:
        raise ValueError("Predictor requires training-only v_fine/v_coarse statistics")
    return max(float(statistics["v_fine"]), 1e-6), max(float(statistics["v_coarse"]), 1e-6)


def _error_row(system, state, truth_s, truth_state, fine_variance, coarse_variance, **metadata):
    positions = (system["H"](state.observation)[0] * 64).double().cpu().numpy()
    estimates = (system["R"](state.memory)[0] * torch.tensor([64, 64, 6, 6, 64], device=state.memory.device)).double().cpu().numpy()
    latent = .5 * ((state.observation.fine - truth_s.fine).square().mean() / fine_variance
                   + (state.observation.coarse - truth_s.coarse).square().mean() / coarse_variance)
    if not np.isfinite(positions).all() or not np.isfinite(estimates).all() or not torch.isfinite(latent):
        raise FloatingPointError(f"Nonfinite diagnostic prediction at {metadata}")
    terminal = bool(truth_state[1] >= 61)
    return dict(metadata, h_abs_error=np.abs(positions - truth_state[[0, 1, 4]]).tolist(),
                r_abs_error=np.abs(estimates - truth_state).tolist(), h_positions=positions.tolist(),
                r_state=estimates.tolist(), truth_state=np.asarray(truth_state).tolist(),
                latent_error=float(latent), terminal_target=terminal,
                terminal_false_negative=terminal and bool(positions[1] < 61))


@torch.inference_mode()
def prediction_diagnostics(system, dataset, device, window_limit=512, seed=7300):
    """Uniform held-out windows; all methods use identical source and target rows."""
    candidates = []
    for episode_id in range(len(dataset)):
        episode = dataset[episode_id]
        candidates.extend((episode_id, t) for t in range(2, len(episode["actions"]) - 4))
    if not candidates:
        raise ValueError("Evaluation data has no five-transition window after three-frame warm-up")
    rng = np.random.default_rng(seed)
    selected = sorted(candidates[i] for i in rng.choice(len(candidates), min(window_limit, len(candidates)), replace=False))
    grouped = defaultdict(list)
    for episode_id, t in selected:
        grouped[episode_id].append(t)
    vf, vc = _statistics(system)
    rows = []
    for episode_id in range(len(dataset)):
        episode = dataset[episode_id]
        observation, memories = _encode_episode(system, episode, device)
        for t in range(2, len(episode["frames"])):
            collision = _collision_types(episode, t - 1, t)
            actual = PlanningState(observation[t:t + 1], memories[t:t + 1])
            rows.append(_error_row(system, actual, actual.observation, episode["states"][t], vf, vc,
                                  method="actual", horizon=0, episode=episode_id, source_index=t,
                                  target_index=t, collision=bool(collision), collision_types=collision))
        for t in grouped[episode_id]:
            initial = PlanningState(observation[t:t + 1], memories[t:t + 1])
            imagined = matched_rollout(system, initial, torch.as_tensor(episode["actions"][t:t + 5][None], device=device))
            for method, states in imagined.items():
                for horizon, state in enumerate(states, 1):
                    collision = _collision_types(episode, t, t + horizon)
                    rows.append(_error_row(system, state, observation[t + horizon:t + horizon + 1],
                                          episode["states"][t + horizon], vf, vc, method=method,
                                          horizon=horizon, episode=episode_id, source_index=t,
                                          target_index=t + horizon, collision=bool(collision), collision_types=collision))
    return {"summary": summarize_prediction_records(rows), "records": rows,
            "window_count": len(selected), "available_windows": len(candidates), "window_seed": seed,
            "readout_population": "All test observations from index 2 onward; no padded or reset frames",
            "collision_definition": "At least one side-wall, ceiling, or paddle reflection in the source-to-target prefix"}


def reconstruction_errors(reconstruction, frames, states):
    """Exact geometric coverage weights are diagnostics, never training inputs."""
    from .env import rectangle_coverage

    predicted = reconstruction.detach().double().cpu().numpy().transpose(0, 2, 3, 1)
    frames = np.asarray(frames)
    states = np.asarray(states)
    if predicted.shape != frames.shape or frames.shape != (len(states), 64, 64, 3):
        raise ValueError("Reconstruction diagnostics require aligned RGB frames and state labels")
    error = (predicted - frames.astype(np.float64) / 255) ** 2
    if not np.isfinite(error).all():
        raise FloatingPointError("Nonfinite perception reconstruction")
    result = {"count": len(states), "global_squared_error": float(error.sum()), "global_scalars": int(error.size)}
    for name, rectangles in (
        ("ball_region", [(s[0]-2,s[0]+2,s[1]-2,s[1]+2) for s in states]),
        ("paddle_region", [(s[4]-6,s[4]+6,56,59) for s in states]),
    ):
        coverage = np.stack([rectangle_coverage(*rectangle) for rectangle in rectangles])
        result[f"{name}_squared_error"] = float((error * coverage[..., None]).sum())
        result[f"{name}_scalars"] = float(coverage.sum() * 3)
    return result


@torch.inference_mode()
def reconstruction_diagnostics(system, dataset, device, frame_batch=128):
    """All held-out raw frames, including warm-up and visible terminal targets."""
    totals, records, readout_errors, readout_records = defaultdict(float), [], [], []
    for episode_id in range(len(dataset)):
        episode = dataset[episode_id]
        episode_totals = defaultdict(float)
        for start in range(0, len(episode["frames"]), frame_batch):
            frames = episode["frames"][start:start+frame_batch]
            states = episode["states"][start:start+frame_batch]
            observation = system["E"](_rgb(frames, device))
            positions = (system["H"](observation) * 64).double().cpu().numpy()
            if not np.isfinite(positions).all():
                raise FloatingPointError("Nonfinite real-frame position readout")
            batch_errors = np.abs(positions - states[:, [0, 1, 4]])
            readout_errors.append(batch_errors)
            readout_records.extend({"episode": episode_id, "frame_index": start+index,
                "h_positions": positions[index].tolist(), "truth_positions": states[index, [0, 1, 4]].tolist(),
                "h_abs_error": batch_errors[index].tolist()} for index in range(len(frames)))
            reconstruction = system["D"](observation)
            for name, value in reconstruction_errors(reconstruction, frames, states).items():
                totals[name] += value
                episode_totals[name] += value
        records.append({"episode": episode_id, **dict(episode_totals)})
    result = {"count": int(totals["count"]), "records": records,
              "population": "Every held-out test frame, including initial/warm-up and terminal observations",
              "definition": "Squared RGB error on [0,1] pixels; region means use exact label-derived rectangle area coverage and three color channels. Diagnostic masks never alter training losses."}
    for name in ("global", "ball_region", "paddle_region"):
        denominator = totals[f"{name}_scalars"]
        result[f"{name}_mse"] = totals[f"{name}_squared_error"] / denominator if denominator else None
        result[f"{name}_scalars"] = denominator
    errors = np.concatenate(readout_errors)
    result["actual_frame_readout"] = {"count": len(errors), "h_mae": errors.mean(0).tolist(),
        "h_p95": np.quantile(errors, .95, axis=0).tolist(), "h_max": errors.max(0).tolist(),
        "source_records": "actual_frame_readout_records.json",
        "population": "All real test frames, including initial/warm-up and terminal frames; H has no velocity supervision mask"}
    result["readout_records"] = readout_records
    return result


@torch.inference_mode()
def train_velocity_probe(system, train_dataset, device, frame_limit=1024, seed=7400):
    if getattr(train_dataset, "split", "train") != "train":
        raise ValueError("Velocity probe may be fitted only on the training split")
    candidates = []
    for episode_id in range(len(train_dataset)):
        episode = train_dataset[episode_id]
        candidates.extend((episode_id, t) for t in range(2, len(episode["frames"])))
    rng = np.random.default_rng(seed)
    selected = sorted(candidates[i] for i in rng.choice(len(candidates), min(frame_limit, len(candidates)), replace=False))
    grouped = defaultdict(list)
    for episode_id, t in selected:
        grouped[episode_id].append(t)
    features, labels = [], []
    for episode_id, indices in grouped.items():
        episode = train_dataset[episode_id]
        for start in range(0, len(indices), 128):
            ix = indices[start:start + 128]
            encoded = system["E"](_rgb(episode["frames"][ix], device))
            features.append(encoded.tokens().flatten(1))
            labels.append(torch.as_tensor(episode["states"][ix][:, 2:4], device=device, dtype=torch.float32))
    probe = fit_linear_velocity_probe(torch.cat(features), torch.cat(labels))
    return probe, {"split": "train", "rows": selected, "count": len(selected), "seed": seed,
                   "method": "Centered linear ridge regression; training-only fit; physical vx/vy units",
                   "ridge": .01, "deployed": False}


def _privileged_candidate_scores(env):
    """Return all exact simulator scores in fixed stay-left-right order."""
    from .planner import PlanningFailure
    candidates = []

    def visit(branch, sequence, squared_sum, movement_count, first_miss):
        if len(sequence) == 5:
            score = (int(first_miss > 0), 5-first_miss if first_miss else 0,
                     squared_sum/(first_miss or 5), movement_count, len(candidates))
            candidates.append((score, sequence))
            return
        for action in (1, 0, 2):
            child = branch.clone()
            if not child.terminated and not child.truncated:
                child.advance(action)
            next_sum, next_count, next_miss = squared_sum, movement_count, first_miss
            if not first_miss:
                x, y, _, _, paddle = map(float, child.state)
                if not all(math.isfinite(value) for value in (x, y, paddle)):
                    raise PlanningFailure("non-finite candidate position prediction")
                # Keep sequence_score's Python float arithmetic/order exactly:
                # vectorized squares/reductions can perturb tied scores by ULPs.
                next_sum += ((x-paddle)/64)**2
                next_count += action != 1
                if y >= 61:
                    next_miss = len(sequence) + 1
            # Truncation freezes physics, but only a miss stops score accumulation.
            visit(child, (*sequence, action), next_sum, next_count, next_miss)

    visit(env, (), 0., 0, 0)
    return candidates


def privileged_plan(env):
    """Exhaustive simulator reference, isolated from learned planner inputs."""
    score, sequence = min(_privileged_candidate_scores(env))
    return sequence[0], sequence, score


def _wilson(successes, count):
    if not count:
        return None
    z = 1.959963984540054
    p, d = successes / count, 1 + z * z / count
    center = (p + z * z / (2 * count)) / d
    radius = z * np.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / d
    return [float(center - radius), float(center + radius)]


@torch.inference_mode()
def run_controller(system, controller, initial_state, device="cpu", max_steps=200, seed=0, planner=None,
                   progress_callback=None):
    from .env import PaddleEnv
    from .planner import ExhaustivePlanner, PlanningFailure

    if controller not in CONTROLLERS:
        raise ValueError(f"Unknown controller {controller}")
    env = PaddleEnv(state=np.asarray(initial_state, dtype=np.float64))
    frames, actions, states = [env.render()], [], [np.asarray(env.state).copy()]
    for _ in range(2):
        frame, _, _, _ = env.step(1)
        frames.append(frame)
        actions.append(1)
        states.append(np.asarray(env.state).copy())
    state = assimilate_history(system, np.asarray(frames), np.asarray(actions), device)
    if planner is None and controller in ("learned", "reset"):
        planner = ExhaustivePlanner(system["P"], system["U"], system["H"])
    rng = np.random.default_rng(seed)
    latencies, decisions = [], []
    first_hit_step, planning_failure = None, None
    invalid_candidates = 0

    def snapshot():
        return {"controller": controller, "initial_state": np.asarray(initial_state).tolist(), "seed": seed,
                "first_hit_before_miss": first_hit_step is not None, "first_hit_step": first_hit_step,
                "total_hits": int(env.hit_count), "episode_length": int(env.step_index),
                "terminated": bool(env.terminated), "truncated": bool(env.truncated),
                "evaluation_capped": not env.terminated and not env.truncated and env.step_index >= max_steps,
                "planning_failure": planning_failure, "invalid_candidates": invalid_candidates,
                "actions": list(actions), "states": [s.tolist() for s in states], "decisions": list(decisions),
                "first_action": actions[2] if len(actions) > 2 else None, "decision_latency_ms": list(latencies)}

    if progress_callback is not None:
        progress_callback(snapshot())
    while not env.terminated and not env.truncated and env.step_index < max_steps:
        planning_state = state
        _synchronize(device)
        started = time.perf_counter()
        try:
            if controller == "reset":
                memory = system["U"](torch.zeros_like(state.memory), state.observation, torch.zeros(1, 3, device=device))
                planning_state = PlanningState(state.observation, memory)
            if controller in ("learned", "reset"):
                result = planner.plan(planning_state)
                action, sequence = int(result.action_id), list(result.sequence)
                invalid_candidates += int(result.invalid_candidates)
            elif controller == "random":
                action, sequence = int(rng.integers(0, 3)), None
            elif controller == "tracker":
                position = system["H"](state.observation)[0] * 64
                if not torch.isfinite(position).all():
                    raise PlanningFailure("Current-frame tracker readout is nonfinite")
                delta = float(position[0] - position[2])
                action, sequence = (1 if abs(delta) <= 2 else 2 if delta > 0 else 0), None
            else:
                action, sequence, _ = privileged_plan(env)
                sequence = list(sequence)
        except PlanningFailure as exc:
            planning_failure = str(exc)
            break
        _synchronize(device)
        latencies.append((time.perf_counter() - started) * 1000)
        decisions.append({"step_index": int(env.step_index), "action": action, "candidate_sequence": sequence})
        frame, _, _, _ = env.step(action)
        actions.append(action)
        states.append(np.asarray(env.state).copy())
        if env.hit_count and first_hit_step is None:
            first_hit_step = int(env.step_index)
        # Raw trajectories permit failure inspection without adopting imagined memory.
        # Ten-decision snapshots bound I/O; an unfinished case restarts from its seed.
        if progress_callback is not None and len(decisions) % 10 == 0:
            progress_callback(snapshot())
        # Assimilate only the real observation, with the real executed action.
        observation = system["E"](_rgb(frame[None], device))
        state = PlanningState(observation, system["U"](state.memory, observation, _action([action], device)))
    result = snapshot()
    if progress_callback is not None:
        progress_callback(result)
    return result


def summarize_control(records):
    summary = {}
    for controller in sorted({r["controller"] for r in records}):
        rows = [r for r in records if r["controller"] == controller]
        successes = sum(r["first_hit_before_miss"] for r in rows)
        latency = [x for r in rows for x in r["decision_latency_ms"]]
        first_correct = [r["first_action"] == r["correct_action"] for r in rows if "correct_action" in r]
        summary[controller] = {
            "successes": successes, "count": len(rows), "success_rate": successes / len(rows),
            "success_wilson95": _wilson(successes, len(rows)),
            "first_action_correct": sum(first_correct) if first_correct else None,
            "first_action_count": len(first_correct),
            "total_hits": sum(r["total_hits"] for r in rows),
            "mean_episode_length": float(np.mean([r["episode_length"] for r in rows])),
            "planning_failures": sum(r["planning_failure"] is not None for r in rows),
            "invalid_candidates": sum(r["invalid_candidates"] for r in rows),
            "evaluation_capped": sum(r["evaluation_capped"] for r in rows),
            "decision_count": len(latency),
            "latency_median_ms": float(np.median(latency)) if latency else None,
            "latency_p95_ms": float(np.quantile(latency, .95)) if latency else None,
            "failure_case_ids": [r["case_id"] for r in rows if not r["first_hit_before_miss"]],
        }
    return summary


@torch.inference_mode()
def paired_probe_diagnostics(system, probe, pairs, device):
    rows = []
    for pair_id, pair in enumerate(pairs):
        final_frames = [member["frames"][-1] for member in pair["members"]]
        if not np.array_equal(*final_frames):
            raise ValueError(f"Paired-history case {pair_id} has unequal last raw frames")
        for member in pair["members"]:
            state = assimilate_history(system, member["frames"], member["actions"], device)
            velocity = probe(state.observation.tokens().flatten(1))[0].cpu().numpy()
            memory_velocity = (system["R"](state.memory)[0, 2:4] * 6).cpu().numpy()
            truth = np.asarray(member["states"][-1])[2:4]
            rows.append({"pair_id": pair_id, "direction": int(member["direction"]),
                         "truth_velocity": truth.tolist(), "frame_probe_velocity": velocity.tolist(),
                         "memory_velocity": memory_velocity.tolist(),
                         "frame_probe_abs_error": np.abs(velocity - truth).tolist(),
                         "memory_abs_error": np.abs(memory_velocity - truth).tolist()})
    return {"count": len(rows), "pairs": len(pairs),
            "frame_probe_mae": np.asarray([r["frame_probe_abs_error"] for r in rows]).mean(0).tolist() if rows else None,
            "memory_mae": np.asarray([r["memory_abs_error"] for r in rows]).mean(0).tolist() if rows else None,
            "records": rows, "population": "Separate identical-current-frame near-interception histories"}


def _write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    with temporary.open("w") as handle:
        handle.write(json.dumps(data, indent=2, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _identity_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class EvaluationLedger:
    """Immutable successful cases/stages; mutable progress and append-only errors."""

    def __init__(self, output, identity, expected_cases):
        self.output = Path(output)
        self.fingerprint = _identity_hash(identity)
        self.expected_cases = expected_cases
        self.stage, self.case_id = "initializing", None
        path = self.output / "evaluation_manifest.json"
        if path.exists():
            existing = json.loads(path.read_text())
            if existing.get("fingerprint") != self.fingerprint or existing.get("identity") != identity:
                raise ValueError("Partial evaluation identity/fingerprint differs; use a distinct output directory")
        else:
            if any(self.output.glob("*.partial.json")) or any((self.output/"cases").glob("*.json")):
                raise ValueError("Partial evaluation lacks a guarded identity manifest; use a distinct output directory")
            _write_json(path, {"schema_version": "paddle-evaluation-resume-v1", "fingerprint": self.fingerprint,
                               "identity": identity, "expected_cases": expected_cases})
        self.completed_cases = len(list((self.output/"cases").glob("*.json")))

    def progress(self, stage, case_id=None, status="running"):
        self.stage, self.case_id = stage, case_id
        _write_json(self.output/"progress.json", {"evaluation_fingerprint": self.fingerprint,
                    "status": status, "stage": stage, "case_id": case_id,
                    "completed_cases": self.completed_cases, "expected_cases": self.expected_cases})

    def cached(self, relative, descriptor, compute):
        path = self.output/relative
        descriptor_fingerprint = _identity_hash(descriptor)
        if path.exists():
            saved = json.loads(path.read_text())
            if (saved.get("evaluation_fingerprint") != self.fingerprint
                    or saved.get("descriptor_fingerprint") != descriptor_fingerprint
                    or saved.get("payload_fingerprint") != _identity_hash(saved.get("payload"))):
                raise ValueError(f"Immutable evaluation record fingerprint differs: {path}")
            return saved["payload"]
        value = compute()
        _write_json(path, {"evaluation_fingerprint": self.fingerprint,
                          "descriptor_fingerprint": descriptor_fingerprint,
                          "payload_fingerprint": _identity_hash(value), "payload": value})
        if Path(relative).parts[0] == "cases":
            self.completed_cases += 1
        return value

    def failure(self, error):
        directory = self.output/"errors"
        directory.mkdir(exist_ok=True)
        index = len(list(directory.glob("*.json")))
        while (directory/f"{index:06d}.json").exists():
            index += 1
        snapshot = self.output/"in_progress"/f"{self.case_id}.json"
        _write_json(directory/f"{index:06d}.json", {"evaluation_fingerprint": self.fingerprint,
                    "stage": self.stage, "case_id": self.case_id, "error": str(error),
                    "type": type(error).__name__, "traceback": traceback.format_exc(),
                    "case_snapshot": json.loads(snapshot.read_text()) if snapshot.exists() else None})
        self.progress(self.stage, self.case_id, status="failed")


def _csv(path, records):
    if not records:
        return
    keys = sorted({key for row in records for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in records:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value for key, value in row.items()})


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.inference_mode()
def render_case_artifacts(system, case, output, device):
    """Compare actual futures with predictions under the same executed actions."""
    from PIL import Image, ImageDraw
    from .env import PaddleEnv

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    env = PaddleEnv(state=np.asarray(case["initial_state"]))
    frames, physical = [env.render()], [np.asarray(env.state).copy()]
    for action in case["actions"]:
        frame, _, _, _ = env.step(action)
        frames.append(frame)
        physical.append(np.asarray(env.state).copy())
    episode = {"frames": np.asarray(frames), "states": np.asarray(physical), "actions": np.asarray(case["actions"])}
    observation, memories = _encode_episode(system, episode, device)
    if len(frames) <= 3:
        return {"case_id": case["case_id"], "status": "No post-warm-up transition to visualize"}

    def panel(source, horizon, predicted):
        target = source + horizon
        target_s = observation[target:target + 1]
        actual_h = (system["H"](target_s)[0] * 64).cpu().numpy()
        predicted_h = (system["H"](predicted.observation)[0] * 64).cpu().numpy()
        actual_r = (system["R"](memories[target:target + 1])[0, 2:4] * 6).cpu().numpy()
        imagined_r = (system["R"](predicted.memory)[0, 2:4] * 6).cpu().numpy()
        recon = (system["D"](target_s)[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        imagined = (system["D"](predicted.observation)[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        image = Image.new("RGB", (768, 354), "#151b25")
        draw = ImageDraw.Draw(image)
        for col, (pixels, title) in enumerate(zip((frames[target], recon, imagined), ("Actual raw frame", "D(E(actual)): reconstruction", "D(predicted S): imagined"))):
            image.paste(Image.fromarray(pixels).resize((256, 256), Image.Resampling.NEAREST), (col * 256, 22))
            draw.text((col * 256 + 6, 5), title, fill="white")
        truth = physical[target]
        errors = np.abs(predicted_h - truth[[0, 1, 4]])
        draw.text((8, 283), f"t={source}->{target}, horizon={horizon}; executed actions {case['actions'][source:target]}; H predicted error xy/paddle: {np.round(errors, 2)}", fill="white")
        draw.text((8, 300), f"H actual xy/paddle: {np.round(actual_h, 2)}; H imagined: {np.round(predicted_h, 2)}", fill="white")
        draw.text((8, 317), f"R velocity after REAL observations: {np.round(actual_r, 2)}; after IMAGINED updates: {np.round(imagined_r, 2)}", fill="white")
        draw.text((8, 334), f"True velocity: {np.round(truth[2:4], 2)}; case {case['case_id']}; first interception={case['first_hit_before_miss']}", fill="white")
        return image

    contact = case.get("first_hit_step") or len(frames) - 1
    source = max(2, contact - 5)
    horizon = min(5, len(frames) - 1 - source)
    initial = PlanningState(observation[source:source + 1], memories[source:source + 1])
    predictions = matched_rollout(system, initial, torch.as_tensor(episode["actions"][source:source + horizon][None], device=device))["prediction"]
    panels = [panel(source, h, state) for h, state in enumerate(predictions, 1)]
    grid = Image.new("RGB", (768, 354 * len(panels)))
    for i, image in enumerate(panels):
        grid.paste(image, (0, i * 354))
    name = str(case["case_id"]).replace("/", "_")
    png, gif = output / f"{name}.png", output / f"{name}.gif"
    grid.save(png)
    animation = []
    for t in range(2, min(len(frames) - 1, 22)):
        initial = PlanningState(observation[t:t + 1], memories[t:t + 1])
        state = matched_rollout(system, initial, torch.as_tensor(episode["actions"][t:t + 1][None], device=device))["prediction"][0]
        animation.append(panel(t, 1, state))
    animation[0].save(gif, save_all=True, append_images=animation[1:], duration=180, loop=0)
    return {"case_id": case["case_id"], "first_hit_before_miss": case["first_hit_before_miss"],
            "png": str(png), "gif": str(gif), "source_index": source, "horizon": horizon,
            "alignment": "Actual future and imagined rollout use identical actually executed actions; R real and imagined labeled separately"}


def _plot_metrics(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for name, values in report["prediction"]["summary"].items():
        if name == "actual":
            continue
        horizons = sorted(map(int, values))
        axes[0].plot(horizons, [values[str(h)]["all"]["latent_error"] for h in horizons], marker="o", label=name)
        axes[1].plot(horizons, [np.mean(values[str(h)]["all"]["h_mae"]) for h in horizons], marker="o", label=name)
    axes[0].set(title="Matched-window latent error", xlabel="Prediction horizon", ylabel="Variance-normalized MSE")
    axes[1].set(title="H predicted position error", xlabel="Prediction horizon", ylabel="Mean coordinate MAE (world units)")
    axes[1].axhline(2, color="gray", linestyle="--", label="Engineering target")
    for ax in axes[:2]:
        ax.legend()
        ax.grid(alpha=.2)
    names = list(report["ordinary"]["summary"])
    rates = [report["ordinary"]["summary"][name]["success_rate"] for name in names]
    axes[2].bar(names, rates)
    axes[2].set(title="Ordinary first interception", ylabel="Success fraction", ylim=(0, 1))
    axes[2].tick_params(axis="x", rotation=30)
    axes[2].axhline(.9, color="gray", linestyle="--")
    for i, name in enumerate(names):
        r = report["ordinary"]["summary"][name]
        axes[2].text(i, min(.98, rates[i] + .025), f"{r['successes']}/{r['count']}", ha="center")
    fig.tight_layout()
    fig.savefig(output / "metrics.png", dpi=140)
    plt.close(fig)


def _targets(report):
    actual = report["prediction"]["summary"]["actual"]["0"]["all"]
    predicted = report["prediction"]["summary"]["prediction"]["5"]["all"]
    ordinary = report["ordinary"]["summary"].get("learned")
    paired = report["paired"]["summary"].get("learned")
    return {
        "h_actual_each_coordinate_lt_1": bool(max((report.get("actual_frame_readout") or actual)["h_mae"]) < 1),
        "r_real_each_velocity_lt_0_5": bool(max(actual["r_mae"][2:4]) < .5),
        "h_horizon5_each_coordinate_lt_2": bool(max(predicted["h_mae"]) < 2),
        "ordinary_first_interception_gte_0_9": bool(ordinary and ordinary["success_rate"] >= .9),
        "paired_first_interception_gte_0_9": bool(paired and paired["success_rate"] >= .9),
        "eligible_as_full_evidence": not report["smoke"] and report["ordinary"]["starts"] >= 500 and report["paired"]["pairs"] >= 100,
    }


def evaluate(config, data, perception, memory, predictor, output):
    from .checkpoints import load_system, read_checkpoint
    from .data import EpisodeDataset, history_pairs
    from .planner import ExhaustivePlanner

    _pin_numeric_mode()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "metrics.json"
    if metrics_path.exists() and json.loads(metrics_path.read_text()).get("status") == "completed":
        raise FileExistsError(f"Completed evaluation exists at {output}; use a new output directory")
    settings = config.get("evaluation", {})
    device = _device(settings.get("device", config.get("device", "auto")))
    smoke = bool(config.get("smoke", False) or config.get("mode") == "smoke")
    test, train = EpisodeDataset(data, "test"), EpisodeDataset(data, "train")
    checkpoint_sources = {}
    for name, path in (("perception", perception), ("memory", memory), ("predictor", predictor)):
        read_checkpoint(path, dataset_fingerprint=test.fingerprint, stage=name)
        checkpoint_sources[name] = {"path": str(Path(path).resolve()), "sha256": _sha256(path)}
    if train.fingerprint != test.fingerprint:
        raise ValueError("Training-probe and held-out evaluation dataset fingerprints differ")
    starts = min(int(settings.get("ordinary_starts", 2 if smoke else 500)), len(test))
    pair_count = int(settings.get("paired_pairs", 2 if smoke else 100))
    max_steps = int(settings.get("control_max_steps", 200))
    controllers = settings.get("controllers", list(CONTROLLERS))
    if starts < 1 or pair_count < 1 or max_steps < 3:
        raise ValueError("Evaluation requires at least one start/pair and one post-warm-up decision")
    if not controllers or len(set(controllers)) != len(controllers) or any(c not in CONTROLLERS for c in controllers):
        raise ValueError("Evaluation controllers must be unique known controller names")
    batch_size = int(settings.get("candidate_batch_size", 243))
    hardware = {"device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "precision": "float32", "python": platform.python_version(), "torch": str(torch.__version__),
                "tf32": False, "cpu_threads": 4, "candidate_batch_size": batch_size, "candidates": 243,
                "privileged_planner_implementation": PRIVILEGED_PLANNER_IMPLEMENTATION,
                "latency_scope": "Synchronized decision only after real observer assimilation; includes H/scoring and reset assimilation for reset policy; excludes real-frame rendering, real E/U, diagnostics, decoding and progress writes; privileged candidates use state-only advance; first decision retained"}
    identity = {"protocol": "paddle-evaluation-resume-v1", "config": config,
                "dataset_fingerprint": test.fingerprint, "hardware": hardware,
                "checkpoint_sha256": {name: value["sha256"] for name, value in checkpoint_sources.items()}}
    ledger = EvaluationLedger(output, identity, (starts + 2*pair_count)*len(controllers))
    ordinary, paired, report = [], [], None
    try:
        ledger.progress("loading_models")
        system = load_system(perception, memory, predictor, device)
        for value in system.values():
            if isinstance(value, torch.nn.Module):
                value.eval().requires_grad_(False)
        model_planner = ExhaustivePlanner(system["P"], system["U"], system["H"], candidate_batch_size=batch_size)
        ledger.progress("prediction_diagnostics")
        prediction = ledger.cached("diagnostics/prediction.json", {"stage": "prediction"},
            lambda: prediction_diagnostics(system, test, device, int(settings.get("prediction_windows", 8 if smoke else 512))))
        ledger.progress("reconstruction_diagnostics")
        reconstruction = ledger.cached("diagnostics/reconstruction.json", {"stage": "reconstruction"},
            lambda: reconstruction_diagnostics(system, test, device))
        pairs = history_pairs(pair_count, seed=8000)
        ledger.progress("velocity_probe")
        def probe_diagnostics():
            probe, fit = train_velocity_probe(system, train, device, int(settings.get("probe_frames", 32 if smoke else 1024)))
            return {"fit": fit, **paired_probe_diagnostics(system, probe, pairs, device)}
        paired_probe = ledger.cached("diagnostics/velocity_probe.json", {"stage": "velocity_probe"}, probe_diagnostics)

        def case(controller, initial_state, seed, metadata):
            case_id = metadata["case_id"]
            ledger.progress(metadata["population"], case_id)
            descriptor = {**metadata, "controller": controller, "initial_state": np.asarray(initial_state).tolist(),
                          "seed": seed, "max_steps": max_steps}
            def compute():
                def snapshot(record):
                    _write_json(output/"in_progress"/f"{case_id}.json", {"evaluation_fingerprint": ledger.fingerprint,
                                "descriptor": descriptor, "record": record})
                snapshot({"initial_state": descriptor["initial_state"], "states": [descriptor["initial_state"]],
                          "actions": [], "status": "initializing_observer"})
                record = run_controller(system, controller, initial_state, device, max_steps, seed, model_planner,
                                        progress_callback=snapshot)
                record.update(metadata)
                return record
            result = ledger.cached(f"cases/{case_id}.json", descriptor, compute)
            ledger.progress(metadata["population"], case_id)
            return result

        for case_id in range(starts):
            initial_state = test[case_id]["states"][0]
            for controller in controllers:
                ordinary.append(case(controller, initial_state, 8000+case_id,
                    {"case_id": f"ordinary_{case_id}_{controller}", "population": "ordinary", "initial_state_id": case_id}))
            _write_json(output/"ordinary.partial.json", ordinary)
        for pair_id, pair in enumerate(pairs):
            for member in pair["members"]:
                for controller in controllers:
                    paired.append(case(controller, member["initial_state"],
                        18000+2*pair_id+int(member["direction"] > 0),
                        {"case_id": f"paired_{pair_id}_{member['direction']}_{controller}", "population": "paired",
                         "pair_id": pair_id, "center": float(pair["center"]), "direction": int(member["direction"]),
                         "correct_action": int(member["correct_action"])}))
            _write_json(output/"paired.partial.json", paired)
        report = {
            "schema_version": "paddle-evaluation-v1", "status": "report_pending", "raw_status": "completed", "smoke": smoke,
            "config": config, "dataset": str(Path(data).resolve()), "dataset_fingerprint": test.fingerprint,
            "evaluation_fingerprint": ledger.fingerprint, "checkpoints": checkpoint_sources, "hardware": hardware,
            "definitions": {"position_coordinates": POSITION_NAMES, "state_coordinates": STATE_NAMES,
                            "position_units": "world units / pixels", "velocity_units": "world units per decision interval",
                            "reset": "Fresh U(current S, zero memory, zero previous-action marker) at every decision/source",
                            "copy": "Initial S at each horizon; U still advanced using matched actions",
                            "paired_uncertainty": "Wilson intervals describe member outcomes; opposite-direction members share a center and are not independent pairs",
                            "control_max_steps_including_warmup": max_steps,
                            "resume": "Completed immutable cases and diagnostic stages are reused only for identical config, checkpoint, dataset and numerical-mode fingerprints; unfinished cases restart from the same initial state and seed"},
            "prediction": {k: v for k, v in prediction.items() if k != "records"},
            "reconstruction": {k: v for k, v in reconstruction.items() if k not in ("records", "actual_frame_readout", "readout_records")},
            "actual_frame_readout": reconstruction.get("actual_frame_readout"),
            "ordinary": {"starts": starts, "summary": summarize_control(ordinary)},
            "paired": {"pairs": pair_count, "seed": 8000, "summary": summarize_control(paired)},
            "velocity_probe": {k: v for k, v in paired_probe.items() if k != "records"},
            "visuals": [],
        }
        report["targets"] = _targets(report)
        # Raw summaries and records must survive every optional CSV/plot/render failure.
        _write_json(metrics_path, report)
        ledger.progress("reporting")
        for name, records in (("prediction", prediction["records"]), ("control", ordinary+paired),
                              ("probe", paired_probe["records"]), ("reconstruction", reconstruction.get("records", [])),
                              ("actual_frame_readout", reconstruction.get("readout_records", []))):
            _write_json(output/f"{name}_records.json", records)
            _csv(output/f"{name}_records.csv", records)
        learned = [r for r in ordinary+paired if r["controller"] == "learned"]
        for success in (True, False):
            selected = next((r for r in learned if r["first_hit_before_miss"] == success and len(r["actions"]) > 2), None)
            if selected is not None:
                report["visuals"].append(render_case_artifacts(system, selected, output/"visuals", device))
            else:
                report["visuals"].append({"first_hit_before_miss": success, "status": "No matching learned-controller case observed"})
        _plot_metrics(report, output)
        pictures = ''.join(f'<h3>{html.escape(v["case_id"])}</h3><img src="visuals/{Path(v["png"]).name}"><img src="visuals/{Path(v["gif"]).name}">' for v in report["visuals"] if "png" in v)
        (output/"index.html").write_text('<!doctype html><meta charset="utf-8"><title>Paddle world model evaluation</title>'
            '<style>body{max-width:1100px;margin:auto;font:16px system-ui}img{max-width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>'
            '<h1>Paddle world model evaluation</h1>'
            + ('<p><strong>SMOKE: execution evidence only; no trained-performance claim.</strong></p>' if smoke else '')
            + '<p>Exact values and source identities: <a href="metrics.json">metrics.json</a>. Raw controls: <a href="control_records.csv">CSV</a>; predictions: <a href="prediction_records.csv">CSV</a>.</p>'
            + '<img src="metrics.png">' + pictures + '<h2>Measured engineering targets</h2><pre>'
            + html.escape(json.dumps(report["targets"], indent=2)) + '</pre><h2>Reconstruction diagnostics</h2><pre>'
            + html.escape(json.dumps(report["reconstruction"], indent=2)) + '</pre>')
        report["status"] = "completed"
        report["prior_error_count"] = len(list((output/"errors").glob("*.json")))
        _write_json(metrics_path, report)
        ledger.progress("completed", status="completed")
        return report
    except BaseException as error:
        # Completed case files remain untouched; preserve the last partial trajectory.
        if report is not None:
            report["status"] = "report_failed"
            report["report_error"] = {"type": type(error).__name__, "message": str(error)}
            _write_json(metrics_path, report)
        _write_json(output/"ordinary.partial.json", ordinary)
        _write_json(output/"paired.partial.json", paired)
        ledger.failure(error)
        raise


def demo(perception, memory, predictor, output, device="auto"):
    from .checkpoints import load_system
    from .env import PaddleEnv

    _pin_numeric_mode()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    device = _device(device)
    system = load_system(perception, memory, predictor, device)
    for value in system.values():
        if isinstance(value, torch.nn.Module):
            value.eval().requires_grad_(False)
    case = run_controller(system, "learned", PaddleEnv(seed=99991).state, device, max_steps=32)
    case["case_id"] = "demo_99991"
    _write_json(output / "control_records.json", [case])
    visual = render_case_artifacts(system, case, output, device)
    result = {"schema_version": "paddle-demo-v1", "status": "completed", "seed": 99991,
              "case": case, "visual": visual, "note": "32-interval bounded qualitative demo; not the 500-start evaluation"}
    _write_json(output / "metrics.json", result)
    return result
