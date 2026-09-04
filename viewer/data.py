"""Build a self-contained microscope over the raw data consumed by E1.

What: render bounded, deterministic samples from stored training episodes, optional paired training
interventions, and both regenerated evaluation probe populations into one offline HTML file.
How: validate the stored tensors against the selected spec, encode evenly spaced RGB samples as PNG
data URLs, and package the pixels, actions, seeds, shapes, and source lineage into a static browser UI.
Why: viewer v0 should let researchers inspect what the model actually sees without labels entering W,
mutating the dataset, or mistaking a displayed subset for experiment evidence (Invariant 11; E1/H1).
The unfrozen ``data/e0_dev`` store stands in for the eventual frozen E0 artifact during phase 1.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import struct
import zlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import yaml

from evaluation.probe_set import ProbeSet, generate_probe_set
from training.data import EpisodeStore, PairedInterventionStore, generate_paired_interventions
from world_state.abi import load_abi

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = ROOT / "configs" / "dev" / "first_slice.yaml"
DEFAULT_OUTPUT = ROOT / "runs" / "data_viewer.html"
TEMPLATE = Path(__file__).with_name("data_viewer_template.html")
MAX_EXAMPLES = 24


class DataViewerError(RuntimeError):
    """The selected inputs cannot be represented truthfully by the data viewer."""


def _read_spec(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = yaml.safe_load(raw)
    except (OSError, yaml.YAMLError) as error:
        raise DataViewerError(f"cannot read spec {path}: {error}") from error
    if not isinstance(value, dict):
        raise DataViewerError(f"{path} must contain a YAML mapping")
    return value, raw


def _read_tensor_payload(path: Path) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError, EOFError) as error:
        raise DataViewerError(f"cannot read tensor store {path}: {error}") from error
    if not isinstance(value, dict):
        raise DataViewerError(f"{path} must contain a mapping")
    return value


def _required(payload: dict[str, Any], path: Path, *keys: str) -> tuple[Any, ...]:
    missing = [key for key in keys if key not in payload]
    if missing:
        raise DataViewerError(f"{path} is missing {', '.join(missing)}")
    return tuple(payload[key] for key in keys)


def _display_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _dataset_dir(cfg: dict[str, Any], root: Path) -> Path:
    try:
        configured = Path(cfg["env"]["dataset"])
    except (KeyError, TypeError) as error:
        raise DataViewerError("spec must define env.dataset") from error
    return configured.resolve() if configured.is_absolute() else (root / configured).resolve()


def _expected_observation_shape(cfg: dict[str, Any]) -> tuple[int, int, int]:
    resolution = int(cfg["env"]["resolution"])
    return 3, resolution, resolution


def _load_episode_store(path: Path, cfg: dict[str, Any], root: Path) -> tuple[EpisodeStore, int | None]:
    if not path.is_file():
        raise DataViewerError(
            f"training episode store does not exist: {path}. Run the selected spec once to collect it."
        )
    payload = _read_tensor_payload(path)
    observations, actions = _required(payload, path, "observations", "actions")
    store = EpisodeStore(observations, actions)
    expected_transitions = int(cfg["env"]["n_transitions"])
    expected_length = int(cfg["env"]["episode_len"])
    expected_shape = _expected_observation_shape(cfg)
    action_dims = load_abi(root / cfg["abi"]).action_dims
    if store.transitions != expected_transitions:
        raise DataViewerError(
            f"{path} has {store.transitions} transitions, selected spec requires {expected_transitions}"
        )
    if store.episode_length != expected_length:
        raise DataViewerError(
            f"{path} episode length is {store.episode_length}, selected spec requires {expected_length}"
        )
    if tuple(store.observations.shape[2:]) != expected_shape:
        raise DataViewerError(
            f"{path} observation shape is {tuple(store.observations.shape[2:])}, expected {expected_shape}"
        )
    if store.actions.shape[-1] != action_dims:
        raise DataViewerError(
            f"{path} action width is {store.actions.shape[-1]}, ABI requires {action_dims}"
        )
    seed = payload.get("seed")
    if seed is not None and not isinstance(seed, int):
        raise DataViewerError(f"{path} seed must be an integer when present")
    return store, seed


def _load_paired_store(path: Path, cfg: dict[str, Any], root: Path) -> PairedInterventionStore:
    if not path.is_file():
        raise DataViewerError(
            f"paired training store is active but does not exist: {path}. Run the selected spec once to collect it."
        )
    payload = _read_tensor_payload(path)
    initial, following, actions = _required(
        payload,
        path,
        "initial_observations",
        "next_observations",
        "actions",
    )
    store = PairedInterventionStore(initial, following, actions, payload.get("seed"))
    expected_groups = int(cfg["counterfactual_data"]["groups"])
    expected_seed = int(cfg["counterfactual_data"]["seed"])
    expected_branches = int(cfg["losses"]["counterfactual"]["k"])
    expected_shape = _expected_observation_shape(cfg)
    action_dims = load_abi(root / cfg["abi"]).action_dims
    if store.groups != expected_groups:
        raise DataViewerError(f"{path} has {store.groups} groups, selected spec requires {expected_groups}")
    if store.branches != expected_branches:
        raise DataViewerError(
            f"{path} has {store.branches} branches, selected spec requires {expected_branches}"
        )
    if store.seed is None or store.seed != expected_seed:
        raise DataViewerError(f"{path} seed is {store.seed}, selected spec requires {expected_seed}")
    if tuple(store.initial_observations.shape[1:]) != expected_shape:
        raise DataViewerError(
            f"{path} observation shape is {tuple(store.initial_observations.shape[1:])}, expected {expected_shape}"
        )
    if store.actions.shape[-1] != action_dims:
        raise DataViewerError(
            f"{path} action width is {store.actions.shape[-1]}, ABI requires {action_dims}"
        )
    return store


def _sample_indices(total: int, examples: int) -> list[int]:
    if total < 1:
        raise DataViewerError("a visualized population must contain at least one example")
    count = min(total, examples)
    if count == 1:
        return [0]
    return [round(position * (total - 1) / (count - 1)) for position in range(count)]


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = zlib.crc32(kind + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)


def _png_data_url(frame: torch.Tensor) -> str:
    if frame.ndim != 3 or frame.shape[0] != 3 or frame.dtype != torch.uint8:
        raise DataViewerError(f"viewer frames must be uint8 CHW RGB, got {tuple(frame.shape)} {frame.dtype}")
    image = frame.detach().cpu().permute(1, 2, 0).contiguous()
    height, width = int(image.shape[0]), int(image.shape[1])
    rgb = image.numpy().tobytes()
    stride = width * 3
    scanlines = b"".join(b"\x00" + rgb[offset: offset + stride] for offset in range(0, len(rgb), stride))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = (
            b"\x89PNG\r\n\x1a\n"
            + _png_chunk(b"IHDR", header)
            + _png_chunk(b"IDAT", zlib.compress(scanlines, level=9))
            + _png_chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def _actions(values: torch.Tensor) -> list[list[float]]:
    return [[round(float(component), 6) for component in action] for action in values]


def _trajectory_section(
        *,
        section_id: str,
        label: str,
        role: str,
        source: str,
        note: str,
        observations: torch.Tensor,
        actions: torch.Tensor,
        examples: int,
        facts: list[dict[str, str]],
) -> dict[str, Any]:
    indices = _sample_indices(int(observations.shape[0]), examples)
    samples = [
        {
            "index": index,
            "frames": [_png_data_url(frame) for frame in observations[index]],
            "actions": _actions(actions[index]),
        }
        for index in indices
    ]
    return {
        "id": section_id,
        "kind": "trajectory",
        "label": label,
        "role": role,
        "status": "available",
        "source": source,
        "note": note,
        "population": int(observations.shape[0]),
        "displayed": len(samples),
        "selection": "Evenly spaced indices; no viewer randomness",
        "facts": facts,
        "samples": samples,
    }


def _paired_section(
        *,
        section_id: str,
        label: str,
        role: str,
        source: str,
        note: str,
        store: PairedInterventionStore,
        examples: int,
        facts: list[dict[str, str]],
) -> dict[str, Any]:
    indices = _sample_indices(store.groups, examples)
    samples = []
    for index in indices:
        samples.append(
            {
                "index": index,
                "initial": _png_data_url(store.initial_observations[index]),
                "branches": [
                    {
                        "index": branch,
                        "action": _actions(store.actions[index, branch: branch + 1])[0],
                        "outcome": _png_data_url(store.next_observations[index, branch]),
                    }
                    for branch in range(store.branches)
                ],
            }
        )
    return {
        "id": section_id,
        "kind": "paired",
        "label": label,
        "role": role,
        "status": "available",
        "source": source,
        "note": note,
        "population": store.groups,
        "displayed": len(samples),
        "selection": "Evenly spaced group indices; no viewer randomness",
        "facts": facts,
        "samples": samples,
    }


def _disabled_paired_section(path: Path, root: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    cf = cfg["losses"]["counterfactual"]
    return {
        "id": "paired_training",
        "kind": "paired",
        "label": "Paired training interventions",
        "role": "TRAIN AUX",
        "status": "disabled",
        "source": _display_path(path, root),
        "note": (
            "This selected spec does not consume paired training data because both counterfactual "
            "loss weights are zero. A store may still exist on disk for another dev spec."
        ),
        "population": 0,
        "displayed": 0,
        "selection": "Not sampled",
        "facts": [
            {"label": "InfoNCE weight", "value": str(cf.get("weight", 0.0))},
            {"label": "Positive MSE weight", "value": str(cf.get("positive_weight", 0.0))},
            {"label": "Store on disk", "value": "yes" if path.is_file() else "no"},
        ],
        "samples": [],
    }


def build_data_snapshot(
        spec_path: Path,
        *,
        root: Path = ROOT,
        examples: int = 6,
) -> dict[str, Any]:
    """Validate and package a bounded view of every raw population used by the selected E1 spec."""

    if not 1 <= examples <= MAX_EXAMPLES:
        raise DataViewerError(f"examples must be in [1, {MAX_EXAMPLES}], got {examples}")
    root = root.resolve()
    spec_path = spec_path.resolve()
    cfg, raw_spec = _read_spec(spec_path)
    dataset_dir = _dataset_dir(cfg, root)
    episodes_path = dataset_dir / "episodes.pt"
    paired_path = dataset_dir / "counterfactual.pt"

    episode_store, episode_seed = _load_episode_store(episodes_path, cfg, root)
    episode_facts = [
        {
            "label": "Full population",
            "value": f"{episode_store.observations.shape[0]:,} episodes · {episode_store.transitions:,} transitions",
        },
        {"label": "Episode length",
         "value": f"{episode_store.episode_length} actions / {episode_store.episode_length + 1} frames"},
        {"label": "Observation tensor", "value": f"{tuple(episode_store.observations.shape)} · uint8 RGB"},
        {"label": "Action tensor", "value": f"{tuple(episode_store.actions.shape)} · float32"},
        {"label": "Collector seed", "value": "not recorded" if episode_seed is None else str(episode_seed)},
        {"label": "Exploration", "value": str(cfg["env"].get("exploration", "unspecified"))},
    ]
    sections = [
        _trajectory_section(
            section_id="episode_training",
            label="Training episodes",
            role="TRAIN",
            source=_display_path(episodes_path, root),
            note=(
                "Training samples random windows from this entire stored population. The viewer shows full, "
                "evenly spaced episodes only for inspection; it does not change the sampler."
            ),
            observations=episode_store.observations,
            actions=episode_store.actions,
            examples=examples,
            facts=episode_facts,
        )
    ]
    del episode_store

    cf_cfg = cfg["losses"]["counterfactual"]
    paired_active = float(cf_cfg.get("weight", 0.0)) != 0.0 or float(cf_cfg.get("positive_weight", 0.0)) != 0.0
    if paired_active:
        paired_store = _load_paired_store(paired_path, cfg, root)
        sections.append(
            _paired_section(
                section_id="paired_training",
                label="Paired training interventions",
                role="TRAIN AUX",
                source=_display_path(paired_path, root),
                note=(
                    "Every outcome branches from the exact same saved E0 state. The only intervention is the "
                    "displayed one-step action; these groups anchor action semantics during training."
                ),
                store=paired_store,
                examples=examples,
                facts=[
                    {"label": "Full population",
                     "value": f"{paired_store.groups:,} groups · {paired_store.branches} branches each"},
                    {"label": "Initial tensor",
                     "value": f"{tuple(paired_store.initial_observations.shape)} · uint8 RGB"},
                    {"label": "Outcome tensor", "value": f"{tuple(paired_store.next_observations.shape)} · uint8 RGB"},
                    {"label": "Action tensor", "value": f"{tuple(paired_store.actions.shape)} · float32"},
                    {
                        "label": "Collector seed",
                        "value": "not recorded" if paired_store.seed is None else str(paired_store.seed),
                    },
                    {"label": "Loss weights",
                     "value": f"InfoNCE {cf_cfg.get('weight', 0.0)} · positive MSE {cf_cfg.get('positive_weight', 0.0)}"},
                ],
            )
        )
        del paired_store
    else:
        sections.append(_disabled_paired_section(paired_path, root, cfg))

    probe_cfg = cfg["probe_set"]
    transition_probe: ProbeSet = generate_probe_set(cfg)
    sections.append(
        _trajectory_section(
            section_id="transition_evaluation",
            label="Fixed transition probes",
            role="EVAL ONLY",
            source=str(probe_cfg.get("path", "evaluation/probe_set_v1")),
            note=(
                "Evaluation regenerates and scores this full population from the configured seed. These frames "
                "are never read by the training sampler."
            ),
            observations=transition_probe.observations,
            actions=transition_probe.actions,
            examples=examples,
            facts=[
                {"label": "Full population", "value": f"{transition_probe.observations.shape[0]:,} trajectories"},
                {"label": "Horizon",
                 "value": f"{transition_probe.actions.shape[1]} actions / {transition_probe.observations.shape[1]} frames"},
                {"label": "Observation tensor", "value": f"{tuple(transition_probe.observations.shape)} · uint8 RGB"},
                {"label": "Action tensor", "value": f"{tuple(transition_probe.actions.shape)} · float32"},
                {"label": "Regeneration seed", "value": str(probe_cfg["seed"])},
                {"label": "Used for", "value": "action sensitivity + transition errors and controls"},
            ],
        )
    )
    del transition_probe

    data_cfg = cfg.get("counterfactual_data", {})
    paired_probe_groups = int(data_cfg.get("probe_groups", probe_cfg["count"]))
    paired_probe_seed = int(data_cfg.get("probe_seed", int(probe_cfg["seed"]) + 8_000_003))
    paired_probe = generate_paired_interventions(cfg, paired_probe_groups, paired_probe_seed)
    sections.append(
        _paired_section(
            section_id="paired_evaluation",
            label="Held-out paired probes",
            role="EVAL ONLY",
            source="training.data.generate_paired_interventions (regenerated in memory)",
            note=(
                "Evaluation regenerates these held-out same-state branches from a seed disjoint from the paired "
                "training store, then scores counterfactual discrimination across every branch."
            ),
            store=paired_probe,
            examples=examples,
            facts=[
                {"label": "Full population",
                 "value": f"{paired_probe.groups:,} groups · {paired_probe.branches} branches each"},
                {"label": "Initial tensor", "value": f"{tuple(paired_probe.initial_observations.shape)} · uint8 RGB"},
                {"label": "Outcome tensor", "value": f"{tuple(paired_probe.next_observations.shape)} · uint8 RGB"},
                {"label": "Action tensor", "value": f"{tuple(paired_probe.actions.shape)} · float32"},
                {"label": "Regeneration seed", "value": str(paired_probe_seed)},
                {"label": "Used for", "value": "held-out K-way counterfactual accuracy"},
            ],
        )
    )

    status = "DEVELOPMENT" if cfg.get("status") == "dev" else str(cfg.get("status", "unknown")).upper()
    return {
        "version": 1,
        "generatedAt": datetime.now(UTC).isoformat(),
        "title": "PATH-WM Data Viewer",
        "spec": {
            "path": _display_path(spec_path, root),
            "sha256": hashlib.sha256(raw_spec).hexdigest(),
            "experiment": str(cfg.get("experiment", "unknown")),
            "status": status,
        },
        "environment": {
            "name": str(cfg["env"].get("name", "unknown")),
            "variant": str(cfg["env"].get("variant", "unknown")),
            "resolution": int(cfg["env"]["resolution"]),
            "dataset": _display_path(dataset_dir, root),
        },
        "sections": sections,
        "disclosure": (
            f"This file embeds at most {examples} evenly spaced examples per population. Training and evaluation "
            "still consume the full validated populations shown in each section."
        ),
    }


def render_data_viewer(snapshot: dict[str, Any], template_path: Path = TEMPLATE) -> str:
    try:
        template = template_path.read_text(encoding="utf-8")
    except OSError as error:
        raise DataViewerError(f"cannot read viewer template {template_path}: {error}") from error
    marker = '{"__PATH_WM_DATA_SNAPSHOT__":null}'
    if template.count(marker) != 1:
        raise DataViewerError(f"{template_path} must contain exactly one {marker} marker")
    # Escaping '<' prevents a spec-supplied string from terminating the inert JSON script element.
    payload = json.dumps(snapshot, separators=(",", ":"), ensure_ascii=False).replace("<", "\\u003c")
    return template.replace(marker, payload)


def write_data_viewer(
        spec_path: Path = DEFAULT_SPEC,
        *,
        output_path: Path = DEFAULT_OUTPUT,
        root: Path = ROOT,
        examples: int = 6,
) -> tuple[Path, dict[str, Any]]:
    """Write one atomic, offline HTML viewer and return its path and packaged snapshot."""

    root = root.resolve()
    spec_path = spec_path if spec_path.is_absolute() else root / spec_path
    output_path = output_path if output_path.is_absolute() else root / output_path
    snapshot = build_data_snapshot(spec_path, root=root, examples=examples)
    rendered = render_data_viewer(snapshot)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(output_path)
    return output_path.resolve(), snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", nargs="?", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--examples", type=int, default=6,
                        help=f"evenly spaced examples per population (1-{MAX_EXAMPLES})")
    args = parser.parse_args()
    output, snapshot = write_data_viewer(args.spec, output_path=args.output, examples=args.examples)
    print(f"Data viewer: {output}")
    print(f"Spec: {snapshot['spec']['path']} ({snapshot['spec']['status']})")
    for section in snapshot["sections"]:
        print(f"- {section['label']}: {section['status']}; {section['displayed']} of {section['population']} shown")


if __name__ == "__main__":
    main()
