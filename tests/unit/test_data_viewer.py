"""The raw-data viewer keeps train/eval populations distinct and embeds only bounded samples."""
from __future__ import annotations

import copy

import yaml

from training.data import collect_episodes, collect_paired_interventions
from viewer.data import build_data_snapshot, render_data_viewer
from world_state.abi import ROOT


def _viewer_spec(cfg, tmp_path, *, paired_training: bool = True):
    spec = copy.deepcopy(cfg)
    spec["abi"] = str(ROOT / spec["abi"])
    spec["env"].update(
        n_worlds=2,
        episode_len=2,
        n_transitions=4,
        resolution=16,
        dataset=str(tmp_path / "dataset"),
    )
    spec["probe_set"].update(seed=11, count=3, horizon=2)
    spec["counterfactual_data"].update(groups=3, seed=17, probe_groups=3, probe_seed=19)
    spec["losses"]["counterfactual"].update(
        k=2,
        weight=1.0 if paired_training else 0.0,
        positive_weight=0.0,
    )
    return spec


def test_data_viewer_packages_exact_train_and_evaluation_populations(cfg, tmp_path):
    spec = _viewer_spec(cfg, tmp_path)
    dataset_dir = tmp_path / "dataset"
    collect_episodes(spec, dataset_dir, seed=7)
    collect_paired_interventions(spec, dataset_dir)
    spec_path = tmp_path / "viewer.yaml"
    spec_path.write_text(yaml.safe_dump(spec), encoding="utf-8")

    snapshot = build_data_snapshot(spec_path, root=ROOT, examples=2)
    sections = {section["id"]: section for section in snapshot["sections"]}

    assert list(sections) == [
        "episode_training",
        "paired_training",
        "transition_evaluation",
        "paired_evaluation",
    ]
    assert sections["episode_training"]["population"] == 2
    assert sections["episode_training"]["samples"][0]["index"] == 0
    assert sections["episode_training"]["samples"][1]["index"] == 1
    assert sections["paired_training"]["population"] == 3
    assert sections["transition_evaluation"]["population"] == 3
    assert sections["paired_evaluation"]["population"] == 3
    assert sections["transition_evaluation"]["role"] == "EVAL ONLY"
    assert sections["paired_evaluation"]["samples"][1]["index"] == 2
    assert snapshot["spec"]["status"] == "DEVELOPMENT"

    rendered = render_data_viewer(snapshot)
    # Two examples from each source: 3 frames per trajectory or 1 + 2 frames per branch group.
    assert rendered.count("data:image/png;base64,") == 24
    assert "See what the model sees." in rendered
    assert "__PATH_WM_DATA_SNAPSHOT__" not in rendered
    assert "no network requests" in rendered


def test_disabled_paired_training_file_is_not_presented_as_consumed(cfg, tmp_path):
    spec = _viewer_spec(cfg, tmp_path, paired_training=False)
    dataset_dir = tmp_path / "dataset"
    collect_episodes(spec, dataset_dir, seed=7)
    spec_path = tmp_path / "viewer.yaml"
    spec_path.write_text(yaml.safe_dump(spec), encoding="utf-8")

    snapshot = build_data_snapshot(spec_path, root=ROOT, examples=1)
    paired = next(section for section in snapshot["sections"] if section["id"] == "paired_training")

    assert paired["status"] == "disabled"
    assert paired["population"] == 0
    assert paired["samples"] == []
    assert "does not consume paired training data" in paired["note"]
