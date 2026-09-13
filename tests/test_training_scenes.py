"""Scene diagnostics must expose fitting failures without changing test gates."""

from copy import deepcopy
import json
import numpy as np
import pytest
import torch

from pathwm.data.memory_output import MemoryOutputEpisodes


def population():
    d = MemoryOutputEpisodes(16, seed=93, curriculum="relocation").with_scenes(
        [{}, {"texture": 8}, {}]
    )
    b = d.batch(range(len(d)))
    logits = torch.cat(
        [torch.nn.functional.one_hot(b["labels"][:, i], n) for i, n in enumerate([4, 2, 2])], 1
    ).float()
    a = {"labels": b["labels"].numpy(), "target": b["target"].numpy()}
    for m in ["ordinary", "reset"]:
        a[m + "_logits"] = logits.numpy().copy()
        a[m + "_image"] = b["target"].numpy().copy()
    return d, a


def test_scene_scores_keep_duplicate_blocks_and_expose_shape_failure():
    from experiments.memory_output import training_scene_scores

    d, a = population()
    # Facts and ordinary images stay perfect; only the middle block's recalled
    # images depict the other shape at the correct color and side.
    for i in range(32, 64):
        y = a["labels"][i].copy()
        y[1] = 1 - y[1]
        j = np.flatnonzero((a["labels"] == y).all(1))[0]
        a["reset_image"][i] = a["target"][j]
    result = training_scene_scores(d, a)
    assert result["scope"] == "training diagnostics only; not a held-out gate"
    assert [b["block"] for b in result["blocks"]] == [0, 1, 2]
    assert [b["examples"] for b in result["blocks"]] == [32] * 3
    assert result["blocks"][0]["scene"] == result["blocks"][2]["scene"]
    for i, b in enumerate(result["blocks"]):
        assert b["metrics"]["ordinary"]["image_accuracy"] == 1
        s = b["metrics"]["reset"]
        assert s["factual_accuracy"] == s["image_color_accuracy"] == s["image_side_accuracy"] == 1
        assert s["image_shape_accuracy"] == s["image_accuracy"] == int(i != 1)


@pytest.mark.parametrize("corruption", ["length", "labels", "targets", "metadata", "block_order"])
def test_scene_scores_reject_misaligned_provenance(corruption):
    from experiments.memory_output import training_scene_scores

    d, a = population()
    if corruption == "length":
        a["reset_image"] = a["reset_image"][:-1]
    elif corruption == "labels":
        a["labels"] = a["labels"].copy()
        a["labels"][0, 0] ^= 1
    elif corruption == "targets":
        a["target"] = a["target"].copy()
        a["target"][0] = 0
    elif corruption == "metadata":
        d.identity["input_augmentation"]["source_data"]["pairs"] += 1
    else:
        d.labels = d.labels.copy()
        d.labels[[32, 33]] = d.labels[[33, 32]]
        a["labels"] = d.labels.copy()
    with pytest.raises(ValueError, match="[Aa]lign|[Bb]lock|[Pp]opulation"):
        training_scene_scores(d, a)


def test_scene_report_is_training_only_and_escapes_labels(tmp_path):
    from experiments.memory_output import training_scene_scores
    from pathwm.evaluation.report import render_report

    d, a = population()
    result = training_scene_scores(d, a)
    result = deepcopy(result)
    result["blocks"][0]["scene"]["note"] = "<script>unsafe</script>"
    for name, value in {
        "run.json": {"identity": {"settings": {"purpose": "development"}}},
        "status.json": {"result": "completed", "step": 4},
        "result.json": {"gate": False, "metrics": {}},
        "training_scene_fit.json": result,
    }.items():
        (tmp_path / name).write_text(json.dumps(value))
    (tmp_path / "metrics.jsonl").write_text("")
    html = render_report(tmp_path)
    assert "Training fit by scene" in html and "not a held-out gate" in html
    assert "Declared capability screen: not passed" in html
    assert "&lt;script&gt;" in html and "<script>unsafe" not in html
    (tmp_path / "training_scene_fit.json").unlink()
    assert "Training fit by scene" not in render_report(tmp_path)
    assert training_scene_scores(MemoryOutputEpisodes(16, curriculum="relocation"), {}) is None
