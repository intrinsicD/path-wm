"""Interrupted R0 must preserve optimizer, EMA and random streams (DDR §24)."""
from copy import deepcopy
import json
from pathlib import Path
import pytest
import torch
import yaml
from contracts import RepresentationBatch, TemporalObservation
from training.common_base import train_common_base
ROOT = Path(__file__).resolve().parents[2]


class TinyData:
    fingerprint = "fixed-test-manifest"
    def __init__(self, fail_after=None):
        self.calls = 0
        self.fail_after = fail_after
    def sample(self, split, stage, batch_size, generator):
        if split == "train":
            if self.fail_after == self.calls:
                raise RuntimeError("simulated interruption")
            self.calls += 1
        def observations():
            return {
                "video": TemporalObservation(
                    torch.randint(0, 256, (batch_size, 4, 3, 64, 64), dtype=torch.uint8, generator=generator),
                    torch.linspace(-0.375, 0, 4).expand(batch_size, -1),
                    torch.ones(batch_size, 4, dtype=torch.bool)),
                "audio": TemporalObservation(
                    torch.rand(batch_size, 1, 2048, generator=generator) * 2 - 1,
                    torch.linspace(-0.128, 0, 2048).expand(batch_size, -1),
                    torch.ones(batch_size, 2048, dtype=torch.bool)),
            }
        return RepresentationBatch(observations(), observations(), {})


def _config():
    cfg = yaml.safe_load((ROOT / "configs/dev/common_base_rank_balanced.yaml").read_text())
    cfg["train"].update(max_steps=4, batch_size=2, held_out_batches=1, checkpoint_every=2)
    return cfg


def test_resume_matches_uninterrupted_optimizer_ema_and_ledger(tmp_path):
    torch.set_num_threads(1)
    cfg = _config()
    complete_dir, resumed_dir = tmp_path / "complete", tmp_path / "resumed"
    complete = train_common_base(cfg, TinyData(), complete_dir, 0, torch.device("cpu"))
    with pytest.raises(RuntimeError, match="simulated interruption"):
        train_common_base(cfg, TinyData(fail_after=3), resumed_dir, 0, torch.device("cpu"))
    assert (resumed_dir / "checkpoint.pt").is_file(), "save before the final step"
    resumed = train_common_base(cfg, TinyData(), resumed_dir, 0, torch.device("cpu"), resume=True)
    left = torch.load(complete.checkpoint, weights_only=True)
    right = torch.load(resumed.checkpoint, weights_only=True)
    for name, value in left["learner"].items():
        torch.testing.assert_close(value, right["learner"][name], rtol=0, atol=0)
    assert complete.metrics == resumed.metrics
    assert complete.final_training == resumed.final_training
    rows = [json.loads(line) for line in (resumed_dir / "training.jsonl").read_text().splitlines()]
    assert [row["step"] for row in rows] == [1, 2, 3, 4]


def test_resume_rejects_changed_config_or_data_and_fresh_run_refuses_overwrite(tmp_path):
    torch.set_num_threads(1)
    cfg = _config()
    train_common_base(cfg, TinyData(), tmp_path, 0, torch.device("cpu"))
    changed = deepcopy(cfg)
    changed["train"]["lr"] *= 2
    with pytest.raises(ValueError, match="config"):
        train_common_base(changed, TinyData(), tmp_path, 0, torch.device("cpu"), resume=True)
    changed_data = TinyData()
    changed_data.fingerprint = "different-manifest"
    with pytest.raises(ValueError, match="manifest|fingerprint"):
        train_common_base(cfg, changed_data, tmp_path, 0, torch.device("cpu"), resume=True)
    with pytest.raises(FileExistsError):
        train_common_base(cfg, TinyData(), tmp_path, 0, torch.device("cpu"))
