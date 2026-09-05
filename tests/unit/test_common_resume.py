"""Interrupted R0 must preserve optimizer, EMA and random streams (DDR §24)."""
from copy import deepcopy
import json
import hashlib
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
        return RepresentationBatch(observations(), observations(), observations() if stage == "representation_av" else {})


def _config():
    cfg = yaml.safe_load((ROOT / "configs/dev/common_base_rank_balanced.yaml").read_text())
    cfg["train"].update(max_steps=4, batch_size=2, held_out_batches=1, checkpoint_every=2)
    return cfg


@pytest.mark.parametrize("prefetch_batches", [0, 1])
def test_resume_matches_uninterrupted_optimizer_ema_and_ledger(tmp_path, prefetch_batches):
    torch.set_num_threads(1)
    cfg = _config()
    cfg["train"]["prefetch_batches"] = prefetch_batches
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


def test_prefetched_batch_does_not_advance_the_checkpoint_random_stream():
    import threading
    import training.common_base as training
    batches = getattr(training, "_training_batches", None)
    assert callable(batches), "no implementation: one-batch prefetch with consumed-only RNG state"
    ready = threading.Event()
    class ObservedData(TinyData):
        def sample(self, *args, **kwargs):
            result = super().sample(*args, **kwargs)
            if self.calls == 2:
                ready.set()
            return result
    expected = torch.Generator().manual_seed(17)
    TinyData().sample("train", "representation_unimodal", 2, expected)
    generator = torch.Generator().manual_seed(17)
    iterator = batches(ObservedData(), "representation_unimodal", 2, generator, count=2, prefetch=True)
    try:
        next(iterator)
        assert ready.wait(5), "the next batch must load while the caller handles the current one"
        assert torch.equal(generator.get_state(), expected.get_state()), "snapshot must exclude the queued batch"
    finally:
        iterator.close()


def test_prefetch_preserves_serial_training_optimizer_and_random_states(tmp_path):
    cfg = _config()
    serial = train_common_base(cfg, TinyData(), tmp_path / "serial", 0, torch.device("cpu"))
    cfg["train"]["prefetch_batches"] = 1
    prefetched = train_common_base(cfg, TinyData(), tmp_path / "prefetch", 0, torch.device("cpu"))
    left = torch.load(serial.checkpoint, weights_only=True)
    right = torch.load(prefetched.checkpoint, weights_only=True)
    for key in ("learner", "optimizer", "data_rng", "corruption_rng", "torch_rng"):
        torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
    assert serial.metrics == prefetched.metrics
    assert serial.final_training == prefetched.final_training



def _r0_source(tmp_path, monkeypatch):
    import training.common_base as training
    cfg = _config()
    # Stub the measured panel only for transfer/recovery mechanics; these are not experimental results.
    metrics = {name: 0.5 if "rank" in name else 0.2
               for name in cfg["curriculum"]["gates"]["unimodal_representation_ready"]}
    metrics.update({name: 0.1 for name in cfg["curriculum"]["gates"]["audiovisual_representation_ready"]})
    monkeypatch.setattr(training, "_panel", lambda *args, **kwargs: dict(metrics))
    result = training.train_common_base(cfg, TinyData(), tmp_path / "source", 0, torch.device("cpu"))
    target = deepcopy(cfg)
    target["train"]["stage"] = "representation_av"
    target["train"]["r0_initialization"] = {0: {
        "checkpoint": str(result.checkpoint),
        "sha256": hashlib.sha256(result.checkpoint.read_bytes()).hexdigest(),
    }}
    return target, result.checkpoint


def test_fresh_r1_refuses_to_bypass_the_r0_source_gate(tmp_path):
    cfg = _config()
    cfg["train"]["stage"] = "representation_av"
    with pytest.raises(ValueError, match="R0|r0_initialization"):
        train_common_base(cfg, TinyData(), tmp_path / "r1", 0, torch.device("cpu"))
    assert not (tmp_path / "r1/checkpoint.pt").exists()


@pytest.mark.parametrize("problem", ["checksum", "incomplete", "failed_gate", "stage", "seed", "fingerprint", "model", "threshold"])
def test_r1_rejects_unbound_incomplete_failed_or_incompatible_r0(tmp_path, monkeypatch, problem):
    cfg, path = _r0_source(tmp_path, monkeypatch)
    source = torch.load(path, weights_only=True)
    if problem == "checksum":
        cfg["train"]["r0_initialization"][0]["sha256"] = "0" * 64
    elif problem == "incomplete":
        source["step"] -= 1
    elif problem == "failed_gate":
        source["metrics"]["audio_effective_rank_fraction"] = 0.0
    elif problem == "stage":
        source["stage"] = "representation_av"
    elif problem == "seed":
        source["seed"] = 1
    elif problem == "fingerprint":
        source["data_fingerprint"] = "different-manifest"
    elif problem == "model":
        cfg["modalities"]["audio"]["encoder"]["layers"] += 1
    elif problem == "threshold":
        cfg["curriculum"]["gates"]["unimodal_representation_ready"]["audio_effective_rank_fraction"]["value"] = 0.0
    if problem != "checksum":
        torch.save(source, path)
        cfg["train"]["r0_initialization"][0]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="R0|initialization|source"):
        train_common_base(cfg, TinyData(), tmp_path / "r1", 0, torch.device("cpu"))
    assert not (tmp_path / "r1/checkpoint.pt").exists()


def test_r1_initial_snapshot_preserves_functions_and_resets_optimizer(tmp_path, monkeypatch):
    cfg, path = _r0_source(tmp_path, monkeypatch)
    source = torch.load(path, weights_only=True)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        train_common_base(cfg, TinyData(fail_after=0), tmp_path / "r1", 0, torch.device("cpu"))
    initial = torch.load(tmp_path / "r1/checkpoint.pt", weights_only=True)
    transport = initial.get("r0_initialization", {}).get("time_reference_transport")
    assert transport, "missing function-preserving R0 time-reference conversion"
    allowed = {f"{name}.projection.{suffix}" for name in transport["modules"] for suffix in ("weight", "bias")}
    assert len(allowed) == 16
    for name, value in source["learner"].items():
        if name not in allowed:
            torch.testing.assert_close(value, initial["learner"][name], rtol=0, atol=0)
    from training.base_model import build_common_world_model
    from training.representation import build_representation_learner
    from dataclasses import replace
    before, after = [build_representation_learner(cfg, build_common_world_model(cfg)).eval() for _ in range(2)]
    before.load_state_dict(source["learner"])
    after.load_state_dict(initial["learner"])
    # Inspect the analytic function before ABI bf16 rounding, which can cross quantization bins.
    for model in (before, after):
        for modality in ("video", "audio"):
            for adapter in (model.core.adapters[modality], model.teachers[modality].module.adapter):
                adapter.abi = replace(adapter.abi, evidence_dtype=torch.float32)
    old_batch = TinyData().sample("eval", "representation_unimodal", 2, torch.Generator().manual_seed(83))
    offsets = {"video": 1 / cfg["data"]["video"]["frames_per_second"],
               "audio": 1 / cfg["data"]["audio"]["sample_rate"]}
    assert transport["offset_seconds"] == offsets
    def translate(observations):
        return {m: replace(o, timestamps=o.timestamps - offsets[m]) for m, o in observations.items()}
    new_batch = replace(old_batch, current=translate(old_batch.current), future=translate(old_batch.future))
    with torch.no_grad():
        views = [model.evaluation_views(batch, stage="representation_unimodal", generator=torch.Generator().manual_seed(93))
                 for model, batch in ((before, old_batch), (after, new_batch))]
    for name in ("online", "masked_source", "teacher_current", "teacher_future", "masked_prediction", "future_prediction"):
        for modality in offsets:
            values = [v[name][modality] for v in views]
            if hasattr(values[0], "tokens"):
                values = [v.tokens for v in values]
            torch.testing.assert_close(*values, rtol=1e-4, atol=2e-5)
    assert initial["step"] == 0 and not initial["optimizer"]["state"]
    assert source["optimizer"]["state"], "source fixture must have trained optimizer state"
    provenance = initial.get("r0_initialization")
    assert provenance and provenance["sha256"] == cfg["train"]["r0_initialization"][0]["sha256"]
    assert provenance["gate"]["passed"] and provenance["step"] == source["step"]


def test_r1_resume_preserves_handoff_without_reopening_source(tmp_path, monkeypatch):
    cfg, path = _r0_source(tmp_path, monkeypatch)
    cfg["train"]["prefetch_batches"] = 1
    complete = train_common_base(cfg, TinyData(), tmp_path / "complete_r1", 0, torch.device("cpu"))
    with pytest.raises(RuntimeError, match="simulated interruption"):
        train_common_base(cfg, TinyData(fail_after=3), tmp_path / "resumed_r1", 0, torch.device("cpu"))
    path.unlink()  # Recovery owns the already initialized target snapshot, not the original file.
    resumed = train_common_base(cfg, TinyData(), tmp_path / "resumed_r1", 0, torch.device("cpu"), resume=True)
    left, right = [torch.load(r.checkpoint, weights_only=True) for r in (complete, resumed)]
    for key in ("learner", "optimizer", "data_rng", "corruption_rng", "torch_rng"):
        torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
    assert left["r0_initialization"] == right["r0_initialization"]
    assert complete.metrics == resumed.metrics and complete.final_training == resumed.final_training



def test_r0_refuses_an_r1_initialization_request(tmp_path):
    cfg = _config()
    cfg["train"]["r0_initialization"] = {0: {"checkpoint":"unused", "sha256":"0" * 64}}
    with pytest.raises(ValueError, match="R1|representation_av"):
        train_common_base(cfg, TinyData(), tmp_path / "r0", 0, torch.device("cpu"))


@pytest.mark.parametrize("offset", [0.125, 1 / 16000])
def test_fourier_time_translation_preserves_function_and_other_axes(offset):
    from encoders.temporal import CoordinateEmbedding
    import training.common_base as training
    shift = getattr(training, "_shift_time_embedding", None)
    assert callable(shift), "no implementation: analytic time-reference translation"
    torch.manual_seed(71)
    embedding = CoordinateEmbedding(3, 32)
    coordinates = torch.rand(2, 19, 3) - 0.5
    expected = embedding(coordinates).detach()
    other_axes = embedding.projection.weight[:, 9:].detach().clone()
    state = torch.get_rng_state().clone()
    shift(embedding, offset)
    assert torch.equal(state, torch.get_rng_state())
    torch.testing.assert_close(other_axes, embedding.projection.weight[:, 9:], rtol=0, atol=0)
    coordinates[..., 0] -= offset
    torch.testing.assert_close(expected, embedding(coordinates), rtol=1e-5, atol=1e-6)
