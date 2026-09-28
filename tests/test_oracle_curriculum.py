"""ORACLE application curriculum (R44 -> C192) as a recipe mode: diagnostic software checks.

Tiny CPU runs verify the workflow only; they say nothing about learned application.
"""

import argparse
import copy
import json
import sys

import numpy as np
import pytest
import torch

from experiments import latent_agent as la
from pathwm.data import rule_world as rw
from pathwm.evaluation import rule_world as ev
from pathwm.io import file_hash, load_component, seed_everything, state_hash
from pathwm.models.latent_core import LatentCore
from pathwm.models.slots import SymbolicSlots

# Recorded in runs/latent_agent_r1/core_relation_probe_20260923/R44/run.json (O-scratch init).
HISTORICAL_INIT = dict(
    perception="b0d8782a8992adb00c0df37a8332c9623e5f9c4cd6c05614b7bf58631e91a2ce",
    core="dee815bfd0d9bfddcb9f47aeb62910da859d9785461d61493e1800fd3656f753",
    codebook="ac566df723dfa08cb783d9c34f898496417bf0e49ba6d7517297f95ba95eb440",
    relation_codebook="e35b24cc631a8f70658237711797e9d2d365156088b0b67e52e035a2de8531ca",
)
TRAIN = rw.split_rules()["train"]


def args(tmp, **changes):
    base = dict(stage="symbolic", seed=1101, device="cpu", size="check", updates=6, oracle_curriculum=3,
                lr=3e-4, max_minutes=None, max_reserved_gib=6.0, stop_after=None, output=tmp, resume=None)
    return argparse.Namespace(**(base | changes))


def run(tmp, **changes):
    return la.train_oracle(args(tmp, **changes), la.SIZES["check"])


def resume(path, **changes):
    return la.train_oracle(args(None, output=None, resume=path, **changes), la.SIZES["check"])


def same(x, y):
    """Exact nested equality of checkpoint structures (tensors bitwise)."""
    if isinstance(x, torch.Tensor):
        return isinstance(y, torch.Tensor) and torch.equal(x, y)
    if isinstance(x, dict):
        return isinstance(y, dict) and x.keys() == y.keys() and all(same(x[k], y[k]) for k in x)
    if isinstance(x, (list, tuple)):
        return type(x) is type(y) and len(x) == len(y) and all(same(u, v) for u, v in zip(x, y))
    return x == y


def load(path, name="last.pt"):
    return torch.load(path / name, map_location="cpu", weights_only=True)


def built(size="check", seed=1101):
    seed_everything(seed)
    model = la.RuleModel(la.SIZES[size], symbolic=True)
    model.oracle = la.OracleCodes(la.SIZES[size], TRAIN)
    return model


def test_full_construction_reproduces_the_historical_initial_state():
    model = built("r1")  # the historical R1 form; "full" is the E4 core since 28 Sep 2026
    assert state_hash(model.perception) == HISTORICAL_INIT["perception"]
    assert state_hash(model.core) == HISTORICAL_INIT["core"]
    assert state_hash(model.oracle.codebook) == HISTORICAL_INIT["codebook"]
    assert state_hash(model.oracle.relation_codebook) == HISTORICAL_INIT["relation_codebook"]
    with pytest.raises(ValueError, match="training rules"):
        model.oracle([rw.split_rules()["validation"][0]])  # no code outside the training rules


def test_symbolic_token_fast_path_is_bit_identical_to_the_rendered_path():
    torch.manual_seed(0)
    symbolic = SymbolicSlots(16)
    batch = rw.sample_episodes(torch.Generator().manual_seed(1103), TRAIN, rw.KIND_SPLIT["train"],
                               episodes=3, support=(8,), queries=6, p_empty=0.5)
    a = ev.encode_episodes(ev.symbolic_perceiver(symbolic), batch, "cpu")
    b = ev.symbolic_episode_tokens(symbolic, batch, "cpu")
    for part in ("support", "query", "chain"):
        for field, x in vars(getattr(a, part)).items():
            assert torch.equal(x, getattr(getattr(b, part), field)), (part, field)
    assert torch.equal(a.keys, b.keys) and torch.equal(a.kinds, b.kinds) and a.episodes == b.episodes


def test_boundary_snapshots_stage0_then_transfers_rows_resets_optimizer_and_reseeds(tmp_path, monkeypatch):
    calls, optimizers = [], []
    original, make = rw.sample_episodes, la.oracle_stage_optimizer

    def record(generator, rules, *a, **k):
        if k.get("p_empty") == la.ORACLE_P_EMPTY:  # training batches (pools use p_empty=0)
            calls.append((len(rules), {r.family for r in rules}, generator.get_state().clone()))
        return original(generator, rules, *a, **k)

    def recorded_optimizer(model, lr):
        params, optimizer = make(model, lr)
        optimizers.append((int(model.oracle.stage), len(optimizer.state),
                           {k: v.detach().clone() for k, v in model.state_dict().items()}))
        return params, optimizer

    monkeypatch.setattr(la.rw, "sample_episodes", record)
    monkeypatch.setattr(la, "oracle_stage_optimizer", recorded_optimizer)
    paused = run(tmp_path / "run", stop_after=3)  # step R, before the boundary
    before = load(paused)
    assert int(before["model"]["oracle.stage"]) == 0 and before["optimizer"]["state"]  # stage-0 moments exist
    resume(paused)
    snapshot, end, initial = load(paused, "relation_stage.pt"), load(paused), built().state_dict()
    record_json = json.loads((paused / "relation_stage.json").read_text())
    # The kept artefact is the PRE-transfer stage-0 state (R44-final analogue), hash-recorded.
    assert all(same(snapshot[k], before[k]) for k in ("step", "model", "optimizer", "sampler"))
    assert record_json["sha256"] == file_hash(paused / "relation_stage.pt") and record_json["stage"] == 0
    stage, moments, after = optimizers[-1]  # built right after the transfer, before any mixed update
    rows = torch.tensor([i for i, r in enumerate(TRAIN) if r.family == "relation"])
    others = torch.ones(len(TRAIN), dtype=torch.bool)
    others[rows] = False
    assert stage == 1 and moments == 0  # fresh AdamW: no stage-0 moments
    assert torch.equal(after["oracle.codebook.weight"][rows], before["model"]["oracle.relation_codebook.weight"])
    assert torch.equal(after["oracle.codebook.weight"][others], initial["oracle.codebook.weight"][others])
    assert torch.equal(before["model"]["oracle.codebook.weight"], initial["oracle.codebook.weight"])  # no decay in stage 0
    assert all(torch.equal(after[k], before["model"][k]) for k in after if k.startswith("core."))  # core carried
    for key in end["model"]:  # frozen modules never change; the stage-0 table is unused afterwards
        if key.startswith(("perception.", "core.key_head.", "core.evidence_mlp.")):
            assert torch.equal(end["model"][key], initial[key]), key
    assert torch.equal(end["model"]["oracle.relation_codebook.weight"], snapshot["model"]["oracle.relation_codebook.weight"])
    # Relation rules only before the boundary; afterwards all training rules from a fresh stream.
    assert [c[:2] for c in calls] == [(44, {"relation"})] * 3 + [(len(TRAIN), set(rw.FAMILIES))] * 3
    assert torch.equal(calls[3][2], torch.Generator().manual_seed(1101 + 1009).get_state())


def test_query_truth_is_not_a_model_input():
    model = built()
    batch = rw.sample_episodes(torch.Generator().manual_seed(4242), TRAIN, rw.KIND_SPLIT["train"],
                               episodes=4, support=(8,), queries=8, p_empty=0.0)
    altered = copy.deepcopy(batch)
    altered.query.outcome = 1 - altered.query.outcome
    altered.query.post = 1 - altered.query.post
    model.oracle.transfer()
    with torch.no_grad():
        logits = []
        for b in (batch, altered):
            q = ev.symbolic_episode_tokens(model.perception, b, "cpu").query
            logits.append(model.core.apply(q.m_pre, q.a, q.b, model.oracle(b.rules)[q.episode])[0])
    assert torch.equal(logits[0], logits[1])
    changed = [ev.symbolic_episode_tokens(model.perception, b, "cpu").query.m_post for b in (batch, altered)]
    assert not torch.equal(*changed)


@pytest.mark.parametrize("pause", [2, 3, 4])
def test_interruption_and_resume_across_the_boundary_match_a_continuous_run(tmp_path, pause):
    continuous = run(tmp_path / "continuous")
    interrupted = run(tmp_path / "interrupted", stop_after=pause)
    assert load(interrupted)["step"] == pause
    resume(interrupted)
    a, b = load(continuous), load(interrupted)
    assert a["step"] == b["step"] == 6 and a["rows"] == b["rows"]
    assert all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"])
    assert same(a["optimizer"], b["optimizer"]) and torch.equal(a["sampler"], b["sampler"])
    for name in ("relation_stage.pt",):
        x, y = load(continuous, name), load(interrupted, name)
        assert all(torch.equal(x["model"][k], y["model"][k]) for k in x["model"])
    for f in sorted(p.name for p in continuous.glob("predictions_*.npz") if "_end_" not in p.name):
        u, v = np.load(continuous / f), np.load(interrupted / f)
        assert all(np.array_equal(u[k], v[k]) for k in u.files), f
    # Session end files are numbered per session; the final state's end evidence is identical.
    u, v = np.load(continuous / "predictions_end_step00006_s01.npz"), np.load(interrupted / "predictions_end_step00006_s02.npz")
    assert all(np.array_equal(u[k], v[k]) for k in u.files)


def test_cli_rejects_the_oracle_mode_outside_the_symbolic_stage(tmp_path, monkeypatch):
    for argv in (["--stage", "core", "--oracle-curriculum", "2", "--updates", "4"],
                 ["--stage", "symbolic", "--oracle-curriculum", "4", "--updates", "4"]):
        monkeypatch.setattr(sys, "argv", ["latent_agent", *argv, "--output", str(tmp_path / "x")])
        with pytest.raises(SystemExit):
            la.main()
    assert not (tmp_path / "x").exists()


# ---------------------------------------------------------------- repairs after independent review


def status(path):
    return json.loads((path / "status.json").read_text())


def enforce_gate(monkeypatch, force_pass=False):
    """Tiny-shaped test of the FULL-size branch: enforce the all44 gate at check size."""
    monkeypatch.setattr(la, "boundary_gate", lambda size: "all44 screen must pass")
    if force_pass:  # test-only: the tiny model cannot pass; the gate decision path is what is tested
        evaluate = la.oracle_evaluate

        def passing(*a, **k):
            rows, metrics = evaluate(*a, **k)
            metrics["all44"]["screen"] = "pass"
            return rows, metrics

        monkeypatch.setattr(la, "oracle_evaluate", passing)


def test_check_size_boundary_gate_exemption_is_explicit(tmp_path):
    path = run(tmp_path / "run")
    settings = json.loads((path / "run.json").read_text())["identity"]["settings"]
    assert settings["boundary_gate"] == "exempt: check size, workflow only"
    assert "boundary gate exempt" in settings["scope"]
    assert "boundary gate exempt" in json.loads((path / "result.json").read_text())["evaluation_scope"]
    assert json.loads((path / "relation_stage.json").read_text())["decision"] == "exempt"
    assert la.boundary_gate("full") == "all44 screen must pass"


def test_failed_relation_gate_is_final_without_any_mixed_update(tmp_path, monkeypatch):
    enforce_gate(monkeypatch)
    path = run(tmp_path / "run")
    state, result = load(path), json.loads((path / "result.json").read_text())
    assert status(path)["result"] == "stopped" and status(path)["report"] == "structural_verified"
    assert result["stop_reason"] == "relation_stage_failed" and result["metrics"]["relation_gate"]["all44_screen"] != "pass"
    assert state["step"] == 3 and int(state["model"]["oracle.stage"]) == 0
    assert not any(r.get("stage") == 1 for r in state["rows"])  # no mixed update, no transfer
    record = json.loads((path / "relation_stage.json").read_text())
    assert record["decision"] in ("fail", "incomplete") and record["sha256"] == file_hash(path / "relation_stage.pt")
    for name in ("predictions_relation_stage.npz", "predictions_end_step00003_s01.npz", "report.html"):
        assert (path / name).exists()
    listing = {p.name: p.read_bytes() for p in path.iterdir() if p.is_file()}
    with pytest.raises(ValueError, match="final"):
        resume(path)
    assert {p.name: p.read_bytes() for p in path.iterdir() if p.is_file()} == listing  # refused before any write


def test_passed_relation_gate_crosses_into_the_mixed_stage(tmp_path, monkeypatch):
    enforce_gate(monkeypatch, force_pass=True)
    path = run(tmp_path / "run")
    assert status(path)["result"] == "completed" and int(load(path)["model"]["oracle.stage"]) == 1
    assert json.loads((path / "relation_stage.json").read_text())["decision"] == "pass"


def test_crash_windows_at_the_boundary_recover_exactly_or_refuse(tmp_path, monkeypatch):
    continuous = run(tmp_path / "continuous")
    # (a) Adapted root reproduction: the crash hits while the snapshot is being written.
    copy_file = la.atomic_copy
    monkeypatch.setattr(la, "atomic_copy", lambda *a: (_ for _ in ()).throw(RuntimeError("crash in snapshot")))
    with pytest.raises(RuntimeError, match="crash in snapshot"):
        run(tmp_path / "a")
    monkeypatch.setattr(la, "atomic_copy", copy_file)
    assert status(tmp_path / "a")["result"] == "failed" and not (tmp_path / "a" / "relation_stage.pt").exists()
    assert int(load(tmp_path / "a")["model"]["oracle.stage"]) == 0 and load(tmp_path / "a")["step"] == 3
    # (b) The crash hits after the snapshot and gate record, before the transfer.
    transfer = la.OracleCodes.transfer
    monkeypatch.setattr(la.OracleCodes, "transfer", lambda self: (_ for _ in ()).throw(RuntimeError("crash in transfer")))
    with pytest.raises(RuntimeError, match="crash in transfer"):
        run(tmp_path / "b")
    monkeypatch.setattr(la.OracleCodes, "transfer", transfer)
    assert (tmp_path / "b" / "relation_stage.json").exists()
    for name in ("a", "b"):  # deterministic recovery equals the uninterrupted run
        resume(tmp_path / name)
        x, y = load(continuous), load(tmp_path / name)
        assert all(same(x[k], y[k]) for k in ("step", "model", "optimizer", "sampler", "rows")), name
        u, v = load(continuous, "relation_stage.pt"), load(tmp_path / name, "relation_stage.pt")
        assert all(same(u[k], v[k]) for k in ("step", "model", "optimizer", "sampler")), name
    # (c) A replay whose snapshot was altered refuses instead of continuing.
    monkeypatch.setattr(la.OracleCodes, "transfer", lambda self: (_ for _ in ()).throw(RuntimeError("crash in transfer")))
    with pytest.raises(RuntimeError):
        run(tmp_path / "c")
    monkeypatch.setattr(la.OracleCodes, "transfer", transfer)
    kept = load(tmp_path / "c", "relation_stage.pt")
    kept["model"]["oracle.relation_codebook.weight"] += 1
    torch.save(kept, tmp_path / "c" / "relation_stage.pt")
    with pytest.raises(ValueError, match="hash|differs"):
        resume(tmp_path / "c")


def test_mixed_stage_resume_refuses_a_missing_or_mismatched_snapshot(tmp_path):
    path = run(tmp_path / "run", stop_after=4)  # paused in stage 1
    kept = (path / "relation_stage.pt").read_bytes()
    (path / "relation_stage.pt").unlink()
    with pytest.raises(ValueError, match="missing"):
        resume(path)
    (path / "relation_stage.pt").write_bytes(kept + b"x")
    with pytest.raises(ValueError, match="hash"):
        resume(path)
    (path / "relation_stage.pt").write_bytes(kept)
    resume(path)
    assert status(path)["result"] == "completed"


def test_pause_time_evidence_survives_resume_byte_for_byte(tmp_path):
    path = run(tmp_path / "run", stop_after=2)
    paused = json.loads((path / "result.json").read_text())
    kept = {name: (path / name).read_bytes() for name in
            ("predictions_end_step00002_s01.npz", "metrics_end_step00002_s01.json", "result_end_step00002_s01.json")}
    assert json.loads(kept["result_end_step00002_s01.json"]) == paused and paused["session"] == 1
    resume(path)
    assert all((path / name).read_bytes() == data for name, data in kept.items())
    final = json.loads((path / "result.json").read_text())
    assert final["session"] == 2 and final["session_result"] == "result_end_step00006_s02.json"
    with pytest.raises(ValueError, match="final"):
        resume(path)  # a completed run is never extended


def test_core_loads_with_the_ordinary_component_primitive(tmp_path):
    path = run(tmp_path / "run")
    s = la.SIZES["check"]
    core = load_component(LatentCore(s["width"], s["heads"], s["loops"], s["code_tokens"], s["key_width"]),
                          path / "last.pt", "core")
    assert all(torch.equal(v, load(path)["model"]["core." + k]) for k, v in core.state_dict().items())


def test_evidence_is_written_once_and_only_an_identical_replay_is_accepted(tmp_path):
    la.retain_json(tmp_path / "m.json", dict(a=1.0))
    la.retain_json(tmp_path / "m.json", dict(a=1.0))  # deterministic replay: accepted, unchanged
    with pytest.raises(ValueError, match="refusing to overwrite"):
        la.retain_json(tmp_path / "m.json", dict(a=2.0))
    la.retain_npz(tmp_path / "p.npz", dict(p=np.arange(3.0)))
    la.retain_npz(tmp_path / "p.npz", dict(p=np.arange(3.0)))
    with pytest.raises(ValueError, match="refusing to overwrite"):
        la.retain_npz(tmp_path / "p.npz", dict(p=np.arange(4.0)))
    assert json.loads((tmp_path / "m.json").read_text()) == dict(a=1.0)


def files(path):
    return {p.name: p.read_bytes() for p in path.iterdir() if p.is_file()}


def test_foreign_or_inconsistent_boundary_snapshots_are_refused_before_any_write(tmp_path, monkeypatch):
    import shutil

    # Root reproduction: a VALID pair (hash + decision) from another run (seed 1102), mixed stage.
    own, foreign = run(tmp_path / "own", stop_after=4), run(tmp_path / "foreign", stop_after=4, seed=1102)
    for name in ("relation_stage.pt", "relation_stage.json"):
        shutil.copy2(foreign / name, own / name)
    before = files(own)
    with pytest.raises(ValueError, match="another run"):
        resume(own)
    assert files(own) == before  # own last.pt, metrics, status and evidence untouched
    # The same in the stage-0 crash window (crash after the snapshot, before the transfer).
    transfer = la.OracleCodes.transfer
    monkeypatch.setattr(la.OracleCodes, "transfer", lambda self: (_ for _ in ()).throw(RuntimeError("crash")))
    for name, seed in (("own0", 1101), ("foreign0", 1102)):
        with pytest.raises(RuntimeError):
            run(tmp_path / name, seed=seed)
    monkeypatch.setattr(la.OracleCodes, "transfer", transfer)
    for name in ("relation_stage.pt", "relation_stage.json"):
        shutil.copy2(tmp_path / "foreign0" / name, tmp_path / "own0" / name)
    before = files(tmp_path / "own0")
    with pytest.raises(ValueError, match="another run"):
        resume(tmp_path / "own0")
    assert files(tmp_path / "own0") == before
    # A record that disagrees with its own checkpoint (another boundary step) is refused too.
    path = run(tmp_path / "record", stop_after=4)
    record = json.loads((path / "relation_stage.json").read_text())
    (path / "relation_stage.json").write_text(json.dumps(record | dict(step=2)))
    before = files(path)
    with pytest.raises(ValueError, match="another run or boundary"):
        resume(path)
    assert files(path) == before
