"""Evaluation lifecycle checks use deterministic tiny diagnostic/controller stubs."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from world_model.paddle import checkpoints, data, evaluation, planner


@pytest.fixture
def harness(tmp_path, monkeypatch):
    calls, stages = [], []
    checkpoint_paths = []
    for stage in ("perception", "memory", "predictor"):
        path = tmp_path/f"{stage}.pt"
        checkpoints.atomic_checkpoint(path, {"schema_version": 1, "tensor_schema": checkpoints.TENSOR_SCHEMA,
            "stage": stage, "dataset_fingerprint": "dataset-A", "models": {}})
        checkpoint_paths.append(path)

    class Dataset:
        fingerprint = "dataset-A"
        def __init__(self, path, split):
            self.split = split
        def __len__(self):
            return 2
        def __getitem__(self, index):
            return {"states": np.array([[20+index, 10, 2, 2, 32]], dtype=np.float64)}

    monkeypatch.setattr(data, "EpisodeDataset", Dataset)
    monkeypatch.setattr(checkpoints, "load_system", lambda *args, **kw: {"P":None,"U":None,"H":None})
    monkeypatch.setattr(planner, "ExhaustivePlanner", lambda *args, **kw: None)
    def prediction(*args, **kwargs):
        stages.append("prediction")
        return {"summary": {"actual": {"0": {"all": {"h_mae": [1,2,3], "r_mae": [1,2,3,4,5]}}},
                            "prediction": {"5": {"all": {"h_mae": [2,3,4]}}}}, "records": []}
    def probe(*args, **kwargs):
        stages.append("probe")
        return evaluation.LinearVelocityProbe(torch.zeros(1,2), torch.zeros(2)), {"count": 2}
    monkeypatch.setattr(evaluation, "prediction_diagnostics", prediction)
    monkeypatch.setattr(evaluation, "train_velocity_probe", probe)
    monkeypatch.setattr(evaluation, "paired_probe_diagnostics", lambda *args: {"records":[], "count":2})
    monkeypatch.setattr(evaluation, "reconstruction_diagnostics", lambda *args, **kw: {"count":2,"global_mse":.1,"ball_region_mse":.2,"paddle_region_mse":.3}, raising=False)
    monkeypatch.setattr(evaluation, "_plot_metrics", lambda *args: None)
    state = {"fail_at": None}
    def controller(system, name, initial, device, max_steps, seed, model_planner, **kwargs):
        calls.append((name, tuple(initial), seed))
        if state["fail_at"] == len(calls):
            raise RuntimeError("intentional controller interruption")
        record = {"controller":name,"initial_state":np.asarray(initial).tolist(),"seed":seed,
                  "first_hit_before_miss":False,"first_hit_step":None,"total_hits":0,"episode_length":3,
                  "terminated":False,"truncated":False,"evaluation_capped":True,"planning_failure":None,
                  "invalid_candidates":0,"actions":[1,1,1],"states":[np.asarray(initial).tolist()]*4,
                  "decisions":[{"step_index":2,"action":1,"candidate_sequence":None}],
                  "first_action":1,"decision_latency_ms":[.1]}
        if kwargs.get("progress_callback"):
            kwargs["progress_callback"](record)
        return record
    monkeypatch.setattr(evaluation, "run_controller", controller)
    config = {"smoke":True,"device":"cpu","evaluation":{"ordinary_starts":2,"paired_pairs":1,
                 "control_max_steps":3,"controllers":["random","tracker"]}}
    def run(config_override=None, output=None):
        return evaluation.evaluate(config_override or config, tmp_path/"dataset", *checkpoint_paths, output or tmp_path/"evaluation")
    return dict(run=run,calls=calls,stages=stages,state=state,config=config,paths=checkpoint_paths,
                output=tmp_path/"evaluation",dataset=Dataset)


def test_evaluation_resumes_completed_cases_without_recomputing_diagnostics(harness):
    harness["state"]["fail_at"] = 2
    with pytest.raises(RuntimeError, match="controller interruption"):
        harness["run"]()
    first = harness["calls"][0]
    harness["state"]["fail_at"] = None
    report = harness["run"]()
    assert report["status"] == "completed"
    assert harness["calls"].count(first) == 1
    assert harness["stages"] == ["prediction","probe"]
    records = json.loads((harness["output"]/"control_records.json").read_text())
    assert len(records) == len({row["case_id"] for row in records}) == 8
    assert len(list((harness["output"]/"cases").glob("*.json"))) == 8
    assert json.loads((harness["output"]/"progress.json").read_text())["status"] == "completed"
    assert list((harness["output"]/"errors").glob("*.json"))


@pytest.mark.parametrize("change", ["config","checkpoint","dataset"])
def test_partial_evaluation_refuses_changed_identity(harness, change):
    harness["state"]["fail_at"] = 2
    with pytest.raises(RuntimeError):
        harness["run"]()
    harness["state"]["fail_at"] = None
    updated = copy.deepcopy(harness["config"])
    if change == "config":
        updated["evaluation"]["control_max_steps"] = 4
    elif change == "checkpoint":
        path = harness["paths"][0]
        value = torch.load(path, weights_only=False)
        value["changed_metadata"] = True
        checkpoints.atomic_checkpoint(path, value)
    else:
        harness["dataset"].fingerprint = "dataset-B"
    before = len(harness["calls"])
    with pytest.raises(ValueError, match="fingerprint|identity|dataset"):
        harness["run"](updated)
    assert len(harness["calls"]) == before


def test_evaluation_refuses_dataset_not_used_by_checkpoint(harness):
    harness["dataset"].fingerprint = "other-training-population"
    with pytest.raises(ValueError, match="dataset"):
        harness["run"]()
    assert not harness["calls"]


def test_raw_metrics_survive_plot_failure_and_resume_repairs_report(harness, monkeypatch):
    def failing_plot(*args):
        raise RuntimeError("intentional plot failure")
    monkeypatch.setattr(evaluation, "_plot_metrics", failing_plot)
    with pytest.raises(RuntimeError, match="plot failure"):
        harness["run"]()
    raw = json.loads((harness["output"]/"metrics.json").read_text())
    assert raw["status"] == "report_failed"
    assert raw["ordinary"]["summary"]["random"]["count"] == 2
    assert raw["reconstruction"]["ball_region_mse"] == .2
    before = list(harness["calls"])
    monkeypatch.setattr(evaluation, "_plot_metrics", lambda *args: None)
    report = harness["run"]()
    assert report["status"] == "completed"
    assert harness["calls"] == before


def test_completed_raw_case_is_immutable_on_resume(harness):
    harness["state"]["fail_at"] = 2
    with pytest.raises(RuntimeError):
        harness["run"]()
    path = harness["output"]/"cases"/"ordinary_0_random.json"
    original = path.read_bytes()
    harness["state"]["fail_at"] = None
    harness["run"]()
    assert path.read_bytes() == original


def test_reconstruction_regions_use_geometric_area_and_coordinate_denominators():
    # One pixel has an error only in red: the region mean averages all channels.
    frames = np.zeros((1,64,64,3),dtype=np.uint8)
    reconstruction = torch.zeros(1,3,64,64)
    reconstruction[0,0,10,10] = 1
    states = np.array([[10.5,10.5,2,2,32]],dtype=np.float64)
    result = evaluation.reconstruction_errors(reconstruction, frames, states)
    assert result["global_squared_error"] == 1
    assert result["global_scalars"] == 64*64*3
    assert result["ball_region_squared_error"] == 1
    assert result["ball_region_scalars"] == pytest.approx(16*3)
    assert result["paddle_region_squared_error"] == 0
    assert result["paddle_region_scalars"] == pytest.approx(36*3)


def test_h_target_uses_all_real_frames_while_r_keeps_warmup_mask():
    report = {"smoke":True,"ordinary":{"starts":1,"summary":{}},"paired":{"pairs":1,"summary":{}},
              "actual_frame_readout":{"count":3,"h_mae":[.2,.3,.4]},
              "prediction":{"summary":{"actual":{"0":{"all":{"h_mae":[5,5,5],"r_mae":[0,0,.1,.2,0]}}},
                                       "prediction":{"5":{"all":{"h_mae":[1,1,1]}}}}}}
    assert evaluation._targets(report)["h_actual_each_coordinate_lt_1"]


def test_all_frame_h_summary_includes_the_two_initial_observations():
    from world_model.paddle.types import ObservationLatent
    def encode(images):
        return ObservationLatent(torch.zeros(len(images),256,64),torch.zeros(len(images),64,64))
    system = {"E":encode,"D":lambda s:torch.zeros(len(s.fine),3,64,64),
              "H":lambda s:torch.zeros(len(s.fine),3)}
    episode = {"frames":np.zeros((3,64,64,3),dtype=np.uint8),
               "states":np.array([[6,12,2,3,18],[12,18,2,3,24],[18,24,2,3,30]],dtype=np.float64)}
    result = evaluation.reconstruction_diagnostics(system, [episode], "cpu", frame_batch=2)
    assert result["count"] == result["actual_frame_readout"]["count"] == 3
    assert result["actual_frame_readout"]["h_mae"] == [12.,18.,24.]
    assert result["global_mse"] == result["ball_region_mse"] == result["paddle_region_mse"] == 0
    assert [row["frame_index"] for row in result["readout_records"]] == [0,1,2]
    np.testing.assert_array_equal(np.mean([row["h_abs_error"] for row in result["readout_records"]],axis=0), [12,18,24])
