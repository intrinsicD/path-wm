"""Exact resume and failure receipts for the bounded evidence-loop recipe."""
import json
import sys
import pytest
import torch
from experiments import evidence_loop as recipe


def run(monkeypatch, path, *extra):
    monkeypatch.setattr(sys,"argv",["evidence_loop","--output",str(path),"--device","cpu",
                                   "--steps","4","--batch","2","--hidden","16",
                                   "--eval-count","2","--variant","linear",*map(str,extra)])
    recipe.main()


def equal(a,b):
    if torch.is_tensor(a): return torch.equal(a,b)
    if isinstance(a,dict): return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)): return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def test_exact_resume_and_development_population(monkeypatch,tmp_path):
    run(monkeypatch,tmp_path/"full","--calibration-steps",3)
    run(monkeypatch,tmp_path/"half","--stop-after",2,"--calibration-steps",3)
    run(monkeypatch,tmp_path/"resume","--resume",tmp_path/"half/last.pt","--calibration-steps",3)
    a=torch.load(tmp_path/"full/last.pt",weights_only=True)
    b=torch.load(tmp_path/"resume/last.pt",weights_only=True)
    for k in ("model","optimizer","step","generator","rng","cuda_rng","calibration_step","variance_optimizer"):
        assert equal(a[k],b[k]),k
    run(monkeypatch,tmp_path/"complete-again","--resume",tmp_path/"resume/last.pt","--calibration-steps",3)
    done=torch.load(tmp_path/"complete-again/last.pt",weights_only=True)
    for k in ("model","optimizer","generator","rng","calibration_step","variance_optimizer"):
        assert equal(b[k],done[k]),k
    for label in ("full","half","resume"):
        result=json.loads((tmp_path/label/"result.json").read_text())
        assert result['evaluation_seed']==240925 and not result['final_evaluation']
        assert json.loads((tmp_path/label/"status.json").read_text())['report']=='structural_verified'
    with pytest.raises(ValueError,match="exceeds"):
        run(monkeypatch,tmp_path/"backward","--resume",tmp_path/"half/last.pt","--stop-after",1,"--calibration-steps",3)
    assert json.loads((tmp_path/"backward/status.json").read_text())['result']=='failed'
    assert not (tmp_path/"backward/last.pt").exists()


def test_bad_resume_and_stop_bounds_leave_no_running_record(monkeypatch,tmp_path):
    with pytest.raises(FileNotFoundError):
        run(monkeypatch,tmp_path/"missing","--resume",tmp_path/"absent.pt")
    assert json.loads((tmp_path/"missing/status.json").read_text())['result']=='failed'
    for value in (-1,0,5):
        with pytest.raises(SystemExit): run(monkeypatch,tmp_path/f"bad{value}","--stop-after",value)
        assert not (tmp_path/f"bad{value}").exists()
