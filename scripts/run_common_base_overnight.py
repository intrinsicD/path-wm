"""Bounded local R0 experiment queue for the common evidence frontend (H1, E1_common_base).

Run the predeclared example duration controls during TAU acquisition, ingest the complete corpus,
then execute paired full-corpus seeds under a GPU lock. Every command has the shared deadline;
failures and partial results remain in runs/overnight/. This script never promotes a failed R0.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]


def compare_r0(records: list[dict], specs: dict[str, dict]) -> dict:
    """Apply DDR §23's rank improvement and unchanged temporal/variance guardrails."""
    balanced = {row["seed"]: row for row in records if row["variant"] == "balanced"}
    control = {row["seed"]: row for row in records if row["variant"] == "control"}
    expected = set(specs["balanced"]["seeds"])
    if set(balanced) != expected or set(control) != expected:
        return {"decision": "incomplete", "r1_ready": False}
    gates = specs["balanced"]["curriculum"]["gates"]["unimodal_representation_ready"]
    deltas = []
    rank_improved, guardrails_passed = True, True
    for seed in sorted(expected):
        for modality in ("video", "audio"):
            name = f"{modality}_effective_rank_fraction"
            delta = balanced[seed]["metrics"][name] - control[seed]["metrics"][name]
            deltas.append({"seed": seed, "metric": name, "balanced_minus_control": delta})
            rank_improved &= delta > 0
        for name, threshold in gates.items():
            if "effective_rank" in name:
                continue
            value = balanced[seed]["metrics"][name]
            guardrails_passed &= (value >= threshold["value"] if threshold["op"] == "greater_equal"
                                  else value > threshold["value"])
    retain = bool(rank_improved and guardrails_passed)
    selected = balanced if retain else control
    return {
        "decision": "retain_covariance" if retain else "covariance_not_supported",
        "rank_deltas": deltas,
        "balanced_guardrails_passed": bool(guardrails_passed),
        "r1_ready": all(row["gate"]["passed"] for row in selected.values()),
        "selected_variant": "balanced" if retain else "control",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deadline", required=True, help="absolute ISO timestamp with timezone")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/overnight/common_base_20260905")
    parser.add_argument("--download-session", help="task-owned tmux downloader to interrupt at deadline")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    deadline = datetime.fromisoformat(args.deadline)
    if deadline.tzinfo is None:
        raise ValueError("deadline needs an explicit timezone")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    state = {"deadline": deadline.isoformat(), "pid": os.getpid(), "completed_runs": records}

    def update(phase: str, **details) -> None:
        state.update(phase=phase, updated=datetime.now(timezone.utc).isoformat(), **details)
        temporary = output / "status.json.tmp"
        temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
        temporary.replace(output / "status.json")
        with (output / "events.jsonl").open("a") as events:
            events.write(json.dumps({"phase": phase, "updated": state["updated"], **details}, sort_keys=True) + "\n")
        print(f"{state['updated']} {phase}: {details}", flush=True)

    def remaining() -> float:
        seconds = deadline.timestamp() - time.time()
        if seconds <= 0:
            raise TimeoutError("overnight deadline reached; saved checkpoints can resume")
        return seconds

    def command(name: str, argv: list[str]) -> None:
        seconds = remaining()
        update(name, command=argv, log=str(output / f"{name}.log"))
        env = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "PYTHONUNBUFFERED": "1"}
        with (output / f"{name}.log").open("a") as log:
            process = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            try:
                code = process.wait(timeout=seconds)
            except BaseException:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                raise
        if code:
            raise RuntimeError(f"{name} exited {code}; see {output / (name + '.log')}")

    def run_pair(corpus: str) -> dict:
        specs = {}
        for variant in ("balanced", "control"):
            specs[variant] = yaml.safe_load((ROOT / f"configs/dev/common_base_{corpus}_{variant}.yaml").read_text())
        for seed in specs["balanced"]["seeds"]:
            for variant in ("balanced", "control"):
                name = f"{corpus}_{variant}_{seed}"
                spec_name = f"common_base_{corpus}_{variant}"
                command(name, ["flock", "-x", str(ROOT / "runs/.common_base_gpu.lock"), sys.executable,
                               "run.py", f"configs/dev/{spec_name}.yaml", "--device", "cuda", "--resume",
                               "--seed", str(seed)])
                run_dir = ROOT / "runs/dev" / spec_name / str(seed)
                summary = json.loads((run_dir / "run_summary.json").read_text())
                metrics_record = json.loads((run_dir / "metrics.json").read_text())
                if summary["metrics"] != metrics_record["metrics"]:
                    raise ValueError(f"unreconciled metrics in {run_dir}")
                records.append({"corpus": corpus, "variant": variant, "seed": seed,
                                "run_dir": str(run_dir), "metrics": summary["metrics"], "gate": summary["gate"]})
                update("seed_complete", completed_run=name)
        selected = [row for row in records if row["corpus"] == corpus]
        comparison = compare_r0(selected, specs)
        if corpus == "examples_long":
            comparison["r1_ready"] = False
            comparison["scope"] = "duration diagnostic only; no promotion or coefficient selection"
        (output / f"{corpus}_comparison.json").write_text(
            json.dumps({"comparison": comparison, "runs": selected}, indent=2, sort_keys=True) + "\n")
        return comparison

    try:
        update("starting", git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
        run_pair("examples_long")
        dataset = ROOT / "data/tau_urban_av_2021"
        while not (dataset / ".full_media_download_complete").exists():
            remaining()
            markers = len(list((dataset / ".archives/completed").glob("*.md5")))
            update("waiting_for_corpus", completed_archives=markers, expected_archives=24)
            time.sleep(min(30, remaining()))
        command("ingestion", [sys.executable, "-m", "training.av_data",
                              "configs/dev/common_base_full_balanced.yaml", "--workers", str(args.workers)])
        manifest = [json.loads(line) for line in (dataset / "manifest_development.jsonl").read_text().splitlines()]
        counts = {split: sum(row["split"] == split for row in manifest) for split in ("train", "eval")}
        if counts != {"train": 8646, "eval": 3645}:
            raise ValueError(f"incomplete official TAU corpus: {counts}")
        update("full_corpus_verified", split_counts=counts)
        comparison = run_pair("full")
        update("r0_comparison_complete", comparison=comparison)
    except (TimeoutError, subprocess.TimeoutExpired) as error:
        if args.download_session:
            subprocess.run(["tmux", "send-keys", "-t", args.download_session, "C-c"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        update("deadline_reached", error=str(error))
        raise SystemExit(124)
    except BaseException as error:
        update("failed", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
