"""Overlap bounded TAU cache decoding with acquisition for H1 / E1_common_base.

Each child caches at most one small batch and publishes no manifest. The canonical
full ingestion serializes on the same shard lock and validates every source later.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    parser.add_argument("--deadline", required=True, help="absolute ISO timestamp with timezone")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-clips", type=int, default=128)
    args = parser.parse_args()
    deadline = datetime.fromisoformat(args.deadline)
    if deadline.tzinfo is None:
        raise ValueError("deadline needs an explicit timezone")
    if min(args.workers, args.max_clips) < 1:
        raise ValueError("workers and max-clips must be positive")
    cfg = yaml.safe_load((ROOT / args.spec).read_text())
    if cfg["data"]["source"]["subset"] != "development":
        raise ValueError("acquisition prefill requires the development source subset")
    completion = ROOT / cfg["data"]["raw_root"] / ".." / ".full_media_download_complete"
    shard_root = ROOT / cfg["data"]["shard_root"]
    failures = 0
    while not completion.exists():
        remaining = deadline.timestamp() - time.time()
        if remaining <= 0:
            print("Prefill deadline reached; cached shards remain reusable", flush=True)
            return
        before = len(list(shard_root.glob("*.pt")))
        print(f"{datetime.now(timezone.utc).isoformat()} cached_shards={before}", flush=True)
        try:
            result = subprocess.run(
                [sys.executable, "-m", "training.av_data", str(args.spec), "--workers", str(args.workers),
                 "--cache-only", "--max-clips", str(args.max_clips)], cwd=ROOT, timeout=remaining)
        except subprocess.TimeoutExpired:
            print("Prefill deadline interrupted decoding; full ingestion will recover temporary shards", flush=True)
            return
        failures = failures + 1 if result.returncode else 0
        if failures >= 3:
            raise RuntimeError("three consecutive prefill failures; inspect the acquisition and decode logs")
        after = len(list(shard_root.glob("*.pt")))
        if after == before or failures:
            time.sleep(max(0, min(30, deadline.timestamp() - time.time())))
    print("Acquisition complete; final ingestion owns full-source verification and manifest publication", flush=True)


if __name__ == "__main__":
    main()
