"""Adapt raw experiment ledgers for the offline instrument panel."""
from pathlib import Path

class DashboardDataError(RuntimeError):
    """Recorded evidence is inconsistent or unreadable."""

def collect_run_results(runs_root: Path):
    raise NotImplementedError("Current-ledger adapter is the next repair step")
