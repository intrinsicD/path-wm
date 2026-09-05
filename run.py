"""Run a Python experiment and refresh its offline HTML instrument panel.

Restores the completion/refresh contract of `eca742a:run.py` without binding the
workflow to a model, dataset or experiment target. Arguments are passed directly
to this Python interpreter; no shell parsing or experiment configuration changes.
"""
from __future__ import annotations

import subprocess
import sys


def run_experiment(arguments, *, execute=subprocess.run, refresh=None):
    if refresh is None:
        from viewer.dashboard import write_experiment_dashboard
        refresh = write_experiment_dashboard
    try:
        result = execute([sys.executable, *arguments])
        code = result.returncode
    except KeyboardInterrupt:
        code = 130
    except OSError as error:
        print(f"Experiment could not start: {error}", file=sys.stderr)
        code = 1
    try:
        artifact, html, receipt = refresh()
        verification = receipt.get("stages", {}).get("verification", "unknown")
        print(f"Dashboard: {html}\nArtifact: {artifact}\nVerification: {verification}")
        if verification != "passed":
            print("Dashboard visual verification remains pending; browser QA did not pass.", file=sys.stderr)
            code = code or 1
    except Exception as error:
        print(f"Dashboard refresh failed: {error}. Raw experiment output is preserved.", file=sys.stderr)
        code = code or 1
    return code


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Usage: python run.py -m <module> [arguments...]\n"
              "       python run.py <script.py> [arguments...]\n"
              "Runs the experiment, then refreshes and verifies the offline dashboard.")
        return 0 if len(sys.argv) > 1 else 2
    return run_experiment(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
