"""Explicit local-browser integration checks; ordinary CPU tests stay fast."""
import base64
import json
import os
from pathlib import Path
import re
import subprocess

import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("PATH_WM_BROWSER_TESTS") != "1",
    reason="Opt-in installed-browser integration check (PATH_WM_BROWSER_TESTS=1)",
)
ADAPTER = Path(__file__).resolve().parents[1] / "viewer/chromium_transport.mjs"


def run_probe(tmp_path, body, budget=3000):
    page = tmp_path / "probe.html"
    page.write_text(body)
    return subprocess.run(
        ["node", str(ADAPTER), "--headless", "--dump-dom",
         "--window-size=913,617", f"--virtual-time-budget={budget}",
         f"--user-data-dir={tmp_path / 'profile'}", page.as_uri()],
        capture_output=True, text=True, timeout=12,
    )


def test_animation_frame_viewport_and_negative_result_preserved(tmp_path):
    result = run_probe(tmp_path, """<!doctype html><html><head></head><body><script>
    requestAnimationFrame(() => requestAnimationFrame(() => {
      const marker = document.createElement('meta');
      marker.id = 'data-analytics-portable-verifier-result';
      marker.setAttribute('data-result', btoa(JSON.stringify({
        ok: false, code: 'deliberate_failure', width: innerWidth, height: innerHeight
      })));
      document.head.append(marker);
    }));
    </script></body></html>""")
    assert result.returncode == 0, result.stderr
    payload = re.search(r'data-result="([^"]+)"', result.stdout).group(1)
    assert json.loads(base64.b64decode(payload)) == {
        "ok": False, "code": "deliberate_failure", "width": 913, "height": 617,
    }


def test_missing_probe_result_fails_without_dump(tmp_path):
    result = run_probe(tmp_path, "<!doctype html><title>No probe</title>", budget=250)
    assert result.returncode != 0
    assert "probe" in result.stderr.lower()
    assert not result.stdout.strip()
