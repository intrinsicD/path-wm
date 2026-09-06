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


def test_mobile_control_bars_remain_visible_with_long_source_run_names(tmp_path):
    from viewer.dashboard import build_dashboard_artifact,_deliver_portable_artifact,find_portable_artifact_builder
    from viewer.ledger import RunResult
    label='overnight_2026-09-06/'+('long_source_protocol_'*5)+'/final'
    results=[RunResult(label,'control','evaluated',4074,
        dict(successes=1,cases=1,success_rate=1.,initial_successes=0,noninitial_cases=1,
             noninitial_successes=1,noninitial_success_rate=1.),
        {'case_set':'frozen case','iterations':30,'checkpoint':'runs/example/checkpoint.pt'},
        ('runs/example/summary.json',),1.)]
    artifact=build_dashboard_artifact(results,[])
    ap=tmp_path/'artifact.json';ap.write_text(json.dumps(artifact))
    hp=tmp_path/'dashboard.html';_deliver_portable_artifact(ap,hp,find_portable_artifact_builder())
    probe='''<script>
    const start=performance.now();
    function measure(){
      const paths=[...document.querySelectorAll('.recharts-bar-rectangle .recharts-rectangle')];
      if(performance.now()-start<1500 || (!paths.length && performance.now()-start<10000)){requestAnimationFrame(measure);return;}
      const boxes=paths.map(p=>{const b=p.getBoundingClientRect();return {x:b.x,right:b.right,width:b.width};});
      const marker=document.createElement('meta');marker.id='data-analytics-portable-verifier-result';
      marker.setAttribute('data-result',btoa(JSON.stringify({width:innerWidth,boxes})));
      document.head.append(marker);
    }
    requestAnimationFrame(()=>requestAnimationFrame(measure));
    </script>'''
    body,ending=hp.read_text().rsplit('</body>',1)
    hp.write_text(body+probe+'</body>'+ending)
    result=subprocess.run(['node',str(ADAPTER),'--headless','--dump-dom','--window-size=390,1400',
        '--virtual-time-budget=15000',hp.as_uri()],capture_output=True,text=True,timeout=25)
    assert result.returncode==0,result.stderr
    payload=re.search(r'<meta id="data-analytics-portable-verifier-result" data-result="([^"]+)"',result.stdout)
    assert payload,result.stdout[-1000:]
    measured=json.loads(base64.b64decode(payload.group(1)))
    assert measured['width']==390
    assert any(b['width']>=80 and b['x']>=0 and b['right']<=390 for b in measured['boxes']),measured
    # Shortened chart labels must never erase the full source/run identity.
    assert artifact['snapshot']['datasets']['control_detail'][0]['run']==label
    assert any(e['field']=='run' for e in artifact['manifest']['charts'][0]['encodings']['tooltip'])
