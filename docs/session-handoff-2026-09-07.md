# World-model continuation handoff — 2026-09-07

The implementations, trained checkpoints, exact datasets, raw results and offline
dashboard are preserved for continuation at home. **The software runs end to end;
the requested learned-control quality targets remain unmet.** Read
[project state](project-state.md), [paddle results](paddle-world-model-results-2026-09-07.md)
and [PushT results](pusht-world-model-results-2026-09-07.md) before choosing the next
experiment. All scheduled training and evaluations in this session have finished.

## Restore and inspect at home

Run these commands from the repository root. Restoration requires only Python's
standard library. It verifies every archive part and object before installing
files, skips existing identical files, and refuses to overwrite different files.

```bash
git pull --ff-only origin main
python3 scripts/session_handoff.py restore --package artifacts/session-2026-09-07 --destination .
```

Open `runs/experiment_dashboard.html` for the self-contained offline report. The
archive also restores the canonical data companion and browser-verification
receipt. Follow the standing workflow in [CLAUDE.md](../CLAUDE.md).

For the recorded Python environment:

```bash
python3 scripts/session_handoff.py requirements --package artifacts/session-2026-09-07 > /tmp/path-wm-observed-constraints.txt
python3.14 -m venv .venv
.venv/bin/python -m pip install -c /tmp/path-wm-observed-constraints.txt -e '.[paddle,pusht,dev]'
.venv/bin/python -m world_model doctor
.venv/bin/python scripts/session_handoff.py verify-restored --root . --output runs/restored-session-verification.json
```

The recorded runtime is Python 3.14.7, Torch 2.14.0+cu130 (installed distribution
version 2.14.0), CUDA 13.0, NumPy 2.5.2 and an RTX 4090. The archive manifest
contains the exact observed distribution versions and platform metadata. Match
those versions and the Torch/CUDA build for numerical continuation; a different
runtime may load the same checkpoints but produce different floating-point
trajectories. The history trainer rejects resuming on a different resolved device.
CPU inference and software checks do not require a GPU.

## Verified portable package

[Manifest](../artifacts/session-2026-09-07/manifest.json) and
[explicit selection](../artifacts/session-2026-09-07.selection.json) preserve
13,870 relative paths, file sizes, SHA256 values and environment provenance.
The snapshot was built from code commit `1793df85bde9c230aa74af38cd701f8e83cc711b`.
Later closeout commits add the archive, final receipts and this final wording.

| Item | Exact count or size |
| --- | ---: |
| Restored files | 13,870 |
| Logical bytes | 1,427,058,893 |
| Unique SHA256 objects | 13,777 |
| Unique object bytes | 1,368,286,032 |
| Compressed bytes | 212,389,903 |
| Archive parts | 5: four at 45,000,000 bytes, one at 32,389,903 bytes |

The archive uses gzip-compressed tar objects addressed by SHA256. Identical
checkpoint aliases share stored objects while restoring their exact original
names. Packaging rejects active trainer locks or changing source files.
Restoration rejects corrupt or missing parts, traversal, symlinks and conflicting
destinations. Exact datasets, selected/last/initial/referenced checkpoints,
failed-run receipts, diagnostics, qualitative figures and canonical HTML are
included. Rebuildable frame/observer caches, environments, credentials, IDE state,
plugin caches and unrelated historical data are excluded. Original local files
were preserved.

The [actual restoration receipt](../artifacts/session-2026-09-07/restoration_receipt.json)
records restoration into a fresh `/tmp` directory: every file/object hash passed,
all 6,000 paddle episodes replayed exactly (220,754 frames / 214,754 transitions),
all 25,650 PushT frames and labels matched the relocated source HDF5, and all
20,493 supplemental observations verified. All four carried inference bundles
ran E→U→P→U plus D/H/R on CPU with finite outputs. Only the paddle continuation
bundle is fully trained; both PushT exports are explicitly smoke evidence.
Restoration took 78.16 seconds and offline source/inference checks took 25.16
seconds. This proves software/data continuity, not learned-control quality.

The final integrated suite passed **336 tests**, including all three installed-
browser checks. The canonical dashboard passed package, data, source-interaction
and 1440/390-pixel verification: 49 datasets, 55 charts and five image blocks.
Additional [visual QA](../artifacts/session-2026-09-07/reporting_visual_qa.json)
and screenshots beside the package were recorded after archive construction.
Native charts and source PNGs were inspected. Optional HTML-panel screenshots
selected hidden iframe containers and are explicitly inconclusive; they do not
replace the passed canonical verification. These supplementary QA files are
available directly in Git and do not alter archive hashes.

## Exact data identities and relocation

- Paddle `data/paddle/baseline`: 5,000 / 500 / 500 episodes, fingerprint
  `c6d255dc9919b1bb9ce38182f9d180047f964754a32c616db072bc965a3be54f`.
- PushT `data/pusht_world_model/cchi_v1`: 206 episodes / 25,650 frames, disjoint
  164 / 20 / 22 initial-configuration groups, fingerprint
  `e6ff0cad101c15ba3e2b839df28bc0bd75dca4063a7d6293141976ab44b4fa9c`.
- Source HDF5 `data/pusht_cchi/pusht_cchi.h5`, SHA256
  `30e442fedacaa4a0d36662b951c326b411a15c3a365b514587b9a738229da8f6`.
- Independent pose supplement `data/pusht_world_model/pose_supplement_v1`,
  20,493 observations, fingerprint
  `34e6e5a352a1355fd6ac7a1a6471bb77f5e5a466e18bf9882a8d9215afb9b7f3`.

Preserve prepared manifests verbatim. PushT retains original absolute source
provenance paths in its fingerprint; rewriting or repreparing it would change
its identity. Training uses the restored relative prepared-data directory.
`verify-restored` supplies the relocated HDF5 explicitly. The equivalent check is:

```bash
.venv/bin/python -c "from world_model.pusht.data import verify_dataset; print(verify_dataset('data/pusht_world_model/cchi_v1', source='data/pusht_cchi/pusht_cchi.h5'))"
```

## Completed results and next choices

Paddle's reference completed E/U at 10,000 updates, P1 at 20,000 total through a
separate budget-only continuation, and P5 at 10,000. Its 3,500-case evaluation
records learned catches of 181/500 ordinary and 41/200 paired cases. Memory,
prediction and control targets remain unmet. The inference export is
`runs/paddle/continuation_v1/inference.pt`.

The history-start U follow-up completed 10,000 updates and selected 9,750.
Cold-start positions improve, but ordinary state error and paired velocity error
worsen. Fresh P1 with the selected new U is still a bounded next experiment under
[the existing plan](paddle-history-start-plan.md); fresh P5 is conditional on its
unchanged copy gate. P1/P5 for this changed U have not run. Existing predictors
and observer caches depend on the original U and cannot be silently reused.
Use a new declared run when changing a budget, population or objective.

PushT's bounded E/U/P1 reference failed its P1 pusher-position gate, so P5 and
full test control did not run. The 1,000-update perception coverage comparison
improved foreground reconstruction and the separate stress grid, but worsened
source localization. It is not adopted and has no downstream U/P expansion.
Selected/final comparisons and different diagnostic populations remain separate
in the [results](pusht-world-model-results-2026-09-07.md). A future objective or
budget experiment needs an explicit plan and preserved reference.

Claude completed independent physics, learning, code, memory and PushT-design
reviews. A further external perception review was rejected by automatic approval
review pending approval for its specific local payload. It did not execute.
The prompt and earlier completed review receipts are restored with the evidence.
Do not retry the pending external transfer without that requested approval.

## Refreshing the dashboard

The restored HTML works offline immediately. Regeneration additionally needs
Node, Chromium and the Data Analytics portable-artifact builder. The installed
plugin is not exported. `PATH_WM_ARTIFACT_BUILDER` selects a compatible builder;
`PATH_WM_CHROMIUM` selects the browser. Check these prerequisites before a long
run, then use `python run.py ...` or refresh via `python -m viewer.dashboard`.
Historical LeWM rows carried by the HTML remain preserved viewing evidence;
their unrelated training populations/checkpoints are outside this task package.
