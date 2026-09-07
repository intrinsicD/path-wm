# Paddle world model

This separate package implements `world_model_codex_implementation_brief.md`.
The existing LeWM implementation and its results remain available. The accepted
architecture is E (visible features), U (persistent observer), and P (stateless
action-conditioned prediction); imagined memory always advances through the same
U. Simulator labels train the permitted readouts and support evaluation.

Use the repository virtual environment. `python` below means `.venv/bin/python`
with that environment activated. `doctor` prints actual Python/PyTorch/CUDA and
device information. The NVIDIA device may require running outside a restricted
sandbox on this machine.

The verified HTML workflow also uses the local Chrome for Testing headless shell
at `.runtime/browser/chrome-headless-shell-linux64/chrome-headless-shell`, version
152.0.7977.82. Its download identity/hash is recorded in
`.runtime/browser/install.json`. The dashboard transport discovers it automatically;
`PATH_WM_CHROMIUM` can select another installed browser.

The examples below use the recorded `smoke_v2` data and checkpoints, whose
environment fingerprint includes the contact-roundoff correction. The earlier
`data/paddle/smoke` and `runs/paddle/smoke` directories preserve preliminary work.
Use new data/run directories for an independent experiment; omit `--resume` from
the training commands when creating a new run.

```bash
source .venv/bin/activate
python -m world_model doctor
python -m pytest tests/paddle -q
python -m world_model generate --config configs/paddle/smoke.yaml --output data/paddle/smoke_v2
python -m world_model verify-data --data data/paddle/smoke_v2
python run.py -m world_model test-history-cases --output runs/paddle/smoke_v2/history_checks
```

Every standalone training/evaluation command uses the shared `run.py` wrapper so
the canonical offline [experiment dashboard](../runs/experiment_dashboard.html)
is refreshed after its result. A dashboard failure does not erase raw evidence;
repair reporting before calling the experiment complete.

```bash
python run.py -m world_model train-perception --config configs/paddle/smoke.yaml --data data/paddle/smoke_v2 --run runs/paddle/smoke_v2/perception --resume
python run.py -m world_model train-memory --config configs/paddle/smoke.yaml --data data/paddle/smoke_v2 --perception runs/paddle/smoke_v2/perception/best.pt --run runs/paddle/smoke_v2/memory --resume
python run.py -m world_model train-predictor --config configs/paddle/smoke.yaml --data data/paddle/smoke_v2 --perception runs/paddle/smoke_v2/perception/best.pt --memory runs/paddle/smoke_v2/memory/best.pt --horizon 1 --run runs/paddle/smoke_v2/predictor_1 --resume
python run.py -m world_model train-predictor --config configs/paddle/smoke.yaml --data data/paddle/smoke_v2 --perception runs/paddle/smoke_v2/perception/best.pt --memory runs/paddle/smoke_v2/memory/best.pt --initialize-from runs/paddle/smoke_v2/predictor_1/best.pt --horizon 5 --run runs/paddle/smoke_v2/predictor_5 --resume
python run.py -m world_model evaluate --config configs/paddle/smoke.yaml --data data/paddle/smoke_v2 --perception runs/paddle/smoke_v2/perception/best.pt --memory runs/paddle/smoke_v2/memory/best.pt --predictor runs/paddle/smoke_v2/predictor_5/best.pt --output runs/paddle/smoke_v2/evaluation
python run.py -m world_model demo --perception runs/paddle/smoke_v2/perception/best.pt --memory runs/paddle/smoke_v2/memory/best.pt --predictor runs/paddle/smoke_v2/predictor_5/best.pt --output runs/paddle/smoke_v2/demo
```

The smoke configuration uses 20/4/4 episodes and 20 optimizer updates per stage.
It checks execution and serialization; its quality-gate bypass is recorded and
does not demonstrate learning. Smoke control uses two ordinary initial states and
two pairs (four directed histories) while retaining full 200-interval episode
limits. Every controller receives the same initial states and two stay actions,
then generates its own trajectory.

The full configuration specifies 5000/500/500 train/validation/test episodes and
four bounded 10000-update stages. Use a new run directory for another experiment:

```bash
python -m world_model run-all --config configs/paddle/baseline.yaml --data data/paddle/baseline --run runs/paddle/baseline
```

`run-all` refreshes and verifies the dashboard internally after each completed
stage and evaluation, so it does not need the outer wrapper. Rerunning the same
command verifies existing data, reuses compatible completed stages, and resumes
an interrupted stage from `last.pt`. Do not start another writer in the same run
directory while a run is active.

Training retains `best.pt` and `last.pt`, including optimizer/sampler/RNG state
and dependency fingerprints. Stage commands use the `--resume` flag shown above;
the saved configuration must match. The five-step stage initializes from selected
one-step P weights with a fresh optimizer; resuming that five-step stage restores
its own optimizer state. Full runs require lower fixed-validation latent loss
and lower MAE for each moving-position coordinate than matched copying before
continuing. A failed gate preserves diagnostics and checkpoints. It does not
count as converged pretraining.

`run-all` exports `inference.pt` in its run directory. To create a separate bundle
from compatible stage checkpoints, use:

```bash
python -m world_model export-bundle --perception runs/paddle/smoke_v2/perception/best.pt --memory runs/paddle/smoke_v2/memory/best.pt --predictor runs/paddle/smoke_v2/predictor_5/best.pt --output runs/paddle/smoke_v2/inference_export.pt
```

The bundle contains all E/D/H/U/R/P weights and fixed scale statistics, without
dependencies on another machine's absolute paths.

Evaluation writes `metrics.json`, raw `prediction_records.json` and
`control_records.json`, diagnostic probe rows, `reconstruction_records.json`,
`actual_frame_readout_records.json`, corresponding CSV files, `metrics.png`,
aligned PNG/GIF panels under `visuals/`, and a local `index.html`.
Prediction/copy/reset comparisons share the exact source windows and executed
actions at horizons 1 through 5. Errors include each coordinate, p95 and maximum,
and separate reflection populations. The `actual_frame_readout` H summary covers
every held-out frame, including initial, warm-up and terminal observations. R
errors and the `prediction.summary.actual` comparison group use observation index
2 onward, after U has assimilated three frames; that group's H subset remains
separately recorded. Readout position units are world units/pixels; velocities
are world units per decision interval. Reconstruction diagnostics also cover all
test frames and report global, ball-region and paddle-region RGB MSE with their
scalar denominators. Those diagnostic masks do not alter training losses.
The preserved `smoke_v2` evaluation predates the per-frame H export; its all-frame
H summary is recorded in `metrics.json`.

Evaluation saves completed controller cases and diagnostic stages independently.
Repeating an interrupted evaluation command reuses them only when the recorded
configuration, data, checkpoints and inference identity match. Raw measurements
survive a plotting/reporting failure. Use a new output directory when those
identities change.

Controls include the learned exhaustive planner, the same planner with a fresh
current-frame observer state, uniform random actions, a frozen-H current-frame
tracker, and a clearly labeled privileged exhaustive simulator reference. The
privileged simulator never supplies model inputs or learned candidate scores.
Records preserve first hit before miss, hits, episode length, exact actions and
initial states, failed cases, and synchronized decision latencies. Ordinary starts
and paired-history cases remain separate. Full budgets use 500 ordinary starts
and 100 pairs; pair seeds 8000–8099 are distinct from the 50 validation-pair seeds
7000–7049 and all ordinary dataset seeds.

The current-frame velocity probe is a separate diagnostic linear regression on
frozen S, fitted only on selected training frames. It is evaluated alongside R on
identical-current-image opposite-history pairs and never joins the deployed
architecture. Its recorded source rows make the training population auditable.

Latency excludes rendering, real E/U assimilation, decoding and diagnostic
readouts; it includes candidate inference, frozen H and scoring. Reset latency
also includes its fresh assimilation. Median/p95, measured device, precision,
candidate batch size and decision count are recorded. These are decision timings;
they do not alone establish a complete 20 Hz observation-to-action loop.

PNG grids and animations align actual frames, reconstructed actual features and
predicted features using the same executed action sequence. Text distinguishes R
after real observations from R after imagined updates. The report includes the
first observed learned-controller success and failure when each exists, and
explicitly records a missing category. A convincing reconstruction by itself is
not evidence of accurate dynamics or useful control.

Software completion and empirical success are separate. Engineering targets are
H actual coordinate MAE below 1, R velocity coordinate MAE below 0.5, five-step H
coordinate MAE below 2, and first-interception success at least 90% on both ordinary
and paired populations. The raw report records target outcomes; a tiny smoke
sample cannot establish those targets. See the active
[implementation plan](paddle-world-model-plan.md) for measured results and scope.
