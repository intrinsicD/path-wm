# Running the CCHI PushT world model

This is the separate E/U/P adaptation described in
[the prospective design](pusht-world-model-design.md). The earlier LeWM CLI and
weights are distinct experiments. A successful command is not a control-quality
claim; check the selected checkpoint's copy gate and held-out controller results.

Use `.venv/bin/python` in this workspace. For another environment, install
`pip install -e '.[pusht,dev]'`. This extra uses pygame-ce because the local
Python3.14 installation could not obtain the old pygame wheel; the import remains
`pygame`. The official CCHI source has already been acquired, converted, and
verified here, with provenance in [the data audit](pusht-world-model-data-audit.md).

```bash
python -m world_model.pusht prepare --source data/pusht_cchi/pusht_cchi.h5 --output data/pusht_world_model/cchi_v1
python -m world_model.pusht verify-data --data data/pusht_world_model/cchi_v1
python run.py -m world_model.pusht run-all --config configs/pusht_world_model/smoke.yaml --data data/pusht_world_model/cchi_v1 --run runs/pusht_world_model/my_smoke
python run.py -m world_model.pusht run-all --config configs/pusht_world_model/baseline.yaml --data data/pusht_world_model/cchi_v1 --run runs/pusht_world_model/my_baseline
```

Choose a new run directory for an independent experiment. `run-all` resumes
existing stage checkpoints only with the identical resolved configuration and
data identity. It preserves failed predictive gates and stops before K5 unless
the configuration explicitly identifies a smoke run. Standalone `train-perception`,
`train-memory` and `train-predictor` accept `--resume`; use `--help` for exact
dependencies. Budget extensions require a separate prospective continuation.

Each stage retains `best.pt`, `last.pt`, immutable selected snapshots,
`training.jsonl`, `validation.jsonl`, `pusht_manifest.json`, and
`pusht_result.json`. Selection includes initial validation step0. Checkpoints
contain models, optimizer, RNG, normalization, dataset/schema and frozen-module
identities. Raw artifacts are authoritative; a reporting error leaves them
available and requires repair of the canonical local dashboard.

`run-all` exports `inference.pt`. Load it with
`world_model.pusht.checkpoints.load_bundle(path, device)` to obtain E/D/H/U/R/P,
latent statistics and normalization. The simulator and planner interfaces are
documented in the design's usage excerpt. Learned control receives current and
goal images plus observed actions; physical reset labels and future replay
actions remain in the evaluation harness. Evaluation uses five primitive
0.1-second predictions and executes one absolute-XY action per decision.
