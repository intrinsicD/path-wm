# CLAUDE.md — how we build PATH-WM

The rules any coding agent or human follows in this repository. The science lives in `docs/PATH-WM_v0.3.md`; bare § numbers refer to it. `DDR §n` refers to `docs/design-decisions.md`, `Invariant n` to §17, `README rule n` to the README's working rules.

## 1. What we are building, and what we are trying to get working

Every plan, PR, experiment log and session summary opens with both. Work that serves neither is not built.

**The product** is a modular world-model research stack: a controlled causal environment (E0); exchangeable encoders and adapters that emit world state in the ABI v1 layout; a predictor, inverse-dynamics head and updater that consume it (trained in E1; frozen for the stitching experiments, where the adapter is the object under training; later experiments replace one module at a time against that frozen reference, DDR §14); path-space planners; the instrument panel (§14); a viewer. Modules talk only through the contracts of §16.1 and `docs/abi/abi_v1.yaml`.

**The research question** is whether perception, dynamics and planning can be made independently replaceable through a transition-compatible latent interface (H1), and whether planning should treat trajectories as persistent objects in path space (H6). E0 builds the environment and E1 the reference model; E2 onward decide the hypotheses in the §11 table.

### Now
<!-- Update at the end of every session. Target and Phase change only when a §20 gate is passed.
     Slice line format: <slice> — step <1–4> — done: <last part> — next: <next part>. -->
- Target: the first gate (§18) — can a predictor trained with one encoder plan using another? Reached through E0 → E1 (P and I_ω frozen, σ_pilot fixed) → E2.
- Phase: roadmap phase 1 (§20). Gate: E0 frozen (preregistration E0 row: ≥ 1e4 transitions/s, save/restore exact, ≥ 2 homotopy classes). The phase's other deliverables (contracts and conformance tests, data policy, instrument panel, viewer v0) are built slice by slice and are not the gate.
- Slice: first slice (§2) — step 1 — done: nothing — next: `contracts.py`, `world_state/abi.py`, `configs/dev/first_slice.yaml`.

A task outside the current phase is allowed, but its plan names the frozen artifact it depends on and what stands in for it, and its numbers are not experiment results.

## 2. We build in vertical slices

A slice is the thinnest end-to-end path from the environment to one number on the panel — the `evaluation/` functions that compute the §14 metrics and write them to the run's results file under `runs/<spec>/<seed>/` (the DDR §19 ledger; the viewer only renders it). The panel grows with the slices: the first slice writes s(w) and transition error; every other §14 row is added by the first slice that needs it. A component task is one part of the current slice: done when that part's tests pass and the Now block names the next part. A slice is widened, never thrown away and rebuilt. Four steps, in this order:

1. **Plan, and get the interfaces right.** The plan is the comment block at the top of `configs/dev/<slice>.yaml`: hypothesis, experiment, modules, the metric that decides success. Then, before any implementation: the §16.1 signatures as `typing.Protocol`s in `contracts.py` at the repo root (one file, no implementations, importable by every test; the environment contract — reset, step, save, restore, render, ground truth — lives there too); the ABI layout dataclass in `world_state/abi.py`; the config schema as the keys of the experiment YAML. Where §16.1 and another section disagree on a signature, §16.1 wins unless it is merely incomplete, and the resolution goes into DDR §13. Known case: the planner takes §7's (W, G, P, V, C, R) — V the critic, injected so `evaluation/budget.py` can wrap it; C the hard constraints. Choices that cannot be widened later (batched-worlds layout, save/restore representation) are made here and recorded as a DDR entry.
2. **Write the tests for the essential parts.** The structural conformance test for every module the slice touches (`tests/conformance/`), plus the one or two behaviour tests (`tests/unit/`) that would catch the failure that matters for this slice. Not full coverage. Tests are written against the interface from step 1, before the implementation, and fail on the missing implementation, not on an import error.
3. **Implement the simplest thing that works end-to-end.** Tiny data, small model, one seed, run from `configs/dev/<slice>.yaml`: a copy of the nearest `experiments/E*.yaml` with sizes, steps and seeds shrunk, the parts the slice leaves out switched off, and every `null` given a dev value marked `# dev`. No spec inheritance. A key the base spec lacks is added to the base spec, which stays editable until its hash is in `docs/preregistration.md`. Numbers from `configs/dev/` are never experiment results. Done when the fast tests are green and the number is on the panel.
4. **Improve iteratively.** One module per iteration, panel before and after; when a metric is bad, look the symptom up in the §14 table before inventing a fix. Once E1 is frozen every change is measured against the frozen reference at equal budget (DDR §14); before that, against the previous iteration. Widen (data, seeds, encoders, environments) only after the thin version works.

**The first slice.** E0 step + save/restore → ViT-S/8 encoder + linear adapter → the E1 predictor and inverse-dynamics head (`experiments/E1_reference.yaml`; §6.1 makes SIGReg + inverse mandatory against collapse), trained on a tiny dataset starting from the LeWM code → action-sensitivity ratio (§4 H1) and transition error (§6.2) on a fixed probe set, regenerated from a committed seed and count at the path `abi_v1.yaml` names. No updater, no counterfactual term, no planner, no viewer, one seed. Done when the fast tests are green and both numbers are on the panel. This is a thin cut across §18 steps 1–3; a slice never freezes anything.

**Freeze tasks** are not slices: run the spec as written (all seeds), commit its hash, σ_pilot, the thresholds it fixes and the artifact (engine version or checkpoint path) into `docs/preregistration.md`, and move Phase in the Now block when the §20 gate is met.

**Planner slices before a trained predictor exists** run against the E0 engine as an oracle predictor (save/restore forks, ground-truth state in place of W, calls counted by the same wrapper). Development configuration only; never an experiment result.

## 3. Code rules

- **Stack.** Python, pytest, PyTorch for the models (E1 starts from the LeWM codebase) and, by default, for the E0 engine too: one framework is simplest, and DDR §15's Warp/JAX/CUDA are a later swap behind the environment Protocol. The first slice's step 1 records that choice as a DDR §15 entry. One entry point runs an experiment spec: `run.py <spec.yaml>`.
- **Predesigned (§16.1, DDR §14):** the contracts, the ABI spec, the instrument panel, the E0 environment, the evaluation protocol and the thresholds. A slice builds a thin version of these against their fixed design; it never redesigns them. Everything else evolves. Until E1 is frozen a §16.1 signature may be corrected — not redesigned — when implementing it shows it is wrong or insufficient; the correction is a DDR §13 entry in the same commit as its tests. After E1 is frozen it changes only when two independent implementations both need the change (§16.1). The ABI spec is not a Protocol: a breaking change to `abi_v1.yaml` is a new major version, never an in-place edit.
- **Do not overengineer.** Build what the current slice needs. The §16.2 layout is the destination, not a scaffold: create a file when a slice needs it, never stub the deferred ones. No plugin systems, decorator registries, base classes or config layers "for later". Swapping a module is one YAML line: a plain `build_<module>(cfg)` per module family with a dict over the names that exist today.
- **Reuse before writing.** In order: modules in this repo; the LeWM code (arXiv 2603.19312; pin URL and commit in the README on first use; if it cannot be fetched, write the minimal version, note it in the plan, and swap LeWM parts in later); the stable-worldmodel harness (arXiv 2605.21800, `envs/external/`; its encode/predict/rollout/criterion/get_cost mapping is in §16.1). Copy and adapt is fine; copied code carries one header line `# adapted from <repo>@<commit>:<path>`. A dependency for one function is not.
- **Extract an abstraction only when it pays.** A Protocol that states a §16.1 signature is an interface and is written first. An abstraction is a base class or helper holding shared logic; it waits until a second implementation actually shares that logic (§16.2 plans the cases: one SMC loop with MPPI/CEM/iCEM as degenerate configs).
- **Keep it simple.** Layout as in §16.2, small files, plain functions and dataclasses, explicit arguments over global state. Dev configs live in `configs/dev/`, never under `experiments/`, so that every file there is a freezable spec.
- **Comment the critical parts.** Every module starts with a header: what it does, how (the algorithm in two or three lines), why (the hypothesis, experiment or invariant it serves, with the § reference). Comment non-obvious decisions and invariants at the line that enforces them. Do not comment the obvious.
- **Make it easy to change.** An open question the current slice touches becomes a config field with the DDR default, never a branch spread across files; the rest stay in the DDR. Numbers that can change live in the spec, not in the code. A module is edited in place until its experiment is frozen; the version frozen with that experiment is the reference (§5), kept as baseline and fallback and never edited again. After that an improvement is a new file selected by one YAML line (DDR §14).
- **Invariants are enforced in code.** Labels never enter W's objective — their only consumers are probe targets, viewer overlays, goal masks (DDR §4) and the §15 text decoder's captions; probes and decoders train behind stopgrad (Invariant 1), a goal mask only selects tokens in the planner cost (Invariant 11); registers reset per predictor call (Invariant 4); predictor and critic calls are counted only by the wrapper in `evaluation/budget.py`, never inside a planner (Invariant 8). Every §17 invariant a slice's module could violate gets a test in the same commit; these are examples, not the list.

## 4. Testing rules

Testing is a gate, not a brake.

- **Conformance tests have two layers.** Structural — ABI shape, dtype, layout, action range, calls within the declared budget and counted separately — runs with random weights in the default fast run and carries no marker. Threshold — s(w), transition error, the §6.5 adapter losses — takes a checkpoint, is marked `threshold`, and while its threshold in `abi_v1.yaml` is null or absent (the adapter losses have no field yet) it records the number and skips with reason `threshold_unset`. Never satisfy a null threshold with a placeholder.
- **Unit tests for the essentials only:** where a silent error would invalidate a result. Environment: save/restore bit-exact including the RNG stream, ≥ 2 homotopy classes on the fixed layout. World model: register reset, loss masking, stopgrad. Planner: no executed action without a verified rollout (Invariant 10). Not for glue, plotting or notebooks.
- **Fast by default.** `pytest -m 'not slow and not gpu and not threshold'` is CPU, tiny tensors, seconds. Markers are declared in `pyproject.toml`.
- **Never delete, skip or xfail a test to make the run green.**
- **Experiments are not tests.** An experiment is a YAML spec in `experiments/`, frozen by committing its hash into `docs/preregistration.md`; a frozen spec is never edited, changes are numbered amendments there. The runner is ordinary code covered by the fast tests; specs do not run in the test suite.

## 5. Session protocol

1. Read the **Now** block. State in one line which product part, hypothesis and experiment the task serves (the first experiment that runs its module).
2. Run the four §2 steps. Steps 1–2 end with the plan and the new tests committed red; step 3 and each step-4 iteration end with the fast tests green and the number on the panel.
3. Commit at the end of every completed step, message naming the slice and the experiment. The repository must be under git from the first task: specs are frozen by commit hash.
4. Report faithfully: what ran, what failed, what was skipped. Negative results are results (README rule 5).
5. Update the **Now** block. A changed DDR default is a DDR entry, not a code comment.
