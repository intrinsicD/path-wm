# Current capability baseline v1 — 12 September 2026

User request: test all currently available capabilities and preserve a comparison
point. This is evaluation, with no weight adjustment, training or winner selection.
The scorecard must not combine separate checkpoints into a fictional single agent.
The directly constructed visual checkpoint is the primary subject; compatible
specialist checkpoints and the briefly trained general belief agent are identified
separately. Missing interfaces/labels receive explicit coverage gaps, not scores.

## Frozen protocol before execution

Extend the ordinary multimodal recipe with `--dataset capabilities`. Reuse its
constructors, existing component evaluators, Run records and standalone renderer.
Small reusable evaluation helpers cover input transformations, observed output
metrics, and strict same-protocol comparisons. No new training framework.

- Visual checkpoint: `runs/direct_weights_v1/timing/weights.pt`. 32 paired scenes,
  seed6301, plus exactly these prespecified transformations: horizontal reflection
  (swap answer labels), RGB red/blue swap, grayscale, brightness ×0.5, independent
  Gaussian pixel noise sigma0.1 with common noise within each pair (seed6401).
  Evaluate full and erased history on each; original accuracy90%, complete pairs80%,
  reversal80%, history advantage30 percentage points are the original-task screen.
  Each changed-input cohort is a separate robustness measurement, using the same
  screen as a declared diagnostic target, not an established deployment requirement.
- Inspect that exact checkpoint's image/audio/text decoders, observed modality
  responses and opposite-action predictions. Log raw outputs and zero/nonzero
  parameter counts. Input or action sensitivity is only a mechanism check; it cannot
  establish correct recognition, language, sound, generation or physical dynamics.
- General belief checkpoints: `belief_v1/{synthetic,instructions,pusht}/last.pt`.
  16 test episodes each; batch2, same history/horizon/image size as saved settings.
  Synthetic seed303 (existing generator test split); real PushT first16 test windows
  (overlapping windows are not independent episodes). Preserve actual predictions
  and targets, image/audio MSE, text exact match and action sensitivity. Image and
  audio screens must beat copy-last and silence respectively. Fixed instruction
  operation screen90%; a lexical reference remains visible. Do not call toy byte
  outputs speech or essays. These are the saved short-pilot weights, not full training.
- Structured facts: direct `fact_grounding_v1/reference`, through-agent
  `event_fact_v1/lr_control`; rerun all32 known development pairs. Frozen reuse for
  regression, not new holdout evidence. Entity/location accuracy90% screens.
- Historical recall: `recall_v1/development/last.pt`, existing15 test episodes;
  raw head without new temperature fitting, split old/recent/absent where provided.
  Screen cost below all-abstain0.25, with answered coverage reported.
- Entity association/state: use the matcher and state cell embedded in
  `key_box_v1/replica/last.pt`; 8 descriptor families seed6501; existing temporal
  reference/reversal/long-toggle/no-information and idle cases. Report matching,
  both-state accuracy (95% screen), raw logits, retry/restore agreement separately.
- Relations: original relation key and original interaction cell (explicitly
  separate specialists), 2 families seed6502, existing reference/rebind/gap/permuted/
  erased cases. Screen95% reference accuracy; report all conditions and rejection.
- Reliability: original frozen gate,16 source worlds seed6601, prescribed calibration
  feedback then frozen test policy. Include cumulative/window drift with fresh seed
  if existing evaluator accepts it. This online table update is part of the tested
  policy, not weight training. Utility must beat stop for the static screen. Preserve
  static-vs-drift tradeoffs, supplied source/outcome assumptions and all raw feedback.
- Planning: saved key-box replica,16 fresh descriptor families seed6701, ordinary
  and relocation feedback; reachable/absent success95%, raw actions/cost and
  no-history/supplied-state controls. Also rerun the already known one-step
  reward-versus-reported-utility counterexample; keep it visible as an open mismatch.
- Real visual input: all12 selected COCO development photographs; decode/re-encode
  compatibility and reconstruction against constant-gray reference. No photo
  identity/memory labels are invented. Six selected video files have existing decode
  receipts; inspect receipt/hash availability and retain this as prior acquisition
  evidence, not a new semantic model score. No live camera recording.
- Separate perception/dynamics reference recipes are engineering/reference paths;
  include fresh compatible checkpoint evaluation if their prepared data can be
  resolved. Do not silently use archived incompatible models or nonexistent aliases.
- Full CPU pytest with per-file counts: architecture, gradients, causal provenance,
  bounded memory, retries, restore, output contracts, modality/capture and run checks.
  Passing software tests never become learned-capability scores.
- Explicit gap rows: raw-image entity discovery/persistent recognition and faces;
  learned graph/concept topology; useful speech/conversation/essay writing;
  goal-directed image/video generation; general software-tool execution; integrated
  learned-dynamics planning; live webcam and labeled real memory episodes.

All new populations are fixed once, with no tuning against results. These are small
screens. Original development exposure and future repeated use remain disclosed.
Record numerical metrics independently of pass/fail; no overall intelligence score.

## Integrity, budget and checks

Checkpoint input paths and hashes, full source snapshot, exact settings, input
digests, RNG and raw predictions saved in `runs/capabilities_v1/reference`. CPU2
threads; evaluation cap20 minutes; no GPU required for this small inference pass.
Full CPU tests cap10 minutes separately. Record CPU time/peak process RSS; prior
40MiB visual GPU inference measurement remains historical, not whole-agent budget
validation. No file overwrites. Resume renders cached results after source/input
identity checks, without reevaluating. Future comparisons require identical protocol
and per-case input identity, while checkpoint hashes may differ explicitly.

Essential RED checks: paired transformations cannot destroy the erased-history
control or mislabel reflections; missing/NaN metrics cannot pass; comparing changed
populations/protocols must fail. Then tiny separate-fixture check, full suite,
commit working code before formal evaluation, inspect raw metrics/artifact hashes
and cached resume. Browser QA currently unavailable: report structural-only.

Claude review is public conceptual only, bounded to one brief; local verification
remains authoritative. Preserve review receipts and reconcile any material issue.

## Pre-execution review and implementation clarification

Claude's abstract review raised conditional exposure, scope and readout-inference
risks. Adopt explicit reused-population labels, checkpoint provenance and negative
result wording; the review did not inspect code and does not establish those defects.
One review round, no private source/results exported. All13 checkpoint roles load
strictly. Tiny development-only paths pass, including state, relations, source
feedback, planning, all modality encoders and CNN/temporal references. These use
validation data or separate seeds106xxx, not the frozen new scoring seeds.

Add the existing variable-candidate matching development set (256 examples) to
measure known/new decisions and false merges/splits, beyond the fresh two-entity
matching screen. This reused population is explicitly a regression diagnostic.
The CNN and temporal references are compatible: evaluate first16 real test frames
and first16 history2/horizon3 windows using the actual prepared path; no alias change.
Input sensitivity tests use fixed equal-length contrasting inputs in all four
encoders. Raw text token IDs and image/audio/video arrays remain inspectable.

No threshold is added after scoring. Methodological gates are diagnostic targets,
not estimates of general competence. Failure to beat a baseline describes the
observed deployed path; it does not prove all underlying representations are useless.

## Completed baseline

Implementation/source895cc86; no scoring or weight-selection deviations. All13
roles load strictly.183 CPU tests pass in167.56s. The formal evaluation finishes in
211.15s with842817536 bytes peak process RSS (803.77MiB); this includes loading and
retaining several model roles and data, not a single-agent deployment footprint.

[Standalone report](../runs/capabilities_v1/reference/report.html),
[raw scorecard](../runs/capabilities_v1/reference/capabilities.json),
[independent verification](../runs/capabilities_v1/verification.json).
The34 case rows comprise21 passing diagnostic screens,11 failing screens and2
measurements without an aggregate gate. These counts are coverage, not an overall
capability score. Ten other abilities/budget claims have explicit coverage gaps.

| Path and evaluated scope | Baseline |
| --- | --- |
| Constructed visual memory: original, reflection, half brightness, noise |64/64 correct per condition;32/32 complete pairs; erased controls50% |
| Same checkpoint, red/blue channel swap |31/64 correct (48.4375%);0 complete pairs |
| Same checkpoint, grayscale |32/64 correct;0 complete pairs |
| Same checkpoint, other modalities and actions |Constant0.5 RGB, zero audio, no generated content tokens; tested text/audio/video encoder contrasts and opposite actions cause zero feature/state change |
| Direct structured facts, reused32 combinations |32/32 entity and location answers |
| Through-agent structured facts, same32 combinations |0/32 entities,24/32 locations |
| Historical recall, reused15 episodes |0 answered, all abstained; forced factual accuracy20%; cost0.25 equals all-abstain |
| Instruction operation selection,16 toy test templates |2/16 correct; lexical reference10/16; enforced output-contract violations0 |
| Variable candidate identity matching |256/256 on reused development descriptors; fresh two-candidate matching16/16 |
| State updates, six temporal/no-information/idle conditions |128/128 entity pairs per condition; runtime readout, retry and restore agreement pass |
| Learned relation addressing/interaction |48/48 three-state outcomes for each non-erased condition; erased relation rejects48/48 and state accuracy50% |
| Static source feedback |Useful source15/16 worlds; utility0.769775 vs stop0.733154; accuracy79.05% |
| Source changes |Window adaptation improves drift utility but harms static utility; all policy traces retained, no universal reliability claim |
| Supplied-mechanics key planning, ordinary and relocation |96/96 reachable and32/32 absent per condition;192/192 relocation correction reads |
| Planner objective alignment |Still fails: selects expected reported utility0.10 while stop would give0.85 |
| Separate short-trained perception reference |RGB MSE0.050805 vs gray0.234020; pose MSE0.014935 |
| General belief-agent prediction, synthetic and real |Image MSE0.236598 vs copy0.005635; real0.237103 vs copy0.000071984; synthetic audio also loses to silence |
| Separate learned temporal reference |RGB MSE0.050845 vs copy0.000461863 |

The learned planning core's ordinary memory utility advantage is0.0314453;
relocation utility0.896875 is below no-history0.909375. Its task-success pass still
does not demonstrate universally useful memory or learned transition planning.

Checksums and primary scores were independently recomputed for all34 cases from
raw logits, targets, arrays or execution/feedback traces. The independent verifier
initially assigned a feedback fee to the no-feedback policy; correcting that
verifier assumption produced exact agreement with the existing explicit fee contract.
No model or measured result was changed. Cached resume leaves raw results and the
report hash unchanged; same-run comparisons give zero deltas and changed populations
are rejected. Reports have structural verification only; browser QA remains unavailable
under the existing restriction. Prior media decode receipts are historical evidence;
this run adds actual image-input/decoder tests, not labeled real-scene memory tests.

### Repeat and compare

Run from the repository with its installed environment, using a fresh output path:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python experiments/multimodal.py \
  --dataset capabilities --check

OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python experiments/multimodal.py \
  --dataset capabilities --output runs/my_capability_check \
  --baseline-reference runs/capabilities_v1/reference

# Cached report regeneration; no repeated model evaluation.
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python experiments/multimodal.py \
  --resume runs/capabilities_v1/reference
```

Use `--capability-weights /path/to/weights.json` to test changed weights. That JSON
maps explicit roles to compatible checkpoint files, for example:

```json
{"visual": "runs/my_new_visual_model/weights.pt"}
```

Unspecified roles keep the original checkpoint paths. Roles with ordinary agent
checkpoints require the accompanying `run.json` settings and the existing schema;
architecture changes need an explicit constructor change. The suite never selects
a winner or fits readouts. Reusing these cases establishes a regression comparison;
reserve a fresh separately declared evaluation for generalization claims. A changed
population, evaluator protocol or metric set cannot silently become a comparable
number. New capabilities require a protocol extension while retaining the v1 cases.

Software verification can be attached with `--capability-software PATH`: this
baseline's `runs/capabilities_v1/software.json` binds its full test run to exact
source hashes and includes per-module counts. It is rejected after relevant source
changes. Raw behavior and software evidence remain separate. `reference/last.pt`
is an evaluation snapshot containing all13 named model roles; it is not a newly
trained unified agent. The original visual `weights.pt` remains unchanged.

Next work should address the measured failures, with this baseline frozen: broaden
visual identity beyond a color circuit, repair useful through-agent recall, and
train/evaluate action-conditioned predictions before combining them with planning.
Real webcam identity/memory still needs a pixel-to-entity path and reviewed temporal
answer keys. Speech, general writing, generative media, software execution, learned
concept/topology formation and a joint GPU budget remain explicitly unestablished.
