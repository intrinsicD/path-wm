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
