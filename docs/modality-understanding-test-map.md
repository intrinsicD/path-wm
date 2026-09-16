# Multimodal capability suites and diagnostic contracts

16 September 2026. User requests the same capability-oriented test structure for
all modalities, with diagnostics that guide repairs. This document specifies the
proposed scope. It does not implement a runner, report new scores or promise a
complete test of unrestricted understanding. Each executable suite must declare
its supported domain, evidence requirements, gates and budget before running.

Use the existing PyTorch recipes, Run records and standalone reports. Extend
ordinary functions in `pathwm/evaluation/` when a concrete test needs them. Avoid
a second trainer, registry or experiment framework. The shared latent reasoning
design remains unchanged; language is not the mandatory intermediate format.

## Three distinct levels of evidence

1. **Mechanics:** valid inputs, masks, shapes, gradients, timing, provenance,
   persistence, causal access and deterministic restore where supported.
2. **Capabilities:** correct content, relations, temporal structure and
   generalization on independently held-out tasks.
3. **Integration:** useful content survives the actual input, shared-core, memory,
   retrieval and output paths, including complementary or conflicting modalities.

Codec reconstruction, semantic understanding and output generation have separate
scores. A passed interface test or four-example fit cannot pass a capability row.
Task reasoning belongs to the shared core; output tests assess whether each
decoder/adapter expresses the requested contents faithfully and with adequate
modality-specific quality.

## Per-modality targets

| Family / case prefix | Input understanding and core use | Output tests | Needed diagnostic examples |
| --- | --- | --- | --- |
| Text / TXT | Entity/attribute binding, subject-object roles, negation, quantities, reference resolution, paraphrases, temporal updates, instructions and dialogue corrections; new combinations and longer contexts | Free generation preserving supplied/retrieved facts, requested completeness, coherent multi-turn references, uncertainty and stopping; fluency separately | Paired sentences differing in one decisive word; token spans, answer facts, evidence refs, generated text and omitted/invented facts |
| Image / IMG | Objects, instance identity, color/shape, spatial relations, counts, fine detail and text when in scope; viewpoint, scale, illumination and occlusion transfer | Reconstruction detail/color/artifacts separately from state-conditioned content/layout/identity and generation with no source image | Original, reconstruction, error maps, stage probes, region/attribute errors and changed-context outputs |
| Audio / AUD | Sound events, temporal order, rhythm/pitch/duration, source separation where supported; speech content, speaker continuity and prosody as separate tasks; noise and variable rates | Waveform/spectral quality for sound; intelligibility, requested content, pronunciation and continuity for speech; generation without source waveform | Playback, waveform, spectrogram, event timings, transcripts when available, missing-source and noise controls |
| Video / VID | Motion, camera/object distinction, identity through occlusion, state changes, interactions, event order, history and future prediction | Observed reconstruction separately from future/action-conditioned generation; temporal consistency, identity, motion and event correctness | Synchronized original/output clips, tracks, timestamped state changes, future-target separation, prediction errors by horizon |
| Structured input/action / VEC-ACT | Units, schema, missing values, timestamps and action/state association; observable effects for the declared environment | Valid typed commands, arguments, constraints, measured outcomes and completion | Input values, state snapshots, proposal/execution/result links, invalid-command and no-effect cases |

The video family is expanded in [the video map](video-understanding-test-map.md).
Speech, music, OCR, metric 3D geometry, tool execution and physical control require
their own declared data/interfaces before they can be scored. Unsupported items
remain explicitly unsupported rather than silently omitted or counted as failures
of a different capability. No speech or music claim follows from tone fitting.

## Shared-core and multimodal cases

Use the same underlying task facts across input variants where appropriate:

- Each modality alone; supported input/output combinations; simultaneous outputs.
- Equivalent evidence in different modalities: preserve the same task-relevant
  answer, not necessarily identical latent vectors.
- Complementary evidence: one modality alone cannot supply all requested facts.
  The combined prediction must use both, with omission controls.
- Conflicting evidence: retain source, time and uncertainty. Targets specify which
  observations are reliable; do not assume the modality with the strongest scores
  is always authoritative.
- Missing, delayed, irrelevant or noisy modalities; misaligned audio/video and
  misleading textual hints. Confirm that one easy input does not hide failure of
  another, and that text cannot supply visual test answers by leakage.
- Persistent identity and state updates, delayed retrieval, stale versus current
  information, remembered versus newly observed versus generated content.
- Predictions and actions grounded in that state; agreement between simultaneous
  textual, visual and auditory outputs, including timing when relevant.
- Novel object/property/action combinations, new sources and longer delays;
  few-shot corrections through memory separately from gradient-based adaptation.

## Diagnose the path, not only the final score

Inspect the actual path supported by a test. The following is a logical checklist,
not a claim that every configuration already traverses these stages:

`preprocessing -> encoder scales -> temporal/fusion/binding adapter -> belief/core`

`core <-> memory/retrieval -> task workspace -> modality readout -> decoder/output`

Use fixed task targets, source splits and declared readout budgets across stages.
Feature shapes may differ; record readout capacity and avoid comparing arbitrary
latent MSE between different spaces. Capture detached traces only for selected
examples; tracing must not alter outputs, gradients or random-number streams.

| Diagnostic | What it can establish | Limit |
| --- | --- | --- |
| Input/annotation inspection and task-specific raw-data reference | Required evidence is present, labels/times/units align | A failed reference does not prove the input is insufficient |
| Frozen stage probes | Target information is accessible to the specified reader at a stage | Probe failure does not prove information destruction; probe training has its own generalization failures |
| Actual downstream task output | The deployed path uses available information correctly | A final failure alone cannot identify the broken component |
| Matched direct-feature versus core/memory path | Extra transitions are associated with changes in task performance | Match training exposure and reader capacity; this is not automatic proof of a unique causal bottleneck |
| No/wrong/reordered context or history | Performance depends on relevant evidence when controls are properly matched | Distribution shifts from the control itself can affect performance |
| Train/evaluate decoder on supplied correct task state | Whether the output path can express the required content in that condition | Supplied-state success is not full-agent success; input-format differences can confound attribution |
| Frozen component refit or bypass | A bounded intervention improves a suspected transition | Only claim the intervention effect supported by that comparison |

For example, readable object identity at the encoder plus poor actual core answers
and a failing core probe should be reported as **a problem localized to the
intermediate path under these readers; cause unresolved**. It should not report
"the updater erased identity" without a controlled intervention. Attention maps,
decoded latent images and confident text are inspection aids, not causal proof.

## Common result and report contract

Every scored case should expose, using the existing run artifacts:

- Stable case ID/version, intended ability, modality, supported configuration and
  task target; model/checkpoint/code identity, preprocessing and source split.
- Independent status for implementation, execution and capability assessment.
  Distinguish `not_implemented`, `not_run`, `unsupported`, `blocked`, execution
  error, measured failure and scoped pass. Report coverage; unrun tests cannot
  contribute passes or disappear from the denominator.
- Primary metric, declared baseline and threshold, sample/source counts,
  per-group failures, seed variation and confidence intervals where appropriate.
  Distinct roles for development data, confirmation and old-source preservation.
- Failure examples with expected/actual outputs and evidence; stage probe/control
  outcomes; suspected transition, alternative explanations and a bounded next
  diagnostic. Explicitly identify unmeasured stages.
- Parameter counts, training exposure, evaluation time, peak memory and inference
  latency; diagnostic overhead separately from ordinary inference.
- Raw JSON/arrays and inspectable text/images/audio/video inside the usual
  standalone `report.html`; report-generation status separate from run completion.

Thresholds must precede results. Review them only in a new versioned protocol,
never to relabel an old failure. A broad suite is a coverage map with individual
gates, not one averaged score that conceals missing or failed capabilities.

## Reuse and current coverage

- `tests/test_modality_foundation.py`, `tests/test_multimodal.py`, World State and
  recurrent-readout tests already cover parts of mechanics and tracing. They do
  not establish general language, speech, image or video understanding.
- `experiments.modality_audit` and `pathwm/evaluation/modality_audit.py` already
  support isolated tiny fits and real-input transport reports. Four words/tones
  are fitting checks, not held-out language/speech tests.
- `experiments.modality_readout` and `pathwm/evaluation/modality_readout.py` already
  provide stage capture, factor probes, supplied-state controls, output scoring
  and panels on controlled tasks. Extend these functions where their contracts fit;
  retain explicit scope for synthetic/template-derived scores.
- Spatial VAE/photo recipes and video recipes supply complementary codec and
  temporal diagnostics. The standalone spatial VAE and the agent's native visual
  encoder are distinct configurations; results must not be transferred silently.
- Existing records report mixed and often failed capability screens. Preserve
  them. This map creates no new pass and does not supersede earlier negative results.

## Bounded rollout

1. Inventory existing checks against the matrix, with exact scope/evidence and
   visible missing cases. Retain separate mechanics/capability/integration status.
2. Implement one small held-out capability path per modality using the existing
   recipes and reports; use real examples plus controlled contrast cases. Keep the
   currently selected video work visible rather than silently reprioritizing it.
3. For those same targets, compare encoder access, shared-core use and output;
   add memory and mixed-input tests without changing the task's meaning.
4. Run a small regression subset after relevant changes. Run slower dataset
   evaluations at milestones; fit expensive probes or refits only for a diagnosed
   gap, with a separate fixed budget. Record full coverage even when runs are skipped.
5. Each repair has a primary outcome, old-source/modality preservation gates and
   a stop rule. Passing closes that scoped item; repeated failure triggers review
   of the overall bottleneck rather than indefinite local parameter changes.

The initial suite is complete only relative to a selected, explicit capability
contract. New supported capabilities add new versioned cases. No finite suite
certifies unrestricted understanding of an entire modality.
