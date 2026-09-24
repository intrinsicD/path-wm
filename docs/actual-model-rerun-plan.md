# Actual-model reruns and missing implementation inventory

24 September 2026. User instruction: test the existing real model; any preliminary
downscale must be followed by the actual full configuration. Missing functionality
must be planned together with Alex before implementation. These requirements are
now in CLAUDE.md and the mandatory experiment workflow.

## Scope and completion criteria

Audit all107 active test files (689 named functions before pytest parametrization)
and the active experiment recipes. Inventory is saved at
`runs/reviews/actual_model_reruns_20260924/test-inventory.json`. Distinguish actual
architecture at reduced dimensions, deterministic/fault-injection unit fixtures,
separate surrogate models, and real components without a complete trained pipeline.
Keep each uncovered obligation visible; a rerun of the unchanged866-test suite
would not close this request. Preserve all previous results.

Do not write a replacement model, new trainer or speculative consumer to make a
rerun possible. Test harness changes may exercise already implemented behavior.
Scientific quality reruns need the actual compatible checkpoint and existing input/
output path. Missing functionality or an undefined full-model configuration must
be discussed before implementation; random weights cannot substitute for trained
quality, and larger dimensions alone do not establish a production configuration.

## Existing references and first executable reruns

- Main categorical multiscale recipe: `experiments.multimodal.build_model`, with
  actual encoders/updater/dynamics/Thinker/decoders/TaskPolicy. Constructor default
  width32, context16, latent8×8, evidence8 and memory32/8/16. These are existing
  configured modules, not a claim of a fully trained deployment checkpoint.
- Existing resource-reference configuration: `experiments.token_budget.build_model`,
  width32 and the same categorical state/memory sizes, image256 and video4×64 in
  its existing encoder comparison. Reuse that recipe, seed71, FP32, no neural updates,
  normal two warmups/seven repeats. Scope: actual-graph execution/resource check,
  not reconstruction quality or closure of every reduced unit test.
- Existing R2 integrated configuration: `experiments.unified_session.build`, width64,
  full `SlotPerception` (seven slots/three iterations, native RGB64), shared main
  multiscale image encoder. Use the existing provisional J perception+identity
  checkpoint `runs/latent_agent_r1/identity_joint_20260923`, and existing
  `--identity-run` path to replace pixel-histogram identity fixtures. The remaining
  R2 core/belief/action/value modules are untrained in this recipe; state that limit.
  Four scenes, seed0, unchanged recipe and existing restart/verification behavior.
- Rerun reusable integration assertions with actual constructed modules at documented
  reference settings where no implementation change is required. Record exactly
  which original assertions run and what fixtures remain; do not call partial
  substitution a complete full-model rerun.

## Known obligations requiring mapping, not silent replacement

| Earlier family | What was simplified | Required actual counterpart / disposition |
| --- | --- | --- |
| Evidence loop, affine transfer, nonlinear fidelity | Independent `DetailCodec`, supplied four-part identity, part-code memory and fixed pose outputs | Trace existing multiscale/slot/decoder and persistent detail interfaces. The original four-part API is not the full model. Equivalent acquisition/correction/transfer cannot be fabricated by swapping class names. |
| Lexical context retrieval | Standalone `LexicalPolicy`,24-token embeddings, supplied entity eligibility and exact copying | Check existing TaskPolicy/interpreter/context consumer wiring and trained checkpoint; missing integration must be planned. |
| Indexed lookup | Exact descriptors; map/session demonstrations with small agent shell | Reuse real store/context; separate descriptor/index correctness from perception-driven retrieval. |
| World-State foundation | Supplied7-value descriptors, small simple image/text adapters, auxiliary property heads | Actual multiscale observation/binding/update/context/Thinker path; descriptor task is not evidence of perception. |
| R1 oracle/slot-core and R2 identity | Supplied symbols/keys or random modules in software checks | Existing learned perception/key path can run; a fully trained unified checkpoint and unsupported tasks cannot be invented. |
| Reduced real-module unit tests | Small widths, image sizes and short memory; some custom stubs | Run existing real-graph counterparts where available; track outstanding same-assertion/full-size coverage separately. |

## Budget and evidence

No new training, tuning or architecture changes. One existing GPU profile invocation
<=10min/6GiB allocation; one CPU R2 life<=5min; actual-module correctness reruns
<=20min CPU total, two threads. Stop failed runs without scientific repairs or gate
relaxation. Keep process exit, source hashes, settings, logs, results and standalone
reports for recipe evaluations. Verification-only pytest jobs own logs/JUnit/results;
they are not scientific training runs. No fresh capability gates are selected.

## Status

Rules written; source/test inventory underway. No claim that all requested real-model
reruns are complete. Missing-path decisions will be presented to Alex before code.
