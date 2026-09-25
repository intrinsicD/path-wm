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

## Initial audit budget and evidence

No new experiment training, tuning or architecture changes. The rollback correctness
test makes two disposable optimizer proposals on the actual model; it produces no
trained checkpoint or quality claim. One existing GPU profile invocation
<=10min/6GiB allocation; one CPU R2 life<=5min; actual-module correctness reruns
<=20min CPU total, two threads. Stop failed runs without scientific repairs or gate
relaxation. Keep process exit, source hashes, settings, logs, results and standalone
reports for recipe evaluations. Verification-only pytest jobs own logs/JUnit/results;
they are not scientific training runs. No fresh capability gates are selected.

## Status

Rules committed in `a5279ee`. Static inventory covers107 files and689 named test
functions; manual counterpart mapping and all-model reruns remain incomplete.
Missing-path decisions below require joint planning before implementation.

### Authorized native visual-memory follow-up

Alex subsequently authorized Codex and actual Claude Opus 5.5 at medium effort to
plan, implement, review, test and repair the first native R2 visual-memory slice.
The [owning plan](real-visual-memory-plan.md) records that agreement, its separate
training budgets, and all failures. Items 1 and 2 below are therefore historical
planning questions for this selected composition, not still-unanswered blockers.

The existing RGB64 multiscale encoder, SlotPerception and broadcast decoder now
connect through source-validated persistent visual slots. Native persistence,
source withdrawal, restart and direct decoder parity pass. Joint6000, using the
original architecture and TRAIN-selected.90/.85/.10 policy, passes all six fixed
development populations2405–2410: relocation62,63,61,63,64,64/64; zero novelmerges;
all other task gates and contracts pass. Retention passesall12. All earlier
failures remain preserved.957 final CPU tests pass with no skips and unchanged source. Unused
confusable-pair augmentation is removed; all six final-source task reruns exactly
reproduce the qualified results. No new codec or
representation is introduced. See the owning plan and its aggregate report.

This closes the missing connection for those specific software contracts. It does
not close fine-detail fidelity, part-wise correction, pose/operator transfer,
lexical-context integration, full-agent learning, or every historical test family.

### Completed executable slice

Evidence directory: `runs/reviews/actual_model_reruns_20260924/`. Every invocation
has a log, command/exit/timing receipt and source manifest; all completed with exit0
and unchanged Python sources during execution.

| Check | Actual path exercised | Result and evidence |
| --- | --- | --- |
|11 belief +4 Gaussian assertions | Existing default-width32 multimodal builder; categorical memory32/8/16 | Passed; original assertion names are retained as pytest parameters |
|2 complete-event checks | Width32, full memory; eval and training modes | Passed; posterior, gradient and RNG equivalence |
|3 actual-model replacements | Full-capacity memory filling, real dynamics planning, whole-model guarded optimizer update | Passed; replaces small-capacity, Toy-transition and scalar-model coverage for these contracts |
|3 R1 module checks +1 query check | Native SlotPerception64/7slots/3iterations and LatentCore64/4code tokens | Passed; gradients, induction and query isolation |
|4 encoder checks | Actual image/video/audio/text multiscale encoders | Passed; exported scales and future masking; visual inputs64×64 |
|8 R1 runtime assertions | Actual native-size perception and core | Passed; restart, frozen weights, planner boundary and feedback contracts |
| Resource recipe | Existing width32 categorical graph; image256, video4×64, dense/packed/local arms | Completed49.7s; `profile.receipt.json`; report in `runs/actual_model_profile_20260924_v1/` |
| R2 session recipe | Native shared perception encoder and existing J learned identity checkpoint | Completed12.1s; restart equal,4/4 successful presses; `integrated.receipt.json`; report in `runs/actual_model_integrated_20260924_v1/` |

The **36 unique reference-configuration checks** live in
`tests/test_model_reference_configuration.py`; `assertions3.xml` records36 passes.
Earlier20/28-pass invocations are subsets, not additional tests. The combined
regression invocation recorded **105 passes** in39.36s (`regression.xml`):36 new
checks plus69 original adjacent unit checks. Those69 original checks are not
additional full-configuration reruns.

Limits: assertion reruns use untrained weights and crafted inputs. R1 feedback
assertions retain deliberate permissive policy thresholds and small scene sets
to exercise branches; they do not validate learned/default-policy performance.
The resource run establishes execution, not reconstruction quality. The J checkpoint
is provisional; R2's other core/belief/action/value modules remain untrained.
Both recipe reports passed structural/self-contained verification; neither has
browser visual QA (`report.qa.json`). The full original suite was not rerun in
this slice; library and recipe code were unchanged. No assertion here closes the
DetailCodec, lexical-policy or other missing-counterpart obligations above.

## Joint planning required before the remaining reruns

The list records the original audit proposals. The native R2 composition and
visual-memory connection (items 1–2) were subsequently authorized as described
above; other counterparts remain open. No substitute model is authorized.

1. **Pin the actual reference composition.** R2 already shares the real multiscale
   encoder with slot perception and supplies one session/store/core composition.
   The categorical multimodal recipe also remains active underneath these pieces.
   R2's existing J checkpoint loads perception and the identity key; its recipe
   explicitly leaves core/belief/action/value modules untrained. Agree which exact
   composition, checkpoint manifest and input/output tasks define the requested
   full-model evaluation. A random initialized full-size graph is suitable for
   software contracts, not trained quality. The reference question has been sent
   to Alex; no answer is assumed.
2. **Persistent visual detail → existing output consumer.** `DetailCodec`'s four
   part IDs,64-value codes, masked replacement and finite-pose decoder are its own
   contracts. The actual `SlotPerception` takes native RGB64, consumes the real
   `FeaturePyramid`, emits seven64-wide slots and uses a nonlinear broadcast decoder.
   `WorldStore` can store tensors, but that alone is not a source-validated consumer
   implementing the four-part recall task. Jointly specify which existing exported
   representation should be retained, how actual instance/region ownership is bound,
   and which existing decoder should consume the recalled values. Reuse the store,
   versioning and output modules; do not add another codec. Then implement only the
   missing connection and rerun acquisition, correction, old-detail preservation,
   restart, transfer and output fidelity through that path. The affine-range SVD
   diagnostic is intrinsically specific to the surrogate linear decoder; it cannot
   be transplanted as a bound on the existing nonlinear decoder.
3. **Context selection through the actual task path.** Main `build_model` constructs
   `TaskPolicy(width)` with `context_selector=None`; the standalone `LexicalPolicy`
   supplies a separate24-token embedding. Its100% lexical result is not a rerun on
   the real task model. Jointly specify query/candidate features from the existing
   interpreter/context path, wire the existing optional selector if needed, and
   agree training/evaluation data and budgets before adding that connection. Do not
   silently attach the old surrogate embedding or claim template copying is language.
4. **Oracle/fixture and historical experimental families.** R1 symbolic slots,
   hand-set concept thresholds, entity `Scorer` fixtures, simple foundation adapters,
   and separate spatial/video/image-readout experiments need explicit counterpart
   mapping against the chosen reference. Fault-injection and arithmetic unit tests
   remain useful but cannot stand in for successful learned operation. Record any
   unavailable full-model counterpart as open, never silently substitute another
   experimental model. The existing R2 life can run without supplied pixel identity,
   and that rerun is already available; it does not close every historical assertion.

The authorized first implementation slice is tracked in the
[real visual-memory plan](real-visual-memory-plan.md). Keep the remaining families
separate: each needs a concrete mapping and gap plan before adding functionality.
The surrogate affine-range result cannot be promoted into a bound on the native
nonlinear decoder, and a native software pass is not a learned-quality pass.

## Subsequent decoder-access comparison

Alex requested full pyramid access in the actual slot decoder and its retraining.
The [native decoder plan](native-pyramid-decoder-plan.md) records two matched seeds:
1.09–1.71% MSE gains versus retrained slots-only, below the20% gate. Software and
full-model execution pass (966CPUtests); the fine-detail quality gap remains open.
No surrogate codec, new encoder or default memory representation was adopted.
