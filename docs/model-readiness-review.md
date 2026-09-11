# Whole-model readiness review — 11 September 2026

**Subsequent implementation:** [the key-box slice](key-box-integration-plan.md) now
connects entity-state retrieval to the belief-agent workspace and a supplied-dynamics
planner. Two iterations run end to end; the repaired model reaches95.83% goal success
but still fails its utility comparison. The review below records the pre-integration
assessment; general perception, learned dynamics and semantic compression remain open.

Assessment of current source and saved results; no new training or test-suite run.
Historical test counts in project-state are not current full-suite results.

## Verdict

An execution smoke test of the multimodal belief agent is already possible. A
meaningful integrated entity-memory-agent test still needs a connecting runtime and
training/evaluation recipe. Component success is not whole-model success. We do
not need to perfect drifting-source selection before building that test.

## What exists, and what remains open

| Area | Implemented | Remaining gap |
| --- | --- | --- |
| Multimodal perception | Image/video/audio/text adapters, reconstruction outputs | Grounded object candidates are not produced for the entity runtime; learned visual instance recognition is untested. |
| Belief core | Recurrent context, categorical prior/posterior, observation correction, thinking and imagination | Useful prediction and factual identity through the full core remain unproven. Saved tiny prediction runs lose to copy-last; warm event fact reader retains0/32 entity accuracy. |
| Session memory | Recent/staged/compressed/protected/consolidated stores; source/belief separation | Delayed useful recall and learned marking remain unproven. Default training history does not span default recent-memory capacity. |
| Entity memory | Descriptor matching, stable IDs, bounded allocation, snapshots, per-entity latent updates | Separate from BeliefAgent; consumes supplied normalized eight-value descriptors. Real candidate extraction, ambiguous tracking/correction and larger-scale retrieval remain open. |
| Relations | Learned stored keys, directed latent interaction, context write gate | One supplied relation type and explicit operations; no general concept hierarchy or learned graph topology. |
| Evidence acquisition | Controlled source choice, noise/drift/coverage experiments | Supplied outcome feedback and synthetic cues; not integrated agent sensing or learned input reliability. Latest coverage screen fails utility gain. |
| Tasks/actions | Task contracts, output heads, action proposals and imagined transitions | No demonstrated integrated observation→entity retrieval→decision→executed action→observed outcome success. |
| Persistence/debugging | Version/hash checks, bounded snapshots, reports and provenance | No unified entity+belief+task session persistence contract; learned latent semantics are not automatically interpretable or invertible. |

Code evidence: `experiments/multimodal.py:build_model` returns EntityMatchReader or
EntityReader/SharedEntityReader before constructing BeliefAgent. EntityMemory,
EntityStateMemory and EntityRelationMemory are distinct runtime wrappers. Neither
BeliefAgent nor MultimodalAgent invokes those wrappers. SourceChoice is used by
controlled evaluators, not the live agent observation/decision path.

Saved evidence: `runs/warm_encoder_v1/verification.json`,
`runs/entity_gate_v1/verification.json`, `runs/entity_source_coverage_v1/verification.json`;
see project-state for original population sizes, training budgets and limitations.
Earlier negative runs show that their configurations failed, not an impossibility
result for the architecture. Later descriptor-based successes do not repair or
localize the earlier agent-level failure.

## Minimum path to a complete controlled test

1. Define one integration boundary: observation candidates enter the entity store;
   retrieved entity states enter the agent workspace; agent requests explicit reads,
   updates or actions; committed outcomes update memory once. Define the authoritative
   owner of changing entity state, provenance, retries and joint session restoration.
2. Build one small episode runner and shared readout for that path, initially using
   the existing synthetic candidates. This avoids making visual discovery a prerequisite.
   Use two or three entities, movement/state changes, a delayed query and an interaction.
3. Train and probe the exact deployed path. Verify entity identity and state are
   recoverable at encoder output, agent state and final workspace. Fix the observed
   failure stage rather than assuming the successful standalone reader transfers.
4. Exercise memory eviction/compression and corrections inside episodes. Keep paired
   histories with identical final observations but different correct answers.
5. Freeze one complete configuration, then evaluate unseen episodes against no-memory,
   last-observation, wrong-entity and supplied-association diagnostic controls. Measure
   identity errors, state/query accuracy, action outcome, abstention, memory use and
   cost. Declare numerical thresholds and budgets before the new run; use separate
   development and final evaluation populations.

This would test one integrated entity-memory agent on controlled inputs. It would
not establish recognition of arbitrary people/objects or general world understanding.
For a raw-image test, candidate extraction and identity across views become additional
requirements. For a general learned knowledge graph, concept learning, relation
creation/deletion, multiple relation types and belief correction become requirements.
Those do not block the first controlled integration test.

Recommendation: prioritize integration and basic full-path learning now. Keep source
headroom/drift tuning as a separate experiment branch until that integration works.
