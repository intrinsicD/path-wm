# Architecture discussion and validation checklist

Red = a dedicated discussion remains. Blue = discussed, with validation still incomplete. Green = validated within the explicitly labelled test scope, not general capability. Discussion coverage is retained separately, including for green parts that still need a walkthrough. Evidence refers to the recorded configurations and must be revisited after relevant changes.

Last updated: 2026-09-15. [Colored overview](architecture-atlas.html#01-overview).

The user requested red for parts still to discuss and green for validated parts. After substantive discussion or validation, update the matching coverage and evidence in `docs/diagrams/architecture-atlas.json`, then regenerate the atlas. Green requires a stated scope, limits and passing evidence for that scope; an overall failed experiment can support only an independently passing subcheck. Remove or revise green when that evidence no longer applies. An assistant-only explanation does not automatically complete a discussion. User corrections override the discussion assessment.

| Part | Discussion coverage | Conversation basis | Remaining walkthrough / follow-up |
| --- | --- | --- | --- |
| World / user / environment | Discussed | Discussed webcam footage, real versus synthetic data, and desired image/video/audio/text interaction. | Revisit concrete sensor and environment choices when integrating them. |
| Observation adapters | To discuss | Desired inputs were discussed; the observation-adapter/event contract has only appeared in the architecture map. | Walk through timestamps, modality synchronization, missing/late packets and event commits. |
| Modality encoders → §2–3 | Discussed | Discussed modality stems, residual transformer blocks per scale, hierarchy merges, fusion, temporal handling and reconstruction probes. On 14 September, the user asked whether architectural structure and a small parameter count can reduce visual training-data requirements. On 15 September, the user specified a spatial VAE with separate lossless rearrangement, residual processing and channel projection; Claude review was reconciled. | Opt-in spatial VAE v2 now implements the R/P/M/C stem, coarsest self-attention/shared loop and detached stage probes. Mechanics, exact GPU resume and ten real-photo learning sanity gates pass; all photo-quality screens fail. See docs/spatial-vae-v2-plan.md and runs/spatial_vae_v2/verification.json. Color diagnosis now finds reduced readout access at the final posterior projection and sampling; see docs/spatial-vae-color-plan.md. Current agent integration, semantic retention, fine detail and matched-resource/data-efficiency studies remain open; the high-level encoder stays blue. |
| Predict and correct → §4 | To discuss | Discussed latent world/entity beliefs conceptually, but not the actual categorical predict/correct mechanism with h, codes and logits. | Walk through one observation: prior prediction, posterior correction, uncertainty and recurrent state. |
| Session memory → §5 | Discussed | Discussed recent inputs, hierarchical history, external entity storage, retrieval, latent versus raw evidence and provenance. | Detailed retention/compression policy can be revisited without treating the whole topic as untouched. |
| Task workspace → §6 | Discussed | Discussed focus on entities and relationships, recalled context, internal thinking and interleaving internal/external actions. | A detailed stopping/budget policy remains open. |
| Task request | To discuss | Discussed broad goals, complex tasks and decomposition; the instruction-to-task contract has not had a dedicated walkthrough. | Define task tokens, output controls, completion criteria and who verifies success. |
| Optional bounded planner → §11 | Discussed | Discussed action DAGs, predicted states, goal reachability, subgoals, new observations and replanning. | General DAG construction and learned search remain proposed, even though the topic has been discussed. |
| Modality outputs → §7 | Discussed | Discussed every encoder/decoder pair, state-conditioned outputs, image detail, conditional generation and replacement interfaces. On 15 September, discussed a mirrored PixelShuffle VAE decoder, Gaussian loss/KL units and a future state-conditioned spatial-latent producer. | Separate v1/v2 VAE decoders support exact saved-latent replay and variable geometry; v2 has a latent-only convolution/PixelShuffle decoder with final RGB projection. Ten photo fits still fail the quality screen; rates and common-target stage probes are measured in docs/spatial-vae-v2-plan.md. Constant latent controls locate decoder grids at learned subpixel phase channels; mean-only decoder fitting improves color but fails sampled-output reliability. See docs/spatial-vae-color-plan.md. State-conditioned generation, fine detail and other modalities remain open; this high-level output node stays blue. |
| Action proposal | To discuss | Discussed examples of physical/software actions and internal/external distinctions; the concrete action/execution interface has not been walked through. | Connect action-head outputs to typed tool/physical commands, durations, execution outcomes and feedback. |
| Generated-content reflection | Discussed | Discussed reusing modality encoders for recalled/generated material while routing it into focus rather than treating it as a new observation. | Distinguish the implemented generated-output reflection path from proposed raw-memory replay. |

## Green: validated scopes

### Observation adapters

Packet deduplication/order, one action/time advance per event, commit-once and masked-input handling.

Still open: Live webcam capture, cross-device synchronization and real streaming are not validated. Discussion walkthrough still pending.

Evidence: [Belief implementation verification (69 CPU tests)](../runs/belief_v1/verification.json), [Event, masking and timestamp checks](../tests/test_belief.py).

### Session memory → §5

Exact recent envelopes, bounded hierarchy, source/belief separation, causal eligibility and state roundtrips.

Still open: Learned compression quality and reliable long-horizon recall are not validated; the photo recall path still loses accessible detail.

Evidence: [Belief implementation verification (69 CPU tests)](../runs/belief_v1/verification.json), [Memory bounds, provenance and causal-read checks](../tests/test_belief.py), [Photo recall limitations](../docs/photo-detail-plan.md).

### Optional bounded planner → §11

Fixed-horizon candidate evaluation, common sampling, caller-state preservation and RNG restoration; supplied key-box search has independent replay.

Still open: This does not validate learned dynamics, general goal achievement or the proposed action DAG. The linked initial key-box integrated capability screen failed.

Evidence: [Belief implementation verification (69 CPU tests)](../runs/belief_v1/verification.json), [Candidate rejection and state preservation](../tests/test_data_and_planning.py), [Independent search replay; integrated screen failed](../runs/key_box_v1/verification.json).

### Generated-content reflection

Re-encoded generated outputs can change workspace while leaving physical belief/history unchanged; generated content cannot enter source evidence.

Still open: Useful self-reflection and quality of generated content are not established by these routing tests.

Evidence: [Belief implementation verification (69 CPU tests)](../runs/belief_v1/verification.json), [Reflection boundary check](../tests/test_belief.py), [Loopback and provenance checks](../tests/test_tasks.py).
