# Architecture discussion checklist

Working assessment from our visible conversation. Red means a dedicated walkthrough is still needed, including topics mentioned only at a broad level. Blue means discussed, not implemented, agreed in every detail, or validated.

Last updated: 2026-09-14. [Colored overview](architecture-atlas.html#01-overview).

The user requested red for parts still to discuss and a color update after we discuss them. After a substantive exchange, update the matching status and its conversation evidence in `docs/diagrams/architecture-atlas.json`, then regenerate the atlas. An assistant-only diagram or explanation does not automatically count as a completed discussion. User corrections override this initial assessment.

| Part | Discussion coverage | Conversation basis | Remaining walkthrough / follow-up |
| --- | --- | --- | --- |
| World / user / environment | Discussed | Discussed webcam footage, real versus synthetic data, and desired image/video/audio/text interaction. | Revisit concrete sensor and environment choices when integrating them. |
| Observation adapters | To discuss | Desired inputs were discussed; the observation-adapter/event contract has only appeared in the architecture map. | Walk through timestamps, modality synchronization, missing/late packets and event commits. |
| Modality encoders → §2–3 | Discussed | Discussed modality stems, residual transformer blocks per scale, hierarchy merges, fusion, temporal handling and reconstruction probes. | Architecture/training improvements remain experimental; discussion coverage does not close them. |
| Predict and correct → §4 | To discuss | Discussed latent world/entity beliefs conceptually, but not the actual categorical predict/correct mechanism with h, codes and logits. | Walk through one observation: prior prediction, posterior correction, uncertainty and recurrent state. |
| Session memory → §5 | Discussed | Discussed recent inputs, hierarchical history, external entity storage, retrieval, latent versus raw evidence and provenance. | Detailed retention/compression policy can be revisited without treating the whole topic as untouched. |
| Task workspace → §6 | Discussed | Discussed focus on entities and relationships, recalled context, internal thinking and interleaving internal/external actions. | A detailed stopping/budget policy remains open. |
| Task request | To discuss | Discussed broad goals, complex tasks and decomposition; the instruction-to-task contract has not had a dedicated walkthrough. | Define task tokens, output controls, completion criteria and who verifies success. |
| Optional bounded planner → §11 | Discussed | Discussed action DAGs, predicted states, goal reachability, subgoals, new observations and replanning. | General DAG construction and learned search remain proposed, even though the topic has been discussed. |
| Modality outputs → §7 | Discussed | Discussed every encoder/decoder pair, state-conditioned outputs, image detail, conditional generation and replacement interfaces. | General output quality and capability still need learning and evaluation. |
| Action proposal | To discuss | Discussed examples of physical/software actions and internal/external distinctions; the concrete action/execution interface has not been walked through. | Connect action-head outputs to typed tool/physical commands, durations, execution outcomes and feedback. |
| Generated-content reflection | Discussed | Discussed reusing modality encoders for recalled/generated material while routing it into focus rather than treating it as a new observation. | Distinguish the implemented generated-output reflection path from proposed raw-memory replay. |
