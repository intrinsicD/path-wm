# Next controlled experiment: temporal correspondence beyond recording context

Product: ABI-v2 evidence for the modality-neutral belief and a replaceable H1 interface.
Research: E1_common_base development only; no H1 or formal freeze claim.

The measured baseline is the two 10,000-update R1 continuations from checksum-bound R0 sources.
Keep their original gates and all-clip timing/future audits unchanged. The current run is not a
basis for beginning B0 until its unresolved temporal correspondence is understood.

Question: can the existing frontend and optimizer recover a known shared temporal signal, and
does supervising both aligned times improve the balanced assignment beyond the current-anchor loss?
The constructed metric counterexample proves that these objectives enforce different constraints;
it does not prove which one limits the trained model on TAU.

1. Build a small deterministic A/V source with a declared physical common cause (for example, a
   moving visible oscillator whose continuous speed controls sound amplitude/frequency). Render
   from that physical state; never encode class names, split IDs, absolute clip time, or evaluation
   targets as sensor channels. Keep distinct trajectory seeds and recording backgrounds across
   train/eval, and record the exact source bytes and sampling clock.
2. Validate that the raw observations contain the shared change using a fixed physical readout,
   and that shifting one modality removes correspondence. A constant-signal construction must
   have zero balanced-time evidence. These are source/metric controls, not trained-model success.
3. Compare exactly one representation-objective component: the present current-anchor ranking
   against a balanced two-time assignment loss that supervises both aligned pairs. Preserve the
   retrieval term, model, common source initialization, other losses, temperature, batch order,
   two seeds and update budget. Write config/interface and essential invariant tests before code.
   Use an explicitly R1-only loss choice in the training config to preserve the exact source
   representation/model/data binding. The balanced margin is dot(v_t - v_s, a_t - a_s) for unit
   embeddings; swapping the time assignment in one modality reverses its sign.
4. Before running, commit the budget and decision: assess held-out balanced-time accuracy with
   recording-group intervals, paired improvement, non-collapse and teacher-copy future controls.
   Check both absolute competence on the known signal and improvement over the matched objective.
   Treat failure on the controlled source as a model/optimization question, not evidence that TAU
   lacks timing. Treat success there plus failure on TAU as motivation for a separate data audit,
   not proof of a unique cause.
5. Only after that thin experiment works, compare the same single change on TAU from the original
   passing R0 sources at the same 10,000-update budget and seeds. Keep the present audit cohort
   for development comparison, disclose previous inspection, and obtain a fresh untouched cohort
   or source before making a confirmatory claim. Preserve unsuccessful outcomes.

This plan is prepared for the next session; its implementation and training remain future work.
