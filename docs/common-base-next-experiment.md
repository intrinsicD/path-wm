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

Progress on 5 September: steps 1–2 completed. The two predeclared 5,000-update controlled R0
sources fail both rank floors and video temporal retrieval, so the conditional R1 comparison
did not run. The R1-only loss is implemented/tested in isolated branch
`dev/controlled-balanced-time` at `57876df`; no candidate training panel or improvement claim
is available. DDR §§38–40 preserve the outcomes. The original plan above remains the
conditional timing comparison, with the following prerequisite now taking priority.

## Next bounded step: physical diagnostics and a compact readout

The shared coordinates remain available in frozen tokens. Two fixed 192-feature projections
outperform 192-feature mean pooling for current and future coordinate readouts in both
modalities and both seeds, with matched fitted-head capacity. An exact-coordinate code also
fails the global rank floor. These findings require a diagnostic/readout experiment before
interpreting the failed R0 gate as missing sensory signal.

1. Define the physical-content and temporal-readout result/config contract and essential tests
   before production evaluation code. Include exact-coordinate, constant, shuffled-target and
   position-only controls. Distinguish current pose, change and future targets; current pose
   alone is not a complete state. Keep labels behind stop-gradient.
2. Compare one compact readout that retains token information across variable layouts against
   mean pooling, with the same frozen encoders, target set, declared parameter/update budgets,
   train-only normalization, two source seeds and recording-disjoint evaluation. The fixed
   projections are a diagnostic baseline, not a production ABI implementation.
3. Preserve original R0/TAU gates and both failed controlled checkpoints. Do not retroactively
   waive rank or temporal retrieval. Any different readiness criterion needs a new explicit
   design and fresh evidence; the current inspected cohort remains development-only.
4. Keep audio-objective diagnosis separate from the readout intervention. Audio current-state
   readout quality declined from initialization; positive teacher-copy gains do not resolve it.
5. Resume the R1 objective comparison only through a valid, explicitly declared prerequisite.
   Do not widen to TAU, bootstrap belief, or claim H1 from these source/readout controls.
