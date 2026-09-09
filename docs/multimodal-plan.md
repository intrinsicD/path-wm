# One multimodal world model

Status: implementation in progress. Authorized 9 September 2026.

Build one small, inspectable trainable system with explicit modality adapters,
structured persistent latent state, episodic memory, internal computation,
stochastic dynamics, action proposals, planning and bounded learning proposals.
The architecture is an experimental starting point, not a claim of intelligence,
physical understanding, semantic entity slots or calibrated uncertainty.

## Concrete interfaces

- Ordinary PyTorch modules constructed directly in `experiments/multimodal.py`.
- Timed, masked image/video, waveform-chunk, byte-text and optional vector inputs.
- Spatial/entity/context/working/reasoning token groups with explicit sizes.
- `observe`, `remember`, `think`, `imagine`, `decode`, `decode_video` operations.
- Caller-owned states/memory; no hidden episode state in model parameters.
- Observation cutoff checks; causal text decoding; imagined branches cannot become
  observed memories. Thinking does not advance environmental time.
- Image/waveform/text generation, probabilistic next-state prediction and bounded
  action proposals. Planning requires explicit candidates, bounds and a cost.
- Optional detached traces of attention, states, retrieval and data flow, plus
  token intervention. Traces are measurements, not semantic explanations.
- Explicit prediction-error supervision and a bounded candidate-update gate with
  rollback. It is an engineered learning loop, not autonomous code modification.

## Training and scope

One readable recipe owns objective, construction, datasets, evaluation and loop;
reuse `Run`, checkpoint/resume and the existing standalone report renderer. No
registry, new orchestration framework, external model downloads or large job.
Train data-space reconstruction and future generation; use a checkpointed EMA
teacher for latent targets, with anti-collapse regularization and raw-space
baselines. Target observations never enter imagined inference. The teacher is a
training copy of the same architecture, not a second deployed model.

Use generated synchronized moving-shape/impact-tone/byte-description fixtures for
all-modality development. Clearly identify their artificial origin and vocabulary.
Also run a short path on existing real PushT observations/actions, with audio and
text absent. No fabricated audio/text labels for real data. Development populations
are explicitly limited; no scientific pass threshold or generalization claim.

Initial budget: CPU only, width 32, 16x16 synthetic images, short waveforms, 8
synthetic updates plus an exact pause/resume comparison; up to 4 real PushT updates.
Additional tiny tests may exercise learning gates and contract failures. No GPU
training or long run is queued. Changes to this budget are recorded below.

## Acceptance

1. First demonstrate failing tests for the new interfaces and commit this red state.
2. Verify masking, time alignment, future rejection, autoregressive causality,
   gradients for every modality, learned dynamics/action conditioning, separate
   thinking time, memory provenance, branch isolation and safe state round-trips.
3. Verify planner choice against an independently calculable toy transition;
   rejection of invalid costs/bounds; rollback and acceptance of learning proposals.
4. Verify the same-runtime full/resumed path including model, EMA, replay priorities,
   optimizer and RNG. Preserve reporting failures and overwrite protection.
5. Complete small synthetic and real-data runs with raw metrics, source snapshots,
   safe checkpoints and self-contained reports. Save multimodal outputs and traces.
6. Run the complete CPU suite, update model/user guides and project state, commit.

## Review and evidence

An actual Claude review is requested using only a generic public design brief,
without code, measurements or private data. Exact receipts belong in
`runs/reviews/multimodal_architecture/`. Adopted corrections and local evidence are
recorded here as implementation progresses.
