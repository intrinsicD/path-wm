# One multimodal world model

## Multiscale conditioned inputs — active slice

Request: all four input modalities expose fine-to-coarse processed feature scales,
with cross-scale attention and residual processing controlled by an agent/user code.
Review actual Claude first; public-concepts-only receipts are preserved under
`runs/reviews/multiscale_inputs/`. The first review is complete; reconciliation
addresses availability bounds, equal-time sequence causality and code provenance.

Implement three scales with shared width: image spatial grids, video space/time
grids, small audio waveform patches merged along sample order, text contiguous
token spans. A scale's conditioned residual transformer finishes before either
pooling or consumers can read it. Masked local pooling initializes the next scale;
optional cross-attention reads only its finished finer pooling footprint, followed
by the next scale's processor. Export all scales and their masks, availability,
support-end ordinals and grid sizes. Dense causal self-attention is acceptable for
these tiny inputs; spatially local support and semantic scale roles are not claimed.

One explicit shared code is produced from pre-observation working/reasoning state
by a replaceable controller, or replaced by user `feature_code [B,C]`. Bias-free
FiLM projections preserve zero-code neutrality after training, with bounded 0.1
scale/shift and live gradients at initialization. Conditioning availability is
state time for the controller, observation cutoff for the user override. No same-call
feedback or persistent hidden code. Existing simple custom encoders remain supported;
new multiscale adapters implement the same explicit pyramid/conditioning interface.

Essential checks: processed-before-consumed order, fine-to-coarse gradients and no
reverse edge; all four pyramids and code gradients; code neutrality/control after
an update; future perturbations, equal-time order, pooled support bounds, masked
NaNs/all-invalid rows, odd sizes and singleton inputs; future rejection before any
encoder; controller ordering; existing teacher detachment and exact resume.
Demonstrate red tests, commit plan/checks, implement, run the full CPU suite.

Development budget: existing CPU checks, two main synthetic updates (one uninterrupted
run plus pause/resume replay), two real PushT updates, small populations and no
learning proposals for these retained smoke runs. Preserve their checkpoints/raw
metrics/reports. Regenerate diagrams and inspect them locally. No GPU job or quality
comparison; no improvement claim until a separately declared matched experiment.

## Diagram follow-up

Completed slice: generate a compact architecture diagram from instantiated modules
and a data-flow diagram from actual values passed between recorded calls. No
profiling dashboard, training changes or hand-maintained graph connections.
Add a small reusable diagram writer and an export flag in the existing recipe.
Emit Mermaid and DOT sources, plus SVG/PNG when Graphviz is available. Save
source/config/input identities so unchanged inputs/code reproduce the diagrams.
The flow view describes recorded call boundaries, not all possible tensor paths.
CPU budget: one tiny synthetic inference example, repeated to verify determinism;
no training/evaluation job or report renderer change. Tests must catch missing fork
edges, repeated-call handling, module replacements and nondeterministic output.

Evidence: three diagram tests first failed on the missing module and were committed
in the red state. All 31 CPU tests now pass and the existing recipe's `--check`
passes forward/backward. Two independent `--diagram` executions produced identical
bytes for all nine JSON/Mermaid/DOT/SVG/PNG artifacts. Both PNGs were visually
inspected for readable labels, complete arrows and unclipped layout. The data-flow
view uses a top-to-bottom layout; module containment uses left-to-right expansion.
The generated source/renderer identities are in `docs/diagrams/diagrams.json`.
No new training run or report renderer change was required for this slice.

## Original implementation scope

Status: implementation and numerical checks complete; HTML visual QA is blocked
by the browser URL policy. Authorized 9 September 2026.

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

### Completed evidence

- 28 CPU tests pass: the 16 retained tests plus 12 multimodal/state/training tests.
  These cover modality gradients and extensibility, masks and temporal causality,
  text generation, state/memory provenance, precise clocks, branch isolation,
  known-transition planning, intervention, acceptance/rejection and exact resume.
- Both synthetic and real PushT forward/backward checks succeed. The EMA target
  receives no gradients; episodic memory has no trainable parameters by design.
- `runs/multimodal_v1/synthetic/`: eight main updates plus two accepted proposals,
  paused at update 3 and resumed. `synthetic_uninterrupted/` matches exactly in
  model/EMA/replay/counters, optimizer, Python/NumPy/Torch/sampler RNG and all
  training/proposal metric rows. Pause-boundary validation rows intentionally differ.
- `runs/multimodal_v1/pusht/`: four main updates, paused at update 2 and resumed;
  eight training and four validation windows, with no audio/text and no proposals.
- All three runs own checkpoints, source snapshots, raw metrics, multimodal
  outputs, inspection tensors and structurally verified standalone reports.
  Each report's 27 embedded media resources decode successfully. Latest validation
  values match the ledger. All 24 source snapshots matched the library/recipe at
  that verification; the subsequent diagram export is a new source revision.
  Exact checks are saved in `runs/multimodal_v1/verify.py` and `verification.json`.
- The synthetic final image MSE is 0.217628 versus the copy baseline 0.004724;
  real PushT is 0.220845 versus 0.000126. Synthetic audio MSE is 0.015613 versus
  silence 0.009775. These small budgets validate execution, not useful prediction.
  Accepted proposals only establish local admission-metric improvements.

### Adopted review decisions

The sandboxed Claude call timed out; a scoped approved retry completed, followed
by reconciliation. Claude accepted the distinction between contract checks and
scientific baselines, and withdrew the assertion that timestamp changes affect
only one attention weight. Softmax couples the keys. Receipts retain both responses.

The implementation rejects future inputs, backward observation time, unobserved
initial memory writes and imagined writes. Equal-time observations are explicitly
allowed for asynchronously arriving modalities and tested. Branch `time` advances
in imagination; `observed_time` does not. Float64 clocks preserve short intervals
at large absolute timestamps. These rules resolve the remaining ordering question;
Claude's suggested rejection of all equal timestamps was not adopted.

Latent spread/rank, prediction error/variance and raw-space copy/silence baselines
are recorded. They are diagnostics, not evidence of calibrated physics or an EMA
advantage. Pixel-only/frozen-target comparisons remain future scientific experiments.
The uncertainty planning penalty defaults to zero. Admission thresholds are fixed
in the recipe, no-op candidates fail, and full rollback includes EMA/replay/RNG.
Runtime episodic stores are caller-owned; training proposals recreate fresh windows
and do not modify deployment memory. Callback file/external side effects cannot be
rolled back and are excluded from the recipe's proposal.

### Environment and budget notes

The previous `data/pusht64` shortcut is absent. The new recipe uses the retained,
manifest-verified `data/pusht_world_model/cchi_v1` directory directly. No data were
moved or removed. The local package was installed into the existing virtual
environment with uv, using no dependency downloads.

The eight-main-update synthetic run pauses at update 3, resumes, and is compared
with an independent uninterrupted run. Each permits two one-update improvement
proposals, as declared before execution. The four-update real PushT run pauses at
update 2 and resumes; extra proposals are disabled. These are CPU-only development
checks. There is no scientific gate for prediction/control quality.

The browser URL security policy rejected direct local HTML navigation. No alternate
browser route was attempted. Reports must retain structural-only verification and
the browser block must be disclosed; their images/audio, embedded resources, raw
values, checkpoint identities and source snapshots are checked locally.
