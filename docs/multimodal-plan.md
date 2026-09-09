# One multimodal world model

## Task decisions, output controls and loopback — implemented slice

Accepted user path: instruction → shared text features → task tokens → learned
operation/output/completion proposals → explicit output controls → attributed
generation → tagged reflection. Keep all four existing output modalities and
ordinary replaceable modules. Low-level reconstruction decoders remain available;
user-facing emission goes through the new contract. No external language model.

Immutable actor, control, request, output and caller-owned task records preserve
who specified controls, who requested each output, who produced it, parent IDs and
purpose. Modes are required/disabled/automatic. Validate the complete emission
batch before decoding; report partial successes on decoder failure, and fulfill
only successful answers. Video may internally decode images while image emission
is disabled. Only caller-supplied controls are authoritative; inferred choices are
agent requests. Raw and gated decisions stay separately inspectable.

Generated loopback updates working/reasoning through shared encoders and learned
metadata features, retaining exact records. It never updates observational clocks
or counts. Generated ancestry persists through derivation and state operations;
observational memory rejects such states, including after later real observations.
This is a trusted-caller provenance contract, not authentication against stripping
metadata. Task/state round-trips preserve attribution and partial fulfillment.

Task interpreter and policy are trained jointly through the existing recipe using
an explicitly synthetic instruction curriculum, disjoint paraphrase families,
operation/modality/completion targets, ambiguous examples and restrictions. Policy
operations are think/recall/imagine/act/emit/ask/finish; a bounded one-step executor
returns action proposals to the caller and never actuates an environment. Report
held-out errors separately, raw versus gated outputs, shuffled instructions and a
lexical nearest-example baseline. This cannot establish broad instruction following.

Claude's public-only initial review is retained at runs/reviews/task_outputs/.
Adopt attribution, partial failure, laundering, round-trip, gate and evaluation
corrections. Keep all modalities/seven operations because they already have real
consumers; use ask probability for clarification instead of another duplicate head.
Reconcile the refined contracts with Claude before closing the slice.

Essential red tests: attribution, atomic prevalidation/partial fulfillment, video
dependency, arbitrary-logit gates, loopback isolation and laundering rejection,
safe serialization/continuation, task gradients and held-out curriculum separation.
Commit these before implementing. CPU development budget: full tests, both existing
data checks, at most 80 main instruction updates plus exact pause/resume replay,
two real PushT updates, no extra proposals. No scientific quality threshold or GPU
job. Preserve raw decisions, checkpoint, metrics, reports and reproducible diagrams.

Implementation evidence: red tests/plan committed as `e3f68dd`; all 51 CPU tests
pass. Nine new checks cover attribution, prevalidation/partial failure, video's
internal image dependency, arbitrary-logit gates, generated derivation/memory
isolation, safe partial-task continuation, task gradients, all seven bounded
consumers, abort and missing-ancestry rejection. Default model: 359,188 parameters;
four computed task tokens, seven operation logits, four output logits and completion.
`ask` is an explicit caller operation; no additional clarification head was added.
The new latent schema is v2; old schemas require their historical source snapshot.

Retained execution: `runs/task_outputs_v1/instructions_full/` finishes 80 updates;
`instructions_resumed/` pauses at 31 and finishes at 80. Both use CPU width 16
(115,012 parameters), seed 42, batch 8, 112 train/112 validation episodes, learning
rate .001, evaluation every 20 and no extra proposals. Model/EMA/replay, optimizer,
scheduler, all saved RNG, training rows and all final task decisions match exactly.
Pause-boundary validation rows intentionally differ. `pusht/` finishes two updates
on eight training/four validation windows with no fabricated task/audio/text labels.
Synthetic, instruction and real-data forward/backward checks pass; the new task
modules receive gradients only with task supervision, and the EMA target remains
detached. No source was edited during these retained runs.

Instruction evaluation contains 112 held-out episodes with 28 distinct instruction
strings from disjoint template families. Raw operation accuracy is 28.57%, versus
92.86% for the lexical nearest-example baseline and 21.43% under mismatched text.
Completion error is 14.29%; modality error on emit examples is 56.25% for image and
50% each for audio/text/video. The low 7.37% overall modality error is dominated by
non-emit examples; it is not good format selection. Raw and enforced gate violation
counts are both zero here because the weak raw policy does not choose emit/finish;
the randomized-logit tests exercise actual enforcement. The report demo accurately
labels its emission as an explicit user request because its learned proposal is act.
This establishes an executable training path, not reliable instruction following or
outcome-calibrated completion. No further training or quality claim was made.

Instruction image MSE ends at .008640 versus copy .005509; real PushT .231872 versus
copy .000126. These are unmatched development results. All three runs own metrics,
checkpoints, task decisions/outputs, exact provenance and standalone reports. Their
65/65/44 embedded media resources decode, all 27 recorded source identities match,
and task reflection preserves clocks/counts/uncertainty and generated ancestry.
Receipts and reproducible checks: `runs/task_outputs_v1/verify.py` and
`verification.json`. The seven diagrams reproduce all 29 files exactly; updated
architecture/task PNGs were visually inspected. Report browser QA remains blocked
by the prior local-URL policy; the renderer was unchanged and reports claim only
structural verification.

Review disposition: two actual Claude responses are retained. Adopted separate
control author/requester/producer, whole-batch validation and ordered prefix
successes, persistent generated ancestry, conservative schema loading, explicit
abort, and raw/enforced metrics. The second response acknowledged the modality and
operation simplifications were preferences and withdrew them as review items.
Its remaining overbroad claims are checked locally: one resolved enum per modality
precludes contradictory modes; fulfillment uses local authors; disabled names an
artifact type (video may contain image frames); memory refusal is immediate and
explicit on a reflected branch; post-mask metrics audit enforcement rather than
learning. The clean observation branch is retained before reflection, never made
by dropping ancestry. No authority comes from parent IDs or learned embeddings.

A final clarification brief was prepared at
`runs/reviews/task_outputs/final-clarifications.txt`, but automatic approval review
rejected its external transmission as potentially non-public and insufficiently
authorized for that payload/destination. No third response or workaround exists;
`final-receipt.json` records the block. That optional follow-up needs explicit user
approval. Local implementation/verification is complete; peer acknowledgment of
those final corrections has not been obtained.

## Multiscale conditioned inputs — completed slice

Request: all four input modalities expose fine-to-coarse processed feature scales,
with cross-scale attention and residual processing controlled by an agent/user code.
Actual Claude reviewed the design first; public-concepts-only briefs, three
responses and CLI receipts are preserved under `runs/reviews/multiscale_inputs/`.
The review and reconciliation are complete.

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

Adopted review decisions: bias-free bounded FiLM with live initial gradients;
finite invalid outputs and safe attention rows; local-footprint cross-scale
attention that can be disabled; conjunctive time/ordinal masks; no reverse pass.
Claude withdrew claims that masked cross-attention raises a query's availability,
that a hardcoded three-stage implementation is required, and that the code needs
an ordinal floor under the explicit fixed-context contract. It accepted window-local
content ordinals with separate conditioning time/source. Its final suggestion of a
shared cross-modal ordinal does not apply: local code inspection confirms that each
pyramid operates on one modality, and the latent consumer makes no cross-modal
ordinal comparisons or causal-prefix claim. No global streaming protocol was added.

Sensor timestamps are retained separately from context-floored availability and are
encoded before fine-scale processing. This avoids losing observation timing when
a newer context code raises availability. Direct tests establish completed-prefix
consistency at fixed code/time, not cache equivalence across changing codes.

Implementation checks: the initial missing-module failure was committed as
`8afca0d` (red). All 42 CPU tests now pass, including ten multiscale checks and four
diagram checks. Ordering hooks cover BOTH pooling and cross-attention consumers;
zero-code neutrality after optimization is compared with a FiLM-disabled reference.
The new default has 325,704 parameters, three input scales and a 16-value code.

Retained execution evidence: `runs/multiscale_v1/synthetic_full/` and
`synthetic_resumed/` both finish two main updates, with identical model/EMA/controller,
replay, optimizer, scheduler, Python/NumPy/Torch/CUDA/sampler RNG state and training
rows. Pause-boundary validation rows intentionally differ. `pusht/` completes two
real-data updates. No learning proposals ran. All reports pass structural validation;
62 embedded media items in each synthetic report and 44 in the PushT report decode.
Inspection contains all 12 synthetic or six available real input scales, masks,
content/availability times, attention, codes and their provenance. All 26 recorded
source identities match this implementation. Receipt and reproducible checks:
`runs/multiscale_v1/verification.json` and `verify.py`.

Final development image MSE: synthetic 0.242622 versus copying 0.006235;
PushT 0.231832 versus copying 0.000122. These tiny runs establish execution only,
not an advantage over the older architecture. No matched quality experiment ran.
The six diagrams (architecture, high-level flow, four input pyramids) regenerate
to identical bytes across all 25 files and their PNGs were visually inspected.
No report renderer source changed; report browser QA remains blocked by the
previous local URL policy, and the retained reports claim structural checks only.

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
