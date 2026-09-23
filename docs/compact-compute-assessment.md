# Compact computation handoff: repository assessment

23 September 2026. Source inspection and proposed comparisons only; no architecture
adoption, training budget, new benchmark result or validation promotion. The named
recent papers in the supplied handoff have not been independently verified here.

## What already exists

- `pathwm/models/latent_core.py`: `LatentCore` repeats one tied attention/MLP block
  in both induction and application; constructor `loops` and call overrides already
  separate parameter count from execution depth. Induction maps evidence `[B,N,D]`
  into `[B,C,D]` concept tokens; application processes three `[B,D]` role tokens
  against those codes and predicts an outcome and residual next-machine token.
  Here C is the number of code tokens, not internal iteration count.
- `pathwm/models/agent.py`: `think(steps=...)` repeatedly uses the same thinker,
  updates working/reasoning tokens and increments `thinking_steps` without advancing
  world time. `imagine(dt=...)` advances time. This Gaussian agent path and the R1
  latent core are distinct implementations, not one already integrated mechanism.
- `pathwm/models/slots.py`: slot attention already uses shared iterative GRU
  refinement. Within-call slot refinement is not cross-observation persistence.
  The older `temporal.py` path also has an explicit GRU memory updater.
- `pathwm/models/multiscale.py`: encoder processing depth, local windows, packed
  merges and multiscale exports already exist. Windows and resampling retain their
  current experimental/default status; do not promote them from resource evidence.
- `experiments/token_budget.py` already measures synchronized forward/backward and
  inference time and peak CUDA allocation/reservation. `evaluation/workload.py`
  accounts for allocated attention shapes and projection/attention FLOPs, explicitly
  excluding MLP, convolution, normalization, transfers and other training work.
- Recipes own construction, settings and losses. `experiments/latent_agent.py`
  owns the R1 stages; `experiments/core_information.py` owns the bounded property
  diagnostic. No new configuration framework is needed.

## Information bottlenecks

The R1 slot consumer reads the multiscale image pyramid. Its documented default
RGB64 path compresses 320 exported source tokens into seven width-64 slots. Those
slots feed role selection and the latent core; the broadcast decoder reconstructs
RGB and mixture masks from slots. Patch projection, pooled hierarchy merges,
slot compression, role selection and concept-code compression are distinct places
to inspect for discarded task information. Dimensions alone do not prove retention.

Preserve the distinction between invertible rearrangement and learned processing:
an ordinary residual block, attention layer or normalization is not automatically
invertible. Keeping fine exports avoids forcing every consumer through the coarsest
grid; it does not undo information already lost in the stem.

## Smallest useful next comparison (proposal)

1. Extend existing profiling to the actual R1/CI1 training path: deduplicated
   parameters by module, optimizer bytes after initialization, activation shapes,
   forward/backward/optimizer time and peak allocated/reserved VRAM. Attribute
   device work with a profiler; do not sum overlapping inclusive hook timings.
   Keep diagnostic tracing separate from steady-state timing. Until measured,
   no module is established as the training bottleneck.
2. Freeze the same perception checkpoint and compare the existing tied core at
   fixed internal depths 1, 2 and 4 against untied blocks with the same executed
   depth and width. Hold data, objectives and optimizer protocol fixed. This
   approximately matches forward arithmetic, not parameter count, optimizer cost
   or wall time. Report quality against total training time as well as steps;
   a separate equal-parameter comparison answers a different question.
3. Use the bounded property task as a learning/stability screen with counterfactual
   and retention controls, then require harder held-out relations or transitions
   to assess depth utility. Existing near-ceiling color results cannot establish
   a useful compute-quality curve or general concept induction.
4. Only after useful fixed-depth behavior, train randomized depth and evaluate
   each depth separately. Add memory or stochastic execution one at a time with
   disabled controls. Existing state should have explicit ownership/reset/detach
   semantics before adding another register system. A skipped iteration must skip
   its associated state update; residual masking alone need not save computation.

This applies prepare-once reuse, explicit information loss/state lifetimes,
useful-progress allocation and actual-execution measurement. Weight sharing saves
parameter/optimizer storage, but unrolled training still retains activations unless
recomputation or another explicit technique changes that cost.

## Other useful ideas and limits

- Frozen-feature reconstruction is a reasonable isolated codec ablation. Freeze
  reference weights and buffers; detach the original target only. Preserve the
  derivative through the frozen encoder on the reconstruction branch. A pretrained
  frozen snapshot needs its own suitability check; an arbitrary early encoder is
  not automatically a meaningful semantic metric. Reconstruction is not a universal
  prerequisite for latent concept learning.
- Mask-aware execution fits existing prepare-once work, but a loss mask is not an
  execution mask. A selected target feature can depend on surrounding pixels and
  other tokens. Preserve those dependencies or label the change an approximation.
- Defer learned routing, adaptive halting, Mamba replacement and new multimodal
  gates. The handoff's combined experiment matrix changes several factors at once;
  first use isolated comparisons, then test interactions.

This assessment changes neither the current integrated-agent priority nor the
existing failed transfer and retention results. No external scientific review is
claimed; a consequential adopted experiment still follows the repository review
and predeclared-budget workflow.

## Reconstruction preference

Alex explicitly wants better reconstruction quality where possible, alongside
concept learning and training efficiency. Retain reconstruction as an explicit
evaluation objective when comparing compression and processing depth. No loss
weight, architecture, acceptable cost trade-off or experiment is selected yet.
Reconstruction quality and downstream usefulness must be reported separately.
