# Ideas retained for later research

The user requested a fresh implementation and fresh evidence on 2026-09-05.
These are research ideas, not validated claims. Previous code and experiment
results are excluded from the active project. Source datasets are retained.

## First establish the baseline

Reproduce a published, learning world model with replaceable encoder, projector,
action encoder, predictor, objective, dataset adapter, and planner components.
Use LeWM as the initial reference and PushT as the first controlled task. Support
multiple datasets with explicit per-dataset action semantics, episode boundaries,
normalization, train/test splits and evaluation protocols. Validate learning and
closed-loop control before opening any of the research directions below.

## Perception and dynamics interfaces (H1)

Can a new encoder and a small adapter drive an unchanged learned predictor and
planner? Judge compatibility by transitions and task performance, not just latent
cosine similarity or CKA. Compare adapter capacity, a random-adapter floor and a
retrained-predictor ceiling at matched budgets. First consider multiple encoders
sharing an interface; later stitch a third. Relative anchor representations are
an alternative. Keep modality-specific spatial structure in perception rather
than forcing audio into an image grid. Transition consistency alone can admit
collapsed or action-insensitive solutions; inverse and counterfactual consumers
are possible later controls, not additions to the initial baseline objective.

## Persistent state and scratch computation (H2)

Separate physical state or belief from predictor scratch registers. Reset scratch
per call and test whether information leaks through it. Keep this comparison
separate from the baseline architecture.

## Passive observations and causal learning (H3)

Measure whether passive video or audio/video pretraining reduces the amount of
action-labelled data required to learn transitions. Passive temporal prediction
and action-conditioned causality are different capabilities. TAU Urban AV is a
passive dataset; latent-action models are a possible later extension.

## Dynamics components (H4)

Compare transformer, recurrent, shared-loop, state-space and hybrid predictors
under matched parameter, training and planning budgets.

## Partial observability and uncertainty (H5)

Compare single-frame state, finite history, predict/correct belief updates and
causal encoders on hidden velocity, occlusion and missing modalities. Begin with
deterministic states. Add ensembles or multimodal distributions only when their
uncertainty improves calibrated prediction and planning.

## Persistent trajectory planning (H6)

Treat candidate trajectories as persistent objects containing states, actions,
cost, uncertainty, provenance, lineage, validity and age. Explore annealed SMC
with proposal, scoring, weighting, resampling, diversity-preserving mutation and
annealing. Candidate sources include random or coloured noise, incumbent
perturbations, shifted reservoirs, policy priors, inverse proposals and gradient
repair. Compare persistence against established warm-start CEM/iCEM.

Maintain distinct solution modes. A cheap critic may rank candidates but verify
executed actions by model rollout and count critic/model computation separately.
After observing the world, shift and revalidate past trajectories, then retain,
repair, reroll or discard according to prediction drift. Reusing an old cost is
not free evidence. Mutations can change actions or segments, preserve prefixes,
reroll crossed-over suffixes, or jump between modes. Splicing latent states needs
a valid connection operator. Learning proposals from verified paths is deferred.

## Hierarchy (H7)

Start with fixed action chunks or a duration-conditioned predictor. Learned macro
actions and subgoals can follow, with a low-level planner verifying connections.
Bidirectional planning is deferred.

## Evaluation principles

Keep training and held-out episodes separate and make any reproduction-protocol
differences explicit. Measure one-step and open-loop errors, compounding error,
action-zero/shuffle and identity controls, collapse diagnostics, task success,
constraints, model calls and wall-clock cost. Later add hidden-state probes,
uncertainty calibration, disturbances and cross-domain generalization. Use
multiple seeds for benchmark claims. Diagnostic labels must not leak into the
baseline representation objective. Existing ingredients do not by themselves
constitute a novel contribution.
