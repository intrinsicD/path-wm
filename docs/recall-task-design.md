# Selective historical recall: first task contract

Implementation follow-up: Alex authorized this slice on 10 September; see the
[implemented interface and commands](recall-task.md) and
[development evidence](recall-implementation-plan.md). The remainder preserves the
reviewed design as written before that implementation.

Status: concrete design recommendation, 10 September 2026. This works out the
cost/abstention choice in [the decision design](decision-design.md). It is not an
implemented task or an executed experiment. The numerical choices below are explicit
research defaults, not estimates of Alex's preferences for every application.

## Decision rule

Use four location labels and `not_observed_in_session` as five factual classes.
The model predicts a probability for each. `abstain` is a separate operation and
is not a sixth class in the training target.

| Submitted decision | Realized loss |
| --- | ---: |
| Correct factual answer, including correct not-observed | 0 |
| Wrong factual answer, including false not-observed | 1 |
| Abstain | 0.25 |

With factual probabilities `p`, choose `argmax(p)` only if `max(p) > 0.75`.
Otherwise abstain, including an exact tie. A wrong answer costs as much as four
abstentions. This states a preference; it does not say that the model's error rate
will be below 25%. Invalid/nonfinite predictions produce a logged error and an
abstention, not a fabricated probability vector. Report those numerical failures
separately even though the attempt incurs the abstention cost.

Keep these costs in the exact caller-owned objective. Learn factual probabilities,
not the value of giving a wrong answer. A later application with different costs
uses `argmin_decision sum_y p(y)*loss(decision,y)`; asymmetric error costs generally
cannot use the same maximum-probability threshold. Positive rescaling of every cost
leaves the decision unchanged; their relative values matter.

Use two fixed task-conditioned retrieval/thinking rounds, then one answer or
abstention. Recall uses existing memory; there is no external question, inspection,
physical action or unbounded reconsideration in this task. The two rounds have the
same compute allowance across matched model comparisons. Record model calls and
latency separately from the task loss; do not introduce an arbitrary monetary
conversion for a fixed shared cost. Cheap baselines can of course use less compute.

Report sensitivity at abstention costs 0.1, 0.25 and 0.5 from the same frozen
probabilities, with thresholds 0.9, 0.75 and 0.5. The 0.25 result is primary.
The other costs are different stated preferences, not candidates from which to
choose the most flattering test result. Do not fit or retrain on each test cost.

## What the task asks

The first input is a stream of short canonical observation records, for example
`saw entity=e07 at location=l2`, carried by the existing text observation adapter.
The query identifies an entity after the historical stream ends. Entity and
location meanings remain learned; there are no semantic slots assigned inside the
world state. This tests temporal association and retention before introducing visual
recognition or free-form language ambiguity. It does not demonstrate visual mapping.

The exact target is the location in the last delivered visible record for the
queried entity at or before the fixed query cutoff. If no such record exists since
the fresh session started, the target is `not_observed_in_session`. The evaluator
derives this label from its complete input log. The policy cannot access that log,
a precomputed seen-entity set, the hidden simulator state or the target label.

The cutoff is inclusive. There is one visible entity/location record per strictly
increasing caller-owned ordinal; the greatest eligible ordinal wins after repeated
observations. This first fixture excludes conflicting simultaneous locations.
Location symbols have stable output meanings across episodes. Entity-to-location
assignments change, not an undisclosed output-code permutation. Shared entity and
location vocabularies are intentional; held-out names are a separate generalization
test. Fresh episodes and hidden future assignments prevent history leakage.

Delivery order and event order coincide in this fixture. Out-of-order packets are
invalid input here, consistent with the implemented event contract. Delayed source
records would need a separate source-time versus arrival-time task definition;
they are not needed to demonstrate ordered historical recall.

All records in the initial task are fully readable; partial-perception variants
are separate. If a later task masks observations, redefine the label using what
was actually delivered rather than silently retaining an omniscient pre-mask label.
The query is task input and never a world observation. Query cutoff includes session
and event ordinal, with timestamp for elapsed-time semantics.

Visible moves update the historical label. Hidden moves change only the evaluator's
simulated current state. They never change the historical answer or leak through a
special observation token. Every delivered step still contains a visible distractor
record when the target is out of view. Two hidden trajectories with identical
delivered records must produce the same historical target and model input.

One independent episode is one complete fresh agent session and one final query.
Consume the whole stream from the empty initial state. Truncated backpropagation
may detach gradients, but must not restart memory midway and then claim a complete
session history. The current short-window recipe will therefore need an explicit
episode replay path before this task can run; it cannot be enabled by labels alone.

## Population and memory coverage

Propose five equally weighted factual classes for the initial diagnostic: each
location is 20% and not-observed is 20%. This is an artificial declared population,
not an assumed real-world frequency. Independently balance seen examples across
recent, compressed and consolidated source-history conditions. Report those groups
separately; not-observed examples have no last-observed age and must not be assigned
an invented one.

Use a fixed stream length per protocol and random entity identities, locations,
move patterns and distractors. Reveal the query only at the end, with equal label
mixtures across split and history groups. The harness can select eligible queries
after constructing the history. Opaque task/session IDs, generator seeds and the
group label stay out of learned inputs. Randomize irrelevant absolute time offsets
while preserving order and elapsed time. Split whole episodes, not windows.

For the current default memory capacities, a 256-record stream is a proposed full
history fixture. Choose last target appearances so the original record is still
recent, is compressed, or has passed through consolidation. Verify these paths
from actual retention bookkeeping rather than inferring them only from elapsed
time. A small-memory development fixture may shorten the stream proportionally,
but cannot stand in for the declared full-capacity comparison.

Cohorts use source-event positions and the fixed reference retention schedule,
chosen before predictions are evaluated. Group sizes and source ages are explicit;
success or confidence never determines membership. For a different memory control,
keep the same episode groups and log that control's actual retention path separately.
Cross-age differences alone are descriptive, not causal effects of compression.

"Passed through consolidation" describes the original source record. The target's
information might also survive in later belief records or the current recurrent
state. Full-agent performance therefore does not by itself attribute success to
consolidated memory. Keep a separate memory-only query diagnostic that excludes the
current world/recurrent state and prior task workspace. Its restricted interface
needs its own training or an explicit intervention-only interpretation.

Start the primary fixture without explicit marks to measure ordinary retention.
User marks and agent-proposed marks are separate conditions with the same total
capacity limit and reported actual occupancy. User marks intentionally reveal a
retention preference and must not be pooled into the unmarked headline. Keep all
memory capacity and fresh-session decisions from the implemented design.

Marks may protect only an already delivered event at write time. They do not reveal
unseen placements or change what counts as observed. The agent's marking score never
receives the eventual query or label; user marks are a separately declared source
of information about retention preferences.

## Learn on every query; calibrate after model selection

Train categorical negative log likelihood on every factual target, including
queries on which the decision rule would abstain. Otherwise the model can stop
learning from its hardest cases. The five-class labels remain available to the
training loss only; they are not inputs to task interpretation or verification
during inference. Reuse appropriate existing state/memory objectives without
creating fake image, action or audio targets for a text-only episode.

Use four disjoint episode partitions with fixed identities before training:

1. Training updates the model, outcome head and relevant memory consumers.
2. Development chooses model settings/checkpoint using factual NLL and declared
   diagnostics, never the final test set.
3. Calibration fits one positive temperature `T` on the frozen factual logits by
   minimizing calibration NLL. Use `p=softmax(logits/T)`.
4. Test applies the frozen model, temperature and cost rule once for final reporting.

The calibration population has the same declared mixture as the primary test.
Fit a single global temperature initially, not per label, memory tier or cost.
Retain raw `T=1` results alongside adjusted results. Save model identity, calibration
episode identities, fitted temperature and fit status with the checkpoint/run.
Numerical bounds and solver tolerance are implementation settings to record before
fitting. If fitting fails, keep the raw result with an explicit calibration-failed
status; do not silently claim calibrated behavior.

Temperature scaling adjusts confidence without changing the top class. It is a
small established post-processing baseline, proposed here rather than assumed to
work for memory tasks. [Guo et al., section 4](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf)
describes fitting it on held-out data. It neither repairs forgotten facts nor
establishes conditional error bounds for long-delay or never-observed examples.

In a multiclass problem it can also change confidence ordering across examples.
Forced-answer top-class accuracy is unchanged, but the selected subset, its accuracy
and the risk/coverage curve may change. For logits `(2,0,0)` versus `(10,9,-10)`,
the maximum probabilities are about 0.787 versus 0.731 at `T=1`, and 0.576 versus
0.622 at `T=2`: their ordering reverses. Report the raw and adjusted rankings
separately. Check reliability among actually accepted answers and within history
groups; do not fit another temperature on test-selected cases.

Use one ordinary posterior trajectory per episode, with fixed evaluation seeds,
then two retrieval rounds and one five-logit factual prediction. No extra imagined
rollouts are needed. Cache those logits once for raw, adjusted and cost-sensitivity
reports. A later multi-draw method must freeze its aggregation rule before fitting
temperature; it is not interchangeable with this prediction path. A finite confidence
guarantee would require a separately specified protocol; none is claimed here.

Development NLL and realized selective cost are different objectives. This initial
choice favors estimating the full factual distribution without tuning a checkpoint
to one abstention preference. Neither improved NLL nor temperature fitting guarantees
lower selective cost; report failures rather than changing selection rules after test.

## Report cost, answering rate and actual errors together

Let `N` be all valid test episodes, `W` wrong factual submissions, and `A` abstentions:

Input validity is fixed before model execution. Every attempted valid episode stays
in the denominator, including numerical failures or timeouts; these are logged
failed attempts with abstention loss. Report harness/data failures separately as
protocol failures, never silently drop difficult model cases to improve metrics.

Log the mechanically known abstention reason: threshold/cost decision, invalid model
output or exhausted execution budget. Retrieved-record availability may be diagnostic.
Do not claim that a latent state contains "conflicting memories" solely because its
probability distribution is diffuse; that explanation has not been verified.

`mean task loss = (W + 0.25*A)/N`.

Coverage is `(N-A)/N`; factual error among answers is `W/(N-A)`. With zero answers,
that conditional error is undefined, not zero. For nonzero coverage:

`mean task loss = 0.25 + coverage * (error_among_answers - 0.25)`.

Always abstaining scores 0.25. An answerer can beat it only through some answered
cases with aggregate error below 25%. Always answering a frequent factual class is
another cheap baseline to report. Its class comes from the training distribution,
never test labels. Under this proposed balanced five-class population its expected
error is 80%, and the cost-aware constant policy would abstain.

Also report factual top-class accuracy before abstention, NLL, multiclass Brier
score, and confidence-versus-correctness plots with sample counts. Show the
coverage/error tradeoff across the fixed confidence ranking; the primary decision
point remains fixed before test. Risk and coverage are distinct selective-prediction
quantities. [Geifman and El-Yaniv, section 2](https://papers.nips.cc/paper_files/paper/2017/file/4a8423d5e91fda00bb7e46540e2b0cf1-Paper.pdf)
also gives a separate procedure for risk guarantees; fitting temperature is not
that procedure.

Break out actual-label groups and history groups. In particular count invented
locations for genuinely unobserved entities, false not-observed answers for seen
entities, incorrect remembered locations, and abstentions, with each denominator.
Do not conflate answered examples with the whole population in any chart.

Beating all-abstain in aggregate is insufficient for a memory claim. A system that
correctly answers only the 20% not-observed cases and abstains on every location
scores 0.20 without recalling any location. A claim of useful old-location recall
must also show lower loss than all-abstain on the preregistered seen/old-history
population, with nonzero location-answer coverage. Report effect size and uncertainty,
not just whether the point estimate is slightly below 0.25.

Use declared fixed weights for any macro-average and report the underlying group
results. Fixed weights remove changes caused solely by prevalence weighting; they
do not make the result independent of the examples chosen within each group.

For a hierarchy advantage, compare to a matched trained recurrent-only/recent-only
control on those same episodes. Removing reads at evaluation measures reliance,
not superiority over a trained alternative. An exact-log lookup is an evaluator-only
oracle reference and never part of the agent. Retain per-episode predictions and
use episode-level uncertainty estimates; formal multi-seed comparisons must not
treat examples from shared trained models as independent training replicates.

This defines the decisions and measurements. Before a scientific run, declare seeds,
split counts, update/replay budget, exact controls, primary old-history group,
effect-size threshold and interval procedure. No statistical pass is possible while
those are unset. Neither this document nor peer agreement authorizes that run.

## Review and implementation handoff

Two actual isolated Claude CLI exchanges completed: `recall-policy` and
`recall-policy-reconcile`, each with independent notes before reading the response.
Only hypothetical public concepts were sent; no private source, dimensions or results.
Receipts are under `runs/reviews/state_memory_design_2026-09-09/recall-policy*`.

Claude supported the reference loss, all-query supervision, separate calibration,
and distinct factual/operational labels. It explicitly withdrew claims that
temperature changes only coverage, that additional gates universally worsen real
loss, that fixed heads require disjoint vocabularies, and that trained controls are
premature for a hierarchy-benefit claim. Corrected the abstention equality boundary
and `1 - unseen_prevalence` baseline algebra. A cost-based threshold minimizes loss
under its estimated probabilities; additional gates could help when those estimates
are wrong, but are not part of this first research policy.

Remaining suggestions were resolved within scope: out-of-order delivery is excluded,
grouping is fixed before outcomes, abstention reasons use observable decision records,
and proper-score checkpoint selection is explicitly different from task-cost
optimization. No arbitrary sample-count estimate became an approved compute budget.
Primary-source checks support the calibration/selective-prediction mechanisms;
exact arithmetic checks support the loss identity and confidence-ranking example,
not learned recall capability.

The next implementation slice is the five-class outcome head and this complete
episode-to-query path, with the fixed decision rule and an independent evaluator.
Reuse the existing task records, source provenance, memory, recipe and reporting.
No new sensing planner, extra memory store or learned abstention head is required.

Essential implementation checks should cover inclusive-cutoff label generation,
not-observed versus forgotten facts, hidden-trajectory input invariance, boundary
probabilities and tie behavior, evaluation-only labels, and completion/verification
separation. Under fixed randomness, detaching gradients at replay boundaries must
preserve forward state and predictions. Separately score cases whose relevant fact
precedes that boundary: an invariant-preserving replay path can still learn poorly.
No new runtime tests or model training were executed in this design turn.
