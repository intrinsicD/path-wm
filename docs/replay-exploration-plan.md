# Replay-guided experimental exploration

16 September2026. User clarified that the requested efficiency work is the
Dream-RSI-style choice of experimental branches, not faster rejection of tests.
The unimplemented screening draft is preserved under
`runs/reviews/efficiency_screen_v1/superseded-plan.md`; no screening code is added.

## Fixed first experiment

Keep the candidate generator, evaluator and interface fixed. Use the existing
VID.order task, full prepared fixture cohort and original joint-native7202 source.
Freeze the shared core and cache one deterministic latent draw per example.
Train only the existing text output decoder/adapter on calibration examples using
the existing normalized byte-CE objective. This is controlled search over a small
readout fit, not a new model, broad understanding repair or training-search oracle.

- Four branches share source weights and sample stream, with Adam learning rates
  0.0003,0.001,0.003,0.01 in that order. Each continuation performs64 updates,
  batch8, gradient clip5; maximum four blocks per branch. Save each actual node's
  weights, optimizer, sampler/RNG, validation outcome, parent and measured cost.
- Validation uses the complete separate validation cohort; score is negative
  normalized byte CE. Test examples never enter optimization or policy selection.
  Score changes on cached states do not establish generalization to other latent
  draws. Cache source, records, seed, code and tensor hashes are part of identity.
- Root action opens the next unobserved branch. A child action continues only a
  revealed frontier. Policies receive only revealed numerical outcomes, depths and
  legal actions. Replay cannot synthesize missing successors; insufficient support
  is explicit failure, never a high score or free stopping reward.
- Collect exhaustive16-node development worlds at seeds9601/9602. Evaluate only
  five preregistered policies: round-robin12 incumbent, round-robin8 simple budget,
  greedy best current frontier, and the same with plateau stopping at patience1/2.
  Plateau threshold0.002 in validation score; every branch gets an initial block.
  All policies have maximum12 actions except short baseline8. Tie break by branch
  index. Utility = best revealed score -0.001*executed blocks. Keep incumbent in
  selection; deterministic candidate order breaks equal utility ties. No sweep.
- Freeze replay winner before fresh seeds9611/9612. Execute incumbent, fixed-short
  and selected policy from scratch, including repeated selected/baseline schedules
  if identical, to verify deterministic interleaving. Root weights and example
  cache remain paired within each world. Training RNG is branch-local; validation
  preserves it. Matched nodes must have identical outcomes and training states.
- Evaluate each selected checkpoint on reserved VID.order examples, source omission
  and the complete quick25-task understanding suite. Keep raw results, source
  snapshots, selected checkpoints and full standalone reports. This is development
  data repeatedly used in earlier research, not unseen benchmark confirmation.
- Optional policy adoption requires in EACH fresh seed: >=20% fewer update blocks,
  >=15% less whole execution wall time, validation CE increase<=0.01 absolute and
  reserved target accuracy decrease<=5pp against incumbent. No quick-suite pass
  loss or accuracy/pair/source-gain decrease>10pp. Full-suite tests occur only after
  selection. Keep all other gates; no aggregate score can conceal regressions.
- If the shorter fixed policy also qualifies and adaptive selection adds <10%
  execution-time benefit, prefer the shorter fixed policy. If none qualifies, keep
  the incumbent and record the negative result. No underlying model promotion.

## Cost and limits

Zero source-weight changes. At most two16-block history collections plus six
12-block fresh runs:6656 decoder updates. Each world<=300s excluding final suite;
each suite<=600s; all formal execution<=1800s; peak torch allocation<6GiB;
artifacts<=500MiB, free disk>=500MiB. Development histories, cache construction,
policy search, review and verification are charged separately. Report total cost
and an illustrative break-even count only if recurring savings exist. Do not claim
this pilot saves net compute when collecting its histories costs more.

This finite policy-selection study is a minimal transfer of the paper's replay
loop. It does not implement unrestricted self-modifying policy code, learned
latent imagination, architecture search, or general action planning. Generalizing
the fixed task/interface needs a new experiment.

## Work sequence and essential checks

Plan/red tests -> pure prefix/replay/policy helpers -> existing recipe integration
and two-block restart/interleaving smoke -> commit -> fixed histories and offline
selection -> fresh execution -> raw state/output/coverage audits -> suite reports.
Test illegal jumps, hidden-future invariance, unsupported replay, nonfinite outcomes,
budget enforcement, incumbent inclusion, train/validation/test separation and
branch-local optimizer/RNG restoration. Use existing Run/report infrastructure;
no new trainer framework. Defaults and existing suite execution remain unchanged.

Actual Claude public-only methodology review is recorded under
`runs/reviews/replay_exploration_v1/`. Agreement is not empirical validation.
Reconciliation explicitly labels the two fresh worlds an engineering demonstration,
not a statistical or reliable-transfer claim. Larger independent tasks/seeds are
required for general adoption; a passing policy remains opt-in. No history results
existed when these candidates/thresholds were fixed.

Before final suite execution, raw checkpoint audit found all three policies select
the identical inference state within each fresh seed. Evaluate the full quick suite
once per unique state/settings/fixture/scorer/RNG contract and retain an explicit
mapping to all six selections. This preserves every intended test and saves four
duplicate suite executions; reused results are not counted as independent runs.
The branch executions themselves were all performed as preregistered. This
identity-based evaluation reuse is recorded before observing these suite results.

## Results

Both development histories, offline selection and six fresh search executions
completed. [Comparison report](../runs/replay_exploration_v1/report.html),
[raw results](../runs/replay_exploration_v1/comparison.json),
[state audit](../runs/replay_exploration_v1/state-verification.json).

Replay selects **round_robin8**, the simple shorter fixed budget. Development mean
utility is-0.015798 versus-0.019008 for the12-block incumbent; the adaptive policies
do not beat the short baseline. This is finite selection using recorded trajectories,
not an LLM rewriting its own policy or a neural latent simulator.

| Fresh world | Incumbent / selected blocks | Incumbent / selected process seconds | Selected validation CE | Cached test accuracy | Full quick suite |
| --- | --- | --- | --- | --- | --- |
|9611|12 /8|15.695 /12.439|0.0125005|87.5%|0/25 pass|
|9612|12 /8|15.646 /12.289|0.0014109|87.5%|0/25 pass|

Both fresh worlds meet the scoped efficiency gates:33.3% fewer updates and20.75%
/21.45% less search-process time. The incumbent, short and selected schedules
produce the **identical selected checkpoint** within each world. Accordingly all
quality/preservation comparisons between those schedules are exactly equal.
The repeated short/selected executions confirm this independently. The full suite
uses new latent draws: VID.order worst-draw accuracy is75%, paired accuracy50%;
both selected models fail all25 overall task gates. There is no capability repair
or model adoption. Retain the simpler short policy as an opt-in demonstration;
default training/agent behavior stays unchanged.

The raw audit checks56 fresh nodes,39,414 exact tensor comparisons,80,544 primitive
comparisons and4,158 frozen parameter/buffer comparisons. Matching actions have
identical optimizer, sampler, RNG and model state despite changed interleaving.
Every continuation references its actual saved parent; no future outcomes are
supplied to the controller. Source checkpoints remain unchanged.

Cost: history collection35.368s/2,048 updates; all six fresh executions80.697s/
3,584 updates; offline policy selection0.000894s (excludes its report); two unique
complete suite runs137.254s. The latter represent all six identical-checkpoint
selections via hashed mappings and avoid four duplicate evaluations. Search times
include CLI startup, cache creation, fitting, snapshots and reports; final suite
cost is separate. Including one full suite per candidate, the observed saving
would be roughly4%, not21%. The pilot costs more overall than it saves.

History-only cost would require approximately11 comparable future searches to
amortize at the observed saving. This illustration excludes review, implementation,
verification and distribution shift. All four actual Claude calls, including the
superseded screening discussion, total31.422 process seconds; token receipts are
saved. Codex/implementation costs are not instrumented. Component durations may
overlap and must not be summed as elapsed time or GPU utilization. Peak decoder
search allocation is below80MiB; the broader suite is accounted separately.

Standalone reports pass structural checks; the aggregate plot is visually
inspected. Interactive browser QA remains unavailable. Larger independent seed/task
confirmation is required before claiming reliable search-policy transfer. Further
policy improvement from partial live histories needs explicit support checks or
additional measured continuations; missing branches cannot be filled by inference.

Final software verification:571 repository tests pass (approximately482.7s while
the final GPU suites ran separately). The29 focused scheduler, grounding,
understanding/request and run/checkpoint checks passed beforehand.52 standalone
reports are structurally verified. Complete source snapshots and raw verification
receipts are saved; no private code or results were exported to Claude.

Formal measurements and the full571-test run use commit `79086d2`, with source
snapshots retained. Final review then added an input guard: this first recipe only
accepts native sampled/context sources without layer aggregation, a video palette
or request adaptation, because its checkpoint export does not preserve those
alternate configurations. The guard leaves the measured native path unchanged;
33 focused checks pass after it, including four unsupported-source cases. The full
suite was not repeated after this guard. Saved policy contracts pin source hashes:
new code revisions need new histories/selection, not reuse of an incompatible
formal policy file.
