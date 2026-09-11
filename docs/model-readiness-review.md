# Whole-model readiness review — 11 September 2026

This replaces the pre-integration assessment. Source and saved results reviewed at
f6e30fb; one direct planner counterexample checked. No training, model changes or
new full-suite run. The latest key-box slice has31 relevant passing tests, not a
current whole-repository test result.

## Verdict

Controlled experiments are already running, with real falsifiable results. The
entity-store → belief workspace → supplied planner → executed feedback loop passes
a small two-box task and one independent training-seed replication. This is useful
integration evidence. It is not yet a complete learned world-model planning test.

The biggest missing piece is the learning/execution contract across components,
not an absence of neural modules. The successful loop still supplies observations,
entity queries, task mechanics and success checking. Different focused recipes
exercise other components; their successes must not be combined as though a single
trained agent performed all of them.

## What is solid enough to reuse

- Actual learned entity-state retrieval reaches BeliefAgent working tokens; there
  is no direct entity-latent-to-final-head bypass in KeyBoxReader.
- Bounded entity records, observation retry handling and entity/belief snapshots
  exist. Generated/reflected content and source observations have separate paths
  in the general agent.
- Bounded supplied expectimax replans after executed feedback. Raw action traces,
  frozen-reference comparisons, no-history controls and externally checked outcomes
  exist. Training uses supervised binary-content cross-entropy, a testable target.
- Ordinary and fixed-relocation tests pass for the new seed;192/192 immediate
  correction reads. Memory's ordinary utility advantage is0.015625, but relocation
  utility is0.896875 versus no-history0.909375. This is not universal memory benefit.
- Checkpoints, source manifests, cached resume and structural report QA work.
  Browser QA remains unavailable under the earlier restriction.

Evidence: [replica verification](../runs/key_box_v1/replica_verification.json),
[active experiment plan](key-box-integration-plan.md).

## Open interfaces and capability gaps

| Area | Current implementation | Still required / scope |
| --- | --- | --- |
| Objective and completion | Key task and fees hard-coded; general TaskSession tracks delivered outputs | One explicit success/failure/unknown contract, terminal loss and action/compute costs. Output delivered is not objective verified. Required for an interpretable integrated planning experiment. |
| Belief and evidence | Frozen learned binary cell; explicit known flags; supplied exact descriptors and content updates | Define unknown vs absent, observation time vs arrival, stale evidence, conflicting sources and action-failure updates. Exact matching/clean labels do not establish calibrated belief. Required before partial/noisy sensing claims. |
| Learned transitions | General BeliefAgent.imagine and prediction losses exist separately | Successful key planner uses supplied mechanics and never calls imagine. Encode actual actions/durations, train grounded outcome predictions and test them inside planning. Required for learned world-model planning. |
| Retrieval and internal actions | Both boxes read in fixed order; two thinking steps per read; task operation heads exist elsewhere | No integrated policy for what to recall, what to inspect, how long to think or when to stop thinking. Fixed scheduling is acceptable for a first dynamics experiment if declared. Learned scheduling requires costed training/evaluation. |
| Memory hierarchy | Recent/compressed/consolidated/source/belief stores exist | Key core receives neutral event text; semantic box content lives in a separate entity store and enters via retrieval. This pass does not demonstrate useful semantic compression, eviction-resistant recall or learned importance marking in that hierarchy. |
| Graph and concepts | Stable IDs, learned descriptors/values and one supplied relation type in a separate wrapper | No general learned topology, concept/instance hierarchy, variable relation vocabulary, skill nodes or learned inspectable meaning. Not a prerequisite to the first controlled dynamics experiment; required for the broader entity-graph goal. |
| Perception | Multimodal encoders and reconstruction heads exist | No integrated pixels→object candidate→stable entity across views path. Supplied eight-value descriptors are not learned visual identity. Required for a visual test, not a structured-observation test. |
| Runtime and persistence | KeyBoxSession snapshots memory/belief/receipts; generic task runner returns physical action proposals | No joint restore of environment, opened boxes, pending action, task verification and remaining budgets. Need execution-result/time/retry ownership before long-running interactive trials. Cached training resume is not mid-action recovery. |
| Experimental scope | Two training seeds for this recipe; fresh descriptors, fixed templates and correction timing | Broader held-out episodes/actions/timings, multiple seeds and uncertainty metrics remain. Fresh descriptor families do not alone establish task or environment generalization. |

## Concrete issues worth resolving first

1. **Planner objective differs from reported utility.** `plan_key` uses retrieval
   reward10, action cost1/inspection0.25 and stop value0. The evaluator reports
   success−0.05×cost and counts correct absent stopping as success. These are not
   equivalent objectives. Direct check: belief(key in open box0)=0.15,
   belief(absent)=0.85, one action left. Current planner selects retrieve (internal
   value0.5); reported expected utility would be0.10 for retrieve versus0.85 for
   stopping. Prior scores remain valid measurements of the implemented policy;
   they cannot establish optimization of the reported utility. Choose one contract
   or explicitly label planner reward as a surrogate before a new comparison.
   Evidence: [planner](../pathwm/models/key_box.py:59),
   [evaluation](../pathwm/evaluation/key_box.py:114).
2. **The successful core does not learn physical dynamics.** KeyBoxReader freezes
   the agent except thinker; the recipe freshly constructs the other agent modules.
   `neutral_event` passes literal `event` and no previous_action. Opening/inspection
   mechanics and opened-state live in the harness. Merely adding more key episodes
   does not train the agent's world transition model. A new experiment must send
   executed actions and meaningful evidence into it, or explicitly test a separate
   learned transition module. Evidence: [reader](../pathwm/models/key_box.py:15),
   [construction](../experiments/multimodal.py:2914).
3. **Uncertainty is partially supplied.** Harness `known` flags determine when a
   box becomes unknown; two Bernoulli readouts are multiplied and normalized over
   box0/box1/absent, excluding two simultaneous keys. Inspection branches assume
   perfect observations. This is a concrete finite task assumption, not a general
   model of uncertainty, identity ambiguity or source reliability. Softmax accuracy
   alone is not calibration. Evidence: [belief mapping](../pathwm/models/key_box.py:150).
4. **The proposed general action DAG is not implemented.** There is bounded cached
   search over supplied task beliefs, and separately candidate-sequence evaluation
   with `imagine`. There is no general learned goal predicate, reusable skill/subgoal
   decomposition, trajectory node equivalence, uncertainty-aware predicted/observed
   matching or persistent branch invalidation. A first short-horizon learned test
   can omit hierarchical skills and a persistent DAG; their absence must be explicit.
   Evidence: [generic planner](../pathwm/evaluation/agent.py:23),
   [task execution](../pathwm/models/agent.py:637), [proposed contract](decision-design.md).

## Recommended next work

Do one specification/implementation slice before another larger training sweep.
Use the existing controlled environment; do not make raw vision, a general graph
or essay/bicycle skills prerequisites.

1. Freeze a small executable task contract: available observations and actions,
   time/identity semantics, success/unknown/failed status, costs and horizon.
   Resolve the planner/metric discrepancy and define delayed/partial correction
   behavior. Keep evaluator truth out of the online agent.
2. Route the executed action plus actual observations through one declared belief
   path. Choose what the next learned model predicts: observable next contents,
   action success and observation outcomes, with explicit supervision masks for
   unavailable labels. Retain the supplied dynamics as a diagnostic control.
3. Train and test those predictions before longer search, including action-sensitive
   counterfactual pairs and multi-step rollouts. Predicting no change/copying the
   last state must be a baseline. Use observable errors/proper probability scores;
   latent distance alone is not task success or a calibrated state-match rule.
4. Freeze the configuration and run the complete controller on held-out episodes
   with no-history, simple reactive/copy-last, learned-dynamics and supplied-dynamics
   controls. Measure success, action cost, recovery, uncertainty and failure types;
   predeclare gates and a final evaluation population. Inspect randomized/delayed/
   partial observations on the fixed current loop first if needed to define the
   contract; do not endlessly tune the binary reader instead of integrating dynamics.

This would be the first meaningful learned world-model control experiment on the
current integrated path. Broader graph structure, visual identity, learned internal
scheduling and hierarchical skill discovery should be separate subsequent claims.

Claude supplied public-only critique and reconciliation; receipts under
runs/reviews/continuation_2026-09-11/readiness-current*. The short brief omitted
supervised loss and existing controls; omissions were not accepted as defects.
