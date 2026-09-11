# Direct factual grounding and binding controls

Alex authorized implementation, Claude review, testing, fixes and iteration on
11 September 2026. Keep the small library and existing multimodal recipe/trainer.
This is a diagnostic reference, not a new agent architecture or memory objective.

## Declared experiment

Train the existing `MultiScaleTextEncoder` architecture from scratch with two learned
attention queries and entity/location classification heads. Inputs contain one
canonical `SeenRecord.text` observation; neutral conditioning, time zero, no recurrent
belief, memory, future query, teacher-forced decoder or evaluator metadata. This tests
trainability of this encoder/readout combination, not information retained by the
previous trained checkpoint. Supervise both entity and location with equal-weight CE.

Enumerate all 32 entities x 4 locations. Training gets the 96 pairs where
`(entity + location) % 4 != 0`; development gets the remaining 32. Each entity and
location is covered in training. Split semantic pairs before encoding; there is no
timestamp augmentation masquerading as new facts. Report entity, location and joint
accuracy, each NLL, and per-entity results on training and held-out combinations.
These finite canonical strings do not test natural-language generalization.

First pilot: seed 23, width 32, AdamW lr 0.0003 / weight decay 0.01, clip norm 1,
batch 16, FP32/two CPU threads, 512 updates. Training metrics at initialization and
every 32 updates; final checkpoint only, one terminal development evaluation, no
calibration/test population or best-checkpoint selection. Fit gate: training entity
and location accuracy each >=95%, joint >=90%, and mean of entity/location NLL <=0.35.
Fresh-combination gate: each accuracy >=90%, joint >=80%, mean NLL <=0.5.

If either extraction gate fails without a software defect, one corrective comparison
may change only lr to 0.001 from the same fresh initialization, with identical data,
seed, batch and 512-update exposure. Preserve the reference. This conditional
optimizer comparison is predeclared; no third training run or extra seeds. At most
450 active seconds per pilot (900 seconds across the two), with the existing
cooperative limit and cumulative pause/resume accounting. Test/QA work is separate.
Stop and report if the second attempt still fails. Do not claim a universal defect
or invent a successful control by weakening the gates.

## Binding reference after extraction passes

No new parameters or training. For two distinct entities at different locations,
read each record independently. The query is an exact entity ID, a legitimate task
input. For query q, normalize the two predicted `P(entity=q | record)` values into
selector weights, then mix the two predicted location distributions. Compute this
in log space. This is an explicit heuristic decomposition, not a calibrated joint
posterior or a learned task-workspace mechanism.

Enumerate distinct-entity/different-location pairs and both queries from cached
single-record logits. Report both-seen, mixed, and both-held-out constituent pairs
separately; do not relabel reused train pairs as held out. Report accuracy and NLL
for each query, paired success on both queries, and coherent location-swap success.
Pair-order invariance is an implementation invariant, not evidence of learned
binding. Coherent location changes must be judged with correctness, not mere flips.
Binding gate: both-held-out query accuracy >=90% and paired success >=80%.
Keep complete raw single-fact logits plus the exact enumeration/selector contract
and raw binding scores. A perfect-logit numeric control verifies the reference.

## Implementation and verification

Add the small fact reader and explicit selector in `pathwm/models/facts.py`;
quantitative scoring in `pathwm/evaluation/facts.py`. Add the finite dataset and
`--dataset facts` branch to `experiments/multimodal.py`, sharing its optimizer,
Run/checkpoint, diagnostic budget, final-cache and report lifecycle. No new trainer.
Essential red checks: held-out semantic-pair separation/coverage; input-only forward
and nonzero encoder/head gradients; selector numeric reference, pair swap/query/
location changes; exact pause/resume and cached terminal evaluation without calibration.
Run relevant tests, then the full CPU suite. Preserve every attempted run and its
report, with browser QA for changed rendering. Fix demonstrated software failures
before interpreting training outcomes.

Claude uses short conceptual briefs under the existing export boundary. Save exact
briefs, responses and receipts in `runs/reviews/continuation_2026-09-11/fact-*`.
The first review requested a fixed selector and finer-grained held-out metrics;
these are included above. Its suggestion that a single failed swap localizes the
cause is too strong: entity recognition, location extraction and selection remain
possible sources. Review agreement is not empirical validation.
