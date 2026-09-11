# Single-event facts through the agent reader

Alex authorized continuing with Claude, implementation, review, tests, fixes and
iteration on 11 September 2026. Reuse the fact dataset, loss, trainer, final cache
and report in `experiments/multimodal.py`. Add `--fact-reader event`; direct remains
the default and its successful historical results remain intact.

## Declared comparison and limits

Use the same 96 training / 32 held-out semantic entity/location combinations from
the direct fact plan. Construct the direct encoder, two attention queries, attention
readout and classification heads first under seed 23. The event arm reuses these
exact initial tensors, then adds the existing BeliefAgent components. No pretrained
weights. Extra components change capacity and optimization, so this is a whole-path
trainability control, not causal attribution to an individual component.

Each forward starts a fresh empty session, observes one canonical text fact at
time zero, and uses the normal event commit and recent-memory write. Pass a fixed
label-free instruction through the existing task interpreter, then two ordinary
thinking/memory-read rounds. Apply the matched attention readout and entity/location
heads to working tokens only. No truth labels, per-fact metadata, raw-text shortcut,
or evaluator fields enter this final readout. Existing event/reader modules remain
unchanged. Only the equal-weight entity/location CE trains this control; no new
memory losses or model architecture.

Training retains ordinary categorical draws. Evaluation retains sampling but seeds
each batch with `seed + 1000000 + start_index`, inside the existing RNG-restoring
evaluation context. This is one repeatable realization for the fixed batch order
and batch size, not an expectation, calibrated uncertainty or noise-robustness claim.
Both terminal split passes use that declared rule. Their identity, raw logits and
results are cached as before. No development checkpoint selection or calibration.

Seed 23, width 32, batch 16, 512 updates / 8192 sampled presentations; AdamW
lr0.0003, weight decay0.01, clip1, FP32/two CPU threads. Training-only evaluation
at initialization and every32 updates; terminal train/development evaluation.
Keep the direct-control gates unchanged: train entity/location each>=95%, joint>=90%,
mean NLL<=0.35; development each>=90%, joint>=80%, mean NLL<=0.5. Only if extraction
passes, run the same fixed cached-logit selector and its both-held-out query>=90% /
paired>=80% gates. This selector does not demonstrate learned entity-query binding.
One-event success does not demonstrate retention across later events or compression.

If extraction fails without a software defect, permit exactly one predeclared
fresh-initialization comparison changing only lr to0.001, with identical seed,
population, update budget and batch order. Preserve both attempts. At most450 active
CPU seconds per attempt,900 total, with cumulative cooperative pause/resume accounting.
No third run, extra seeds, new objective or gate relaxation. Tests and report QA are
separate. Never change shared source during an active run.

## Essential checks and review

Red checks: exact shared initialization, normal event/interpreter/think calls,
constant label-free metadata, final output boundary at working tokens, gradients
to source encoder/updater/thinker/heads, fresh-state isolation, fixed evaluation
repeatability without training RNG consumption, and exact pause/resume/cache reuse.
Run the full CPU suite and real CLI check, then the capped pilot(s). Reports name
the actual reader and sampling contract, distinguish the explicit selector, and get
structural/browser QA. Preserve raw metrics, snapshots, checkpoints and receipts.

Two short abstract Claude exchanges use the adopted public-only boundary; receipts
are `runs/reviews/continuation_2026-09-11/event-*`. Accept the whole-path capacity/
optimization confound. Reject unused parameter padding as a remedy and reject a
pre-training decodability requirement: neither establishes causal identification.
Record a failed gate without inventing a localized representation diagnosis.

Claude explicitly withdrew the padding and pre-training-decoding recommendations.
Its remaining cautions are addressed by the declared contract: zero is the fixed
input timestamp, not deterministic dynamics; the intervention replaces exactly
the final working-token input. No further review round is needed for these points.

## Outcome, 11 September 2026

Red checks were committed in `aed1688`; implementation is `f09400c`. All 86 CPU
tests passed, including both fact-reader resume/cache paths. The initial forward
check has finite losses and gradients in the agent, matched attention reader and
both heads. Tests verify exact shared initialization, ordinary event/interpreter/
think calls, constant task metadata, fresh-state reset, gradients to the updater
and thinker, and fixed evaluation draws without training-RNG consumption.

The first boundary test incorrectly expected a fresh event ordinal of zero. The
existing agent starts at -1 and `observe()` commits ordinal zero; the test was
corrected to that contract. After the first negative pilot, the gradient test was
strengthened to retain the first encoder call's scale tensors. It passes: gradients
reach the actual observed fact, not merely the later constant instruction that
shares the encoder. No experiment source changed between the two pilots.

Both attempts finished all 512 updates / 8192 fact presentations:

| Reader / learning rate | Held-out entity | Held-out location | Held-out joint | Mean held-out NLL | Active seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prior direct / 0.0003 | 32/32 | 32/32 | 32/32 | 0.211649 | 10.3502 |
| Agent reference / 0.0003 | 1/32 | 8/32 | 0/32 | 2.431158 | 56.5757 |
| Agent comparison / 0.001 | 0/32 | 24/32 | 0/32 | 2.171213 | 52.8277 |

Training entity/location/joint accuracy was 3.125% / 25% / 1.0417% for the agent
reference, and 5.2083% / 75% / 4.1667% for the comparison. Both extraction gates
failed in both runs; the two-record selector was correctly skipped. The higher
learning rate learned some location discrimination but did not recover entities.
Total new active training/evaluation time was 109.4034 seconds, excluding the
declared checkpoint/report overhead. No third run, extra seed, new objective,
calibration or final-test population was used.

The audit confirms the same initial model hash and final sampler state between
the two agent runs, with only learning rate changed in resolved settings. The
prior direct control and agent reference share the data identities and core
settings; shared component initialization is covered by the exact tensor check.
All trained source/snapshots match current source. Both CLI resumes preserve the
prediction caches, result JSON and metric rows byte-for-byte. Saved logits reproduce
all reported entity/location/joint metrics and NLLs; checkpoint floats are finite.

Both standalone reports passed structural and 1280x720 browser checks, including
expanded examples, with no broken images or horizontal overflow. Receipts, raw
results, checkpoints, exact review exchanges and screenshots are bound by
`runs/event_fact_v1/verification.json`:

- [Reference report](../runs/event_fact_v1/reference/report.html)
- [Learning-rate comparison](../runs/event_fact_v1/lr_control/report.html)

The whole existing agent route failed this fixed-budget task. This does not isolate
the source encoder, categorical bottleneck, task interpreter, memory reader or
optimization dynamics. Source review confirms recent-memory writes detach stored
records even during replay; replay keeps newly created compression/consolidation
graphs, not original observation graphs (`HybridMemory.write`). The live categorical
path still gives observed-fact gradients, as tested. This existing storage policy
was preserved and is not established as the cause of failure.

Next proposed single-factor test: initialize only the source text encoder from the
successful direct-control checkpoint, keep it trainable, and leave the agent route,
classification loss and budget fixed. This tests whether learned input features
help downstream learning. It is not implemented or run here; neither new memory
losses nor compression changes are justified by these results alone.
