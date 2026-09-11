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
