# Frozen-writer recall repair

13 September 2026. Alex approved repairing memory reading and workspace formation
while preserving the encoder and stored-state writer. This is a bounded native
facts/image-output experiment, not evidence of general entity understanding.

## Comparison declared before training

Use both completed balanced-relocation source checkpoints7801/7802 unchanged.
For each source compare identity against fixed channel normalization of retrieved
memory tokens. Both arms continue the SAME existing weights, with the same sampler
seed8001, optimizer, update count and trainable parameters. Freeze the entire model
except Thinker, factual readout and image feature producer; the image decoder head,
all visual encoders, initial state, state updater, dynamics, direct diagnostic and
all prior calibration buffers stay fixed. No auxiliary write loss or direct-reader
training. Alternate ordinary and reset query losses, as in the original recipe:
factor CE + foreground-weighted RGB MSE +0.1 standardized target feature MSE.
Targets enter losses only. There is no target/encoder-detail input at recall.

Preserve raw snapshots, keys, top-k selection, and state context. The adapter only
subtracts/divides retrieved values using channel moments over BOTH stored snapshots
and all tokens from TRAIN histories, float64 accumulation, std floor1e-4 (fixed from
the probe protocol; not tuned). Identity has zero/unit buffers and the same class.
No timestamp or positional features are added. Lack of explicit snapshot time is
an unresolved alternative, not changed simultaneously. Empty memory remains empty.
This fixed-budget comparison can show useful improved fitting or output accuracy;
it cannot establish asymptotic superiority or recover information proven absent.

Train128 pairs seed7701, validation32 pairs7722, test64 pairs7723, relocation
curriculum. All16 tuples/motions familiar; fresh noisy backgrounds only. Exact
frame hashes must be disjoint. Source checkpoints are evaluated on this same new
test population. A separately reported direct readout applied to the same stored
values (the existing native factual head and the previously fitted probes) is a
conditional access control, never a perfect information ceiling. No probe retuning.

Four runs1536 updates, batch16, AdamW lr0.001 wd0.0001, global trainable gradient
clip1, FP32, max300 training-loop seconds each,1200s total. Final checkpoint only;
validation128 updates. Same seeds/samples within each source pair. GPU cap4GiB with
1GiB headroom. Development is a separate16-update run, seed18001/data17701/17722/
17723; no development accuracy gate or architecture tuning. One formal768+768
resume; CPU exact resume/standalone reload regression. Commit source before formal
runs; do not edit source while any test/training/evaluation runs.

## Gates and causal checks

Use existing relocation screen: ordinary AND reset facts/images>=90%, complete
selection and relocation pairs>=80%, +30 points against erased history, reset bank
erasure, cue erasure and later-view erasure; swapped bank alternate answers>=80%.
Weighted pixel error must beat background and pair-mean baselines. Both sources
must pass to promote a repair. Separately call calibration beneficial at this budget
only if reset factual AND image accuracy each improve>=10 points over identity
retraining in BOTH source pairs. Report native baselines, absolute scores and all
failures even if neither gate passes. Do not select a winner by test outcomes.

Audit unchanged all frozen parameters/buffers, source file hashes, raw bank values,
train-only calibration, detached storage, no cross-episode mixing, erased/swapped
causality, independent pixel classifications and saved-logit metrics. Strict export
loads without donor data; GPU replay max error<=1e-4 and categorical equality.
CPU replay uses the existing1e-4 tolerance but preserves known raw-encoder device
limits as a separate result. Every run keeps raw arrays, weights, optimizer/RNG/
sampler, curves, timing/resources and a verified standalone report. Renderer is
unchanged; structural QA only because browser QA remains unavailable. Inspect PNGs.

Claude's public-only review accepts the bounded comparison with caveats: report
learning curves and fix the std floor before evaluation; improved conditioning or
faster fitting is a valid bounded outcome, not proof of erased-information recovery.
No private code, data or measurements were exported.
