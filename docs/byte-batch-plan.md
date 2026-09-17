# One-transfer byte batches

17 September 2026. The understanding/request suite repeatedly builds observation,
task-instruction, metadata and answer byte batches during complete model updates.
Inspection finds a device allocation/copy per string plus the padded destination.
Test one host-built rectangular batch and one tensor transfer, without changing
token IDs, padding, masks, API, neural operations, model or learning objective.
This is a performance slice before the separately proposed request-form repair.

## Fixed comparison before implementation

Reference: `e7c51d0` byte batching (same implementation as `0144cc9`). Candidate:
encode each string once, pad its Python integer list on the host, construct one
long tensor on the requested device and derive validity from nonzero IDs. Keep
empty-batch rejection, empty strings, generators, Unicode, right padding, BOS/EOS,
device placement and RNG exact. No feature/model-state cache or frozen sampling.
Use independent old-expression references for compatibility checks; unchanged
behavior tests may pass before implementation. No speed assertion based on timing
in unit tests. Preserve reference source and execution scripts before measurement.

Sources: fixed `runs/request_meaning_v1/seed720{1,2}/balanced` checkpoints, with
existing balanced VID.order calibration rows only. No validation/test data enter
training. Batch8, Adam0.001, interpreter-only, gradient clip5; normal core/decoder
and global RNG behavior. Source7202 is the primary timing source;7201 checks
transfer with the identical fixed procedure. Starting/model/optimizer/RNG state
restored outside each timed block. Predeclare16-update training blocks, nine paired
repeats with alternating order after three warmup blocks per arm. Replay identical
calibration clip/request indices and CPU/CUDA RNG streams. Verify all per-update
losses, final model/optimizer tensors and RNG states BITWISE equal each pair.

Inference: same source, first8 balanced calibration rows, data construction plus
actual requested core and16-token greedy generation, no gradients;16 calls per
timed block, same3 warmups and9 alternating paired repeats. Preserve stochastic
core draws and all generated token arrays. Synchronize before/after timed CUDA
regions. Record raw pairs, phase seconds, setup/reset costs separately and peak
allocation. Do not claim that accumulated independent repetitions constitute a
training budget or independent model-quality evidence. Completed training blocks
own the measured final checkpoint, raw losses/settings, source snapshot and report.

Per-source gate: training OR inference paired median latency reduction >=5%, with
neither workload slowing by >5%, AND all exact compatibility checks passing. Enable
the shared implementation only if both sources pass. All endpoints remain visible;
no threshold, population, repetition or batch-size changes after measurement.
Tokenizer-only CPU/CUDA microbenchmarks (batch1/8/64, same recorded strings) are
secondary; no training/inference speed claim from them. No quality improvement or
new capability claim; same weights/inputs imply preserving current limitations.

Essential checks: independent ragged/Unicode/empty/generator reference, exact
source routes and seeded model-update compatibility. Full existing software suite
and its resume/report tests must pass. Actual Claude public-only methodology review
and reconciliation saved under `runs/reviews/byte_batch_v1/`. No shared source edits
while measurements run. If a gate fails, restore the original byte builder and
preserve the negative trial; do not repeat until a favorable time appears.

Budget: <=600 process seconds for paired measurements and exact audits, <6GiB
PyTorch allocation, <250MiB artifacts. Full CPU tests separately<=1200s. No source
data/download changes. Existing standalone report renderer, structural checks,
figure inspection; disclose if interactive browser QA is unavailable.

## Progress

Candidate is committed as5e7979a after independent reference checks;54 focused
tests pass. Actual Claude review/reconciliation identifies alias coverage, exact
RNG/gradient preservation and warmed allocator behavior as required checks.
PAD0, BOS1, EOS2 and byte+3 remain unchanged; all input strings include BOS/EOS.
`seed_everything` resets Python/NumPy/CPU/CUDA streams; sample generators have
explicit fixed seeds. Raw repeated timings are retained; no inferential claim.

First measurement attempt `runs/byte_batch_v1` stopped after a PyTorch warning
revealed noncontiguous GRU parameters introduced by benchmark model deep copies.
This is a benchmark setup defect, not tokenizer evidence. Partial7202 changes
(training+2.23%, inference+1.47%) are preserved, not used for adoption. Completed
representative training blocks retain their checkpoints/reports. Before a fresh
attempt, explicitly flatten RNN parameters outside timed regions after each copy,
as device loading does for ordinary models, and turn that warning into an error.
Fresh attempt lives under `runs/byte_batch_v2`; same cohorts, seeds, repeats,
workloads and acceptance gate. No tuning to previous times. Keep combined
attempt cost visible and within the original600s measurement budget.

The request-form/EOS learning repair remains a separate next step; do not infer
it is solved by this performance work.

Corrected tokenizer result: exact checks pass, but both sources fail adoption.
7202 training/inference improve3.27%/1.95%;7201 improves1.19%/1.02%. Preserve v1
and v2 artifacts, revert the byte builder. Combined elapsed upper bound344.63s.

## Second hypothesis: bounded frozen metadata reuse

Before measurement, register a separate mechanism under the remaining original
budget (<=255s for its formal comparison). Repeated task metadata is identical
while MetadataEncoder is frozen. Memoize only the most recent ordered serialized
batch when the encoder and every child are in evaluation mode, every parameter is
frozen, and autocast is disabled. Key includes records, parameter identities and
versions, device and dtype; return a clone so caller mutation cannot poison reuse.
Changed records/weights/load/device/training state invalidate reuse. No graph,
checkpoint buffer, input/latent cache, new parameters or gradient-path change.
No support for bypassing PyTorch version tracking via `.data` or external storage
mutation; callers doing that must disable reuse. Provide an explicit opt-out.

Keep the SAME source/indices/16-update or16-call blocks,3 warmups,9 paired repeats,
5% end-to-end gate and exact comparisons. Reference is original metadata forward,
candidate frozen reuse; byte batching remains original in both. GRU layout is
prepared outside timing as repaired above. Reject unsupported precision modes;
test cache invalidation, caller mutation, frozen gradients, parameter loading,
changing batch/metadata, opt-out and bounded single-entry behavior. Trainable
metadata paths keep their original expression. No source edit during comparison.
This is a new hypothesis, not repeated timing of the failed tokenizer candidate.


## Final adoption decision

Metadata reuse also misses the unchanged5% gate:7202 training/inference latency
reductions -0.61%/+0.93%;7201 +1.36%/+1.67%. All31,266 pairwise exact checks pass.
A separate four-call real training-path audit confirms one miss followed by three
hits, so ineligibility is not the explanation. No general speedup claim is supported.
The candidate additionally fails a fresh-interpreter cuDNN submodule import test:
its precision-key guard assumes an attribute that can disappear after import.
The formal timing setup seeded precision after model construction, masking this
issue. Saved red receipt: `runs/metadata_cache_v1/import-order-red.txt`.

Remove the unearned caching machinery and restore the original MetadataEncoder.
Retain independent byte/metadata behavior, gradient, mutation, hook and inference
mode checks; candidate-only reuse-count assertions move with the rejected source
into the ignored experiment archive. This removes an abandoned feature, not a
passing claim for its failing speed or import-order checks. No cache flag remains
in the production API. Preserve both candidate sources, tests, raw results and
Claude receipts. No model weights, objective, default, or capability changes.
