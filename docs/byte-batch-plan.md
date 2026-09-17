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

Protocol recorded. Candidate and comparisons not yet run. The request-form/EOS
learning repair remains a separate next step; do not infer it is solved here.
