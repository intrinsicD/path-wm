# Observation-to-output runtime audit

This read-only diagnostic addresses the missing encoder cost in cached-feature
training and prepared-input decoder timings. It adds no fits and changes no
quality selector. Albedo stays excluded. Use seed9107 checkpoints for five fixed
packages: CNN P1, ViT P1, ViT D2early, ViT D2conditioned and ViT D3split. The last
three use the same original ViT P1 pose head to make output scope comparable.
Do not select a faster checkpoint from timing observations.

Input is the first32 existing COCO test RGB64 frames, already GPU-resident as
FP32 in[0,1]. Batch sizes1 and32. Measure RGB, foreground, pose and all three
outputs. Each timed invocation recomputes the encoder, its preprocessing and
appropriate decoder computation. CNN uses its two native grids; ViT resizes to224,
runs the pinned backbone, then uses final features with the same FP16 roundtrip
as fitted readouts. D2/D3 early input is computed by the same frozen patch embed
from the same preprocessed observation. No disk/cache feature reads are timed.
This is observation-to-output compute, not capture/JPEG/CPU-transfer/planning or
complete application latency. No claim about text-preserving input resolution.

Five warmup invocations followed by20 measurements per output/package/batch.
CUDA synchronize immediately around every invocation; preserve all observations,
median and p95 milliseconds per batch, normalized milliseconds per image, model
parameter counts, active CUDA allocation and incremental peak allocation. PyTorch
eval/no-grad, FP32, TF32off, four CPU threads, same GPU; only one GPU workload.
Clear unused models and allocator cache between packages. Desktop GPU use remains
an explicit timing environment limitation. Twenty repeats give a descriptive
sample, not a latency-service guarantee or cross-device performance claim.

Before measuring, compare each independently requested output with the same output
from the combined request on actual inputs (max absolute difference<=2e-5), and
fingerprint all participating frozen modules before/after. In development use only
the already available ViT D2early package on CPU, batch1, one warmup/two measurements;
exercise checkpoint load, output-equivalence, raw receipt and verified HTML. This
uses existing tested forward paths and introduces no trainable behavior; no
implementation-mirroring unit test is added. Commit the development slice before
the formal GPU audit. Formal starts after the geometry queue completes, seals all
source/checkpoint/data identities, caps at10minutes and preserves partial results.

Every standalone run refreshes the canonical dashboard. Report results in a
separate source-backed table, keeping prepared-input timings distinct.
