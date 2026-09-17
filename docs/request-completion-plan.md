# Request completion and decoder cost

17 September 2026. Suite-directed continuation of the completed routing study.
Preserve the shared model, sources, completed runs, acceptance gates and defaults
for scientific variants. This iteration has three small, separately checked slices.

## Plan before execution

1. Repair masked question byte lengths after the observation builder's existing
   trailing-whitespace normalization. Preserve full and neutral text behavior,
   evidence, leading/internal question whitespace, masks and observation times.
   Cover empty, whitespace-only and multibyte questions. Existing fixture outputs
   must remain exact; no finished evaluation needs rerunning for this fix.
2. Inspect frozen decoder continuation on the saved working states from all six
   `runs/request_routing_v1/seed720{1,2}/{full,neutral,masked}` runs. Use every saved
   row, original ordering and batches of 64. No new core samples or neural updates.
   Cross ground-truth first-word prefixes with first/sequence requests, routes,
   wording, draws and existing controls. Record raw next-token probabilities for
   EOS, space and the top legal token, their margin, and correct complete-answer
   termination. Also record the generated-first-word prefix as a distinct on-policy
   diagnostic; only prefixes actually reached by generation qualify as on-policy.
   Full-vocabulary probabilities and PAD/BOS-masked greedy decisions are distinct.
   Teacher-forced prefixes are interventions, not proof of usable content or a
   unique failing module. Group repeated rows by original clip/pair, not as new data.
   This is descriptive diagnosis: no capability gate or training decision is passed.
3. Reduce redundant inference work in the existing text decoder: after its causal
   self-attention, allow reading state and projecting vocabulary at only the final
   prefix position. Reuse the same forward path, weights, validation and attention
   implementation. Teacher-forced training and ordinary full logits/traces remain
   unchanged. This is not a KV cache or a change to the learned model. Benchmark
   against the original full-prefix generation with identical states, budgets and
   hardware; require all saved generated tokens/strings/EOS decisions unchanged.
   Check masked/NaN contexts, mixed EOS, budget termination, causality, state/RNG,
   and full-path gradients against the previous expression. Floating-point last
   logits may differ with changed matrix shapes: report max error, use FP32
   `atol=2e-6, rtol=1e-5`, never relax these after measurement. Optimization gate:
   numerical checks and exact generation on all saved rows, plus >=5% median
   latency reduction on the primary CUDA batch-64, 16-token request workload.
   If this fails, retain the original generation path; do not claim a speed gain.

No source modification during a running measurement. Save baseline/final timing
blocks with synchronization, 3 warmups and 9 alternating paired repeats on the
first 64 full-condition rows of source7202/neutral (fixed before inspection).
Also report batch-1 CUDA and CPU batch-64 timing; neither substitutes for primary.
Measure one unmodified full teacher-forced forward/backward baseline and final
path on the same batch/prefix as a compatibility check, not a training-speed claim.
No training, new corpus, fitted probes or checkpoint selection. Hash saved state
files, source checkpoints and decoder state before/after; reproduce original
generated strings before interpreting the cached states. Full suite correctness
and independent Claude methodology criticism are required before closure.

Compute budget: <=600 process seconds for frozen diagnostics, comparisons and
benchmarks, <6GiB allocated CUDA memory, <250MiB new artifacts. CPU software suite
has a separate <=1200s budget. Stop and record any overrun without narrowing the
population or relaxing checks. Every completed diagnostic run owns raw rows,
source identity, relevant checkpoint and a standalone report through the existing
renderer. Record result and report status separately. Existing renderer structural
QA is mandatory; disclose any lack of interactive browser QA.

Review: actual Claude via the established isolated CLI, hypothetical public-only
briefs, tools/MCP disabled. Exact briefs, responses and receipts live under
`runs/reviews/request_completion_v1/`. Verify and reconcile substantive criticism.

## Progress

Essential red checks saved and committed before their implementations. The first
52 focused checks pass. Claude acknowledged the architectural, probability,
ground-truth and exact-token clarifications. After the final fixed repetition and
effect-size clarification, it withdrew its decision-rule objection; uncertainty
intervals remain a preference, not a blocker for this narrow engineering gate.
All warmups are excluded; exactly 5% passes; same-protocol reruns use the same gate.

**First performance iteration, not adopted:** last-position slicing alone passes
all 9,216 saved answer reproductions, all exact greedy token comparisons and full
training outputs/gradients. It FAILS the primary speed gate: paired median CUDA
batch-64 reduction -3.67% (slower); batch-1 +5.02%, CPU batch-64 +9.77% cannot replace
the primary. Raw maximum logit difference 7.63e-6 passes the fixed mixed absolute/
relative tolerance. Receipts: `slice-only-verification.json`, `verify_slice_only.py`.

**Second performance iteration, fixed before measurement:** keep the same
population, budget, repetitions and adoption threshold. Combine vocabulary and
right-padding validity flags into one device-to-host boolean check on the valid
path, keeping the original specific errors on invalid inputs. Current forward
validates separately with up to three host synchronizations per generated token.
This consolidation preserves every numerical decoder operation and all forward
hooks, and applies to both ordinary training and inference. No trusted-prefix
bypass or mutable state. Compare the combined candidate with the ORIGINAL
validation and full-prefix expression, not with an already modified baseline.
Rerun token/logit/gradient checks and matched timings. Primary failure retains
the original generation default; do not tune gates. No training-speed claim from
the single compatibility timing. Diagnostics remain independent of this gate.

**Second performance result, not adopted:** same9,216 exact token/answer checks,
fixed logit tolerance and exact full training outputs/gradients pass. Primary
paired median change -1.94%; secondary CUDA batch-1 +2.36%, CPU batch-64 +9.14%.
Both candidates miss the declared primary gate. Restore original validation and
full-prefix generation defaults. Retain `TextDecoder.generate(..., last_only=True)`
as an explicit experimental path, not a promoted optimization. `combined-verification.json`
and its exact candidate source preserve the failed comparison. No GPU or training
speedup claim. The small CPU gain is descriptive and does not override the gate.
