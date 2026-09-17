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

## Frozen diagnostic results

All six evaluations completed at implementation commit `c959c63`, with no core
calls or neural updates. They reproduce all9,216 saved answer rows in22.56 process
seconds (including process startup, source loading and reports). Each run owns its
checkpoint, raw `continuations.json`, source snapshot and verified standalone
report. [Summary report](../runs/request_completion_v1/report.html).

For source7202/neutral, all384 full-condition sequence requests start with the
correct color. At that exact first-word prefix, EOS is the greedy legal token for
four of eight wording variants, in all16 clips and all3 draws per variant. Space
is greedy for the other four. Failing wording cells have mean space-minus-EOS
logit margins from-2.71 to-5.58; successful cells+1.51 to+6.05. Both color orders
show the same split. Thus these are systematic wording-associated decisions, not
observed near-ties. Masked routing has the same pattern. Every first-only request
chooses EOS correctly; every teacher-forced complete two-word answer chooses EOS.
The latter is a conditional intervention, not proof that the model can freely
generate the answer. See the exact questions, ranges and cells in `result.json`.

The weaker source retains content errors: first-word accuracy62.5% on the neutral
full-condition requests, sequence exact42.71% averaged over the repeated draws.
Its EOS rate after the correct teacher-forced first word is18.23% for sequence
requests, so stronger-source premature EOS is not a sufficient explanation of
both sources' failures. These means are descriptive, not the earlier worst-draw
acceptance endpoints. No scientific model variant, broad capability or color is
promoted; the historical quick suite remains the reference.

Raw audit:296,076 checks, including all9,216 original answer rows and reached-prefix
next-token consistency,5,010 exact prepared-fixture text/mask/time comparisons,
and4,548 unchanged checkpoint tensors. Six local run reports plus the summary use
the unchanged renderer; structural checks and figure inspection pass. Interactive
browser QA was not performed. Actual Claude review and reconciliation are saved;
peer agreement is distinct from empirical checks. Software suite status is recorded
separately in `software-cost.json` / `full-software-tests.txt`.

Final verification:605 full software tests pass in594.83s, including existing
training/resume/report paths;53 final focused checks pass. Targeted Ruff checks and
`git diff --check` pass. Architecture atlas regenerated with16 graphs and62 source
references; colors/scopes unchanged. Full tests overlap formal diagnostics, so
their process durations are not additive wall time. All compute/artifact budgets
were met. No training/evaluation job remains running.

## Use and next iteration

To run this diagnosis on another compatible saved cache, choose a NEW output:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m experiments.modality_readout \
  --stage request-completion \
  --core runs/request_meaning_v1/seed7202/balanced \
  --request-cache runs/request_routing_v1/seed7202/neutral \
  --seed 9701 --device cuda --output runs/my_completion_diagnosis
```

Source/cache identity and weights must agree. This stage checks saved generation
before interpreting probabilities; it cannot resume training or resample the core.
Do not rerun the six completed caches without a new reason. Optional decoder
slicing is available through `TextDecoder.generate(..., last_only=True)`; normal
recipe/runtime calls keep the original full-prefix default.

Next preregister a matched request-form/continuation repair with paired first-only
and sequence objectives, content and broader-task preservation, and fresh wording
controls. Existing inspected wording cells are development evidence, not a fresh
test set; do not simply fit the four failed templates and call that generalization.
Keep weaker-source video-content access as a distinct problem. No new fit is part
of this iteration, and no training-time improvement has been established.
