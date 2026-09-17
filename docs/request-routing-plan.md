# Frozen request/evidence routing comparison

16 September2026. Follow the request-meaning suite result: balanced TaskInterpreter
training changes output form, but generated first words remain unchanged and both
sources fail joint accuracy. Test the already proposed routing diagnosis before
more fitting. Preserve the shared latent design and every source weight.

## Fixed protocol before execution

Sources: `runs/request_meaning_v1/seed720{1,2}/balanced`, fixed final checkpoints.
No fitting, optimizer steps, checkpoint selection, downloaded data or new neural
modules. One factor: how the question enters observation text. The task path always
receives the real request.

- `full`: unchanged question in both observation and task paths.
- `neutral`: observation question is `.`, task request unchanged.
- `masked`: observation question is one period per original UTF-8 byte; preserves
  actual byte-token sequence length, validity and times, not token statistics.

Keep existing textual evidence, non-text arrays and times untouched. Only replace
the explicit question substring. Labels, answer choices and fixture IDs never
enter neural inputs. Same question wrapper and fixed task metadata in all routes.
Use existing request corpus,16 reserved development VID.order clips,16 direct and
4 separate order-stress requests, three paired draws with the existing19701 seed
mapping. Same omitted-video/last-frame/constant-request controls as before. All
rows and full EOS-terminated generated strings saved. Exactness requires complete
string equality plus EOS; word count and first word remain separate diagnostics.

Primary screen for each candidate, in BOTH sources and each draw: first, sequence
and both-opposite-request exact >=80%; paired joint benefit >=10pp versus full;
first-word content decline <=5pp. Omitted/last-frame first and sequence exact <=60%;
constant-request joint<80%. Order stress reported separately, not substituted.
Broader25-task quick suite for all6 source/route combinations, fixed9401 seed.
No passed task lost and no accuracy/paired/source-gain decrease>10pp vs full.
All weights and source hashes unchanged, fresh full-route saved answers and quick
arrays equal previous balanced runs. Both full baselines are recomputed because
code-based comparison contracts change. Defaults stay full regardless of this
small development result; a passing signal would motivate matched training and
independent validation, not establish general understanding.

Save posterior logits, physical tokens and final working tokens aligned to every
request answer. Measure paired opposite-request state equality and differences;
neutral/masked physical states must match within equal-length pairs under the same
draw. Working differences alone do not show useful task interpretation. Disabling
duplicate observation conditioning changes the distribution seen during training;
a loss does NOT show that training a separate route cannot work, and a gain does
NOT uniquely identify harmful interference. Prior fixture reuse, shared vocabulary
and repeated draws/wordings are explicit limitations, not independent samples.

## Implementation and checks

Extend UnderstandingData.inputs with a small optional question mode. Reuse the
existing request-evaluate and understanding recipe stages. Log routing separately
from the scoring contract so deliberate variants can be compared under the same
fixtures/metric contract; include it in run identity and source/evaluation metadata.
Non-full routing requires an actual instruction path; fail before creating output
otherwise. CLI rejects use for unimplemented training stages. Default behavior
remains exact. No parallel evaluation framework or generic abstraction.

Essential checks before implementation: preserves textual evidence/byte length/
non-text tensors; actual model task path preserves physical state across opposite
questions while working state remains request-sensitive; neutral suite invocation
passes real requests and records the route; invalid sources fail before writes.
Record focused and full checks, exact source preservation and raw-array audits.
Actual Claude public-only methodology review and reconciliation under
`runs/reviews/request_routing_v1`; no private code, photos, data or results exported.

Budget:12 formal evaluations <=300s each, total<=1800 process seconds, PyTorch
allocation<6GiB, artifacts<500MiB, free disk>=500MiB. Full CPU software checks and
reviews separate; no training this iteration. If a process exceeds budget stop and
report remaining work; do not silently reduce populations/gates. Every completed
run owns raw evidence, source snapshot, checkpoint and standalone report. Use the
existing report renderer; structural/media checks and chart inspection. Interactive
browser QA availability must be disclosed.

Claude review narrowed two concerns: tokenization is explicitly byte-based rather
than BPE, and full-string+EOS equality rules out length-only false passes. Inspect
actual token counts/masks. Physical tokens/logits must be BITWISE equal within
paired neutral/masked calls; any mismatch stops the interpretation and triggers
implementation/nondeterminism debugging, never a relaxed tolerance. All effect
gates apply in each of three draws and both sources. Reused fixtures remain
exploratory development evidence. No unique causal attribution or confirmation
claim. Exact briefs, criticisms and reconciliation receipts are retained.

Implementation checks:21 focused tests pass. A GPU smoke check on both fixed
sources confirms64 exact comparisons: capturing state leaves working tokens
unchanged; neutral/masked physical pairs are bitwise equal on all three draws;
model/source hashes stay exact. Same hardware/environment and scalar core-call
batching as the reference, no tolerance relaxation. Claude acknowledged the
remaining equality/draw clarifications; actual CUDA equality remains measured.

## Results — completed16 September, reviewed17 September

[Standalone report](../runs/request_routing_v1/report.html),
[raw comparison](../runs/request_routing_v1/comparison.json),
[verification](../runs/request_routing_v1/verification.json).
All12 fixed evaluations completed at source0144cc9; no neural training.

| Source / observed question | First-color exact | Sequence exact | Joint exact | Quick suite |
| --- | ---: | ---: | ---: | ---: |
|7201/full|46.875%|42.1875%|35.9375%|0/25|
|7201/neutral|53.125%|42.1875%|35.9375%|0/25|
|7201/masked|53.125%|40.625%|34.375%|0/25|
|7202/full|84.375%|46.875%|46.875%|1/25|
|7202/neutral|100%|50%|50%|1/25|
|7202/masked|100%|50%|50%|1/25|

Worst-of-three-draw values; same development clips, not independent repetitions.
Both candidate routes fail primary and minimum-benefit gates. In7202, neutral
routing loses16.67pp on MIX.agreement.text+image+video; masked loses16.67pp on
MIX.agreement.text+video. Source7201 improves VID.order choice scores and recovers
33.33pp on REAL.scene.image, but no new quick task passes. Defaults stay full.

All768 opposite-request pairs per source/candidate have bit-identical physical
tokens and posterior logits; working states remain request-dependent. Weights
remain identical across9,096 tensor comparisons. Fresh references exactly reproduce
3,072 old answer rows and600 broader-suite arrays.584 full software tests pass in
609.90s;64 GPU smoke checks pass.98,905 raw checks,13 standalone reports and373
embedded media checks pass. Plot visually inspected; interactive browser QA not
performed. Formal processes total1,045.66s; peak torch allocation269.18MiB, excluding
CUDA runtime/reservation. No formal budget exceeded.

Exploratory aggregation of saved outputs (no additional model runs): for7202 both
candidate routes give the correct first word in every full-condition answer.
Exactly192 of384 sequence requests fail by ending after that first word; none of
those failures supplies the wrong second color. This narrows the next question to
request-dependent continuation/termination in that source, while7201 still has
content/order errors. It does not prove where a representation loses information.
See `error-analysis.json`. No further fits or changes to gates.

The user requested a new-session handoff before another experimental slice.
See [session handoff](session-handoff.md) for remaining review and proposed next
experiments. One untested API edge identified during review: full observation text
is stripped, whereas masked length currently counts raw question bytes. Questions
with trailing whitespace need a focused check/fix; current normalized fixture
questions are unaffected. Do not change model defaults or rerun completed runs to
address this preprocessing-only edge.
