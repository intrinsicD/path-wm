# PATH-WM — new-session handoff

Updated17 September2026. Repository: `/home/alex/Documents/path-wm`, branch `main`.
Read this first, then `CLAUDE.md`, `docs/experiment-workflow.md` and
`docs/project-state.md`. The user requested this handoff before starting a fresh
session. Preserve completed experiments; do not restart the research from scratch.

## Objective and working agreement

Build a modular multimodal world model that can learn and run on Alex's own
RTX3050 (8GiB VRAM). Keep the shared latent reasoning/state design, interchangeable
modality encoders/decoders/adapters, persistent entity memory and graph direction.
Do not replace the system with a text-only thinker or assume bigger decoders are
the answer. Use the recurring understanding suite to select concrete failures,
form a falsifiable hypothesis, implement a small comparison, review with actual
Claude, test and repair. Preserve existing alternatives and promote only demonstrated
improvements. Be economical with both assistants' weekly limits.

Current focus is **request interpretation, evidence routing and answer completion**.
Image/video codec, memory, planning and general modality understanding remain wider
open topics; they were not solved by this narrow experiment. The Dream-RSI-derived
experiment scheduling pilot is already complete and optional; see
`docs/replay-exploration-plan.md`. It is separate from agent runtime thinking/memory.

## What just finished

Prior iteration (`docs/request-meaning-plan.md`) trained only the existing
TaskInterpreter on balanced German first-color versus two-colors-in-order requests.
Both encoders and the rest of the model stayed frozen. Novel paired exact answers
rose from0% to35.94%/46.88% in two source checkpoints, still below80%. First words
were unchanged: the gain was output form, not video content. This selected the next
question: does putting a request into BOTH observation text and the task path cause
avoidable sensitivity?

The **frozen request-routing comparison is now complete**:

- Source models: `runs/request_meaning_v1/seed7201/balanced` and `seed7202/balanced`.
- `full`: original question in observation text and task interpreter.
- `neutral`: observed question replaced with `.`, real task request retained.
- `masked`: one period per question UTF-8 byte in observation; real request retained.
- Textual evidence, other modalities and times stay unchanged. No new model module,
  neural optimizer update, parameter fitting or checkpoint selection.
-16 reserved development video clips,16 direct and4 separate phrase-order requests,
  three paired draws; existing omission, last-frame and constant-request controls.
- All three routes also ran the broader25-task quick suite for each source.
-12 formal runs completed. No experiment/test process remains running.

Protocol and results: [request-routing-plan.md](request-routing-plan.md).
Report: [request_routing_v1/report.html](../runs/request_routing_v1/report.html).

### Main measurements

Values are worst of three draws; draws and wording variants reuse clips.

| Source / route | First-color exact | Sequence exact | Both requests exact | Quick suite passes |
| --- | ---: | ---: | ---: | ---: |
|7201 / full|46.875%|42.1875%|35.9375%|0/25|
|7201 / neutral|53.125%|42.1875%|35.9375%|0/25|
|7201 / masked|53.125%|40.625%|34.375%|0/25|
|7202 / full|84.375%|46.875%|46.875%|1/25|
|7202 / neutral|100%|50%|50%|1/25|
|7202 / masked|100%|50%|50%|1/25|

Both variants **fail adoption**: neither reaches80% joint exact or10pp paired
benefit in both sources. For7202, neutral loses16.67pp on
`MIX.agreement.text+image+video`; masked loses16.67pp on
`MIX.agreement.text+video`. Source7201's task-only route improves video-order
choice scores and recovers33.33pp on `REAL.scene.image`, but no new task passes.
All model defaults and broad validation colors remain unchanged.

Important error breakdown from SAVED predictions, not a new experiment:
for7202/neutral and7202/masked, every full-condition generated first word is
correct. Exactly192/384 sequence requests fail by ending after that first word
(e.g. expected `rot blau`, generated `rot`). The other192 are complete and correct.
Source7201 still has order/content mistakes and sometimes old symbolic phrases.
See `runs/request_routing_v1/error-analysis.json`, including per-wording breakdown.
This motivates inspecting request-dependent continuation/EOS, but does not by
itself identify whether the task interpreter, shared working state or decoder is
responsible.

### Integrity, cost and limits

-584 full software checks passed at implementation commit `0144cc9`.
-21 focused checks and64 exact GPU smoke checks passed beforehand.
-3,072 old answer rows and600 old broader-suite arrays reproduce exactly.
-9,096 source tensor comparisons show unchanged model weights.
- All768 opposite-request pairs in each candidate/source have bit-identical
  physical tokens and posterior logits. Working states remain request-sensitive.
-98,905 raw audit checks;13 standalone reports;373 embedded media checks.
- Chart visually inspected; reports structurally checked. Interactive browser QA
  was not performed. Existing renderer was reused.
- Formal process time1,045.66s; full tests609.90s, overlapping formal runs. These
  durations must not be summed as elapsed wall time. Peak torch allocation
  282,256,896bytes (~269.18MiB), excluding runtime/reservation. Artifacts~67MiB.
- No neural training. Diagnostic ridge readers in the broader suite are separate
  from actual model answers.
- These are repeatedly used DEVELOPMENT fixtures, shared vocabulary and only two
  source checkpoints. Do not claim independent confirmation, natural video or
  general-language understanding, or count repeated wordings as independent data.
- Neutral/masked routes change the training distribution. Their failures do not
  show that training separated routing is impossible; gains do not uniquely prove
  harmful interference. Masked controls sequence length, not token statistics.

## Implementation and Git state

Committed on `main`:

- `f171821`: preregistered routing plan and essential red tests.
- `0144cc9`: implementation and focused/GPU verification. All formal measurements
  and the full584-test pass use this commit; each run retains source snapshots.
- Later handoff/results commits contain documentation/recordkeeping. Inspect
  `git log -5 --oneline` and `git status --short` on entry. Nothing was pushed.

Main changes:

1. `pathwm/data/understanding.py`: `UnderstandingData.inputs(...,
   question_mode="full"|"neutral"|"masked")` replaces only the explicit question;
   actual textual evidence is preserved. Default is unchanged.
2. `pathwm/evaluation/request_meaning.py`: `validate_request_route` rejects altered
   routing without an actual instruction source, before creating output files.
3. `experiments/modality_readout.py`: `--observation-question` for
   `request-evaluate` and `understanding` only. Training stages reject it. The
   request evaluator records actual observed text, task request and token counts,
   plus aligned posterior/physical/working arrays in `request_states.npz`.
4. `pathwm/evaluation/understanding.py`: applies and logs routing while retaining
   scoring/fixture comparison contracts. Route is intentionally a variant field,
   not a reason to refuse a controlled comparison.
5. `tests/test_request_routing.py`: evidence preservation/byte lengths/no label
   leakage, actual physical/task-path behavior, invalid-source guards and broader
   suite wiring. Existing request and understanding tests also pass.

`Core.forward` itself is unchanged: encode/observe → physical state → optional
TaskInterpreter goal tokens → shared Thinker → output decoder. Task requests still
reuse the existing text encoder. This iteration changes evaluation preprocessing,
not the reasoning architecture or codec weights.

## Claude collaboration

Actual installed CLI, isolated temporary cwd, no tools/MCP/browser/session persistence.
Reviewed public hypothetical methodology only; no private code/data/results exported.
Exact briefs, responses and receipts:
`runs/reviews/request_routing_v1/{public,reconcile,final}-*`.

Claude retracted BPE concerns after the byte-tokenizer clarification, accepted
full-string+EOS scoring, and accepted bitwise state checks plus three paired draws.
Retained caveats: shifted input statistics, repeated development fixtures,
non-independent repetitions, and possible nondeterminism. Actual physical equality
and old-route reproduction passed; no tolerance was relaxed.

Follow `docs/claude-collaboration-workflow.md`. Reuse the scoped CLI method when a
new consequential decision needs review; do not invent Claude's opinion or send
private repository contents. No need to review routine formatting with Claude.

## Exact next steps

1. **Read the report/error breakdown; do not rerun the12 finished evaluations.**
   Sources, outputs and gates are already audited. No variant is adopted.
2. **Small preprocessing review remains:** `inputs` strips assembled observation
   text, but masked length currently counts raw question bytes. Trailing whitespace
   can therefore break the intended length match for future questions. Add a focused
   regression test, then the smallest normalization-aware fix. Current fixture
   questions have no trailing whitespace, so measured comparisons are unaffected.
   This edge has NOT yet been tested/fixed. Keep it separate from neural results.
3. **Propose the next bounded learning diagnostic using the suite:** on the preserved
   models, compare EOS versus continuation probabilities at the first-color prefix,
   crossed with first-only/sequence requests and wording families. Separate correct
   content, response form and termination. Start by aggregating existing outputs;
   any new model calls/training need a fixed protocol before execution. This is a
   proposal, not implemented or approved as a successful repair.
4. If continuation evidence warrants training, compare a small request/readout repair
   against a matched control, preserve other tasks/legacy outputs and retain the
   weaker-source video-content problem as a separate failure. Do not enlarge encoders
   or decoders solely because overall task scores are poor.
5. Keep broader multimodal regression in the loop; the agreement regressions prevent
   adopting the cleaner route now. Do not spend more fits optimizing only word count.

The architecture atlas follow-up now points at this result/handoff, with discussion
and validation statuses preserved. Main project state and routing plan contain the
result. The root report uses the existing renderer; no UI framework was introduced.

## Commands and artifact map

Run from the repository root. `.venv/bin/pytest` does not exist; use Python's module
entry point. Set modest CPU threading:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m pytest \
  tests/test_request_routing.py tests/test_request_meaning.py \
  tests/test_request_readout.py tests/test_understanding_suite.py -q
```

Example for a NEW output directory only (not necessary to repeat now):

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m experiments.modality_readout \
  --stage request-evaluate \
  --core runs/request_meaning_v1/seed7202/balanced \
  --understanding-suite data/understanding_v1/full \
  --observation-question neutral --seed 9701 --device cuda \
  --output runs/my_new_request_routing_check
```

Broader suite uses `--stage understanding`, prepared `data/understanding_v1/quick`,
seed9401, and optionally `--reference` pointing at a FRESH full-route quick run with
matching code/fixture/scoring contract. Historical contract hashes differ after
code changes; compare original raw arrays explicitly instead of bypassing guards.

Authoritative current artifacts under `runs/request_routing_v1/`:

- `comparison.json`, `verification.json`, `error-analysis.json`.
- `seed720{1,2}/{full,neutral,masked}/request_meanings.json`, `request_states.npz`,
  source snapshots, `last.pt`, `report.html`; `quick/` holds each25-task suite.
- `execution.json`, `software-cost.json`, `full-software-tests.txt`,
  `test-collection.txt`, `focused-tests.txt`, `red-tests.txt`, `smoke.json`.
- `audit.py`/`audit.txt`, `build_summary.py`, `execute.py`, `smoke.py` are ignored
  experiment-local utilities, not a new library interface. `execute.py` targets
  existing run paths: do NOT rerun it unchanged.

`runs/` and datasets/checkpoints are intentionally outside Git. A new local session
on this machine has them; cloning the Git repository alone will not recover them.
No ongoing jobs need resuming. The end-of-turn research record lives in `ara/`;
only read/write it during the research-manager epilogue, not mid-task.

## Suggested opening prompt

> Read docs/session-handoff.md and CLAUDE.md. Continue with actual Claude using the
> existing testsuite. Preserve the completed routing experiments and defaults.
> First fix the documented preprocessing edge with a focused test, then define the
> smallest diagnostic that separates request-form interpretation from premature EOS
> while preserving content and broader modality behavior. Plan, implement, review,
> test and fix; do not rerun finished experiments without a new reason.
