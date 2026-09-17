# PATH-WM — new-session handoff

Updated 17 September 2026. Repository `/home/alex/Documents/path-wm`, branch `main`.
Read `CLAUDE.md`, `docs/experiment-workflow.md`, this handoff and the current plan.
Preserve completed experiments; do not restart the research from scratch.

## Latest requested continuation: visual residual adapters

Start with the focused [visual-adapter handoff](visual-adapter-handoff.md) and
[design](visual-adapter-design.md), continued from2beb97f. Alex now wants one shared
trainable base plus per-application residuals trained jointly from the first update,
with equal status for every task. Feature versus weight residuals at multiple scales
is the open architecture question. Proposed first test: shared multiscale feature
readouts with small private residuals and matched head controls. The old separate-
task frozen/unfrozen Phase A is optional, not a prerequisite. RGB is equally weighted
if selected; no additional frozen RGB anchor enters the joint objective. Actual
Claude reviewed public methodology and acknowledged corrections; no training or
implementation occurred. Next select real labels/data and fill the joint contract.
Preserve completed learning work and its unresolved findings below.

## Objective and working agreement

Build the modular multimodal world model for Alex's RTX3050 (8GiB). Preserve the
shared latent reasoning/state design, interchangeable modality modules, persistent
entity memory and graph direction. Use the understanding suite to select concrete
failures, form a falsifiable hypothesis, implement a small comparison, review with
actual Claude, test and repair. Promote only demonstrated improvements. Keep the
Python library and readable experiment recipes; no new trainer/reporting framework.
Be economical with both assistants' limits.

Claude collaboration follows `docs/claude-collaboration-workflow.md`: actual
installed CLI in an isolated temporary cwd, tools/MCP/browser/persistence disabled.
Public hypothetical methodology only; do not export private code, data or results.
Save exact briefs/responses/receipts and reconcile substantive criticism.

## Latest learning iteration

[Paired request-boundary protocol/results](request-boundary-plan.md),
[report](../runs/request_boundary_v1/report.html). Four matched512-update fits use
only the existing9,936-parameter interpreter. Actual Claude reviewed the public
hypothetical design and reconciliation; receipts in
`runs/reviews/request_boundary_v1/`. Source snapshot2d43133 is preserved in
`/tmp/pathwm-boundary-20260917` and run snapshots. Final code63b7321 removes the
rejected additional boundary CE term/flag; paired calibration sampling, neutral
training questions and opt-in fresh-prefix evaluation remain. Defaults unchanged.

- Stronger7202 source: worst-draw joint exact56.77% untouched,76.56% matched ordinary
  CE,74.48% boundary CE. First-only accuracy98.44%,93.23%,95.83%; both continuations
  regress preservation and miss80% joint. Boundary-minus-control is-2.08pp.
- Weaker7201 source:37.5% joint in every arm. First-word joint accuracy62.5% in every
  arm;7202 is100%. This slice does not repair weak-source content access.
- Broad quick coverage remains0/25 and1/25, with seven unimplemented domains.
  Boundary7202 regresses REAL.scene.image accuracy and source-gain minima by33.33pp.
  No candidate, language/general capability or architecture-color promotion.
- All11,520 fresh/control/historical-order rows independently reproduce from saved
  working states;8,986 tensors,2,216 arrays,2,208 legacy arrays and94,940 primitives
  checked. Physical/posterior state and every non-interpreter tensor stay exact.
- Original formal cap stops at1500.46s during final7202 CE evaluation, after all four
  final512-update checkpoints were saved. That run remains stopped with its report.
  Separate213.83s evaluation-only recovery passes its360s cap; original resource gate
  stays failed. `recovery.json` maps every arm to its authoritative weights, request
  and quick artifacts. Recovered paths are `seed7202/ce/recovered_requests` and
  `seed7202/ce/recovered_quick`; never relabel the interrupted parent as complete.
- Successful audit process11.94s; earlier failed artifact-key lookup is preserved
  but unclocked. Recorded execution/recovery/audit sum1726.23s is a lower bound,
  not complete cost accounting. Three reported fits peak87.06MiB; stopped fit lacks
  exported training-only/peak memory receipt. No speed claim.
- 640 tests pass on immutable2d43133 (853.82s pytest;855.45s process includes291.11s
  pause).30 final focused checks pass after removal/CLI guard. Six-update CUDA resume
  and simplified-current-CE versus frozen-control checks each match848 tensors and
  1,321 primitives;42.58s combined development. Eighteen standalone reports pass
  structural/hash checks; comparison figure inspected, unchanged renderer, no browser QA.

Current interfaces and safe fresh-directory examples are in the plan. Archived
boundary-weight commands require their frozen source; the current recipe supports
ordinary CE only. No new scientific runs are scheduled. Before another learning
comparison, use the existing cache to distinguish wording/continuation mistakes
from the weaker source's content failures; preregister a new factor and preservation
controls, without tuning against this now-inspected fresh corpus. Independent
explorer changes are committed separately; inspect Git status before editing.

## Earlier performance iteration

[Byte batching and frozen metadata](byte-batch-plan.md),
[combined report](../runs/metadata_cache_v1/report.html). Both candidates were
implemented, reviewed with actual Claude, tested against unchanged expressions,
and measured on the two fixed request-meaning sources. Both fail the preregistered
5% whole-workload gate. Original byte batching and MetadataEncoder are restored
exactly; there is no metadata cache flag in the current library. Implementation
closure is commit `124e507`; candidate sources and tests remain in Git and ignored
run archives. No speed, quality, architecture or validation-color promotion.

- One-transfer bytes:7202 training/inference reduction3.27%/1.95%;7201 1.19%/1.02%.
- Frozen metadata:7202 -0.61%/0.93%;7201 1.36%/1.67%. A real-path four-call audit
  confirms three cache hits. Its precision guard additionally fails after a cuDNN
  submodule import without precision reseeding; removing caching fixes that case.
- The first byte trial was stopped for benchmark-created GRU compaction overhead.
  All completed blocks are retained; corrected trials flatten outside timing.
- Two complete comparisons each pass31,266 runtime equality checks. A stricter
  saved-artifact byte audit checks5,088 tensors,160 arrays and8,850 primitives,
  across six final training pairs and five available inference pairs. Runtime
  equality used `torch.equal`; strict signed-zero identity was not checked for
  every intermediate repeated block. No adoption relies on that missing closure.
- Recorded measurement upper bound487.21s, successful audits1.66s, peak95,126,016
  allocated bytes; ~42MiB before summary reports. One failed import-order inspection
  lacks a separate elapsed-time receipt, so exact all-process budget closure is
  unavailable. No source data or source checkpoints changed.

634 full software tests pass in542.05s (543.67s process receipt);31 focused checks
pass. Fifteen reports are structurally verified; the comparison plot was inspected.
The renderer is unchanged and browser QA was not performed. Unrelated model-explorer
changes remain in the checkout; the full test result includes the tests collected
during that concurrent work, not a guarantee for later edits. Inspect status before
editing those files.
The request-form/EOS follow-up is completed above; the earlier diagnosis below remains historical evidence.

## Latest completed measurements

[Request completion and decoder cost](request-completion-plan.md) follows the
[frozen routing comparison](request-routing-plan.md). The older routing study
is complete and does not need rerunning. Current report:
[runs/request_completion_v1/report.html](../runs/request_completion_v1/report.html).

Three slices were implemented/reviewed/tested:

1. **Preprocessing repair:** masked question lengths now account for the existing
   observation builder's trailing-whitespace stripping. Leading/internal question
   whitespace, evidence, full/neutral routes, masks and times are preserved.
   Empty, whitespace-only and multibyte regressions are covered. All5,010 existing
   prepared-fixture route comparisons remain exact; historical results unaffected.
2. **Frozen EOS diagnosis:** new `request-completion` stage in the existing recipe
   reads saved working states, checks source/cache weights and reproduces saved
   answers before measuring continuation. No core sampling or neural training.
   Six source/route evaluations,9,216 rows,22.56 process seconds including startup,
   source loading and reports. Raw next-token EOS/space probabilities, margins and
   legal argmax are saved for target-first-word, generated-first-word and complete
   target prefixes. Generated prefixes are labelled on-policy only if reached.
3. **Two inference-cost trials:** last-position state attention/vocabulary projection,
   then that slicing plus combined validation flags. All9,216 greedy token arrays,
   fixed logit tolerance and full-path output/gradient compatibility checks pass in
   each trial. Both miss the preset primary CUDA batch-64 speed gate: paired median
   latency reductions -3.67% and-1.94% (slower). CPU batch-64 reductions9.77%/9.14%
   are secondary and do not justify promotion. Original validation and generation
   defaults remain. `TextDecoder.generate(..., last_only=True)` preserves slicing
   as an explicit experiment, with no GPU or training-speed claim.

The final implementation is commit `c959c63`; earlier `08cdaf0` and `1860144`
record plan/essential red tests. Later commits update documentation and research
records. Nothing is pushed. Inspect `git log -5 --oneline` and `git status --short`.

## What the diagnosis establishes

Source7202 with neutral or masked observed questions correctly starts every
full-condition answer with the first color. Yet four of eight sequence wording
variants greedily choose EOS at that prefix for every clip and draw. The other
four choose space. For neutral routing, failed wording mean space-minus-EOS
margins range -2.71 to-5.58, successful means+1.51 to+6.05. Both color orders show
the same pattern. Examples:

- `Schreibe mir alle Farben nacheinander auf.` stops early in every case.
- `Schreibe mir beide Farbtöne nacheinander auf.` continues in every case.
- Both sequence variants beginning `Nenne mir bitte` stop early.

All first-only requests stop correctly. Every teacher-forced complete two-word
answer chooses EOS. This is a conditional intervention, not proof of free content
generation or of a unique failing neural module. The wording pattern comes from
reused development fixtures,16 clips and3 draws, not independent confirmation.
Do not interpret the probability or mean summaries as newly passed task gates.

Source7201 still has content errors: neutral first-word correctness62.5%, sequence
exact42.71% averaged across draws. Its teacher-forced first-prefix EOS rate for
sequence requests is18.23%. Stronger-source termination alone does not explain
both sources. Historical quick-suite scores remain0/25 and1/25; no new broader
capability evaluation or promotion was made in this iteration.

## Verification and limits

All605 full software tests pass in594.83s, including training/resume/report paths;
the final focused selection has53 passing checks. Full output and exit status are
in `full-software-tests.txt` and `software-cost.json` under the run root. No training
or evaluation job remains running. Targeted Ruff checks and diff checks pass.

Raw audit:296,076 checks,9,216 preserved rows,5,010 prepared-fixture route
comparisons,4,548 unchanged checkpoint tensors. An additional768 neutral/masked
full-condition first-prefix decisions match. Seven standalone reports are
structurally verified; comparison figure visually inspected. Existing renderer
unchanged; interactive browser QA not performed. Formal diagnostic peak torch
allocation37,914,112bytes; this excludes runtime/reservation. New artifacts~32MiB.
Decoder cost trials total31.61s; full tests overlap other work, so do not sum their
duration as elapsed wall time. Source snapshots for formal diagnoses are complete.
The combined performance candidate source was saved immediately; the slice-only
source was reconstructed from it and original validation afterward, as explicitly
recorded in `source-provenance.json`. Neither candidate was adopted.

Actual Claude reviewed and acknowledged the per-position equivalence, distinct
teacher-forced/generated measurements, fixed temperature/constraints and exact
output checks. It withdrew an objection after clarification of the prespecified
three warmups/nine paired repeats/5% threshold. No significance or unmeasured
production-speed claim. Receipts: `runs/reviews/request_completion_v1/`.

Architecture atlas/checklist follow-up updated and regenerated; discussion status,
validation scopes and colors remain unchanged. No assistant-only walkthrough was
marked as completed with the user.

## Next iteration

Preregister a matched request-form/continuation repair with paired first-only and
sequence objectives, content and broader-task preservation, and fresh wording
controls. The inspected failed templates are development evidence: fitting them
alone cannot establish generalization. Keep weaker-source content access separate.
Do not enlarge encoders/decoders simply because overall task accuracy is poor.
No new neural fit or successful repair is implied by this diagnostic.

If returning to performance, keep the rejected CUDA trials and their fixed gate;
use a new declared workload/profiling hypothesis rather than repeatedly timing
until a favorable result appears. Full-prefix validation and generation are the
current reference. Training-time improvement remains unmeasured.

## Commands and artifact map

Use `.venv/bin/python -m pytest`; there is no `.venv/bin/pytest` executable.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m pytest \
  tests/test_request_completion.py tests/test_text_decoder.py \
  tests/test_request_routing.py tests/test_request_meaning.py \
  tests/test_request_readout.py tests/test_modality_foundation.py -q
```

For a NEW compatible cache/output only; the six current evaluations are complete:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m experiments.modality_readout \
  --stage request-completion \
  --core runs/request_meaning_v1/seed7202/balanced \
  --request-cache runs/request_routing_v1/seed7202/neutral \
  --seed 9701 --device cuda --output runs/my_completion_diagnosis
```

Under `runs/request_completion_v1/`:

- `result.json`, `verification.json`, `report-verification.json`, `report.html`.
- `seed720{1,2}/{full,neutral,masked}/`: raw `continuations.json`, settings/source
  identity, source snapshots, unchanged `last.pt`, standalone report and QA receipt.
- `slice-only-verification.json`, `combined-verification.json`, corresponding
  candidate source copies and scripts; `source-provenance.json` labels provenance.
- `execution.json`, `software-cost.json`, full/focused/red test logs.
- `execute.py`, `audit.py`, `build_summary.py`: experiment-local utilities, not a
  new library interface. Do NOT rerun `execute.py` unchanged into completed paths.

Datasets/checkpoints and `runs/` are intentionally ignored by Git; a clone alone
cannot recover them. Source data and completed runs are preserved. End-of-turn
research records are in `ara/`; read/write them only during the research-manager
epilogue, not while selecting or implementing the next experiment.
