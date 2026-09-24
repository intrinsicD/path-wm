# Evidence, instance memory and reactive episodes

24 September 2026. Authorized: multi-turn design, implementation, review and repair
with actual Claude Opus 5.5 at **medium** effort. This plan implements the next
architecture contracts from the walkthrough. Public-only external review remains
in force: abstract briefs and Claude-authored generic code, no private source,
datasets or local measurements exported. Exact briefs/receipts live in
`runs/reviews/architecture_completion_20260924/`.

## Scope and owners

- Reuse `WorldSession`/`WorldStore` for the authoritative clock, evidence, revisions
  and exact restart. An episode client owns no second store. Conversation episodes
  and instance detail use existing entities/components/events.
- Keep concept/shared transform weights, individual detail codes, requested pose,
  source evidence and inferred uncertainty distinct. Functional roles do not
  require a new generic module hierarchy.
- Local/Global remain bounded internal working contexts; persistent World State is
  external. Active context loads selected tensors plus pinned version references.
- Ordinary response selection is reactive. Typed utterance boundaries, interruption,
  correction and restart are implemented without pretending a protocol is learned
  natural language. Text and audio references share an episode; speech recognition,
  natural dialogue and waveform generation are not established by those mechanics.
- The learned visual demonstration tests familiar transformations on new instance
  textures, not natural human rendering, learned identity, new concept induction,
  inverse-view alignment or the complete earlier R1/R2 goal.

## First implementation and training task

Synthetic RGB16 tiles contain four independent RGB8 detail regions. An observation
exposes selected complete regions at canonical orientation. Visibility and tracked
part/instance association are explicit supplied inputs. The target generator alone
uses geometric rotation. A learned shared encoder compresses each region and a
learned pose-conditioned decoder reconstructs quarter-turn views. The model has no
analytic inverse renderer, geometric warp or target image input.

Deterministic evidence updates replace only newly observed detail records; duplicate
views do not accumulate confidence. Shared transformation knowledge is trained once,
then frozen during memory acquisition, reconstruction, correction and restart.
Unobserved detail uses a learned population estimate with predicted variance. No
generated image is promoted to observation. Independent random unknown details are
not expected to beat their conditional-mean oracle.

Current proposal after Claude round one: start much smaller than its multi-class
deformation task. Its 60k-step wall-time estimate is unverified; learned precisions
do not guarantee non-interference and correlated views invalidate naive confidence
multiplication. Round two asks Claude to correct these and supply a small core.

## Checks before implementation (red, then green)

1. Only selected observed codes change; no gradients cross persistence; decoder
   gradients reach encoder during training. Invalid shapes/masks fail explicitly.
2. Evidence-backed detail and conversation state share the session/store/clock.
   Interrupted/incomplete utterances do not execute; episode switches do not leak.
3. Source correction/retraction invalidates dependent state and output pins;
   unrelated detail survives; no fallback silently resurrects invalidated heads.
4. Matching model versions restore the same state/output. Changed versions, future
   evidence and stale pins are rejected. Memory capacity failures stay explicit.
5. Retries preserve identity/content, conflicting retries fail; generated messages
   remain derived, and a person's assertion is evidence of speech, not world truth.

## Predeclared development comparison

Two training seeds (17, 29), maximum 3000 updates each, batch 64, AdamW, FP32.
Per-stage wall-time ceiling 30 minutes; inspect measured throughput before longer
work. One short workflow check precedes training. At most two diagnosed repair
comparisons initially, each separately recorded with unchanged quality gates and
fresh run directories. Further comparisons require an updated rationale/budget in
this plan, not silent extension. Training and validation streams are disjoint from
512 final test instances (fixed seed 240924); validation seed 240925. Do not select
settings from final test results. All stochastic source settings are saved.

Primary scope: new textures, same known four quarter-turn requests. Full observed
reconstruction MSE <= 0.003 in [0,1] RGB on each seed. After revealing missing parts:
their MSE must decrease by >= 50%; previously observed target-region MSE must not
increase by > 0.003. Full-memory MSE must be <= half latest-only, no-update and
wrong-instance memory MSE. Nonidentity-pose output must beat copying canonical RGB
and ignoring requested pose by >= 50% MSE. Unknown predicted variance must exceed
known variance by >= 3x. Interval coverage is reported separately, not called
calibrated solely for passing a variance ordering gate. Publish subgroup metrics,
not just averages. Software stale-read, restart and correction checks must pass.

Controls retain the trained model; alter access/updates/instance codes/pose one
factor at a time. Report analytic prior and lossless-copy reference limits without
requiring impossible unknown-detail recovery. Record full raw predictions/targets,
metrics, checkpoint, source identities, optimizer/RNG state, parameters, step time,
throughput and CUDA peak memory. End result status before report generation. Use
the existing report renderer and structurally validate the standalone report.

No scientific success is inferred from a passing API test. General dialogue/audio,
natural images, learned association, uncertainty calibration under domain shift and
the earlier failed R1 transfer remain open even if this bounded demonstration passes.

## Completion and review

Claude rounds: design → objections/reconciliation and generic implementation →
implementation-contract review → remaining corrections. Local audit checks actual
source and tests under the export boundary. Commit red plan/checks, working slice
and final evidence separately. Full `python -m pytest` is required before committing
shared code. Update the walkthrough, model/experiment guide, state and atlas with
scope; do not mark the whole architecture green.

Applied principles: prepare each observed part once; retain source evidence separately
from compact state; explicit state ownership; version derived reads; separate
generation from verification; count memory traffic/retrieval and real runtime cost.
The small explicit library and recipe remain the user interface.

### Claude round-two reconciliation

Claude withdrew the unmeasured budget, naive correlated precision fusion, geometric
non-interference and impossible unknown-detail oracle claims. It supplied a shared
part encoder, pose-FiLM decoder and variance-weighted Gaussian objective. Start with
that small MLP implementation and independent-part procedural texture; do not adopt
its additional hierarchical-prior gates or its silent fallback to older evidence.
A retracted latest head stays unavailable until an explicit rederivation is published.
The registered gates above remain fixed; decoder interference is measured.

### Development audit correction (before inspecting first trained result)

The smoke recipe used seed240924, originally reserved for final testing, even for
five-step checks. This is a split-handling defect, not evidence of generalization.
Retain those outputs but treat seed240924 and the currently running first fit as
development only. **New untouched final population: seed240927, 512 instances**,
registered now before the first trained result is inspected. Default subsequent
checks use validation seed240925; final evaluation must be explicit. Quality gates
are unchanged. Do not tune settings on the new final population.

The first fit overlaps a CPU test run: its timing is operational context, not an
uncontended efficiency comparison. Resume validation/status, stop-after bounds and
stored/direct decoder parity also need explicit negative/positive checks before
calling the implementation complete.

### Repair comparison 1: simpler learned finite-view decoder

First FiLM fit (seed17, 3000 steps) failed fidelity, retention, memory, pose and
uncertainty gates; its development reconstruction MSE was 0.01822. Preserve
`runs/evidence_loop_film_s17_v1/`. It passes only the software checks and revision
error decrease, not the target capability. The initial full suite was intentionally
interrupted at 272 passes to fix review findings; that is not a full-suite pass.

Following Claude's review, compare a pose-specific learned linear decoder with the
same encoder, code width, mask distribution, seed, optimizer and objective. Compute
all four affine heads in one matrix multiplication; no per-item expanded weights
or analytic geometry in the model. A separate presence/pose variance head uses a
softplus variance floor instead of a hard clamp. This combined decoder/variance
change is a repair comparison, not isolated attribution to linearity alone.
Budget remains 3000 steps/seed, two seeds; validation only until configuration is
selected. Existing gates unchanged. The learned prior is marginal because parts
are independent, not a learned relational concept. Unknown variance is not claimed
calibrated from a scalar confidence or from variance ordering.

All final population codes pass through the portable WorldStore serialization.
Live WorldSession parity/restart and per-output source checks are separate checks;
the dataset is not stored as one unbounded runtime session. Each output chunk is
an internal authorization record, not evidence or proof of physical playback.

### Repair comparison 2: variance-only calibration

The linear seed17 development fit passes all original gates (validation MSE
0.0002413, PSNR36.17); however, hidden nominal90% coverage is only73.94%. Do not
call variance ordering calibration. Before inspecting the untouched final set,
predeclare an additional 600 fresh training-stream batches, batch64, Adam lr0.03,
ordinary Gaussian NLL for the variance head only. Freeze encoder, prior and mean;
verify their weights and output means unchanged. All presence subsets remain in
the training distribution. No calibration data from validation or test are used.
New diagnostic target for hidden population coverage: nominal90% within85–95% on
validation and final held-out textures. Original reconstruction gates unchanged.
Checkpoint calibration progress/optimizer/RNG; a completed resume must not repeat
the phase. Known-pixel or domain-shift calibration remains a separate question.

## Implemented result (24 September)

Five actual Claude Opus5.5 medium rounds covered design, objections, generic code,
adversarial implementation contracts and calibration. Private implementation and
measurements were reviewed locally; Claude did not inspect them. Receipts are in
`runs/reviews/architecture_completion_20260924/claude-receipt.json`.

Implemented `models/detail_memory.py`, `data/detail_views.py`,
`world_state/episodes.py` and the ordinary `experiments/evidence_loop.py` recipe.
The episode client shares the existing session/store/clock. Incoming assertions,
inferred state and output authorizations remain distinct. Source/head/version
pins are checked before each output commit. Interrupted, stale and already-answered
turns cannot emit another current response. Abort and explicit unknown/rederivation
permit deliberate recovery. Budget omission is never silently treated as ignorance.
Actual codec weights are included in the neural session's signature.

### Untouched final population

Both selected models use936,768 parameters. Train seeds17/29; same512 previously
untouched test textures, seed240927; all four familiar quarter-turns. Quantitative
reveal pattern is first two parts → all four. All population codes undergo portable
WorldStore serialization; the separate live-session check tests acquisition,
correction, restart and output readiness/commit. These scopes are not conflated.

| Metric | Seed17 | Seed29 |
| --- | ---: | ---: |
| Full observed reconstruction MSE | 0.00024258 | 0.00023893 |
| PSNR | 36.151 dB | 36.217 dB |
| Newly revealed part MSE, before → after | 0.031745 → 0.000242 | 0.031763 → 0.000240 |
| Old-visible MSE increase | 0.00003048 | 0.00002888 |
| Latest-only MSE | 0.016113 | 0.016112 |
| Wrong-instance MSE | 0.063906 | 0.063873 |
| Hidden nominal90% interval coverage | 91.849% | 91.897% |
| Instance-bootstrap95% coverage interval | 91.598–92.072% | 91.646–92.137% |
| Hidden predicted variance / empirical error | 1.048 | 1.039 |

All original gates plus the predeclared hidden-coverage gate pass in both seeds.
The known-part variance/empirical-error ratio is0.977/0.991; this is not a claim
of per-instance or domain-shift calibration. Calibration leaves all reconstruction
weights bitwise unchanged. Unknown-part predictions match the population prior's
error rather than inventing the missing independent texture.

Resource receipts: parameters3,747,072 bytes; both optimizer states7,617,076 bytes;
peak CUDA allocation217,815,552 bytes (reservation247,463,936). Training plus
calibration and periodic validation took16.02/16.38 seconds in these invocations,
excluding final persistence/evaluation/report work. These are operational timings,
not a matched speedup claim. Retained raw evidence adds storage beyond compact
codes. A separate diagnostic profile reports supported-operator FLOPs, activation
shapes and CUDA timings with its measurement limits.

Results/reports: `runs/evidence_loop_final_s17/`, `runs/evidence_loop_final_s29/`,
and comparison `runs/evidence_loop_summary_v1/report.html`. Raw predictions,
checkpoints and source identities are retained. Reports use the existing renderer,
structurally verified; reconstruction contact sheet visually inspected. No new
browser interaction or changed-renderer QA is claimed.

Eighteen focused tests pass, including exact uninterrupted/split/completed-resume,
invalid resume/stop receipts, unavailable-pixel invariance, code-version changes,
correction before emission, interleaved text/audio references, abort/retry and
explicit unknown-state recovery. The unchanged source passed the full repository
suite: **838 tests in660.49 seconds**. Exit receipt and log are retained in
`runs/reviews/architecture_completion_20260924/full-suite-exit.json` and
`full-suite-detached.log`. An independent NumPy audit recomputes the main final
metrics from saved predictions; source hashes and run artifact hashes match.

### Reproduction

```bash
python -m experiments.evidence_loop --output runs/my_final17 --seed 17 --variant linear --calibration-steps 600 --final-eval
python -m experiments.evidence_loop --output runs/my_final29 --seed 29 --variant linear --calibration-steps 600 --final-eval
python -m pytest tests/test_detail_memory.py tests/test_episodes.py tests/test_evidence_recipe.py
```

### Still open

This closes a controlled evidence/instance/output integration slice. It does not
complete the general agent: learned identity/view alignment, new concept induction,
natural-human reconstruction, learned context selection, natural conversation,
speech generation/playback and general planning remain unproven. The earlier R1
transfer and retention failures remain in force. Do not promote the whole atlas
or infer that finite affine view maps solve general world dynamics.
