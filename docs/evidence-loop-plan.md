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
