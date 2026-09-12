# Current work

**Image detail and request-only output interface pass scoped screens, 12 September:**
the opt-in detail channel reduces real COCO reconstruction MSE0.012395→0.000879
(92.9%,128 reused images, zero updates). This transports source pixels; it is not
learned compression or evidence of state-mediated photo reconstruction. A learned
state-to-spatial adapter supplies every feature to the same frozen RGB head with
no image encoder in the agent. Four training stripe requests fit to MSE0.000110
after512 updates; erased/shuffled requests0.138158/0.266283. This is memorization,
not novel-prompt generation. All212 donor tensors are unchanged.35 targeted checks,
exact CPU resume, actual GPU256+256 resume, standalone reload and8-metric audit pass.
Training12.12s; peak106MiB reserved. Public-concepts-only Claude review completed;
HTML QA structural-only, comparison PNGs visually inspected.
[Protocol/results/load instructions](image-output-plan.md#completed-screens),
[report](../runs/image_output_v1/report.html). Next: compact learned visual codec and
held-out paired image/request training; audio/video generation training still open.

**Handwritten initialization trained and diagnosed, 12 September:** the exact source
binary now supports strict initialization, component freezing and resumable RGB
training. The original decoder rate collapsed both seeds; validation-only selection
of0.000003 (encoder0.0003) stabilizes training. Joint RGB MSE0.012395/0.012447 improves
10.52%/10.15% from the handwritten baseline. Decoder-only improves9–10%, encoder-only
5%; joint fails the5% advantage screen against decoder-only. Initial patch features
retain only three color averages and collapse an equal-mean checkerboard pair.
Both information loss and decoder optimization matter; this is not an additive error
attribution. Ordinary initialization at its original rate remains better0.007319/0.007859.
[Protocol/results/load instructions](hierarchy-training-plan.md#completed-diagnosis),
[report and trained binaries](../runs/hierarchy_training_v1/decoder_rate_repair/report.html).
203 full-suite tests plus15 focused rate/freeze/resume checks pass; original90 and
repaired81 raw scores independently verify. Source/frozen/export hashes and paired
sampling pass. Seven repair runs complete; the last ordinary control stops at103/384
updates under the600-second cap, so its same-rate paired comparison is incomplete.
190MiB peak reserved for completed repair runs; original521s plus repair600s.
Mask head frozen, image-only scope, reused test population, HTML QA structural-only.
Claude retry was rejected by automatic approval review; no external review occurred.

**Direct hierarchy weights completed, 12 September:** wrote the same1.80M-parameter
image hierarchy as explicit numbers, plus a separately labeled132-coefficient fitted
readout. [Plan/results/load commands](hierarchy-weights-plan.md#completed-comparison),
[report](../runs/hierarchy_weights_v1/report.html). Four GPU evaluations,25.91s,
zero optimizer updates; handwritten RGB MSE0.013853/IoU0.204784, fitted0.013693/0.072054.
Both fail perception gates and trail trained references; color transport works,
foreground perception remains weak.197 tests,36-metric audit, strict binary reload
and cached GPU resume pass;120–136MiB reserved. Previous checkpoints unchanged.
HTML QA structural-only. Claude send rejected by automatic approval review; no
external peer review was performed for this comparison.

**Hierarchy/fusion comparison completed, 12 September:** implemented the requested
per-scale transformer depth plus optional final all-scale stack.194 CPU tests pass,
including causal video/audio/text, invalid gradients and exact resume; disabled
fusion matches pre-change outputs exactly. Eight real COCO RGB/foreground runs,
two seeds,589s: fusion improves IoU0.184→0.258 at7401 but worsens0.304→0.238 at7402
with18.6% worse RGB. Extra depth helps neither seed. Both proposed improvements fail
the prespecified screen; longer shallow training improves RGB, mask effects mixed.
All mask IoUs remain below the full-foreground reference.170–184MiB reserved GPU;
independent raw metrics/hash audit and completed GPU resume pass. [Results/plan](hierarchy-fusion-plan.md#completed-comparison),
[report](../runs/hierarchy_fusion_v1/report.html) (structural-only). Keep fusion
default0 and the option available. Encoder recurrence remains a separately specified
cross-window test; no new speech, video-memory or integrated-agent claim.

**Encoder–decoder review completed, 12 September:**
[Pair-by-pair review and proposed comparisons](encoder-decoder-review.md) distinguish
short-pilot undertraining, decoder mismatch, state-path failures and missing speech
context. Every multimodal feature scale already has attention/MLP processing.
Historical extra CNN depth helped geometry but missed its gate; frozen-encoder
decoder repair recovered COCO reconstruction with a PushT retention tradeoff.
Proposed next: reuse useful vision donors, compare direct versus state-mediated
readouts, and freeze/probe the successful text donor before adding capacity.
Audio needs a meaningful temporal contract before speech training. Claude reviewed
public concepts and accepted corrections; no model changes or new training/tests.
The capability baseline remains the frozen comparison point. Numerical gates and
budgets for the next formal experiment still require a predeclared specification.

**Capability baseline completed, 12 September:**34 behavioral/mechanism checks across
13 separately identified checkpoint roles,183 passing software tests,10 explicit
coverage gaps. [Report](../runs/capabilities_v1/reference/report.html) and
[protocol, results and comparison commands](capability-baseline-plan.md#completed-baseline).
Direct visual weights retain100% accuracy on original/mirrored/dim/noisy scenes but
fall to48.44% with red/blue swapped and50% grayscale. Their other outputs are gray,
silence and empty text, with zero response to the tested audio/text/video input contrasts or opposing actions.
Structured entity/state/relation and supplied-mechanics planning screens pass;
through-agent fact identity0%, historical recall answers0/15, instructions12.5%,
and learned prediction loses to copy-last. Known planner reward/utility mismatch
persists. No weight changes. CPU evaluation211s/804MiB peak process RSS. Independent
raw checks across all34 cases, checkpoint hashes, cached resume and comparison guards
pass. Browser QA remains structural-only. This is a baseline of separate components,
not a complete jointly trained agent or evidence of general webcam capability.

**Direct weight test completed, 12 September:** at the user's explicit request,
constructed binary model weights and adjusted them without backpropagation. First
candidate family fails (39/64 final answers). A bounded numeric timing/routing revision
with34 label-fitted readout parameters passes on a fresh set:64/64 answers,32/32 complete
pairs and32/32 reversals; ordinary/erased controls50%. Zero optimizer updates,40MiB
inference allocator peak;180 CPU tests and independent artifact/metric checks pass.
[Weights and scope](direct-weights-plan.md#completed-revision),
[report](../runs/direct_weights_v1/timing/report.html) (structural-only).
This is a hand-built fixed-task circuit in the small existing model configuration,
not general pretrained-weight generation. Real webcam transfer remains open. The
previous optimizer-based visual curriculum is deferred by this user-requested test.

**Assistant-authored curriculum discussed, 12 September:** the user proposes having
the assistant generate teaching modalities and train the learner. [Curriculum proposal](webcam-memory-data-plan.md#assistant-authored-teaching-data-12-september-discussion)
combines controlled histories and checked questions/images with separate real-footage
evaluation. One brief Claude review emphasized pixel/label checks; the local
reconciliation avoids assuming a perfect automated verifier. Text/image generation
is available; realistic video/speech generation is not yet connected. No new corpus
or training run. Visual-memory interfaces, objective, gates and GPU budget still need
a concrete experiment specification; prior real-footage preference remains intact.

**Real-footage development pack prepared, 12 September:** the user prefers real
webcam footage for observation memory. Selected 12 COCO train photographs and 6
Charades official-train indoor clips, preserving source hashes, annotations and
split/subject groups. Contact sheets inspected; six imported video episodes have
timestamps and pending annotation templates. [Data and recording guide](webcam-memory-data-plan.md)
defines the first 28-take pilot and independent answer-key checks following Claude's
review. Three capture/import tests pass. No camera activation or training. Own
recordings, reviewed entity/event/query labels and the experiment gates remain open;
this selection is development material, not a trained or validated memory capability.

**S01 sequence adopted, 12 September:** first remember observations, then predict
action consequences, then plan toward a goal using those predictions. Observation
memory is the next capability target. First environment/input and numerical gates
remain open. [Recorded answers](agent-specification-questions.md#recorded-answers).
Both training and inference must fit the current local GPU; scale later.

**S03/R06 preference recorded, 12 September:** comfortable fit on the user's GPU
is required; prefer own components, with pretrained model plus adapter if quality
is inadequate. Local hardware query reports RTX3050/8192MiB. User confirmed both
training and inference must fit now; scaling comes later. Numerical headroom and
quality gates remain open. [Recorded answers](agent-specification-questions.md#recorded-answers).

**S02 long-term scope recorded, 12 September:** eventual webcam vision, speech and
writing, image/video creation and software-tool use. [Questionnaire answer](agent-specification-questions.md#recorded-answers)
is partial: first experiment and learned-versus-supplied boundaries remain open.

**Specification questionnaire prepared, 11 September:**
[85 numbered questions](agent-specification-questions.md) cover scope, goals,
evidence, perception, identity, graph learning, memory, focus, dynamics, planning,
training, execution, inspection and experiment gates. Begin with S01–S05 and then
the task/evidence/dynamics contracts. Explicit deferrals define what the first
experiment does not claim. Answers remain pending; no architecture change or
training was launched. Claude reviewed integration-level coverage.

**Whole-model audit updated, 11 September:** controlled integration is working,
but learned dynamics, uncertainty semantics and general task execution remain
separate or supplied. Concrete planner/metric objective mismatch confirmed: under
85% absence belief, current one-step planner retrieves although reported expected
utility favors stopping0.85 over retrieval0.10. Prior measurements remain valid;
they do not establish optimization of that utility. [Current readiness review](model-readiness-review.md)
replaces the stale pre-integration assessment. Next: resolve the task/cost/evidence
contract, then connect action-conditioned prediction to the integrated path.
Review only; no model changes or new training/full-suite result.

**Independent key-box replication and correction screen pass, 11 September:**
new training seed2302, fresh evaluation2431: ordinary and longer-history relocation
cases both96/96 reachable and32/32 absent; correction readout192/192. Frozen prior
model also passes (correction191/192). Ordinary memory utility advantage remains
0.015625; under relocation memory costs more than no-history, so no universal
utility benefit.31 tests, lint, cached resume and independent time-varying execution
and correction audit pass. [Report](../runs/key_box_v1/replica/report.html)
structural-only; [active plan](key-box-integration-plan.md). Next: varied change
and observation timing. Still supplied descriptors, corrections and action mechanics.

**Interleaved key-box training passes controlled screen, 11 September:**
four read pairs separated by observation events preserve working state during
training. Fresh seed2421: known content100%, reachable96/96, absent32/32;
utility0.9203125 beats no-history0.9046875 by0.015625 (required0.01).
Frozen previous model fails the same screen.30 tests, lint, cached resume and
independent execution replay pass. [Report](../runs/key_box_v1/history/report.html)
structural-only; [active plan](key-box-integration-plan.md). Next: replication and
stronger state-change/history checks. Supplied descriptors and action mechanics;
this does not validate learned dynamics or general action planning.

**Integrated key-box loop implemented, 11 September:** entity state now reaches the
actual belief-agent workspace; supplied expectimax action mechanics execute and
replan from real feedback. First training screen failed. Query-switch repair raises
known content82.81%→100% and reachable success76.04%→95.83% on matched fresh cases,
but utility0.891016<no-history0.904688, so full screen still fails. Four false stops
follow later readout confidence loss.29 distinct relevant tests, resumes and independent
execution replay pass. [Report](../runs/key_box_v1/switched/report.html) structural-only.
[Active plan](key-box-integration-plan.md). Next: full action-history readout stability.
Source-selection tuning is paused. This is controlled descriptor/explicit-dynamics
integration, not visual discovery or learned world-model planning.

**Matched-budget coverage fails utility gain, 11 September:** eligible drift
source checks29/32 versus21/32, but late utility gain0.000977<0.01. Static cost
and reset guards pass.11 source tests, cached resume and independent allocation/
feedback replay pass; costs exactly matched. [Report](../runs/entity_source_coverage_v1/reference/report.html)
structural-only. No promotion; result conditional on forced acquisition/epsilon0.5.
Next proposal: source-selection headroom diagnostic before further tuning.

**Source-local diagnosis complete, 11 September:** of32 drift source records,14
never reached an eligible all-new-feedback block,10 reached checks but stayed below
threshold,8 reset. This separates check availability from threshold outcomes; it
does not establish a repair.9 source tests, resume and independent full-feedback
reconstruction pass; original artifacts unchanged. [Report](../runs/entity_source_diagnosis_v1/reference/report.html)
structural-only. Next proposal: matched-budget feedback coverage with detector fixed.

**Variance-aware forgetting fails adaptation margin, 11 September:** development
selects z2; held-out static resets4/16 meet25% (matched fixed-trigger8/16). Late
drift utility0.731096 beats cumulative0.722993 by0.008103, below required0.01.
Other utility guards pass.29 distinct relevant tests, cached resume and independent
development/held-out replay pass. [Report](../runs/entity_source_uncertainty_v1/reference/report.html)
structural-only. No promotion or tuning; next diagnose missed/late changes from
existing traces before choosing another detector. Original gate unchanged.

**Triggered forgetting fails false-reset guard, 11 September:** late drift utility
0.742107 versus frozen0.713721 and cumulative0.723066; static utility loss0.011832
is within0.02. But10/16 unchanged worlds reset (62.5%>25%), so full screen fails.
26 relevant tests, cached resume and independent action/feedback/reset replay pass.
[Report](../runs/entity_source_change_v1/reference/report.html) structural-only.
Original gate unchanged. Next candidate: uncertainty-aware change checks with
separate stationary calibration and held-out false-alarm controls; no tuning here.

**Online recency adapts but fails stable-source guardrail, 11 September:** drift late
utility window0.760835 versus frozen0.723169 and cumulative0.733477. Static
utility falls0.780920→0.754734 (loss0.026187>0.02), so full screen fails.25 relevant
tests, resume and independent action/feedback replay pass.
[Report](../runs/entity_source_drift_v1/reference/report.html) structural-only.
Reference unchanged; fixed window is not learned drift detection. Next candidate:
evidence-triggered forgetting rather than unconditional recency, with stable-source
controls and feedback costs preserved.

**Outcome-trained source choice passes, 11 September:** a per-world action-value
table selects the useful opaque source in16/16 worlds from calibration feedback.
Evaluation accuracy79.91% versus best fixed77.00%; utility0.777112 versus0.748059.
Combined calibration/evaluation utility0.752104 exceeds stop0.728353.23 relevant
tests, resume and independent selected-feedback audit pass.
[Report](../runs/entity_source_choice_v1/reference/report.html) structural-only.
This is static-source adaptation with supplied outcome feedback, not a neural
selector or general reliability estimator. Next candidate: source-quality changes
and online updating, with explicit feedback availability and stale-value controls.

**Alternate evidence is worth its declared cost, 11 September:** selective high-noise
accuracy80.86% alternate versus73.44% same-source, both reread42.58%. Utility
0.787305 versus0.725859 despite alternate cost0.05 versus0.02. Paired descriptive
95% interval for utility gain[0.024129,0.101331].21 relevant tests, resume and
independent audit pass. [Report](../runs/entity_evidence_sources_v1/reference/report.html)
structural-only. Fixed sensor properties/policies; no learned source selection.
Next candidate: learn acquisition choice from outcome feedback without giving the
policy hidden noise/correlation labels. Reliability-estimation proposal remains separate.

**Correlated rereads expose the independence limit, 11 September:** selective
high-noise gains9.77/5.08/1.17 points at rho0/0.5/0.9; rho0.9 fails the2-point
gate. Rho1 adds no information and loses sensing cost.19 relevant tests, four
resumes and independent shared-observation/decision/cost audits pass.
[Report](../runs/entity_gate_correlation_v1/rho0.9/report.html) structural-only.
Reference unchanged; no memory integration. Next candidate: compare acquiring a
different evidence source with repeating the same source, without assuming the
agent already knows error correlation.

**Reobservation diagnostic passes, 11 September:** high-noise accuracy72.27%→82.81%
with selective rereads42.58%; ignore recall93.75%→98.44%. Duplicate replay gives
no gain. Always-two reaches87.11% and higher utility at the declared cost0.02.
18 relevant tests, cached resume and independent decision/cost audit pass.
[Report](../runs/entity_gate_reobserve_v1/reference/report.html) structural-only.
This is a supplied static-context sensing policy, not learned deferral or a new
memory operation. Next candidate: test correlated second-observation noise before
integrating sensing; independent reread benefit may not survive shared errors.

**Clean retention does not repair the tradeoff, 11 September:** weight-one teacher
KL leaves clean-runtime NLL unchanged0.0237295 and low-noise ignore90.625% at
sigma0.15; high-noise accuracy81.25%→80.08%. Full criteria fail in both arms.
16 relevant tests, two resumes and independent objective/metric audits pass.
[Report](../runs/entity_gate_retain_v1/retained/report.html) is structurally verified;
visual QA remains unavailable. Reference unchanged. Next candidate: diagnose
whether the binary write decision needs an explicit defer/reobserve option under
ambiguous cues, before another loss-weight experiment. This remains a proposal.

**Two conditional replications retain the augmentation tradeoff, 11 September:**
high-noise gains+8.98/+11.72 points over controls, but both full acceptance gates
fail. Rep1 clean-runtime NLL0.152775 exceeds0.15; rep2 NLL0.269396, development
and low-noise ignore checks fail.15 relevant tests, four cached resumes and paired
audits pass. Reference unchanged. Reports: [rep1](../runs/entity_gate_replicate_v1/rep1_augmented/report.html),
[rep2](../runs/entity_gate_replicate_v1/rep2_augmented/report.html), structural QA only.
Next candidate: explicit clean-behavior retention during noisy continuation, tested
against these preserved results on fresh contexts. No broad robustness claim.

**Noise augmentation improves recall but fails acceptance, 11 September:** on fresh
contexts high-noise accuracy80.08% versus matched control71.48% (frozen75.78%).
Low-noise0.15 ignore recall94.53% misses95% criterion; reference stays unchanged.
Both arms preserve100% clean runtime state accuracy.14 relevant tests, both cached
resumes and independent paired audit pass. [Treatment report](../runs/entity_gate_augment_v1/augmented/report.html)
and [control](../runs/entity_gate_augment_v1/control/report.html) are structurally
verified; visual QA remains unavailable. Next proposed: independent-seed replication
of the recall/false-write tradeoff before another repair or threshold choice.

**Frozen gate noise shift finds a limit, 11 September:** the primary per-class
robustness gate fails at sigma0.30 and0.60. Accept recall drops92.97% then55.47%;
ignore recall remains≥95%. Sigma0.03/0.15 pass at100%. No retraining or threshold
selection.12 relevant tests, cached resume and independent metric audit pass.
[Report](../runs/entity_gate_shift_v1/reference/report.html) is structurally verified;
visual QA remains unavailable under the prior browser policy denial. Next proposed:
noise-augmented training with fresh evaluation contexts; high-noise ambiguity
means this failure does not identify a unique architectural defect.

**Context write gate passes the bounded screen, 11 September:** the gate learns
from source-selection loss with matcher, key and interaction weights frozen.
All96 held-out cases pass source/state prediction, including allocation permutation
and repeated irrelevant cues; always/never-write controls score50%. Soft/hard
source choices agree100%. Full145-test suite and cached resume pass.
[Report](../runs/entity_gate_v1/reference/report.html),
[verification](../runs/entity_gate_v1/verification.json). Report structural checks
pass; browser policy blocked visual QA. This tests separated context regimes,
not ambiguous semantic relevance. Next proposed: predeclare a context-noise shift
comparison before extending the claim or adding graph operations.

**Remembered relation keys pass, 11 September:**140 tests pass. A learned key supports
destination-only recall after the source cue disappears:100% source/state accuracy
through replacement, gaps and allocation changes. Erasing keys yields0% source
accuracy. Only key addressing trained; persistence and write policy remain explicit.
[Report](../runs/entity_relations_v1/reference/report.html) is browser verified. Next
proposed: learn whether a new cue should overwrite or preserve a relation.

**Frozen source retrieval passes, 11 September:**136 tests pass. Descriptor queries
select the source among three records at100% accuracy, matching oracle outcomes.
Allocation permutation passes; all unknown queries roll back. No model fitting.
[Report](../runs/entity_source_v1/reference/report.html) is browser verified. The
source query is still supplied. Next proposed: remember a relation from an earlier
cue and retrieve it for a later destination-only action.

**Directed state interaction passes, 11 September:**132 tests pass. A learned
interaction uses another entity’s latent state:100% across all tested conditions,
including a second copy. The matched source-zero control scores50% reference and
29.3% composition. Frozen dynamics, transactions and browser QA pass.
[Report](../runs/entity_interaction_v1/full/report.html). Endpoints and copy type are
still supplied. Next proposed: selecting a source among distractors before claiming
learned relational retrieval or graph structure.

**Explicit idle preservation passes the bounded screen, 11 September:**128 tests
pass. With31 extra idle events, accuracy improves25%→100%; all seven frozen
conditions pass. Latents remain exactly stable across idle stretches; resets and
actions still work. This is an explicit deterministic rule, not learned belief
persistence. [Report](../runs/entity_noinfo_v1/adapted/report.html) is browser verified.
Next proposed: an update depending on another entity’s remembered state, before
claiming learned interaction edges or graph structure.

**Mixed-history adaptation improves ordering but still fails idle stability,
11 September:**123 tests pass. Reset-order accuracy rises62.5%→100% and unseen
composition38.7%→100%; long no-information remains75%, with worse NLL. Recognition
is frozen; adapted runtime/transaction checks pass. [Report](../runs/entity_state_varied_v1/adapted/report.html)
is browser verified. Next: a task-specific state-preserving no-information update,
keeping this failed baseline and fresh idle-length tests.

**Frozen temporal screen exposes state-update limits, 11 September:**121 tests pass.
Reference and repeated-toggle histories score100%; reset-order histories62.5% and
extra no-information events75%. All routing, retry/restore and latent-agreement checks
pass; model weights are unchanged. [Report](../runs/entity_temporal_v1/reference/report.html)
is browser verified. Next: broaden state-update training with fresh held-out histories;
retain this failed frozen baseline. No general temporal or graph-learning claim.

**Learned persistent state succeeds on the bounded task, 11 September:**119 tests
pass. A frozen recognizer routes observations into learned per-entity16-float states;
256/256 development pairs and persistent-runtime outputs are correct. Retry/restore,
rollback and reversed allocation-order checks pass. [Report](../runs/entity_state_v1/reference/report.html)
is browser verified. Splits share16 temporal templates and differ in descriptors;
this does not establish temporal generalization. Next: frozen held-out composition
and length tests before broader attributes or graph learning.

**Variable-count adaptation succeeds, 11 September:**115 tests pass. One256-update
run gives100% development matching/coverage and passes every fresh lifecycle gate
at capacities1/2/4/8. The matched original model still fails capacity8. Claude reviewed
the design twice. [Adapted report](../runs/entity_variable_v1/adapted_pinned/report.html)
is browser verified; weights, descriptor isolation and resume are checked. Geometry
and candidate counts changed together. Next: learned changing state bound to stable
IDs; visual discovery and learned graph structure remain open.

**Frozen growth screen completed, 11 September:** capacities1/2 pass; capacities4/8
fail the declared lifecycle gate. Capacity8 allocation/revisit accuracy is98.44%,
with no wrong-ID matches; uncertain novelty leaves some stores underfilled. All
retry/restore checks pass. Fixed checkpoint loading and evaluator ID-offset errors;
113 distinct tests pass. [Corrected report](../runs/entity_growth_v1/corrected/report.html)
is browser verified. Next: predeclare variable-cardinality training; preserve this
failed frozen reference. No new model training or graph learning yet.

**Entity lifecycle implemented, 11 September:** a bounded Python `EntityMemory`
allocates stable IDs, reuses them on confident matches, defers at capacity or low
confidence, and supports idempotent retries and JSON snapshot restoration. Claude
reviewed the transaction contract twice. Recognition prototypes remain frozen;
variable-cardinality accuracy and learned belief/graph updates remain unvalidated.
See [usage and scope](entity-learning-task.md).

**Known-versus-new matching succeeds, 11 September:** `--dataset entity-matching`
is implemented and reviewed with Claude. All 105 CPU tests pass; the 256-update run
gets 128/128 known identities and 128/128 novel rejections correct, with zero false
merges/splits on development. Novel selection coverage is 96.1%; memory swaps preserve
probabilities and metrics. [Report](../runs/entity_novelty_v1/reference/report.html)
passed browser QA. The distance-separated task is a sanity check, not general novelty
or calibration. No memory records are allocated yet; the next proposed slice is
transactional allocation/revisit behavior. See [task record](entity-learning-task.md).

**Bounded descriptor variation succeeds, 11 September:** the learned/shared reader
trained at `--entity-noise 0.2` passes every declared development gate after 256 updates;
matching and identifiable task accuracy are 100%. All 102 CPU tests pass. Oracle,
geometry, zero-noise compatibility, raw scores, resume and reorderings are verified.
[Report](../runs/entity_variation_v1/reference/report.html). Claude completed both
review and reconciliation after explicit approval of the follow-up; no conceptual
objection remains.
See [task record](entity-learning-task.md). New-entity allocation remains next,
not implemented. This is bounded synthetic drift, not general visual robustness.

**Learned association succeeds on stable descriptors, 11 September:** the
Claude-reviewed `--entity-reader shared --entity-association learned` mode passes
all fixed development gates after 256 updates. Matching improves from 37.5% to 100%
on eight held-out descriptor groups; task accuracy is 100% on 128 identifiable cases.
Task-loss gradients reach the matcher; no exact lookup is used by its forward path.
All per-frame reorderings preserve probabilities. [Report](../runs/entity_learned_v1/reference/report.html)
and [task record](entity-learning-task.md) retain the scope: stable synthetic features,
fixed two-object slots and enumerated hypotheses; graph structure is not learned.
Next proposed: feature variation and unmatched/new entities.

**Shared entity reader succeeds, 11 September:** the Claude-reviewed
`--entity-reader shared --entity-association observed` path passes all declared
development gates after 256 updates: 100% identity/state/effect accuracy on 128
identifiable cases and correct bounded ambiguity handling. All 96 CPU tests pass.
Predicted probabilities are unchanged under all eight per-frame reorderings after
undoing output order. It uses 30,021 parameters and 128 persistent state floats.
[Report](../runs/entity_shared_v1/reference/report.html) and
[task evidence](entity-learning-task.md) retain limitations: association, two object
slots and the finite hypothesis set are supplied; graph learning remains unimplemented.
Next proposed: learn association while preserving this working reference.

**Supplied association complete, 11 September:** the Claude-reviewed
`--entity-association observed` diagnostic is implemented; 94 CPU tests pass.
Matched 256-update development identity accuracy improves to 88.3%, but state/effect
remain 64.8%/57.8% and combined gates fail. The trained recurrent reader changes its
probabilities under candidate reordering. This is a useful diagnostic gain, not
reliable binding or learned graph structure. See the
[updated task record](entity-learning-task.md) and
[report](../runs/entity_alignment_v1/observed/report.html). Next proposed diagnostic:
shared per-entity updates and permutation-consistent readout.

**Entity baseline built and evaluated, 11 September:** `--dataset entities` now runs
the [controlled two-object task](entity-learning-task.md) through the existing recipe.
All 92 CPU tests pass. The fixed 256-update baseline reaches development identity/state/
effect accuracy of 60.2%/62.5%/54.7%; every combined development gate fails. Training
state accuracy reaches 100%, but identity is only 66.8%. The graph comparison is
deferred under the declared stop rule. Source, cached resume, oracle, raw scores and
browser report checks are recorded in `runs/entity_learning_v1/verification.json`.
[Open the report](../runs/entity_learning_v1/reference/report.html). Next diagnostic:
separate descriptor association from state updating; no extra run is authorized by
this result, and the earlier entity-reader bottleneck remains unresolved.

**Task definition complete, 11 September:** [the first entity-learning task](entity-learning-task.md)
defines two-object identity persistence, state updates and action-effect prediction.
Both graph structure and latent values are intended to be learned; readable labels
are inspection aids, not imposed semantics. Two actual Claude reviews are reconciled.
A finite 512-case specification check confirms an exact history oracle and final-view-only
bounds of 50% identity and 25% state-pair accuracy. These are contract checks, not model
results. Next implementation step is the task generator/reference path and existing-reader
diagnosis, followed by a bounded recurrent baseline before any graph comparison.
No new neural training or graph implementation was started in this definition slice.

**Entity-design discussion, 11 September:** the user accepted diagnosing accessible
identity first, then testing controlled two-object binding. Subsequent discussion
proposes per-entity learned beliefs and external retrieval. Two actual Claude
exchanges are reconciled in [the design note](entity-memory-design.md): candidate
extraction, uncertain association and persistent keys are separate mechanisms.
Explicit entity storage remains a proposal; no new architecture or run was started.
The follow-up runtime/payload review adds four reconciled Claude exchanges: small
read/propose/commit interfaces, explicit gradient boundaries, and optional raw,
latent or readable payloads. Recalled media may reuse modality encoders but cannot
silently enter as new observations. Source IDs, historical time and typed revisions
remain distinct; a memory-origin label alone is insufficient.

**Current slice complete, 11 September:** the
[trainable encoder initialization comparison](warm-encoder-plan.md) is implemented
as `--fact-encoder-weights` for the event fact reader. Two brief actual Claude
reviews are reconciled; all 88 CPU tests pass. Exact transfer, unchanged remaining
initialization/RNG, encoder updates and donor-bound resume checks pass. Both new
512-update recipients finished in 103.5041 active CPU seconds total. At lr0.0003,
held-out location accuracy improves from the saved cold run's 8/32 to 31/32; at
lr0.001 it improves from 24/32 to 32/32. Held-out entity and joint accuracy remain
0/32 in both warm runs. Training entity accuracy is only 3/96 in each. Both
extraction gates fail; binding is skipped and the declared two-run slice is complete.

The [reference report](../runs/warm_encoder_v1/reference/report.html) and
[learning-rate comparison](../runs/warm_encoder_v1/lr_control/report.html) passed
structural and 1280x720 browser QA. `runs/warm_encoder_v1/verification.json` binds
raw-score, source, donor, matching cold settings/sampler, cached-resume, test and
browser checks. Development combinations are reused; the donor adds 512 upstream
updates / 8192 presentations. The shared encoder also reads the fixed instruction.
These results leave entity learning unresolved without isolating its cause.
Next proposed: freeze the donor encoder in one otherwise matched diagnostic to
test whether preserving its features changes entity learning.

**Prior single-event slice, 11 September:** the
[single-event agent-reader control](event-fact-plan.md) is implemented as
`--dataset facts --fact-reader event`. Two brief actual Claude reviews are
reconciled; all 86 CPU tests pass. A strengthened gradient check confirms the loss
reaches the observed fact, and exact resume preserves cached outputs. Two matched
512-update runs took 109.4034 active CPU seconds total. At lr0.0003, held-out entity/
location accuracy was 3.125% / 25%; at lr0.001 it was 0% / 75%. Joint accuracy was
0/32 in both runs. Both extraction gates fail and binding evaluation is skipped.
No third training run, new objective or memory change was started.

The [reference report](../runs/event_fact_v1/reference/report.html) and
[learning-rate comparison](../runs/event_fact_v1/lr_control/report.html) passed
structural and 1280x720 browser QA. `runs/event_fact_v1/verification.json` binds
source snapshots, settings/data/init/sampler matching, raw-score checks, exact
resume, tests and screenshots. The existing whole path does not learn the task
under this budget; the failing component is not isolated. Recent records are
detached on storage by the existing memory policy, while the live categorical
path still carries gradients. The trainable encoder initialization follow-up is
now complete above.

**Prior direct-control slice, 11 September:** the
[direct fact extraction and binding controls](fact-learning-plan.md) run in the
same recipe (`--dataset facts`). Two short actual Claude reviews are reconciled.
All 83 CPU tests passed; 12 focused fact/run checks passed after the report revision.
The first 512-update reference passed every declared gate in 10.3502 active CPU
seconds: entity, location and joint accuracy are 100% on 96 training and 32 held-out
combinations. Mean held-out NLL is 0.211649. The fixed selector answers every
enumerated two-record query correctly, including 768 queries / 384 pairs whose
constituent facts are both held out. Coherent location swaps also pass. These
reused fact combinations are not independent samples or natural-language evidence.
No second learning-rate run was needed.

The [standalone fact report](../runs/fact_grounding_v1/reference/report.html) has
verified tables, curves and expandable examples at 1280x720, with no broken images
or horizontal overflow. `runs/fact_grounding_v1/verification.json` binds the intact
training snapshot, separate final renderer, checkpoint/results, tests and browser
receipts. CLI resume reused cached predictions and unchanged result/metric files;
an independent probability-space calculation matches every saved binding score.
Next proposed: test the same factual task through the existing agent event/reader
path. Success of this freshly trained encoder and explicit selector does not prove
the world model's learned binding, recurrent retention or memory compression.

**Prior diagnostic slice, 11 September:** the
[current/recent factual recall diagnostic](recall-learning-plan.md) is implemented
in the same recipe (`--dataset recall --recall-mode current-recent`). Actual Claude
completed one review and one reconciliation after Alex approved the export. All 79
CPU tests pass; the actual diagnostic forward/backward check has finite losses and
the expected gradients. The 256-update CPU pilot finished in 144.58 active seconds.
The final checkpoint gets 71.875% seen-location accuracy on its 40 training episodes,
but only 17.5% on the 80 seen cases among 100 fresh development episodes: current
12.5%, recent 22.5%. Both declared gates fail. Overall development task loss is 0.31
at 12% coverage, worse than always abstaining (0.25). No calibration/test data was
loaded and no new memory loss or budget extension was started.

The [standalone report](../runs/recall_diagnostic_v1/pilot/report.html) passed
structural and 1280x720 browser checks. `runs/recall_diagnostic_v1/verification.json`
binds source/results/checkpoint/report hashes, tests and screenshots. Next focus is
basic entity/location binding and generalization; this pilot does not isolate the
input encoder, query binding, readout or insufficient optimization budget.

**Prior recall slice:** historical recall is implemented in the same recipe. Read the
[usable guide](recall-task.md) and [implementation record](recall-implementation-plan.md).
All 76 CPU tests pass, including exact pause/resume, held-out split isolation and
report-failure recovery. A 256-event default-memory forward/backward check completed.
The eight-update CPU development run selected update 7. On its 15 test episodes,
factual accuracy is 20%, every decision abstains, and task loss is 0.25. The model
does not yet demonstrate useful recall. Calibration selected its upper bound T=20;
this small sample establishes no calibration guarantee.

**Overnight work complete:** all seven topic reviews and their combined reconciliation
with actual Claude are complete. The [decision agenda](remaining-decisions-2026-09-10.md)
starts with five choices and the full design, followed by the detailed alternatives,
interfaces, evidence and remaining empirical questions. Recommended next: grounded
local memory learning with current/recent recall controls, then learned marking if
its marginal signal is useful. Preserve fixed capacity, exact historical records and
separate live, hypothetical and offline replay state. Freeze the whole selected
procedure before final calibration; each later task configuration needs its own evaluation.

The final consistency check completed on 10 September at 08:31 Berlin, before the
09:00 deadline. All 21 overnight Claude responses succeeded; saved evidence and source
identity are unchanged. The heartbeat `overnight-agent-design-proposals` is paused
with its persisted status verified. These proposals remain unadopted. Receipts are
`overnight-integration-local-checks.json` and `morning-handoff-check.json` under
`runs/reviews/state_memory_design_2026-09-09/`. No model run or test suite was repeated
for the morning handoff.

**Implemented:** the categorical belief and bounded session-memory design authorized
by Alex on 9 September 2026. Read [the model guide](belief-model.md) and the
[implementation plan/record](belief-implementation-plan.md). One editable
[recipe](../experiments/multimodal.py) still owns construction, targets, training,
evaluation, checkpoint/resume and reporting. No second trainer or runtime LLM was added.

The CLI defaults to `--state-model belief`: recurrent context, grouped categorical
prior/posterior, source-only evidence and a separate task workspace. Ordered event
transactions advance executed action/time once, merge partial packets against a
fixed prior and memory snapshot, and commit once. Shared dynamics power imagination.
Thinking/reflection leave the physical belief and observational history unchanged.
Both evidence age and inferred-state age/ordinal constrain causal memory reads.

Memory contains exact recent latent envelopes, chronological compression staging,
compressed history, protected user/agent marks and gated consolidation. Separate
perception/prediction/thinking readers have learned scale gates and a null choice.
Full categorical probabilities reach readers and compression. Source-only features
remain independently encoded and compressed. The default tensor payload is bounded
at 318,040 bytes per FP32 stream, plus bounded metadata and temporary computation.
Fresh `initial_state()` starts empty memory and workspace with the same model weights;
callers discard prior task progress and plans. Individual memory resets remain withdrawn.

Learning now includes observable reconstruction/prediction likelihoods, split
categorical KL, isolated full/partial teacher targets, delayed recall, frozen-reader
compression distillation and delayed marginal mark utility. These mechanisms are
implemented; effective long-horizon memory and calibrated uncertainty are unproven.
The ordinary two-step default history is too short to train delayed recall across
the default 32-record recent store. The guide gives explicit small-memory settings
that exercise all memory scales within an eight-step development history.

**Earlier belief-slice verification:** 69 CPU tests passed at that stage. Exact
pause/resume reproduces model, optimizer, sampler, RNG and training rows, including the optional extra-update gate. New tests
cover event retry/order, masked inputs, source/belief separation, mixed batches,
full-distribution reads, memory bounds/consolidation, provenance, snapshot loading,
common planning samples/RNG restoration and future memory at equal timestamps.
Final real PushT forward/backward check is saved with the run receipts.

Three short CPU development runs completed: eight synthetic updates, eight
instruction updates and two real PushT updates; no extra proposals or GPU runs.
Every run has raw metrics, checkpoint, source snapshot, media and an offline report.
The current report renderer has now passed browser visual checks at a 1280×720
viewport, with no broken images or horizontal overflow on the three reports.
See [verification](../runs/belief_v1/verification.json),
[synthetic report](../runs/belief_v1/synthetic/report.html),
[instruction report](../runs/belief_v1/instructions/report.html), and
[real report](../runs/belief_v1/pusht/report.html).

**Learning remains weak:** synthetic held-out image MSE is 0.237670 versus 0.007451
for copying the last image. The tiny real check gives 0.236499 versus 0.000050895.
The four-example instruction evaluation has 75% operation error. These development
populations are too small and training too short for capability conclusions. They
confirm execution and expose poor current predictions, not successful world learning.
The earlier negative instruction result is unchanged and preserved in the historical
[task implementation record](multimodal-plan.md); its old run paths are unavailable
in this local checkout and were not reverified or reconstructed.

**Claude collaboration:** two actual isolated CLI exchanges reviewed abstract
implementation invariants. No private source, dimensions or results were exported.
Claude withdrew overbroad demands for parameter-disjoint encoders, hard gates,
mask tokens and a particular RNG mechanism. The adopted properties are forward
information separation, frozen event inputs, bounded replay graphs and side-effect
free planning. Remaining fixed/variable-rollout and sealing concerns were resolved
by the concrete fixed-horizon interface, open-event type checks and local tests.
Receipts: `runs/reviews/state_memory_design_2026-09-09/implementation*`.

The Gaussian reference, existing task controls/output attribution and multiscale
adapters remain usable. Python `build_model()` retains its Gaussian default; use
`state_model="belief"` explicitly in code. Its SVG diagram exporter still describes
the Gaussian reference; the new guide has the categorical flow. Checkpoint schemas
are distinct, with no implicit Gaussian conversion. Completed runs retain their
own source snapshots and must be resumed with compatible source/settings.

**Active design proposal:** [decisions from belief and memory](decision-design.md)
connects the task workspace and planner through an exact objective/cost contract,
a learned observable-outcome head, and verification of actual results. This is the
Claude-reviewed discussion requested on 9 September, completed on 10 September;
it is not implemented or a new capability claim. Historical last-observed recall
with a separate factual not-observed answer and operational abstention is the
recommended first slice. Active current-location inspection is a subsequent task
requiring observation-conditioned continuations and isolated hypothetical updates.
Existing memory bounds, reset decisions and negative results are unchanged.

**Implemented first-task contract:** [selective historical recall](recall-task-design.md)
specifies four locations plus factual not-observed, separate abstention, loss 0/1/0.25,
two retrieval rounds, complete text-observation episodes and independent calibration.
Two further abstract Claude reviews reconcile cost/calibration and data-split claims.
Report seen/old-history performance separately: recognizing only unseen entities can
beat all-abstain without remembering any locations. The subsequent implementation
and development evidence are recorded in the guide linked above.

**Next step:** declare one bounded input/binding/readout diagnostic before any new
memory-learning objective; preserve the failed current/recent pilot. Numerical
costs are explicit research defaults, not inferred application preferences. Deployment-length replay,
general mark selection, calibrated probabilities and closed-loop behavior remain open.
