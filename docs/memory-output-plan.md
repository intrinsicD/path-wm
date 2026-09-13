# Remember a selected object and render its last observed state

## Active relocation comparison (13 September, approved continuation)

Hypothesis: independently varying an object's final side while preserving its
appearance and initial selection removes the appearance-to-location shortcut.
This is a data intervention in the existing three-frame task. No architecture,
loss, trainable initialization or normalization change is included.

Add an opt-in `relocation` curriculum; preserve the original `parity` generator
and completed runs. Each quartet shares background and initial object arrangement:
select left/right, then repeat both selections with the final objects swapped.
All four colors, both shapes, both initial sides and both final sides are balanced.
Eight initial appearance arrangements form a complete cycle of16 selection pairs.
Require complete cycles. Whole quartets stay in one dataset; independent seeds
give disjoint noisy backgrounds. Train128 pairs7701, validation32 pairs7702,
evaluation64 pairs7703. All16 target tuples and both movement types occur in every
split. Evaluation tests fresh rendered histories within the trained support;
it does NOT test novel tuples, motion rules, categories or longer histories.

Audit counts and exact inputs before training: within each quartet, relocation
counterparts have identical first frames and appearance but opposite target side;
selection counterparts share their final two frames but differ in target identity.
Bayes majority predictors using appearance alone or the exact initial frame
achieve at most50% final-side accuracy. Final-frame-only joint accuracy is at most
50%. Check split input hashes and complete quartet membership. The previous
parity held-out results remain failed; those tuples now occur in training and
cannot be called held-out composition in the new comparison.

Use the same frozen handwritten visual donor. BOTH input normalization and output
feature calibration continue to use the original parity TRAIN population, exactly
as before, keeping their buffers fixed across this intervention. Fresh trainable
initialization seeds7801/7802; no fine-tuning from the failed agent. AdamW and all
settings unchanged:1536 updates, batch16,300s training-loop cap per seed,600s total,
FP32, CUDA cap4GiB leaving1GiB free. Final checkpoint only; no checkpoint selection.
Development checks use separate seeds and at most16 updates. Source committed
before formal execution. Score the preserved normalized7801 checkpoint on the
same new evaluation population as a frozen control (no updates).

Keep every existing gate and additionally require ordinary and reset factual/image
accuracy on BOTH members of each relocation pair >=80%. Report per-attribute,
static/moved, and complete-selection/relocation-pair scores. Preserve all controls
and raw histories/outputs. Independently recompute counts and image-template scores;
strict standalone GPU reload must match saved outputs within1e-4. CPU/GPU pixel
agreement is separately checked against the existing1e-4 threshold; prior failure
remains visible and cannot be silently relaxed. No general generation or learned
retrieval-selection claim follows from a pass. Reports remain structural-only
while browser QA is unavailable; inspect the actual comparison PNGs.

Essential RED checks: factorial/input counterfactual alignment, exact split
separation, shortcut bounds, per-attribute/relocation scoring and gates. Extend
the existing exact resume/reload fixture to exercise the new dataset. Preserve
old tests and strict default behavior.

Public-only Claude review adds hidden-background prediction bounds and cue removal.
Also remove the later visible frame: neither single visible frame suffices for
the full task. Evaluate both removals after a state reset; require full reset
accuracy to exceed each by>=30 points for facts and images. Hidden-frame-only
side accuracy must be<=50%, joint<=25%. These are pre-run controls. A follow-up
corrects Claude's claim that the cue frame alone suffices and limits causal
interpretation of a data-balancing intervention. Receipts under
`runs/reviews/continuation_2026-09-11/relocation-*`.

13 September 2026. Alex approved the proposed observation-memory-output slice.
Use the actual Gaussian reference agent, bounded detached episodic snapshots and
the existing state-to-spatial RGB decoder. This is a controlled synthetic learning
experiment, not visual entity discovery, general language, search or learned physics.

## Task and split

Each episode has three64px RGB frames. Two distinctly colored objects (square and
cross) are visible; a neutral border selects one in frame0. Frame1 removes the cue
and may swap their positions. Frame2 hides both. Query: render the selected object's
last observed appearance/location and classify its color, shape and side. The
request is fixed by the recipe; no natural-language understanding is claimed.
Paired episodes differ only in the initial selected object. Final two-object and
hidden frames are identical; answers differ. Targets contain only the selected
object on a neutral background and never enter model observations.

Four colors; paired colors share parity but differ (0/2 or1/3). Shape IDs are0/1,
positions left/right. Training selected-object triples satisfy
`color % 2 XOR shape XOR side == 0`; test triples satisfy1. This assigns BOTH
members of every pair to one split. Familiar factors, disjoint joint combinations;
no claim of recognizing unseen categories. Train128 pairs seed7701, validation32
pairs7702 on training combinations with fresh backgrounds, test64 pairs7703 on
held-out combinations. Balance selection, initial side and movement within groups.
Keep all members of each pair together. Save every image/label hash and tuple list.

## Learning and diagnostics

Initialize the frozen visual hierarchy and RGB head from the exact trained
handwritten donor used by image-output v1. Ordinary initialization for the new
state updater, thinker, factual readouts and feature producer. Existing weights
remain unchanged. No detail/source-image features bypass agent state at query.

Train state/thinker/producer plus a three-factor factual head. Auxiliary factual
loss on the last visible state teaches useful writes despite detached snapshots;
this is supervised representation learning, not an independent probe. Alternate
ordinary and reset-before-recall conditions from the start. Reset replaces every
running token with a fresh hidden-frame state while preserving only the bank.
Store frame0 and frame1 states with observed provenance; retrieve both (capacity4,
retrieve_count2). This evaluates reading/binding within supplied memories, not
learned selective retrieval or autonomous storage policy. Decode working/reasoning
tokens only. No generated image is written back to observed memory.

Loss: mean factor cross-entropy at output +0.5 pre-storage factor CE + foreground-
weighted RGB MSE (foreground10, background1) +0.1 standardized feature MSE. Fit
feature normalization on training targets only. An independently parameterized
two-attention-layer direct reader sees the frozen encoder features of frames0/1
with explicit time positions and trains on factual CE; it does not feed the agent.
Its success is positive diagnostic evidence; its failure cannot prove lost data.

At evaluation report direct-reader, stored-state, ordinary, ordinary-without-bank,
reset-recall, reset-erased-bank, reset-paired-swapped-bank and fully erased-history
conditions. Targets stay fixed under ablations; additionally compare swapped-bank
outputs to the paired alternate targets. Report detached stored-state readout and
teacher target-feature reconstruction to distinguish writing/readout/output limits.
Do not infer retrieval-selection failure from a before/after recall gap alone.

Independent rendered-image scoring compares output pixels against all16 canonical
color/shape/side templates, alongside foreground/full RGB MSE. Report image and
factual joint accuracy, both-members-correct pair accuracy and factual/image
agreement. A decoded image must beat background-only and pair-mean MSE controls.

## Predeclared screen and budget

Two initialization seeds7801/7802, identical data, AdamW lr0.001, weight_decay0.0001,
batch16, gradient clip1, FP32,1536 updates each. Validation every128 updates; final
checkpoint only, no selection/search. Each run has a300s training-loop cap, total
600s. Explicit512+1024 resume on seed7801; exact CPU resume on a small fixture.
CUDA allocator cap4GiB with1GiB headroom; source/encoder preprocessing, final
evaluation, reports and CPU checks are recorded separately from training-loop time.
Incomplete runs retain results but cannot pass. Development-only check uses
different seeds and may repair implementation before formal test exposure.

Pass separately for each seed: ordinary AND reset-recall factual joint accuracy
>=90%, image joint accuracy>=90%, both-correct pair accuracy>=80% for both outputs;
each full condition exceeds fully erased history by>=30 percentage points.
Reset-recall also exceeds reset-erased-bank by>=30 points. Swapped-bank outputs
match alternate targets on>=80% of examples. Foreground-weighted image MSE beats
both background-only and pair-mean controls. All outputs finite, frozen tensors
unchanged and independent saved-output metrics agree. Software integrity and
learned-capability gates are separate. No broad capability baseline is replaced.

Essential RED checks: paired rendering/split/label consistency; post-storage state
reset really isolates bank, erase makes pair inputs equal, swapping bank is batch
local and cannot alter source; gradients teach writes but memory stays detached;
future inputs rejected; frozen donor/head unchanged; exact resume and standalone
reload. Reuse Run and existing report renderer; HTML QA structural-only while
browser access is unavailable. Save target/history/prediction panels and raw arrays.

Claude public-only review identified useful intact-state/no-bank and pixel-scoring
controls. Locally reject treating intended composition or a decoder prior as
inherent leakage; distinguish supervised writes from independent probes. Follow-up
acknowledgment received: Claude withdrew the two leakage characterizations and
accepted the auxiliary-training distinction. Its requested distractor-scaling test
is deferred because this slice explicitly makes no retrieval-selection claim.
Briefs/receipts: runs/reviews/continuation_2026-09-11/memory-output-*.
No private source/data/results sent.

## Implementation checks before formal execution

Four new tests pass, including paired split/target checks, reset/erase/swap and
future-memory isolation, supervised-write versus frozen-module gradients, exact
CPU optimizer/model/RNG/sampler resume, standalone strict reload and report content.
Together with image-output, visual-memory, multimodal and run regressions,28 tests
pass; lint passes. A separate GPU check and16 development updates (seed17801,
data17701/17702/17703) complete. Training-loop time2.925s, peak402MiB reserved.
This is workflow evidence, not a capability pass or model selection. Development
episodes inspected visually. Reports embed raw-result summaries and labeled
history/target/ordinary/reset/erased/swapped panels; structural HTML QA only.

## First comparison and bounded normalization repair

Both reference seeds completed1536 updates (273.897s/257.496s). Neither passes.
Seed7801 ordinary/reset factual accuracy25%, image0%, complete pairs0%; seed7802
results retained unchanged in its run. Teacher-feature decoding classifies every
held-out object correctly. Validation factual readouts remain weak as well, so
this is not merely a held-out distribution problem. Initial independent audit
reproduces60 saved metrics to4.98e-9 and CPU reload to6.56e-6; actual GPU resume
ledger prefix matches. Reports/predictions show averaged objects rather than
selection-dependent output. No learned memory capability is claimed.

Development-only feature inspection (seed17704) finds per-level RMS1049.8/1070.9/
1092.8. Paired selection contrast after token LayerNorm is0.01794/0.01136/0.00965;
fixed per-channel standardization before LayerNorm gives0.669/0.966/0.765. This
numeric contrast motivates a test; it is not proof of better learning or the sole
cause of failure. Preserve raw encoder and decoder weights. Fit per-level channel
mean/std from TRAIN OBSERVATIONS only, floor std at0.01; freeze and save buffers.
Preserve all feature validity/time/content-time/grid metadata. Encoder normalization
does not alter the teacher or decoder's expected raw feature representation.

Predeclare ONE added repair run, seed7801, same1536 updates, data, losses, batch,
optimizer, initialization of trainable tensors and300s training-loop cap. This is
an additional bounded stage after the original two-run budget; no sweep or best-
checkpoint selection. Gates unchanged. The previously tested combination split
is reused for the explicit preprocessing comparison, not a newly untouched test.
This single seed cannot establish robust general benefit. No further repair run
is part of this slice if it fails. Source committed before the repair run.

Essential new checks: the no-op normalization wrapper yields identical metadata,
outputs and trainable initialization; calibration ignores invalid entries, rejects
empty/nonfinite observations and never updates during inference; teacher unchanged;
normalized full/checkpoint-resume/standalone routes agree. An independent direct
reader uses the same changed input preprocessing. No new trainable parameters.
Claude recommends an identity-transform control; exact no-op equivalence is checked
in software. Effects on optimization are a possible mechanism, not evidence that
the task was independently solved. Statistics are never fit on validation/test.
Follow-up Claude acknowledgment accepts this framing and the exact no-op control.
All30 targeted tests pass before the repair run, including normalized replay and
standalone loading, exact no-op gradients and unchanged teacher weights. The GPU
normalization check passes. Traces distinguish raw encoder outputs from normalized
published features and retain the fixed channel statistics.

## Completed results

Reference source ab22ff5; normalization source dd26b05. All three runs complete1536
updates within their declared300s caps. Existing runs and donor are unchanged.

| Condition | Recall factual joint accuracy | Recall image joint accuracy | Training seconds |
| --- | ---: | ---: | ---: |
| Raw features,7801 |25%|0%|273.897|
| Raw features,7802 |12.5%|0%|257.496|
| Fixed channel calibration,7801 |0%|0%|279.418|

All held-out gates FAIL. The normalized model reaches100% factual/image accuracy
on fresh-background validation episodes with familiar combinations. This is a
learning improvement under this setup; it does not establish held-out composition
or separate representational effects from optimization. The normalized checkpoint
is an opt-in experimental artifact, not a replacement for the frozen broad baseline.

Frozen-checkpoint CPU validation replay saves all64 familiar-combination episodes
and controls separately. Ordinary and reset recall each give64/64 factual answers
and images,32/32 complete pairs. Reset-erased-bank gives8/64; swapped banks give
64/64 correct alternate answers/images. Thus this trained path uses the supplied
memory for familiar combinations. Raw validation arrays and scores are retained
under normalized_7801/validation_predictions.npz and validation_result.json.

On128 held-out examples after reset/recall, factual color128/128, shape128/128 and
side0/128. Image-derived color123/128, shape127/128, side0/128. All128 factual
predictions are members of the eight training triples. The independent direct
reader instead gets color/side128/128 but shape3/128;122/128 stored-state predictions
and125/128 direct-reader predictions also belong to training triples. These patterns
strongly suggest learning the parity correlation instead of independently binding
every property. The training design permits this shortcut: any two target factors
determine the third. Joint failure does not mean that identity is entirely absent.
Teacher-feature decoding classifies every held-out target correctly (RGB MSE0.00010315).

The next proposed curriculum must vary location independently of appearance, with
paired counterfactual moves and held-out histories. Any new compositional split
needs its own identifiability/shortcut audit. Do not relabel this failed parity
benchmark as passing, or replace its held-out data retrospectively. No additional
training beyond the single declared normalization repair was performed.

Software and numerical verification:30 targeted checks pass, including exact CPU
model/optimizer/RNG/sampler resume for both configurations, exact no-op outputs and
gradients, trainable-initialization equality, metadata/trace preservation and frozen
parameters/statistics. All180 saved test metrics independently recompute within
5.10e-9. GPU512+1024 resume preserves its ledger prefix. The normalized standalone
checkpoint reproduces all128 saved GPU images and logits exactly at batch16.
Its CPU outputs differ by up to0.003258884 per pixel, failing the preexisting1e-4
audit threshold; all128 factual labels and nearest-template image labels agree.
This failed cross-device tolerance remains visible, not silently relaxed. Raw-
feature checkpoints differ by at most7.08e-6. All checkpoints strictly reload;
the normalized encoder/decoder hashes match the raw-feature reference exactly.

Peak CUDA reserved402MiB; three formal training loops total810.811s, plus2.925s
development training. Preprocessing, final evaluations/reload audits and reports
are outside loop timing. Combined and per-run HTML are structurally checked only;
static and moving comparison PNGs were visually inspected. Inspection does not
substitute for browser QA.

[Combined report](../runs/memory_output_v1/report.html),
[independent audit](../runs/memory_output_v1/verification.json),
[attribute/combination diagnosis](../runs/memory_output_v1/combination_diagnostic.json),
[normalized checkpoint](../runs/memory_output_v1/normalized_7801/weights.pt).
Checkpoint SHA256 ad1e4d9115eb569755d73bde150126ddd1139929bb41909a4e1dddadaabe9238.

Reproduce the normalized comparison into a fresh directory:

```bash
.venv/bin/python -m experiments.memory_output --weights runs/hierarchy_training_v1/decoder_rate_repair/seed_7501/hand_both/weights.pt --output runs/memory_output_v1/new_normalized --device cuda:0 --seed 7801 --normalize-input
```

The exported model needs no donor file or target data for inference:

```python
import torch
from experiments.memory_output import load_model, fact_labels

model = load_model("runs/memory_output_v1/normalized_7801/weights.pt")
with torch.no_grad():
    output = model(history_rgb, mode="reset")  # [B,3,3,64,64] observed frames
    image = output["image"]
    color_shape_side = fact_labels(output["facts"])
```

This fixed task interprets the visual selection cue; it has no language-request
parser. The last frame is hidden context, not the target. The harness stores two
detached observed states and retrieves both; learned search, entity discovery,
general image generation and natural-video transfer remain untested.
