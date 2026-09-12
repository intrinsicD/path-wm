# Remember a selected object and render its last observed state

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
acknowledgment pending. Briefs/receipts: runs/reviews/continuation_2026-09-11/memory-output-*. No private source/data/results sent.
