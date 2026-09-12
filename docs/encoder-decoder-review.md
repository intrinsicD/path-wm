# Encoder–decoder review — 12 September 2026

**Recommendation:** preserve useful trained components, isolate failures before
replacing them, and compare modest extra processing in vision. Real speech needs
a different temporal input/output contract. Structured text needs a diagnosis of
the path through the agent before a larger encoder. Unrestricted language and
media generation require additional training objectives and data.

This is a source and retained-results review at code revision53dc797, with an
independent public conceptual Claude critique. No new model training, capability
evaluation or architecture change was performed. Recommendations below are proposed
experiments, not measured improvements or a frozen run specification.

## What the baseline does and does not establish

The [latest baseline](capability-baseline-plan.md) tested separate checkpoint roles.
The general synthetic/instruction models had8 optimizer updates each, PushT2,
the separate CNN perception example12, and its dynamics example6. Those results
establish their present weakness, not a capacity ceiling. The manually constructed
visual checkpoint has deliberately zeroed branches and fixed-task routing; keep it
as a diagnostic control, not the default initialization for general learning.

Older trained CNN, DINO and decoder-recovery alternatives were not included in that
baseline. Their preserved experiments inform this review but are not fresh tests of
the current integrated agent, nor interchangeable checkpoints in its scorecard.

There are two distinct interfaces:

- Separate perception recipes decode spatial encoder features directly.
- The multimodal agent normally decodes its state tokens after observation updating,
  and sometimes after dynamics. A bad image can originate in the encoder, state
  update, dynamics, or decoder. Its RGB loss does not locate the failure.

## What is already processed before feature publication

[FeatureHierarchy](../pathwm/models/multiscale.py) processes the finest scale with
attention and an MLP, merges that finished scale into the next, optionally reads
the finer features through cross-attention, and processes the new scale again.
Every published scale is processed. `depth` already controls blocks per scale;
the ordinary multimodal constructor currently uses its default1. Fine features
remain available alongside coarser ones.

Thus, the proposed extra processing is a useful **depth/structure comparison**,
not a missing processing stage. Spatial scale and semantic abstraction are also
different: a coarser grid does not automatically represent objects or concepts.
Learning objectives must reward identity, properties and temporal consistency.
Reconstruction alone does not require those abstractions.

The current implementation constructs dense attention masks and dense pooling
membership matrices for tiny inputs. Increasing camera resolution and applying
global attention to all patches is not a comfortable-GPU strategy. Prefer bounded
views/crops, convolutional downsampling or windowed spatial processing, with local
details retained. Windowed hierarchical attention has an established architectural
precedent in [Swin](https://arxiv.org/abs/2103.14030); its benefit here remains a
hypothesis, not an argument to replace the whole model with Swin.

## Pair-by-pair assessment

### 1. Multimodal image encoder and patch image decoder

**Implemented:** patch convolution → three processed scales → agent state →
learned output-patch queries → linear RGB patches. The general pilots use16×16
images and width16. The output decoder has no convolutional refinement between
neighboring output patches. The hand-built checkpoint's gray output is expected
from its zeroed decoder, not evidence that its input encoder recognizes nothing.

**First action:** establish a properly trained direct image pair and compare it
with the state-mediated path. Use actual image detail sufficient for the target;
upsampling an already reduced image cannot restore removed visual evidence.
Continue useful ordinary weights or use an ordinarily initialized small pair;
do not train all the zeroed diagnostic branches as if they were a pretrained model.

**Architecture comparison:** one versus two blocks per scale, then a lightweight
convolutional/residual stem or output refinement if the direct pair still has
specific spatial errors. Test stage width separately from depth; changing both
would obscure what helped. Expose this through existing constructors/recipes.

**Training:** combine reconstruction with the actual observation-memory questions
and, when labels exist, entity matching/property readouts. Preserve color if the
task asks about color; enforce invariance only to changes that preserve the target.
If encoder features pass while state-mediated answers fail, repair the adapter,
state update or readout instead of spending the next budget on deeper vision.

### 2. Separate CNN and convolutional reconstruction/dense heads

**Implemented:** a compact convolutional hierarchy with fine/coarse maps, optional
residual depth within each branch, cross-scale exchange, and spatial reconstruction
or task heads. These already have a stronger local spatial structure than the tiny
multimodal patch decoder. See [encoder](../pathwm/models/encoders.py) and
[decoders](../pathwm/models/decoders.py).

**Relevant retained evidence:**

- In the three-seed [depth/exchange study](../runs/encoder_study_2026-09-08/evaluation/factorial_summary.json),
  deeper branches with exchange reduced normalized pose error `q` from2.47–3.08
  to1.07–1.41. The required gate was `q≤1`; all12 factorial arms failed it.
  Extra depth with exchange took about30% more training time. Validation comparisons
  at matched elapsed time also favored it; those are not matched-time test results.
- The [decoder recovery experiment](../runs/decoder_recovery_2026-09-08/evaluation/metrics.json)
  held the task-adapted encoder fixed. On4146 reused COCO test images, refitting its
  existing decoder reduced RGB MSE from0.272195 to0.006105; a fresh decoder reached
  0.006390. Encoder weights remained unchanged. This demonstrates substantial
  recoverable image information, not general object understanding.
- The recovery had a tradeoff: refitting the existing decoder increased PushT RGB
  MSE from0.000188 to0.001635. A fresh decoder was worse still on PushT. Training
  on one domain did not preserve every previously learned output behavior.

**Recommendation:** use the retained trained CNN as a candidate donor, and test
frozen-encoder decoder repair before discarding it. Maintain mixed-domain replay
and measure retention when fine-tuning. Extra branch processing is supported as a
candidate; its success on one geometry task does not establish webcam identity.
An adapter from spatial maps into the agent's token/time contract is still needed.

### 3. DINOv2 and a trained reconstruction/task head

**Implemented:** local pretrained ViT-S/14; the wrapper reduces its input contract
to64×64 then resizes to224. Published `local` features are patch embeddings, `fine`
features are final transformer tokens, and `coarse` is their average pooling.
These are not three successive transformer stages.

**Evidence:** the historical [frozen-feature probes](../runs/encoder_study_2026-09-08/evaluation/probe_summary.json)
on512 images found DINO mask IoU0.5832 versus0.3183 for the selected custom encoder,
but worse reconstruction MSE:0.037109 versus0.007349. This is task-dependent evidence
from limited reused data, not a general ranking. The historical DINO pose references
also missed their [readiness target](../runs/encoder_study_2026-09-08/evaluation/reference_summary.json).

**Recommendation:** retain DINO as a frozen pretrained comparator/fallback; train
the adapter and appropriate head first. Check real input resolution, selected
intermediate transformer layers, feature normalization and access to local detail
before interpreting reconstruction errors. Compare early/middle/final features
with matched readout capacity. The official [DINOv2 repository](https://github.com/facebookresearch/dinov2)
provides pretrained features and dense-task heads; it is a representation backbone,
not a pretrained inverse image generator. Do not start its pretraining from scratch
as the first experiment on this GPU. Using it would also not confer persistent
identity automatically.

### 4. Video encoder and frame decoding

**Implemented:** a separate image-style patch encoder with temporal/spatial pooling
and causal attention. Its weights are separate from the still-image encoder.
`decode_video` applies the image decoder to an ordered sequence of states; there
is no separate learned video decoder. Current examples use short frame windows.

**Recommendation:** share a trained frame backbone, then train a small causal
temporal processor on real sequences. Preserve order, timestamps, missing frames
and action alignment; test reversal and occlusion, not just per-frame RGB.
Separate spatial/temporal processing has a precedent in
[TimeSformer](https://arxiv.org/abs/2102.05095). Our proposed causal streaming version
is an engineering adaptation; that paper does not validate our memory/dynamics.

Train observed-frame reconstruction first, then action-conditioned future states
and short decoded sequences. Score moving objects separately from static background
and compare with copy-last. A better image decoder alone cannot fix incorrect
future states. Longer coherent, goal-directed video creation is a later capability.

### 5. Audio encoder and waveform decoder

**Implemented:** four-sample linear patches and processed temporal scales; the
decoder reads four queries and linearly emits a fixed32-sample waveform. At the
toy data's8kHz rate that is **4ms per chunk**. The examples are artificial signals,
not a speech corpus. More training can improve that toy task but cannot establish
speech understanding or generation with this contract and data.

**Recommendation:** define stream timing, sample rate and meaningful speech context
first. Use convolutional waveform or time-frequency processing over appropriate
windows; retain timing through any downsampling. For output, use a temporal waveform
decoder or a codec decoder with a learned sequence generator. A codec only decodes
supplied acoustic codes; speech generation still needs text/state-to-code learning.
[EnCodec](https://arxiv.org/abs/2210.13438) is a concrete streaming encoder/decoder
reference. It is an audio reconstruction component, not a complete speech recognizer
or conversational model.

For an own-model path, start with narrow sound classes and short reconstruction.
For useful speech sooner, compare a pretrained audio component with trainable
adapters. Evaluate content recognition, acoustic quality and state retention
separately. Include speech-only segments: silence-heavy average waveform MSE can
reward a silent output. Spectral errors and inspected/listened reconstructions
are useful alongside waveform metrics. Deeper attention alone is not the first fix.

### 6. Text encoder and autoregressive byte decoder

**Implemented:** UTF-8 byte embeddings, processed scales, and a small decoder with
one causal attention read followed by a state read. No language pretraining.
The ordinary generation default is32 tokens. This is a short structured-language
interface, not an already trained writer.

**Evidence:** the direct structured fact model correctly reads all32 held-out
entity/location combinations. Merely transferring that encoder and leaving it
trainable was already tried: both warm event-model runs had0/32 entity accuracy,
while location reached31/32 and32/32. See the
[warm-encoder study](warm-encoder-plan.md). Do not repeat that transfer as if it
were an untested fix. The transfer experiment did not freeze the encoder and
cannot uniquely identify which downstream stage fails.

**Recommendation:** freeze the successful donor, train compatible downstream
readouts, and probe encoder features → observation evidence/codes → world tokens
→ working tokens. Check learned conditioning, bottleneck usage and gradient flow
at the first failing boundary. Keep exact output-length and autoregressive tests
separate from teacher-forced losses; use shuffled/absent state to detect a decoder
that predicts from its prefix while ignoring memory.

Continue training the small model for bounded facts/instructions. A deeper byte
decoder and longer context can be tested for that scope. Essays and conversation
need broad language training; my practical fallback is a compact pretrained language
model with a state adapter. That does not require replacing the research agent's
memory/planning architecture. A particular language model and its whole-stack GPU
fit have not been selected or measured.

### Other adapters

`VectorEncoder` is a small MLP for numeric/vector inputs; it has no dedicated
inverse decoder in the default recipe. Normalize known units and add an explicit
numeric target/readout when a task needs one; current evidence does not justify
a larger encoder. Metadata, feature-conditioning and task modules are also not
independent sensory encoder/decoder pairs. General tool execution is a separate
missing capability, not something a text decoder already supplies.

## Comparison sequence and decision rules

Start with vision and the already available structured-text diagnostic, consistent
with the selected observation-memory-first goal. Use existing recipes and reports.

| Test | What a positive result supports | Next action if the observed path fails |
|---|---|---|
| Frozen features → fitted task head / direct decoder | Information is accessible to that tested head on that held-out task | Validate head optimization/capacity and input evidence before attributing failure to the encoder |
| Same encoder → state → equivalent task head | The observation update preserves useful information | Probe intermediate stages; test adapter/update/readout before increasing encoder depth |
| Delayed or occluded history → answer | Task information survives and can be retrieved over time | Separate retrieval, binding, compression and readout errors |
| Action → predicted state → output | The complete learned prediction path helps the task | Compare observed-state decoding and future prediction; repair dynamics if only the latter fails |

Include null/shuffled features or states, a random frozen encoder where compatible,
and an ordinarily initialized trainable baseline. Keep head family, populations,
data exposure and optimization budget explicit. A successful probe establishes
accessibility for its actual decoder family; a failed probe does not prove absence.
No one of reconstruction, recognition and memory success establishes the other two.

For extra processing, first compare shallow/deeper with identical inputs and
decoder. Also spend the deeper model's elapsed-time budget on more updates of the
shallow model. If deeper wins, compare extra depth against extra width to investigate
parameter allocation. Report parameter counts, exposure, wall time, peak allocated
and reserved GPU memory, and process/device usage; do not pretend all budgets can
be identical simultaneously. Keep pretrained donor cost/exposure visible separately.

Use episode/identity/recording-group splits, validation-only selection and fresh
final evaluation. Historical test sets here have already been inspected repeatedly.
The12 photos and6 clips in the development pack are useful for checking ingestion,
not enough by themselves to establish general webcam capability. Numeric quality
gates, sample counts, seeds and a bounded compute schedule must be declared before
the next formal run. This review does not invent a pass threshold after the fact.

For the8GiB GPU, train one pair/adapter at a time initially; cache frozen features
where valid and use small batches. Include an integrated training/inference profile
before claiming comfortable fit. A proposed initial process budget is at most4GiB,
with at least1GiB device headroom after other processes; lower it if necessary.
This is a proposed budget, not a measured allocation or a changed user requirement.
Cached features must record source/preprocessing/encoder version. If feature
conditioning changes, recompute conditioned stages or include that condition in the
cache key. Preserve source/time and recalled-versus-observed attribution at adapters.

## Review provenance and remaining uncertainty

Claude received only public conceptual questions, without private code, results or
architecture dimensions. Adopted criticism: compare extra training at equal elapsed
time; account for pretrained exposure; use input-prior controls; distinguish
architecture hypotheses from measured diagnoses. In a short reconciliation Claude
withdrew claims that a successful arbitrary decoder necessarily establishes
linear/shallow accessibility, that scratch training always requires failure across
all tasks, and that persistence is universally harder than reconstruction.

Exact briefs, responses and execution receipts are retained under
`runs/reviews/continuation_2026-09-11/encoder-decoder-{review,reconcile}-*`.
The compact [local evidence extract](../runs/reviews/continuation_2026-09-11/encoder-decoder-local-evidence.json)
records the inspected input hashes and selected retained measurements. No claim of
independent replication or new capability testing is attached to this review.
