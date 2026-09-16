# Image-to-video codec reuse: evidence and limits

16 September2026. Follow-up to the [implemented shared codec](shared-video-vae-plan.md).
The user asks why the small temporal addition barely helps, whether published work
and weights exist, and where the temporal component sits in our implementation.
No additional model run or external weight download in this literature follow-up.

## Our actual connection

`RGB[B,T,3,H,W] → shared image encoder per frame → mu/logvar[B*T,Z,h,w]`.
The optional temporal module reads current/two preceding mu grids, validity and
log(1+elapsed seconds). A left-time-padded3×3×3 convolution, SiLU and1×1×1 projection
predict a residual for the current mu. Output projection starts at zero. Variance
remains frame-local. Independent Gaussian sampling follows; the shared image decoder
then reconstructs each frame. No early-scale temporal features, persistent cache,
temporal downsampling or learned video prior. This is a minimal late-fusion control,
not a reproduction of the larger published video codecs below.

## Primary sources and released weights

- **Stable Video Diffusion**,2023: the official model card describes fine-tuning the
  image f8 decoder for temporal consistency and provides both temporal and standard
  framewise-decoder weights. This is a concrete precedent for reusing image-codec
  components while modifying temporal reconstruction. Its generative diffusion
  network is a separate component, not evidence that a codec alone predicts video.
  [Official model/weights](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt),
  [paper](https://arxiv.org/abs/2311.15127).
- **Improved Video VAE**,CVPR2025: the authors study image-weight inflation and
  spatial/temporal capacity conflicts under temporal compression. Their KTC divides
  keyframe and temporal processing; group-causal convolution permits interactions
  within a frame group. Their failure analysis is relevant motivation, not a proven
  explanation of our result: we retain every frame and do not compress time.
  Group causality also does not imply strict frame-level streaming causality.
  [Peer-reviewed paper](https://openaccess.thecvf.com/content/CVPR2025/html/Wu_Improved_Video_VAE_for_Latent_Video_Diffusion_Model_CVPR_2025_paper.html),
  [readable preprint, sections1/3](https://arxiv.org/html/2411.06449v1),
  [authors' code and trained-model link](https://github.com/ali-vilab/iv-vae).
- **CogVideoX**,2024/2025: section2.1 describes causal3D processing in encoder and
  decoder, spatial and temporal compression, and staged clip-length training.
  Its VAE reconstruction objective includes L1, perceptual and later adversarial
  terms. This differs materially from our small posterior-mean-only residual and
  RGB-squared-error baseline. Full generator attention findings should not be
  attributed to its VAE. [Paper](https://arxiv.org/html/2408.06072v3),
  [official release including VAE](https://github.com/zai-org/CogVideo).
- **HunyuanVideo**,2024: section4.1.1 trains its3D VAE from scratch, mixing video
  and still-image data4:1 with a resolution/length curriculum. It supports joint
  image/video training but does not show that initializing a VAE from image weights
  is always best. [Technical report](https://arxiv.org/html/2412.03603v1).
- **Mathieu et al.**,ICLR2016: examines blurry future-frame prediction with MSE
  and alternative objectives. This is background on ambiguous predictions, not a
  demonstration that our observed-frame reconstruction blur has that same cause.
  [Paper](https://arxiv.org/abs/1511.05440).

## What the local evidence does and does not say

Both continued arms worsen RGB on the single reserved source, while their separate
validation source improves. Therefore the worsening cannot be attributed solely to
adding the temporal layer. Only four source videos train the model, and every
current target frame is fully observed; useful historical reasoning is not required.
The latter is a task-design concern, not an identified causal failure mechanism.

A source-settings audit adds a concrete confound relative to the untouched image
checkpoint: original beta=0.1 and color_weight=6; the registered new simple baseline
uses beta=0.01 and color_weight=0. Training also changes the data domain. Thus the
paired temporal-versus-frame comparison is matched, but comparison to the frozen
source is not an isolated test of video fine-tuning or temporal architecture.
The recorded source and run JSON files substantiate these settings; no corrective
run was silently added after the fixed budget.

Before claiming an architecture problem, compare continuation with the original
image loss preserved, freeze the shared image codec in a separate control, and
use a task with missing/occluded current-frame content where history can actually
help. Randomized/wrong history and exact no-future checks remain essential.
These are proposed next comparisons; thresholds/data/budgets still need registration.
A temporal decoder is a reasonable later ablation, motivated by SVD, not yet adopted.

## User follow-up: images as single-frame or repeated-still videos

Both fit the existing VideoVAE interface. A single-frame example includes the temporal
module's current-frame branch but cannot train its past-frame kernel taps: their
inputs are zero. Repeating a still at successive times also trains past-frame taps
and provides a constructed constant-scene target. It does not reveal actual motion
or establish that the photographed real scene stayed stationary. A new gradient
contract test confirms this difference with an active residual at1 vs4frames.
Eight video-wrapper tests pass; together with the prior54 regressions this is62
unique scoped tests. No new training or efficacy measurement was performed.

Proposal: mix single-frame images, real moving clips and a small declared share of
synthetic repeated-still clips, matching frame exposure/loss normalization. Excess
static examples might favor weak motion response; measure that rather than assuming
an optimal mixture. The implemented recipe's auxiliary image loss currently bypasses
the temporal module, so this common-path training would be a new comparison, not
something already measured. CogVideoX section3 explicitly treats images as one-frame
videos; HunyuanVideo section4.1.1 gives a separate joint image/video VAE precedent.
Neither source validates a particular repeated-still ratio for our small model.

The follow-up feature-space distinction: before temporal mixing, identical frames
use identical image weights and deterministic spatial features. After mixing, the
same current image can have different codes depending on its history. Geometry
and decoder remain shared; there are no explicitly reserved appearance/motion
channels. Useful direction retention and preserved appearance do not follow from
matching shapes alone. This is consistent with the current residual design, not
an adoption of separate image/video semantic spaces.

The user additionally asks whether different image/video features require different
image weights or pixel-level video retraining. They do not: a shared frame encoder
can feed a learned temporal feature module. An alternative to modifying mu in place
is to retain spatial features and append temporal features/tokens. This is proposed,
not implemented. Freeze the shared image codec first to isolate temporal learning;
if its compressed features lack useful detail, compare earlier-scale features and
then cautious joint fine-tuning. No architectural requirement forces training from
scratch. Equal feature dimensions alone do not guarantee useful motion semantics.

## User follow-up: spatial neighborhood versus temporal horizon

The current mixer has one3x3x3 convolution (time, height, width), followed by
a1x1x1 output projection. It reads current/two previous feature grids. At the
experiment's4fps, these three slots span0.5seconds, with no state across calls.
Its3x3 spatial neighborhood is on the latent grid: the current48x48 inputs become
12x12 grids. One grid step corresponds to four input-pixel steps, but each feature
already has a larger overlapping encoder receptive field. Therefore3x3 is neither
a3x3 pixel neighborhood nor a strict maximum speed that can be represented.

Large inter-frame displacement makes direct local feature correspondence harder.
Wider spatial communication or coarser grids can help; a wider spatial kernel alone
does not increase the temporal horizon. Three stride-one, undilated3x3 layers have
a theoretical7x7 spatial receptive field, including when weights are shared across
iterations, but are not equivalent to a single7x7 filter. Extra iterations still
cost computation. Global attention already has global spatial access; repeating it
refines processing rather than enlarging that access. Equal-channel9x9 convolution
has nine times the spatial kernel taps of3x3, not nine times the whole model cost.

Proposed small comparison: local3x3 processing with additional depth/shared
iterations, then coarse-scale context or a modest5x5 alternative. Do not add all
kernel sizes by default. Vary displacement, cadence and occlusion with tasks that
require history; compare resources and appearance retention. No variant is adopted
or measured here. Forecasting/generation additionally needs an appropriate learned
transition/prior, conditioning and objective; a larger encoder neighborhood alone
does not provide it. IV-VAE section3.4 motivates enlarging receptive fields as
resolution increases and uses parallel dilated convolutions plus compressed-space
attention. That precedent does not identify our local failure cause.
[Primary paper](https://arxiv.org/html/2411.06449v1).
