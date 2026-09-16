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
