# VAE color and grid-artifact diagnosis

15 September 2026. User sees improved reconstructions with artifacts and weak
color; asks why. Diagnose the saved v2 codec before changing its architecture.
This is an attribution/reparability experiment, not a new quality claim.

Reference: C/beta0.1, seed57101,512 updates, from `runs/spatial_vae_v2/formal`.
The existing ten arms provide descriptive beta/attention comparisons. Reuse the
same source-group train512/validation64/test96 populations, explicitly treating
the previously viewed test population as diagnostic, not a pristine selection set.
No checkpoint/architecture selection or alteration of original artifacts.

1. Audit raw output clipping, channel order, saved vs model vs PNG, global RGB and
   opponent-color errors, saturation slope, interior/border errors. Compare mean
   and sampled decoding. Period2/4 phase-conditioned residual means are diagnostic
   grid statistics, not automatically artifacts; also measure target phase energy.
2. Decode constant latent fields (zeros and spatially averaged encoded colors)
   with no encoder input at decode time and no noise. Hook every decoder stage,
   inspect the first shuffle-generated phase variation, and retain matched shapes.
   Constant27 RGB patches at levels0.15/0.5/0.85 test color response. PixelShuffle
   itself only rearranges values; learned channel differences may become a grid.
3. Frozen color readouts:256 training images,64 validation,96 test; fixed linear
   ridge1e-3, image-level mean RGB common targets. Include input identity and shuffled
   heldout rows. Compare pooled features at stem, both sides of each C, posterior mu,
   decoder input/stage outputs and final RGB. These show accessible global color,
   not all spatial color information or irreversible loss.
4. Three fixed512-update continuations from the reference: encoder-only,
   decoder-only, joint. All use fresh AdamW3e-4/wd1e-4, clip1, seed57201, batch8,
   original Gaussian MSE+beta0.1 KL; identical sampler/noise. Frozen parameters and
   buffers must remain bitwise unchanged. Do not reset weights or change widths,
   normalization, shuffle, loss or data between arms. Save separate exports/reports.

Predeclared reparability screens: at least20% lower global chroma MSE, and at least
50% lower period4 interior phase-residual RMS, each with total raw RGB MSE no more
than5% worse. Treat these as separate outcomes. Brief-fit failures are inconclusive;
successful decoder-only repair shows some relevant information was already usable.
No universal causal attribution or full repair from a small matched comparison.

Budget:<=120s per continuation,<=2GiB GPU reserved,>=1GiB GPU free; <=10min total
diagnostics/continuations,<=500MiB added artifacts,>=3GiB disk reserve. No deletions.
Existing recipe, Run, RidgeReader, report renderer and snapshots; add focused
numeric/frozen-component tests, retain independent audit and labelled panels.
Actual Claude reviews receive only public conceptual methodology. Report browser
QA remains unavailable under the existing local-file restriction.

## Declared follow-up after the three fixed continuations

All three stochastic continuations fail the20% global-chroma improvement screen.
The frozen mu global-color reader is better than final RGB; this does not establish
local-color accessibility or availability under sampled z. Before additional runs,
add one matched decoder-only512-update continuation using deterministic mu instead
of sampled z (same weights, fresh optimizer/seed/data, same frozen encoder). This
is an accessibility/sampling diagnostic, not a replacement VAE training objective.
Compare against the already completed sampled decoder-only arm; keep the20% color
screen and RGB nonregression guard unchanged. No new hyperparameter selection.

Add fixed local color readers at posterior input, mu, and sampled z:64 training
images x4 cells,16 validation and16 diagnostic test images x4 cells, ridge1e-3,
linear/RBF. Targets are corresponding4x4 RGB patch means. Report global and local
readouts separately; global averaging provides extra context and noise reduction.
This uses existing diagnostic populations; no claim of pristine generalization.

## Results and attribution

The observed issues are numerical output properties, not just the HTML view. The
original JPEG->RGB->LANCZOS shorter-side64->center crop preparation reproduces16
selected prepared arrays exactly. Model tensors use RGB/255 with no explicit
linear-light/gamma transform; renderer clips to[0,1] and displays RGB. Clipping
affects0.532% of reference values and changes MSE0.01714363 to0.01713224 (negligible
relative to color errors). A first independent resize audit used a floating scale
which rounded some short sides down to63; inspecting the archived preparation
revealed the documented minimum64 guard. Correct integer geometry reproduces the
inputs. This was an audit reconstruction error, not a prepared-data defect.

**Grid mechanism:**28 constant latent fields, including zero and means from27
encoded constant RGB inputs, decode without noise or encoder skips. Central
pre-shuffle spatial phase RMS is below2e-8; first shuffle output is0.15734 in
feature units, second shuffle0.16278, final RGB0.00718. Different channel widths
make these magnitudes unsuitable as a cumulative loss score; the informative
transition is spatially constant -> periodic at the shuffle. Learned expansion
produces unequal subpixel-channel values which rearrangement maps to 2/4-pixel
positions. Later processing does not fully remove that pattern. PixelShuffle is
still an exact permutation. This does not prove normalization or initialization
is the unique training cause. Pattern is present with zero sampling noise.
[Constant-field control](../runs/spatial_vae_color_v1/constant_control_fp32/report.html).

**Color bottleneck:** the frozen global mean-RGB linear reader error is about
9e-6 before the posterior,0.000700 at mu and0.003026 from final RGB. A more local
matched target (4x4 RGB patch mean) produces:

| Reference representation | Channels | Local linear reader MSE |
| --- | ---: | ---: |
| Before Gaussian projection |24|0.001091|
| Posterior mean mu |4|0.007396|
| One sampled latent z |4|0.014088|

Targets/sample positions are identical across representations:256 training points,
64 heldout points from16 diagnostic test images, fixed ridge. RBF sensitivity
checks also deteriorate. Global averaging adds context and reduces noise; it is
not directly comparable to the local reader. This identifies reduced practical
color access at the posterior projection and sampling, not mathematical proof
that four channels cannot encode color or that all color information is absent.

**Controlled continuation:**

| Training intervention | Mean RGB MSE | Global chroma MSE | Period4 residual RMS |
| --- | ---: | ---: | ---: |
| Reference | 0.017144 | 0.004909 | 0.010226 |
| Sampled training: decoder | 0.016446 | 0.004880 | 0.009659 |
| Sampled training: encoder | 0.016932 | 0.004886 | 0.010200 |
| Sampled training: all | 0.016215 | 0.004735 | 0.009727 |
| Mean-only decoder training | 0.013110 | 0.003063 | 0.010495 |

All continuations use512 updates, identical starting weights/data/sampler/seed,
fresh optimizers; component freezing verified bitwise. Extra ordinary training
reduces global color error by only0.5–3.5%, failing the20% screen. Mean-only decoder
training reduces it37.6% and total mean RGB error23.5% without changing the encoder.
Some useful color therefore remains in mu, but the sampled-training decoder does
not use it fully. Crucially, sampling through the mean-trained decoder yields
RGB MSE0.09944 versus reference0.01836: this is a failed stochastic-VAE replacement,
not a repair to adopt. None meets the50% grid reduction screen.

The supported working explanation is a combination: learned decoder phase
imbalance creates grids; color is weakly transported through the trained posterior
and noise bottleneck, with remaining color imperfectly used by the decoder. The
current objective favors a noise-robust but desaturated reconstruction in this
short fit. Exact attribution to beta, latent width, normalization or long-run
optimization still needs separate interventions; no universal MSE/architecture
claim is made. More ordinary training of this duration alone did not fix it.

## Verification, corrections and next experiments

45 relevant tests pass. 813 independent audit checks cover original data, export/checkpoint
equality, frozen components, matched sampling streams, source snapshots, NumPy
color/phase metrics, exact GPU output replay and report hashes. A precision mismatch
was found in initial standalone reference/follow-up evaluations (default GPU math
versus the training path's deterministic IEEE math; largest pixel differences
0.000235/0.000514). Entry points now initialize math settings consistently.
`reference_fp32` and `decoder_mean/evaluation_fp32` are authoritative; original
results remain stored. Four training fits/weights were not rerun or changed, and
the interpretation is unchanged. 18 reports and scientific panels are verified
structurally/visually; browser QA unavailable under the previous local-file policy.

Four training loops total21.98s; maximum recorded training reserve180MiB, including
allocator cache. Artifacts~196MiB, disk reserve retained. Actual Claude methodological
review reconciled; private code/data/results stayed local.

[Full diagnosis and tradeoff images](../runs/spatial_vae_color_v1/report.html).

Next candidates, not yet implemented/adopted: phase-symmetric initialization of
the learned subpixel groups, with fine-pattern controls so suppression does not
merely blur detail; and posterior/rate/color-objective comparisons that preserve
color under actual sampling. Keep the explicit rearrange/process/compress design.
Do not silently remove sampling or replace reconstruction with display saturation.

```bash
.venv/bin/python -m experiments.spatial_vae --color-diagnosis --weights runs/spatial_vae_v2/formal/C_beta0.1/training/weights.pt --output runs/my_color_diagnosis --device cuda
.venv/bin/python -m experiments.spatial_vae --color-followup --weights runs/spatial_vae_v2/formal/C_beta0.1/training/weights.pt --output runs/my_color_diagnosis --device cuda
```
