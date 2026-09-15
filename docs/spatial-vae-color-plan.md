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
