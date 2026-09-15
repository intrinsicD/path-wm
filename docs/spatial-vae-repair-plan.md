# Small VAE color/phase repair and loss-path cleanup

15 September 2026. User prefers an efficient solution and requests Claude review,
implementation, tests, duplicate-code reduction and compilation-time work.

Keep the current R/P/M/C model, spatial Gaussian sampling, parameter count and
latent-only decoder. First test training-only supervision; no larger architecture
or mean-only decoder substitution. Existing recipe and report renderer remain.

## Implementation and essential checks

1. Cache the KL tensor once per loss call (currently four identical evaluations).
   Verify old loss, all logged terms and gradients; preserve default behavior.
   Consolidate the duplicated reconstruction-panel/report call in the recipe.
2. Add opt-in color and phase residual objectives to the existing loss. Color is
   the average of global and 4x4-cell opponent-color squared error; pad partial
   cells by replication for odd geometry. Opponent axes are orthonormal RGB axes,
   not perceptual DeltaE. Weight6. Phase is squared DC-centered residual mean for
   each of16 phase classes modulo4, on the interior excluding8 pixels per border,
   trimmed at bottom/right to whole cells. Weight20. Use the full image if too
   small for that interior and disable phase measurement if smaller than4.
   Residual means prediction minus target, so correct checkerboard texture costs
   zero. This only targets coherent periodic residuals, not every local artifact.
   Log raw and weighted terms. Defaults zero; no extra inference computation.
3. Test numeric controls, old-value/gradient equivalence, a single KL evaluation,
   sampled gradient flow, export and exact paused/resumed new-objective training.
4. Benchmark eager loss and cold AOT graph capture/autograd compilation separately,
   using identical tensors, isolated processes, three runs per old/new variant.
   Existing training has no torch.compile; do not enable a whole-model compiler
   or promise shorter startup from fewer source lines. Count graph operations.

## Fixed experiment (declared before fitting)

Reference: `runs/spatial_vae_v2/formal/C_beta0.1/training/weights.pt`.
Use the same diagnostic train512/validation64/test96 source groups. Previously
viewed heldout images do not become a fresh confirmation set. Two continuation
seeds57301/57302, fresh AdamW3e-4/wd1e-4,512 updates,batch8,beta0.1,variance0.5.
Each seed runs a2x2 factorial: standard; color6; phase20; color6+phase20. All start
from identical saved weights; same data and posterior-noise stream within seed.
Joint encoder+decoder training with real posterior draws throughout.

Evaluate all arms before choosing: deterministic mean and three fixed posterior
draw streams56171/58171/60171, raw RGB/global chroma errors, KL bits/original pixel,
constant-latent grid RMS, unchanged synthetic pattern/edge diagnostics, native
96x128 and odd63x79 source crops. Save first-draw arrays and all draw metrics;
reuse frozen posterior tensors across additional draws. Equal updates are not
equal compute: report timing and memory for each fit and require <=25% overhead
for an efficiency pass. No compute-matched superiority claim from this design.

For each seed the standard continuation defines28 constant latent fields; decode those identical fields through every candidate for the grid screen. Also retain each model's endogenous constant-field diagnostic.

Per-seed screens against that seed's standard continuation: >=20% lower sampled
global chroma MSE averaged across the three draws; >=50% lower constant-field
phase RMS; <=5% mean and sampled raw RGB and pattern-edge regression; <=25% higher
KL rate and training time. Both seeds must pass all for a full repair/efficiency
screen. Report subcriteria independently; do not tune weights after seeing test.

Budget: eight512-update fits, <=120s each, <=2GiB reserved GPU, >=3GiB free disk,
<=450MiB total new artifacts. No deletions or replacement of earlier runs. If a
screen fails, preserve the candidate and report the remaining issue, not a repair.

## Claude method review

Actual Claude received public conceptual methodology only. Requested explicit
pooling/border/loss scales, individual seed results and compute accounting; these
are fixed above. Phase labels can permute under translation without changing the
aggregate phase penalty; the criterion does not certify that all artifacts or
real high-frequency details are treated correctly. Keep pattern and native checks.
Results, private source and data stay local. Second review reconciles these limits.

## Results

Eight fits completed. Identical122,979 parameters and latent geometry; all use
actual posterior sampling. Source weights remain unchanged. The simple color
term alone is the useful candidate from this diagnostic, not a full repair or
pristine heldout selection. The phase term does not add a consistent benefit.

| Seed | Arm | Sampled RGB MSE (3 draws) | Global chroma MSE | Fixed-field grid RMS | KL bits/pixel | Training s |
|---|---|---:|---:|---:|---:|---:|
|57301|standard|0.017512|0.005048|0.004720|0.138119|7.366|
|57301|color|0.012767|0.000850|0.004373|0.194398|7.973|
|57301|phase|0.017574|0.005063|0.003827|0.138318|8.014|
|57301|both|0.012828|0.000854|0.004465|0.194821|7.930|
|57302|standard|0.017203|0.004617|0.006007|0.140853|7.256|
|57302|color|0.012774|0.000856|0.006035|0.193052|7.956|
|57302|phase|0.017307|0.004733|0.004954|0.139550|7.953|
|57302|both|0.012783|0.000889|0.003647|0.194002|7.928|

Color alone reduces sampled global chroma error81.5–83.2%, pixel chroma error
55.7–58.9%, and sampled raw RGB error25.7–27.1% relative to each matched ordinary
continuation. Mean raw MSE is0.01140/0.01144. Native mean raw MSE is0.00919/0.00917;
odd geometry0.00843/0.00847. Saved photos visibly recover more color, but remain
blurred, some saturated local colors remain wrong, and grids persist. The raw
pattern-edge nonregression screen passes; this is not proof of full detail.

Color requires36.9–40.7% greater achieved KL, failing the predefined25% rate guard.
This is a rate-distortion tradeoff, not an increase in latent tensor size, model
parameters or inference computation. No entropy coder is present. Training is
8.2–9.7% slower, passing the25% runtime screen. Neither phase-only nor combined
training reaches50% fixed-field grid reduction in both seeds. All composite screens
therefore fail. Keep original and candidate weights; no automatic global default
or full-quality claim. Next research question is a decoder phase repair that
preserves the sampled-color gain, followed by achieved-rate matched comparisons.

The proposed mechanism is narrowed, not proven unique: the same architecture can
learn substantially better sampled colors with a targeted objective. The current
four-channel geometry is not a hard barrier to this measured improvement. Whether
better rate allocation, initialization or decoder structure closes the remaining
gap still needs controlled comparisons.

## Code efficiency and verification

KL is now evaluated once per loss call instead of four times. Loss, metrics and
gradients retain the baseline contract. Training and evaluation share one RGB
report path; color diagnostics and supervision share opponent coordinates. Extra
posterior draws reuse encoded tensors, and a test compares them exactly to three
independent full evaluations without consuming training RNG.

Three isolated-process CPU measurements per loss variant: median eager
forward/backward0.440ms ->0.325ms (26.1% lower). Captured graph64 ->32 nodes,
KL exp calls4 ->1. Median first AOT-eager call including autograd compilation,
after graph inspection/reset,0.910s ->0.822s (9.7% lower); import/startup excluded.
Warm AOT-eager1.319ms ->0.975ms, still slower than eager. This is not an Inductor
or whole-model compilation measurement; default execution remains eager. No
project-wide compilation speedup is claimed.

50 relevant tests pass, including exact new-objective pause/resume and legacy
loss gradients. Independent checks cover eight exports/source snapshots, fixed
budgets, matched sampler/CUDA-noise streams, NumPy color/rate/phase metrics, every
logged objective, gates and exact GPU mean/three-draw replay. The audit initially
loaded one export onto CUDA while comparing CPU snapshots; corrected the audit
loader only. Training and results were unaffected. Reports are structurally
verified and PNG examples inspected; browser QA remains unavailable under the
existing local-file restriction. Reused renderer unchanged.

Total training62.38s; peak GPU reserve254MiB includes allocator cache. Artifacts
about300MiB within450MiB budget. Actual Claude reviews and reconciled limitations
are saved in `runs/reviews/spatial_vae_repair_v1/`. Review API-equivalent cost is
not a subscription bill; private source, data and results stayed local.

[Report and examples](../runs/spatial_vae_repair_v1/report.html),
[raw performance](../runs/spatial_vae_repair_v1/performance.json),
[verification](../runs/spatial_vae_repair_v1/verification.json).

Reproduce the fixed comparison in a new directory:

```bash
.venv/bin/python -m experiments.spatial_vae --color-repair --weights runs/spatial_vae_v2/formal/C_beta0.1/training/weights.pt --output runs/my_color_repair --device cuda
```

Use a saved candidate without any extra inference module:

```python
from pathwm.models.spatial_vae import SpatialVAE
model = SpatialVAE.load("runs/spatial_vae_repair_v1/formal/seed57301/color/training/weights.pt", "cuda")
# Same model API and spatial latent contract as the reference.
```
