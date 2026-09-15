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
