# Read-only decoder input reliance

Within the authorized overnight architecture investigation, distinguish offering
multiple feature sources from a fitted decoder actually using them. No optimizer
updates or checkpoint selection. Evaluate the fixed D2 endpoints for all four
arms and three seeds on the same512COCO test crops. Preserve all raw per-image
RGB errors, foreground metrics, fixed image panels and checkpoint fingerprints.

Use one fixed derangement of the512rows, generated with NumPy seed20260911 by
rejecting permutations with fixed points. Targets and displayed images remain
in their original order. The conditions are: normal; replace only input slot1
with its donor image's slot1; replace both final-token and derived-coarse slots
with the same donor image's paired context. This retains within-donor spatial
and fine/coarse consistency. For late, slot1 is another final-token projection,
so it is not a distinct local sensor. All models share the same permutation.

For the conditioned arm additionally replace the FiLM context with zero or flip
its sign while retaining the requested RGB/mask output layer. This tests the
learned use of task-dependent shared computation. It does not swap the final
output type. Scoped hooks must restore on success or exception; no model buffer,
parameter or cached source tensor may be mutated.

All perturbations are evaluation-time distribution changes. A degradation shows
dependence of the fitted readout, not indispensability of a branch for a separately
trained architecture. A small effect can reflect redundancy or weak use, not
absence of information. No invented pass gate. Report paired absolute/relative
changes separately for RGB and masks, and all three seeds. No future prediction,
encoder conditioning, queried-instance segmentation or control claim.

First test donor-slot routing and context-hook cleanup/output-type preservation,
including an informative failure. Run a16-image CPU development diagnostic on
the first available D2 conditioned endpoint, reconcile normal outputs against
the saved GPU endpoint on those exact rows, produce verified HTML, commit and
seal before the full read-only diagnostic. CPU FP32/TF32off, four threads; require
baseline RGB MSE and foreground IoU agreement within1e-5 (probabilities themselves
are checked on saved panels where available). Keep any threshold disagreement
visible rather than silently changing masks. Cap the complete diagnostic at
20minutes; preserve completed conditions if the time budget is reached.

Render fixed first-six COCO RGB and mask comparisons from the saved D2 panels,
without selecting examples based on how an architecture looks. Those panels
show output differences, while the intervention table addresses input reliance.

## Development completion

The conditioned seed9107 endpoint's five16-image CPU development conditions
completed in0.671seconds of diagnostic execution (42.2seconds with setup and
canonical reporting). Baseline foreground IoU was identical to saved GPU results;
RGB MSE differed by9.31e-10. Maximum saved-panel differences were1.79e-6 RGB and
5.01e-6 mask probability, below the declared check scale. Routing/context-hook
tests pass and HTML verification passed. These small intervention outcomes are
development evidence only. Commit this slice, freeze source/data/checkpoint
identities, and retain the20minute full-evaluation cap before the42conditions.
