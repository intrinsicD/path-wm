# Overnight architecture report plan

Audience: technical researcher; decision: select the next encoder/decoder package
from measured local comparisons, including results that contradict adding depth
or conditioning. Primary surface: self-contained portable HTML through the
existing canonical report builder and local browser adapter. This follows the
project's standing offline-report workflow. No external publication. Supporting
Markdown, numerical snapshot, code/config identities and figures stay local.

The report must be complete before9September07:00Europe/Berlin. Final architecture
recommendations wait for the declared decoder comparisons; earlier snapshots
must clearly say partial. Albedo is excluded. Do not imply that pretrained ViT
package comparisons isolate architecture, that pooled category AP measures all
semantics, or that perception gates establish world-model/control quality.

## Reading path and specification mapping

1. Title and technical summary: concrete architecture recommendation, measured
   gains/tradeoffs and scope. These fulfill title/summary roles.
2. Scope and metric definitions before numerical comparisons: source populations,
   seeds, q versus per-case tolerance, RGB range, foreground union masks versus
   object-query masks, category AP and crowd unknowns. Move this ahead of findings
   because the thresholds and denominators are needed to interpret them.
3. Frozen representations and typed readouts: all paired seed values, generic
   reconstruction/segmentation/category accessibility, and fresh-pose error tails.
4. Extra encoder processing: joint continuation versus added convolution and
   transformers; retained generic outputs, fresh results, gradients, gates and
   compute. Keep exact matched controls and adaptive selection explicit.
5. Local decoder evidence and task FiLM: fixed endpoints, four-arm paired results,
   pixel/mask tradeoff, fewer raw-slot parameters and decoder timing scope.
6. Internal representations: train-fitted PCA, corrected per-head entropy and
   bilateral/directional reliance interventions. State the limits of visual and
   post-training ablation interpretations next to the evidence.
   Also report the adaptive location-distribution follow-up: paired frozen P1
   reference, unchanged pose architecture, exact map-target expectations, loss
   scales, map manipulation checks, physical errors and boundary coverage.
7. Model specification and proposed integration: named spatial/depth features,
   typed consumers and conditioning, plus causal future-state requirements.
   Distinguish tested prototypes from untrained world-model integrations.
8. Limitations and robustness: reused grouped holdouts, one source checkpoint,
   three head/module seeds, decoder capacity/pretraining differences, calibration,
   paired sampling/source checks, losses/optimization and no control evidence.
9. Recommended next steps and further questions: derive priorities from results;
   identify the exact new data/tests needed for geometry tails, prediction and
   eventual software/computer tasks. Keep additional levels, encoder conditioning,
   memory/register changes and queried-object segmentation explicitly untested.

These sections implement the technical-report specification's finding,
methodology, validation, limitations, next-step and further-question roles.
Every quantitative visual needs adjacent interpretation. Full exact values and
source identities remain available, with provenance in the shared source drawer.

## Chart contract and QA

Use native grouped bars for three paired seed comparisons, one metric and unit
per chart. No misleading confidence intervals from three fixed-source head seeds.
Use exact tables for architecture/data mappings and detailed per-seed values.
Use training trajectories only to diagnose convergence/budget, preserving sampled
recorded updates. Reuse inspected scientific PNGs for RGB/mask/PCA/output maps;
no visual similarity or attention concentration is a model-quality score.
For descriptive budget diagnostics, compare the mean of validation updates
3100–3500 with3600–4000, five points per non-overlapping window. Report q/RGB
ratios and absolute IoU changes without significance or convergence thresholds.
This is post-hoc descriptive analysis and changes no trained-model selector.

Recompute saved predictions/targets into physical metrics, mask probabilities
into published per-image metrics where available, AP from raw scores/unknown
masks, selected steps from validation ledgers, exposure counts and paired sample
streams. Preserve failed publication attempts and stopped fits separately.
Require canonical artifact validation, exact embedded snapshot, desktop/narrow
browser/source interaction checks and actual inspection of scientific figures.
The existing reader supplies system light/dark appearance. Do not replace the
shared renderer, relax its size limit, or silently drop exact evidence.

The geometry follow-up also needs before/after location-map panels. Reuse the
already declared first-seed case indices in `figures/fresh/summary.json`: four fixed
cases and the two original package-specific worst cases. Do not select cases from
the new objective's outcomes. For CNN and ViT separately, show the same observed
scene, old/new poses and old/new pusher/body distributions on common log-probability
scales. Assert identical case identities/targets and normalized maps before
plotting. This is a read-only visualization from saved arrays; it changes no model,
metric or selection rule. Inspect the figures and verify HTML after all six fits.
