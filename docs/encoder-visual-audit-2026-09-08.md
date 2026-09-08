# Encoder visualization audit and hybrid architecture review

8 September 2026. The user asked why coarse PCA looks unusual, whether attention
entropy is correct/useful, and whether convolutional stages should include deeper
transformers. This slice audits the existing frozen artifacts and develops a
proposal. It does not train a new encoder or change the adopted latent contract.

## Bounded audit plan

- Preserve the original first-seed panels, arrays, checkpoints, and training runs.
  Reuse their six fixed test frames and 256 training PCA frames in all four arms.
- Independently reconstruct all four attention heads, check their probability
  normalization and projected output against the real attention operation, and
  compare normalized per-head entropy with the previously saved mean maps.
  Include per-head means, key counts, effective attended-key counts, and the
  relative output change under uniform attention. These are local diagnostic
  measurements, not trained ablations or control evaluations.
- Compare the original joint PCA with training-fitted separate-scale PCA and
  training-fitted PCA after subtracting each image's spatial mean. Report variance
  captured, spatial-versus-image-mean variance, and original color clipping.
  Centering removes image-level information deliberately; it is an inspection
  transform, never an adopted model change or a proof of semantic quality.
- Add essential checks against independent attention outputs, entropy boundary
  cases/head averaging, and synthetic known variance decomposition before running.
- Budget: four frozen encoders, 256 training frames and six already-selected test
  frames each, CPU only, zero optimizer updates; up to ten minutes including HTML.

## Figure contract

Question: how much of the unusual coarse map is projection/global variation,
and what do the actual heads attend to? Static scientific image panels compare
the same first three declared test frames using original joint PCA, separate
scale PCA, and image-centered PCA. RGB feature colors encode three projection
coordinates, not classes, and have training-fixed percentile bounds; independent
bases are not color-aligned. Attention heatmaps use a fixed 0–1 entropy scale;
selected-query probability maps identify direction, head, and query. Raw arrays
and exact metrics accompany the panel, which is embedded in the canonical offline
dashboard and visually checked. The other fixed views remain in the raw audit.

## Architecture and interpretation

Pending audit and public-only critical Claude consultation. The exact public
brief and replies are saved under
`runs/encoder_visual_audit_2026-09-08/collaboration/`. No implementation-source or
private-result export is part of this consultation.
