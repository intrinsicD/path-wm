# Native decoder access to the existing feature pyramid

25 September 2026. Alex asks to pass the multiscale feature set **also** to the
actual decoder for reconstruction, then explicitly identifies decoder retraining
as necessary. Codex and actual Claude Opus 5.5 medium continue as partners, with
the same permissions. This authorizes this bounded connection/training comparison,
not a replacement encoder, codec, memory representation or full-agent redesign.

## Concrete connection and comparison

The actual native checkpoint is `runs/real_visual_joint_repair_3501_u6000_v1`:
RGB64, width64, seven slots, three slot iterations, decoder hidden32. It exports
fine16×16×64 and coarse8×8×64 features. These already feed slot attention but do
not reach the existing broadcast decoder. The complete information-preserving
filter-bank design in `multiscale-modality-design.md` is not thereby implemented.
This test uses the actual implemented pyramid, without claiming invertibility.

Add optional pointwise projections of coarse features at the decoder's8×8 stage
and fine features at its16×16 stage. Preserve the entire finest grid: no pooling
it down to eight. Start both projections at zero, preserving the pretrained
output and existing weights. Enabled decoding requires matching pyramid input;
there is no silent fallback to slots-only recall. Default model state and behavior
remain unchanged. Raw pixels and hidden labels never enter the decoder.

Freeze encoder, slot attention and semantic heads. Train the decoder, including
its new connections in the pyramid arm. Compare with the same parent's slots-only
decoder retrained on identical images, objectives and update counts. Record the
added parameters; this is a comparison of a concrete added connection, not proof
of an information-only benefit independent of capacity. Claude's initial suggestion
of a different reconstruction head is not selected: Alex requested the actual
decoder with additional feature access.

## Registered protocol — before implementation or training

- Two training seeds:3601 and3604. Two arms each:slots andpyramid.
- Existing procedural TRAIN texture generator, randomization1.0, actual full
  configuration, batch32,2,000updates, AdamW lr3e-4, last checkpoint only.
- Decoder objective: RGB MSE plus0.5matched mask cross-entropy. Frozen semantic
  outputs may be reported but add no decoder gradient. Same objective in both arms.
- Maximum10minutes/arm and6GiB reserved GPU memory. Incomplete runs do not pass.
- Evaluate256fixed TRAIN-kind scenes(seed3602) and256validation-kind scenes(seed3603).
  All are procedural development data. Preserve scene hashes, per-image metrics,
  parent/checkpoint/source identities and illustrative reconstructions.
- Primary metric:full-image MSE. Also report PSNR, machine-body MSE and adjacent
  pixel gradient error with exact formulas. No threshold or checkpoint selection
  from evaluation, no automatic longer fit.
- Success:at least20%lower MSE than the equally retrained slots-only arm on BOTH
  populations for BOTH seeds; source remains frozen, numerical values finite,
  software/resume checks pass. Report failures without changing this gate.
- Feature-use diagnostics:zeroed and shuffled-across-image pyramid values with
  slots fixed. Label these as interventions, not valid operational inference.
- Report pretrained output as reference. Preserve all outcomes using the existing
  report renderer; structural verification is required, no new renderer.

## Verification and scope

Before quality runs: native full-config checks for default compatibility, zero
initialization/output equality, frozen source gradients and hashes, source shape/
mask/batch validation, no feature mutation, real feature sensitivity after learning,
and exact save/resume. Use actual checkpoint and existing data. Freeze source for
all comparisons and the required full CPU suite. A smaller number of software
check steps is not a reduced architecture or a learned-quality result.

This tests reconstruction while the source features are available. It does not
establish compact memory fidelity: current memory stores slots, and pyramid
recovery requires retained frame evidence and re-encoding. No change to the default
identity/memory path or adoption of the experimental decoder is implied.

## Current state

Actual Claude's read-only plan is preserved in
`runs/reviews/native_pyramid_decoder_20260925/claude-plan.md`. Implementation is in
progress after Alex's decoder-retraining follow-up and the concrete native wiring
plan communicated in conversation. No new reconstruction result yet.
