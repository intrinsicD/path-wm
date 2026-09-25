# Native decoder access to the existing feature pyramid

25 September 2026. Alex asks to pass the multiscale feature set **also** to the
actual decoder for reconstruction, then explicitly identifies decoder retraining
as necessary. Codex and actual Claude Opus 5.5 medium continue as partners, with
the same permissions. This authorizes this bounded connection/training comparison,
not a replacement encoder, codec, memory representation or full-agent redesign.

Follow-up: [Actual-encoder retention diagnosis](reconstruction-diagnosis-plan.md)
recovers the tested images from the existing fine features; the learned decoder
remains the target for diagnosis. This does not alter the negative result below.

## Outcome

The experiment is valid and complete, but **fails the registered quality target**.
All four runs completed2,000updates with frozen encoder/slots/heads and identical
matched inputs. Adding the pyramid reduces MSE by only1.09–1.71%, below20% in
every comparison. No threshold, schedule or checkpoint was changed after results.

| Training seed | Evaluation | Retrained slots MSE | Retrained pyramid MSE | Reduction |
| --- | --- | ---: | ---: | ---: |
| 3601 | TRAIN kinds | 0.0075951073 | 0.0074979726 | 1.279% |
| 3601 | Validation kinds | 0.0088383982 | 0.0086871480 | 1.711% |
| 3604 | TRAIN kinds | 0.0075882750 | 0.0075054044 | 1.092% |
| 3604 | Validation kinds | 0.0088211629 | 0.0087115620 | 1.242% |

Each evaluation has256images. The pretrained parent's MSE is0.0083487/0.0094587
on the same TRAIN/validation populations, so decoder retraining itself improves
both arms. Against the equally retrained control, additional feature access gives
only the small gains above. Zeroing or shuffling features worsens both pyramid
models: validation MSE becomes0.0100351/0.0093991 for3601, and
0.0099443/0.0094397 for3604 (zero/shuffled respectively). Thus the connections
affect reconstruction, but this connection and training budget do not meet the
quality target. This does not identify the limiting stage, prove convergence, or
show that useful detail is absent from the pyramid.

[Comparison report](../runs/native_pyramid_decoder_comparison_v1/report.html),
[commands](experiments.md) and raw run files preserve all outcomes. The optional
connection remains an explicit experimental path; it is not promoted as the
default decoder or as a validated fine-detail memory solution. No further fits were run.

966 full CPU tests pass, zero failures/errors/skips,716.36s with source unchanged.
Independent audits verify101source files and74frozen tensors per arm, per-image
metric means,32saved-example pixel errors, identical training hashes and exact
state resume. Every arm's report and the aggregate are structurally verified using
the existing renderer; no browser QA is claimed. GPU runs took172.8–179.8s of
recorded recipe time and used at most about1.8GiB reserved, while the CPU suite ran
concurrently; this is not a speed comparison.

Machine-body error includes visible panels and lamp pixels (entity1/2), not
texture alone. Gradient error uses full-frame adjacent differences. The larger
parameter set also changes the effect of the shared gradient-norm clipping rule;
this is a concrete connection comparison, not a capacity-independent causal claim.

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

## Implementation record (before measured outcome)

Actual Claude's read-only plan is preserved in
`runs/reviews/native_pyramid_decoder_20260925/claude-plan.md`. Implementation is in
progress after Alex's decoder-retraining follow-up and the concrete native wiring
plan communicated in conversation. No new reconstruction result yet.

### Software verification before registered training

71 focused tests pass on unchanged source (zero skips). Full native GPU two-update
checks establish initial output equality across arms, identical training batches,
frozen components and exact pause/resume of model, optimizer, all RNG streams and
sampler. A full default perception run from the real parent also completes,
covering the earlier downscaled default smoke. Source snapshots, checkpoints,
reports and command receipts are preserved.

One resume invocation omitted the required `--stage` flag and exited before
loading; the corrected command passes. An initially overbroad comparison of first
training rows differed solely in gradient norm: the additional parameters add
gradient components. Forward metrics and input batches are exactly equal. This is
recorded in `software-audit.json`, not hidden by dropping the measured norms.

Final decoder counts are47,556 versus53,796 (+6,240). These reconstruction runs
contain perception only (219,928 versus226,168 parameters); the parent's separate
identity-key head is not a reconstruction consumer and is excluded from these
counts. Both arms use the actual trained parent perception, not a substitute model.

The registered four training runs and full CPU suite subsequently completed on
that frozen source; their outcome is recorded above.

### Existing identity path under current source

The optional connection changes the source hash even though the default decoder
path is unchanged. Old binding manifests remain historical; their guards are not
weakened. Fresh `native_identity_confirmation_after_decoder_v1` uses the same
parent, TRAIN3405/64, fixed match.90 and both original margins. Its policy rows
equal the original16-policy selection's .90 rows, and its raw scores equal the
previous final-source confirmation exactly. Normal four-scene R2 life under the
new manifest has identical `life.json` and raw metric bytes to the previous life.
All101 source files are sealed for each new run. This is compatibility evidence,
not a rerun of the six identity-quality populations. Current working manifest:
`runs/native_identity_confirmation_after_decoder_v1/binding.json`.

### Independent final review

Actual Claude Opus5.5 medium independently recomputed the per-image means, relative
gains, input hashes and default-path compatibility; it agrees with valid=true and
quality_gate=false. The five collaboration responses and reviews are preserved in
`runs/reviews/native_pyramid_decoder_20260925/`.

On individual images the pyramid arm has lower MSE on188/256 and225/256 images
for seed3601 (TRAIN/validation), and200/256 and235/256 for3604. These are
descriptive diagnostics, not additional gates. Shuffled features perform better
than zero features, but this alone does not identify frame-independent learned
content: zeroing also changes feature statistics. The +6,240 parameters confound
a pure information-access claim; their effect on optimization is not assumed
to favor either arm. The report's comparison images show seed3601 only.

No proposed follow-up probe, different connection, longer fit or joint encoder
training was run. The remaining cause of the small gain is unresolved.
