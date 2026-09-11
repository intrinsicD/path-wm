# Trainable text-encoder initialization for the agent reader

Alex authorized continuation, Claude review, implementation, tests, fixes and
iteration on 11 September 2026. Reuse `--dataset facts --fact-reader event` and the
existing trainer, cached final evaluation and report. Add one explicit checkpoint
argument, `--fact-encoder-weights`; omission preserves random initialization.

## Declared comparison

Transfer only `agent.encoder` from `runs/fact_grounding_v1/reference/last.pt`
(SHA256 `eec43cc215980aff9bd31023c39bfc41364f0d4ec79244972728318fc2305508`)
into the event model's shared text encoder. Keep every encoder parameter trainable.
Construct the ordinary seed-23 model first so all non-encoder initial tensors and
post-construction RNG match the cold recipient. Never transfer donor heads, optimizer,
sampler, RNG or session state. This encoder also reads the constant task instruction;
effects cannot be attributed specifically to observed-entity feature quality.

The donor completed 512 updates / 8192 fact presentations on the same 96 training
pairs, at seed23, width32 and lr0.0003. It passed its fixed final-checkpoint gates.
No held-out pair entered its gradient training. The 32 development combinations
have already been inspected, so recipient evaluation is reused development evidence,
not an independent final test. Count the donor's additional exposure separately;
this comparison cannot establish better total-compute efficiency.

Recipient: seed23, width32, batch16, 512 updates / 8192 presentations; lr0.0003,
AdamW weight decay0.01, clip1, FP32/two CPU threads. Same event, task interpreter,
two thinking rounds, final attention heads, equal-weight entity/location CE, data,
fixed batch evaluation seeds, and training-only probes every32 updates. Final
checkpoint only; no best-checkpoint selection, calibration or final test.

Compare with the saved cold recipient `runs/event_fact_v1/reference`. If extraction
fails without a software defect, permit one recipient from the same initialization
changing only lr to0.001, compared with the matching saved `event_fact_v1/lr_control`.
Each cold/warm comparison changes encoder initialization only. Both cold outcomes
stay visible; do not select a new cold baseline after inspecting warm results.

Unchanged gates: train entity/location each>=95%, joint>=90%, mean NLL<=0.35;
development each>=90%, joint>=80%, mean NLL<=0.5. Only after extraction passes,
evaluate the existing explicit cached-logit selector with both-held-out query>=90%
and paired>=80% gates. It does not establish learned query binding or later retention.
At most450 active seconds per recipient,900 across at most two recipients, using
the existing cumulative cooperative timer. No third run, extra seeds, encoder
freezing, new objective or gate relaxation. Tests and report QA are separate.

## Implementation and verification

Use the existing `load_component` helper in the model factory after ordinary
construction. Accept weights only for the event fact reader. Bind the resolved
donor path, file hash, component name and initial encoder tensor hash to run
settings; check that the file did not change during loading. Resume must verify
the donor again and restore the full recipient checkpoint, rejecting changes at
the same path before altering stored progress. Check and report show initialization.

Red checks: exact transferred tensors; every other recipient tensor and construction
RNG unchanged; encoder remains trainable and updates; no donor optimizer/heads;
exact paused/full recipient replay, final cache reuse, and changed-donor rejection.
Run the full CPU suite and actual CLI check. Preserve source/checkpoint/raw predictions
and validate reports structurally and in the browser. Audit shared initialization,
core settings, data identities and sampler state against the preserved cold runs.

Two compact abstract Claude exchanges are saved under
`runs/reviews/continuation_2026-09-11/warm-*`. Accept the instruction-encoding and
upstream-exposure caveats. Prior held-out evaluation makes this adaptive development,
not automatic gradient-training leakage. No causal bottleneck claim is intended.
Claude acknowledged that distinction. Its final caution about resume checks not
restoring statistical independence is already part of this contract: resume checks
establish execution integrity only; no independent-test claim is made.

## Outcome, 11 September 2026

Completed the declared two-recipient comparison. The red plan/tests were committed
as `29f27c7`; implementation and the tested run source are `4c3aebf`. All 88 CPU
tests pass, including 9 focused fact checks. The actual warm CLI forward/backward
check is finite and reaches the encoder, agent reader and both output heads.
Transfer tests verify fresh optimizer/heads, encoder updates, exact paused/full
replay, cache reuse and rejection of a changed donor without altering saved progress.

The actual donor audit matches the saved cold initialization exactly. It changes
80 active encoder tensors (160 including the inert target copy), preserves the
other 1392 tensors and consumes no additional construction RNG. Initial warm model
SHA256 is `b837d0eb4ccac7707c5ce56cb6a55d6fa975eb4039e520d0ecb670cfcaed84d3`;
initial encoder SHA256 is
`93f955cdef171ca400a163a6302586da61a5b6c456b3e2e932f76e9beb000248`.
Both recipients retain the same data, core settings and final sampler state as
their corresponding saved cold run. The original donor file is unchanged.

Each row below is the fixed final checkpoint on the same 32 development combinations.

| Learning rate | Initialization | Entity correct | Location correct | Joint correct | Mean NLL |
| --- | --- | ---: | ---: | ---: | ---: |
| 0.0003 | Saved cold reference | 1/32 | 8/32 | 0/32 | 2.431158 |
| 0.0003 | Trainable transferred encoder | 0/32 | 31/32 | 0/32 | 2.063643 |
| 0.001 | Saved cold comparison | 0/32 | 24/32 | 0/32 | 2.171213 |
| 0.001 | Trainable transferred encoder | 0/32 | 32/32 | 0/32 | 2.393122 |

Warm training entity/joint accuracy is 3/96 in both recipients. Training location
accuracy is 95/96 at lr0.0003 and 96/96 at lr0.001; mean training NLL is 1.860383
and 1.648691 respectively. Both training-fit and development gates fail. The higher
learning rate improves location accuracy but worsens development entity NLL
(3.840849 to 4.774020), so it does not rescue extraction. Binding evaluation is
skipped in both runs under the unchanged gate.

The reference used 51.9130 active CPU seconds; the conditional comparison used
51.5911, totaling 103.5041 seconds. Each completed 512 updates / 8192 recipient
presentations. The reused donor contributed another 512 updates / 8192 presentations
and 10.3502 active seconds upstream. That is additional exposure for each warm
lineage, not new training repeated in this slice. No cold run was repeated and no
third recipient was started.

Both [reference](../runs/warm_encoder_v1/reference/report.html) and
[learning-rate comparison](../runs/warm_encoder_v1/lr_control/report.html) passed
structural and 1280x720 browser checks. Curves, metrics and expandable donor identity
and examples render without broken images or horizontal overflow. Completed CLI
resume leaves prediction, result and metric files byte-identical. Independent raw
logit calculations reproduce entity/location/joint accuracy and NLL, and all saved
checkpoint floats are finite. `runs/warm_encoder_v1/verification.json` binds these
checks to source, settings, donor identity, tests, snapshots and screenshots.

This initialization intervention improves location extraction in these two runs
but leaves entity extraction unsuccessful, including on the training facts. It
does not isolate a failing component: the encoder also processes the instruction
and continues changing during training, while categorical evaluation uses one fixed
realization. Previously inspected development pairs remain disjoint from gradient
training, but these adaptive results establish no independent generalization claim.

Next proposed diagnostic: hold the successful donor encoder fixed in one otherwise
matched event-reader run to test whether preserving its features changes entity
learning. That would test a different training condition; it would not by itself
establish forgetting or memory detachment as the cause. Keep architecture, objective
and existing storage policy fixed until a bounded comparison supports a change.
