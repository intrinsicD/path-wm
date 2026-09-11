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
