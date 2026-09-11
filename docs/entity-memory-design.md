# Entity discovery, association and persistent belief

Proposed design, discussed with actual Claude on 11 September 2026. This is not
implemented, and it does not replace the current bounded latent-memory model.
The user accepted diagnosing the existing entity reader before a controlled
identity-binding comparison; a full entity-store architecture remains a proposal.

## The missing connection

Do not train a classifier to predict arbitrary database keys from sensory input.
Learn observation-conditioned entity candidates, then infer which previous entity,
if any, each candidate corresponds to. A database key names the resulting record;
it is assigned by storage and carries no perceptual meaning. A finite type vocabulary
does not imply a fixed vocabulary of individual instances.

| Component | Role |
| --- | --- |
| Candidate extraction | Produce features for possible entities from observations; candidates may be incomplete, duplicated or absent. |
| Candidate retrieval | Use recognition features and context to retrieve a bounded set of previous records. |
| Association | Score existing matches, new entities and unresolved cases using current evidence and history. |
| Belief update | Update the corresponding learned state from attributed evidence; preserve uncertainty and time. |
| Persistence | Keep a stable local key, recognition evidence, versioned latent belief and source references. |

Recognition and changing state have different jobs. They need not use disjoint
networks or fixed slow/fast vector partitions. Recognition features themselves can
evolve. A reader trained for a particular question extracts an answer from the
retrieved belief; a stored latent does not automatically support arbitrary questions.
Known observation timestamps can be recorded directly. Preserve observations
separately from inferred beliefs and version latent states against compatible models.

## Avoiding identity mistakes

Allocator collisions are duplicate database keys and can be controlled through
ordinary unique-key allocation. False associations are different: the system can
use a perfectly unique key for the wrong person or object.

Do not force every candidate into its nearest existing record. Compare forced
nearest-neighbor matching, confidence-gated matching with abstention, and bounded
multiple hypotheses. Soft scores are not inherently safer than hard decisions with
abstention. Avoid writing an ambiguous mixture into several canonical entity states;
retain bounded provisional alternatives and source evidence instead. Even confident
decisions can be wrong, so define correction and replay behavior for affected beliefs.
Record merges must not silently destroy the original attribution history.

Appearance alone may not distinguish two similar objects. Motion, temporal context
and other observations can help, but identical observable histories can remain
unresolved. No proposal guarantees error-free semantic identity. A fixed granularity
is sufficient for the first task; nested part/whole entities are an extension to test,
not a mandatory hierarchy or universal definition of objecthood.

## Smallest useful comparison

First finish the accepted frozen-model readout diagnosis. Then isolate association
and belief updates with controlled proposals for two simulated objects: movement,
property changes, crossings, occlusion, disappearance/re-entry and similar-looking
instances. Ground-truth correspondence is a training target, never an observation
feature or preassigned global identity class. Randomize proposal order and use novel
episodes/instances at evaluation to expose index and appearance shortcuts.

Hold observations, feature access and candidate budgets fixed across association
policies. Measure false merges, duplicate tracks, identity switches, false-new and
missed-entity rates, retrieval recall at the candidate limit, association coverage,
and whether an update changes the correct object's attributes while preserving the
other object's state. Include deliberately ambiguous cases. Abstaining on everything
must not count as success. Freeze exact splits, thresholds and compute caps before
running; none are declared or launched by this design note.

This first comparison assumes candidate proposals and therefore does not establish
discovery. Subsequently replace them with learned proposals and separately measure
localization/segmentation and missed/duplicate proposals. Simulator masks, locations
and track correspondences can supply initial supervision without requiring masks
as inference inputs. Reconstruction or temporal consistency alone does not establish
the desired persistent identity or task granularity. Start with an in-memory mapping;
database engineering is unnecessary for this comparison.

## Review and grounding

Two isolated public-only Claude exchanges are preserved under
`runs/reviews/continuation_2026-09-11/entity-association*`. No private source, data
or results were exported. Claude accepted corrections to mandatory slow/fast
partitions and hierarchy, retrieval against mutable beliefs, conflation of type
ontology with instance vocabulary, and equating soft assignment with safety. The
reconciliation also adds coverage and retrieval metrics and separates association
tests from discovery tests. No remaining review disagreement was reported.

[Slot Attention](https://arxiv.org/abs/2006.15055) supplies evidence for learning
exchangeable, object-centric candidates from perceptual features in studied tasks;
a slot position is not automatically a persistent identity.
[SAVi](https://arxiv.org/abs/2111.12594) studies sequential object-centric learning
using optical-flow targets and initial object-location cues on synthetic video.
These are relevant starting points, not evidence of universal entity discovery or
a ready-made persistent entity database. Peer agreement is not empirical validation.

## Runtime and learned-model boundary

A subsequent user-requested review covers how to connect the proposed graph to the
existing memory hierarchy and which payloads to retain. Four compact actual Claude
exchanges (`graph-interface*` and `entity-payload*` in the same review directory)
reconcile the following contract. This remains design work, not implementation.

The library would provide a small caller-owned entity store alongside existing
session memory. Runtime code owns key allocation, versioning, capacity limits and
transaction execution. Trainable modules produce recognition queries, association
scores and belief/link proposals. The experiment recipe owns construction and
training; no separate trainer or graph framework is needed.

| Proposed operation | Contract |
| --- | --- |
| Read | Query a snapshot with explicit candidate/neighbor limits; return tensors, masks, opaque handles, source kinds and provenance. No mutation. |
| Propose | Learned association, new/unresolved scores and candidate belief/link changes. No committed side effects. |
| Commit observed event | Validate expected version and event identity; apply attributed updates once. Real observations only. |
| Revise belief | Explicitly revise an inference using cited evidence, including old evidence. Do not fabricate a new observation. |
| Snapshot / branch | Isolate planning from live state. Speculative branches cannot commit observational history. |

Exact recent evidence, compressed history and entity beliefs may share a transport
envelope while retaining different provenance, temporal masks and reliability rules.
They are not interchangeable evidence. Capacity pressure must not force false entity
merges; reject/defer or evict under a declared policy. Bound tentative hypotheses,
retrieved records and adjacency expansion as well as active tensor storage.

Training uses the same forward operations. Stored snapshots are detached: ordinary
backpropagation trains their consumers, not the historical writer automatically.
The first comparison can supervise association/retrieval scores using correspondence
targets. To train earlier writers from later outcomes, explicitly recompute a bounded
observed sequence under current weights or retain a declared bounded autograd history.
Hard candidate lookup and ID allocation do not provide gradients to query keys by
themselves. Policy gradients are not a prerequisite. Evaluate predicted retrieval and
assignments; identical APIs do not remove teacher-routing distribution shift.

Checkpoint compatibility includes model/latent/schema versions, allocator state,
tentative records, retrieval configuration and RNG where used. Essential checks are
idempotent commits, branch isolation, capacity behavior, provenance round trips,
declared gradient boundaries and exact compatible resume.

## Entity payloads and historical retrieval

| Representation | Useful for | Limitation |
| --- | --- | --- |
| Original text/image/audio or a retained reference | Re-examination and source-grounded questions | Storage and re-encoding cost; may be evicted. |
| Versioned recognition features and latent belief | Fast candidate search and learned readout | Model compatibility and stale beliefs; cannot recover missing source details automatically. |
| Optional supported facts, relations or summaries | Explicit queries and inspectable links | Derived summaries are lossy; conflicts and derivations need provenance. |

Do not require every payload kind for every entity. A first prototype can combine
selected evidence references with a versioned latent cache. A source-only design is
also possible at higher encoding cost. A directly recorded measurement or exact quote
is not automatically an inference merely because it is structured; a quote supports
what was reported, not necessarily the truth of the reported event.

Historical image/text may reuse the same modality encoder weights. The result enters
the memory-read/working-context path, not the ordinary fresh-observation transaction.
In particular, a retrieved image must not call the current event-observation path
as if the camera had just seen it. Preserve original source IDs and observation time;
keep retrieval time separate. Distinguish historical evidence, inferred summaries and
hypothetical content through routing, masks and tests, not only a learned origin label.

Retrieval alone must not advance world/event time, append observational history or
count the same source as independent support again. A deliberate belief revision may
use old evidence while preserving its origin. Carry stable evidence IDs and derivation
links through crops, summaries and re-encodings; retrieval counters alone cannot detect
shared evidence. Test repeated reads and overlapping derived sources explicitly.

Pin compatible latent versions or regenerate caches from retained evidence. Compare
task outputs, retrieval behavior and semantic invariants under a declared migration
policy, not raw latent distances alone. When required evidence is gone, invalidate or
mark a belief stale rather than claim it can always be reconstructed. Selective raw
retention has a separate budget from active latent memory.

Local source inspection confirms that current `MemoryRecord` stores encoded evidence
and belief tensors with provenance. It is not an archive of every original image or
audio sample. `HybridMemory.write` detaches stored records and accepts sealed observed
events; `commit_event` advances observation counts. The proposed raw-evidence retention
and graph adapter are therefore additions, while their read path must preserve those
existing event boundaries.

Claude accepted all corrections: no mandatory payload trio, no automatic gradient
through a database, no mandatory policy-gradient algorithm, no fixed identity-class
vocabulary, no guarantee from origin tags or soft scores, and no universal recovery
after evidence loss. No new model run or database implementation was started.
