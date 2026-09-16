# Video understanding: proposed evaluation map

16 September 2026. Requested capability overview, not an adopted run protocol.
No new training, benchmark download or validation promotion. Numerical gates,
populations and budgets must be declared before any experiment using this map.

## Objective and present evidence

For PATH-WM, video should provide persistent, queryable and predictive information
about entities, changes and interactions to the shared latent core and World State.
Direction is one genuine component of this objective. Coverage of a constructed
direction task does not establish the other components or natural-video transfer.

The shared image/video codec has tested mechanics. Reconstruction studies do not
establish understanding. The latest fixed correspondence rule and structured reader
pass a narrow fresh controlled-pan screen, but learned refinement fails source
preservation. See [results](video-evidence-plan.md). Natural object trajectories,
streaming memory and useful integration of this codec into the agent remain open.

## Capability matrix

| Capability | Concrete task | Measurement |
| --- | --- | --- |
| Visual state and detail | Locate objects and distinguish task-relevant properties, including small objects | Localization/property accuracy by object size and resolution |
| Motion | Stillness, horizontal/vertical/diagonal motion, rotation, speed and acceleration | Direction/stillness accuracy and displacement/velocity error with explicit coordinate and time units |
| Camera versus object motion | Moving camera/static object, static camera/moving object and both moving | Foreground/background motion error; correct ambiguity handling where separation is unobservable |
| Identity and tracking | Similar objects cross, leave view, become occluded and return | Track position, identity switches, duplicate/merge errors and reacquisition performance; point tracking alone does not prove semantic identity |
| State and relation changes | Open/close, empty/fill, pick up/put down, inside/outside and contact | Per-entity state/relation correctness and transition timing |
| Interactions and event order | Who moved which object, before/after, repeated actions and multi-step events | Actor/object binding, temporal localization, order and count accuracy |
| Persistent history | Same final image after different relevant histories; delayed queries beyond recent context | Correct historical answers and entity binding versus memory delay/distractors |
| Prediction | Predict future positions, states, contacts and events from prefixes only | Error by horizon and appropriate distribution scores versus persistence and constant-velocity references; forecast uncertainty |
| Action effects and causal alternatives | Same initial condition with different recorded actions or controlled interventions | Predicted outcome and task success; observational video alone is not a ground-truth causal test |
| Uncertainty | Occluded, noisy, ambiguous or unseen cases | Error versus declared confidence, coverage versus error when abstaining, correction after new evidence |
| Shared-core and World State use | Feed video evidence through adapter, binding, update and retrieval, then query/predict/act | End-task correctness and benefit over matched no-history/shuffled-history controls, plus stage diagnostics |
| Multimodal correspondence | Associate visual events with sounds or instructions, including mismatches and missing modalities | Synchronization, association and grounded-answer accuracy; report video-only and audio-only controls |

## Checks across all capability tests

- Identify targets, coordinate conventions, timestamps and annotation quality. Tiny
  crops can remove decisive evidence; compare supported resolutions rather than
  imposing 48x48 as the capability contract. Avoid claims about metric 3D velocity
  without the required calibration/observability.
- Use single-frame and unordered/shuffled-history controls where appropriate. Build
  matched-history cases needing different answers. A temporal benefit must depend
  on the relevant sequence, not merely extra scene appearance. Reversing video is
  useful only for tasks whose target actually transforms that way.
- Hold out people/objects/scenes/source recordings as appropriate; near-duplicate
  clips cannot cross splits. Separate development from untouched confirmation,
  including previously inspected sources as regression checks.
- Stress different speeds, lighting, blur, compression, viewpoints, occlusions,
  frame rates, dropped frames, scene cuts, clip lengths and distractors. Report
  per-group failures and several seeds; account for source-level sample dependence.
- For online use, test no future leakage, state across chunks, explicit reset,
  episode isolation, missing/late inputs, bounded memory, latency and GPU memory.
  Existing finite-window causal checks do not validate streaming state.
- Fix readout capacity/training exposure for feature probes. Read comparable targets
  from frame features, temporal features, core state and retrieved memory. A failed
  probe is not proof of erased information; a successful probe is not proof that
  the deployed agent uses it. Test the actual downstream path too.
- Record performance versus training data and compute, plus preservation of image
  capabilities. Reconstruction and generation quality have separate reports; neither
  is a substitute for state/behavior tests. Understanding need not be tested solely
  through a language decoder.

## Proposed sequence and stopping discipline

1. Preserve controlled pans as a diagnostic/regression test. Establish natural
   object motion and tracking first, including stillness, variable speed, camera
   motion and short occlusion, with independent temporal references.
2. Feed that evidence through the actual shared core and entity memory; test
   persistent identity, state changes and history-dependent queries.
3. Add future state/event prediction at multiple horizons; require relevant-history
   benefit over matched simple references before attributing success to dynamics.
4. Add action-conditioned tasks, longer events and audiovisual interaction as their
   relevant interfaces become testable. These need their own scope and gates.

This sequence is proposed, not authorization to execute all stages. Each bounded
milestone needs a fixed primary endpoint, source split, minimum meaningful effect,
preservation limits, resource budget and stop rule before running. Passing advances
that scope; failure permits a targeted cause-localizing diagnosis with a declared
budget, not an automatic chain of new architecture variants. Preserve negative
results and revisit the overall bottleneck before extending the experiment series.

## External reference tasks

- [TAP-Vid](https://tapvid.github.io/) supplies real and synthetic annotated point
  tracks for trajectory/occlusion evaluation; it is not a full entity-memory test.
- [Something-Something v2](https://www.qualcomm.com/developer/software/something-something-v-2-dataset)
  supplies videos of object interactions for fine-grained action tasks.
- [V-JEPA 2](https://ai.meta.com/research/publications/v-jepa-2-self-supervised-video-models-enable-understanding-prediction-and-planning/)
  evaluates understanding, anticipation and action-conditioned planning separately.

These are methodology references, not downloaded data, adopted architectures or
capabilities already measured in PATH-WM. Existing Charades clips can support a
small initial natural-video study, but their available annotations must first be
checked against the desired trajectory/state targets.
