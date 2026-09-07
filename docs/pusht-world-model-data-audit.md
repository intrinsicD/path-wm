# PushT data and interface audit — 2026-09-07

The repository retains PushT source configurations, simulator/loader code, and
historical evidence tables. **The large LeWM PushT source and old checkpoint
directories are absent from the current workspace view. The coordinator has
reacquired the compact official CCHI source, which this audit verified directly.**
No training, dataset modifications, or changes to the LeWM implementation were
made by the audit. Read-only source checks and four small CPU replay probes are
recorded in [the machine-readable audit](../runs/pusht_cchi_data_audit_2026-09-07.json).

## Availability and exact recovery identities

The configured file `data/pusht/pusht_expert_train.h5`,
the prefix/pilot files, and the referenced
`runs/diagnostics/pusht_epoch1_2026-09-06` checkpoint directory do not exist here.
Bounded filename searches under Documents, Downloads, the local/Hugging Face
caches, `/mnt`, `/media`, and `/tmp` did not locate a moved PushT HDF5/archive.
This establishes current unavailability in those locations, not that originals
were deleted. Historical inventories must not be treated as current file checks.

The exact main-source identity is retained in
[configs/datasets/pusht.yaml](../configs/datasets/pusht.yaml):

- Hugging Face dataset repository: `quentinll/lewm-pusht`.
- Revision: `655cd446b9929369d7d406001da85c15d1457850`.
- File: `pusht_expert_train.h5.zst`.
- Archive SHA256: `7cfbd6d90fa2f27876379a5ff169715a36ed82edbda64f9e5b5bfa34d212f318`.
- Historically verified archive size: **13,136,247,974 bytes**.
- Historically extracted HDF5 size: **46,300,921,856 bytes**.
- Historically verified population: **18,685 episodes / 2,336,736 frames**.

These are supported by [source-data.md](source-data.md) and the retained
[reference evidence](../ara/evidence/tables/reference_validation_2026-09-05.json),
including contiguous-offset and matching-row checks from that earlier audit.

The recovered CCHI source is
`https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip`, historically
and now directly verified as **206 episodes / 25,650 frames** at 96 pixels.
The acquired archive's SHA256 is
`63d52a114a3f010861f0181309d165b7d69133ccae426ece2fc94caed147bdf9`.
The new HDF5 is **99,922,306 bytes**, SHA256
`30e442fedacaa4a0d36662b951c326b411a15c3a365b514587b9a738229da8f6`.
[convert_pusht_cchi.py](../scripts/convert_pusht_cchi.py) preserves
Zarr `data/img`, `data/action`, and `data/state`, deriving episode offsets/lengths
from `meta/episode_ends`. This audit compared every converted image, action and
state block with its original Zarr array using exact equality; all match, and
episode ends/offsets match. See the new
[conversion receipt](../data/pusht_cchi/conversion.json).

The full source scan measured block angles from **0.000215591251617 to
6.2830877304 radians**. Causal angle differences must therefore wrap across
0/2π. No source-wide terminal-success rate was measured in this audit.

The measured filesystem had **24,150,200,320 bytes available** at the audit.
The full legacy HDF5 alone exceeds that space; retaining its archive and HDF5
together needs about 59.44 GB before derived caches. Recovering the source on a
different volume or defining a verified bounded conversion is necessary before
that full-storage workflow. No source or old result was removed to make space.

`data/e0_dev` is present, but is not a verified PushT source. Read-only,
memory-mapped inspection of its local tensor containers found:

| File | Actual stored tensors | Labels/provenance limitation |
| --- | --- | --- |
| `episodes.pt` | uint8 observations `[640,33,3,64,64]`; float32 actions `[640,32,2]`; seed 0; 64 worlds | No state labels or PushT source identity |
| `counterfactual.pt` | uint8 initial observations `[2048,3,64,64]`; next observations `[2048,4,3,64,64]`; float32 actions `[2048,4,2]` | Branching fixture, no verified PushT identity |

These can support generic shape checks, but cannot substitute for requested
PushT training or verify the missing HDF5 schema.

## Source schema and privileged labels

The table distinguishes the **expected historical LeWM interface**, established
from retained code/evidence, from the **directly verified CCHI schema**. Actual
LeWM HDF5 dtypes, compression/chunking, and terminal-action values still require
an accessible source. The CCHI file has exactly the five fields shown below.

| Field | LeWM PushT expected shape | CCHI verified shape/dtype | Meaning |
| --- | --- | --- | --- |
| `pixels` | `[2336736,224,224,3]`, expected RGB uint8 | `[25650,96,96,3]`, **float32** | CCHI raw values are integer-valued 65..255 |
| `action` | `[2336736,2]`, floating dtype unverified | `[25650,2]`, float32 | Relative target displacement versus absolute target XY |
| `state` | `[2336736,7]`, floating dtype unverified | `[25650,5]`, float32 | Agent XY, block XY, block angle; LeWM additionally agent velocity XY |
| `proprio` | `[2336736,4]` | Not written by converter | Agent XY and agent velocity XY |
| `ep_len`, `ep_offset` | `[18685]` each | `[206]` each, int64 | Episode observation counts and contiguous first-row offsets |
| `step_idx`, `episode_idx` | `[2336736]` each | Not written by converter | Recorded local step and episode identity |

The LeWM source's seven-state/four-proprio dimensions are independently present
in the archived reference normalization arrays. Its earlier receipt explicitly
reports **no seed column**. No archived shape/dtype inventory supports claiming
the missing LeWM HDF5 storage dtypes were confirmed today.

CCHI image chunks are one frame, `[1,96,96,3]`, using LZF compression. Actions
and states are LZF-compressed float32; episode arrays are contiguous int64.
Every RGB scalar is finite and integer-valued, so converting the source to uint8
before the declared resize loses no values. Divide by 255 exactly once for E.
Actions span x=12..511 and y=25..511, all finite; final unused action rows are
also finite. Episodes range from 49 to 246 observations (median 122), giving
**25,444 valid one-step transitions**. No source terminal flag is available.

[PushT._get_obs](../third_party/swm/pusht.py) emits float64
`[agent_x, agent_y, block_x, block_y, block_angle_mod_2pi, agent_vx, agent_vy]`.
The last two values are **agent**, not block, velocity. The misleading local
variable name `vel_block` in `_set_state` is assigned to `self.agent.velocity`.
Block linear and angular velocity are omitted from this observation and from the
documented source labels. Pymunk exposes them in a live simulator, but they must
not be presented as existing source targets. CCHI omits even agent velocity.
Finite-difference labels would be separately defined estimates, with explicit
time/angle wrapping and validity masks; they are not simulator ground truth.

## Exact simulator action timing and reset limitations

The checked local simulator is adapted from stable-worldmodel commit
`6f1e499e9cc0c898d326112f485c1062c3d20f24`. Its world is 512 units square.
One action interval is **0.1 seconds / 10 Hz**, comprising ten 0.01-second physics
steps. The PD controller uses `k_p=100`, `k_v=20`.
`_setup` explicitly sets `space.damping = 0`; `reset` changes that value only
when a non-None damping override is supplied. The checked default wrapper uses
that local simulator behavior, rather than assuming Pymunk's library default.

- LeWM relative mode computes one fixed target at interval start:
  `target_xy = current_agent_xy + 100 * action_xy`. This is a target displacement,
  not a guaranteed realized agent displacement. Nominal action bounds are
  `[-1,1]^2`, but `step` does not clip them.
- CCHI absolute mode treats `action_xy` directly as the target in world/pixel
  coordinates. Passing these values into relative mode would multiply them by
  100 and change their meaning entirely.
- The existing LeWM loader takes observations five source rows apart, and packs
  the five consecutive 2D actions beginning at each observation into width 10.
  Its model interval is therefore 0.5 seconds. The proposed E/U/P adaptation
  should explicitly choose a single source interval first; inheriting five-frame
  skipping accidentally would change its transition horizon and memory timing.
- A length-L source episode has L observations and L stored action rows. Only
  rows `0..L-2` have a recorded successor inside that episode. The final action
  may be finite or unused padding; it must never become a cross-reset target.
  Existing loaders tolerate NaNs only in the unused final block, not in executed
  transition actions. New adapters should preserve a separate terminal flag only
  when supported by source metadata, rather than infer failure from file end.

`_set_state` advances physics by 0.01 seconds after assigning poses/agent
velocity. It neither restores block velocities nor supplies a general simulator
snapshot. Starting from a mid-trajectory seven-vector therefore does not establish
exact simulator replay. Before controller evaluation, compare initial pixels,
post-reset state, and recorded-action prefixes with actual source data; report
residuals and hidden-state omissions. Prefer episode-prefix replay for observer
warm-up rather than appending invented stay frames to a mid-trajectory source.

Rendering includes a green goal shape by default (`with_target=True`). Its
`goal_pose` comes from variation settings; `_set_goal_state` changes the numeric
goal state without changing `goal_pose`. Source/background/goal rendering must
be compared before claiming a match. Turning the goal overlay off globally would
be an unverified rendering change, not a harmless adapter detail.

Four preselected CCHI episode-start probes (episodes 0, 1, 102, 205) used absolute
actions, 96-pixel rendering, and explicit zero agent velocity appended solely for
the simulator's seven-coordinate reset/goal API. Their first ten recorded
actions were replayed. Episodes 0, 1 and 102 reset to exact source poses and their
ten-step position RMSE stayed below 1e-5 world units. Episode 205 begins in contact:
the reset physics step changes pose, with position RMSE 0.205641 and angular error
0.000427645 radians; after ten actions these are 0.00937632 and 0.00108914 radians.
Initial image MAE ranges from 0.0486473 to 0.113824 in 0..255 units; 0.217% to
0.445% of pixels differ. This is useful compatibility evidence, not exact image
or all-trajectory replay. The raw report retains every step and the initially
rejected five-coordinate goal probe; that API error was corrected by explicitly
padding the missing reset velocity, without inventing source velocity labels.

## Population integrity and preserved LeWM evidence

The full-source reproduction config intentionally uses **random windows** and
full-source action normalization. Adjacent observations and episodes can cross
that train/validation assignment; it is interpolation evidence, not an
episode-held-out or configuration-held-out protocol. Do not reuse its split for
a new claim of independent E/U/P generalization.

The prior source audit found only **185 exact first-frame `state[:5]` groups**
across 18,685 episodes. This is a configuration-group proxy, not proof of unique
trajectories. [prepare_pusht_pilot.py](../scripts/prepare_pusht_pilot.py) already
groups exact/near starts using agent/block distances and circular angle distance;
its historical bounded pilot selected 160 groups, with 128 training and 32
held out, one episode per group, totaling 20,015 frames. That derived HDF5 was
3,014,189,180 bytes and is also currently unavailable. Its recorded SHA256 is
`af224b18f0bf54039983618bddf62ec4f8ca446ee549570ab824caa7a31d0ae5`.

For a new full-source experiment, freeze train/validation/test **groups before
sampling frames or windows** and keep all variants of each selected group in
one split. Any proposed 128/28/29 split assumes the 185-group count is reverified.
Fit action/velocity normalization and latent statistics only on training groups.
Test goals/cases and all caches must carry the new split identity.

The new CCHI audit found **206 distinct exact initial states and 206 near-state
groups**, using the existing 5-pixel agent/block and 0.05-radian circular-angle
thresholds. Its nearest distinct configuration has scaled distance 7.61337,
above the grouping threshold of 1; each group contains one episode. This supports
the coordinator's declared **164 train / 20 validation / 22 test episode** split
under this grouping rule, while not proving absence of similar later segments.

Historical LeWM results remain useful references: the 13,933-update continuation
achieved **30/50** frozen source goals, versus **17/50** at 8,404 and **45/50** for
released weights. Its checkpoint was recorded as SHA256
`151b356addea1a9bc7c939fcd102986ed1e7dfca693212463308b8456b3ca4f0`.
The [follow-up report](tworoom-followup-2026-09-06.md) and
[retained raw evidence table](../ara/evidence/tables/tworoom_followup_2026-09-06.json)
remain readable; the original checkpoint files are not available here to resume
or rehash. These numbers describe the prior LeWM system and source-goal protocol,
not the new E/U/P architecture or an independent new test population.

## New E/U/P data interface

The coordinator selected and acquired the compact CCHI source for the new
adaptation. The normative decisions belong to
[pusht-world-model-design.md](pusht-world-model-design.md). They are not
implemented or trained behavior in this audit. Preserve the paddle baseline
and all old LeWM interfaces.

- Keep E/D's 64-pixel RGB interface and fine/coarse token ordering. Resize raw
  source frames deterministically with one recorded antialiasing implementation;
  retain raw source identity. A separate module/config should name the PushT
  action/tensor schema and its 0.1-second interval.
- Provide absolute executable floating action `[target_x/512,target_y/512]`
  in `[0,1]^2` to P with its own 2-to-64 projection. The approved U uses the same
  two preceding-action coordinates and an initial `[-1,-1]` sentinel outside
  the executable domain; its GRU input is 64+2=66. A real `[0,0]` command remains
  distinguishable from episode start. Reuse the same real/imagined P-then-U
  ordering and never reset memory mid-episode.
- Use H targets `[agent_x/512, agent_y/512, block_x/512, block_y/512,
  sin(angle), cos(angle)]`. CCHI's R has these six pose outputs plus four causal
  backward XY displacements and one wrapped angular change, totaling 11 outputs;
  motion terms at t<2 are masked. Before first training, the coordinator adopted
  per-coordinate physical-motion RMS scales fitted only on training groups at
  t>=2, with floors of 1 world unit for XY and 0.01 radians for angle. Divide
  motion labels by these recorded scales; the earlier `/512` and `/pi` proposal
  is superseded for prepared training labels. Also fit normalized action-offset
  RMS `(action_world - pusher_xy)/512` on training transitions for the declared
  CEM initialization. Record fit counts and exact source episode IDs. These motion
  targets are **estimated displacements per raw interval**, not stored velocity
  labels. This requires task-specific readout sizes, not replacing the recurrence.
- An episode adapter should expose raw frames, `L-1` causal executable actions,
  pose/motion targets with validity masks, episode/group/source identities,
  timing, and separate evaluation metadata. Model inputs remain RGB and actions.
- Continuous PushT actions require a new declared planner and goal evaluator;
  the paddle's three-action exhaustive interception score cannot be reused.
  The evaluation/design task owns that decision and its budget.

Storage arithmetic, excluding file/index overhead:

| Population | Raw uint8 RGB64 | Full float32 S cache | Float32 128-memory cache |
| --- | ---: | ---: | ---: |
| Full historical LeWM source | 28,713,811,968 B | 191,425,413,120 B | 1,196,408,832 B |
| Verified CCHI source | 315,187,200 B | 2,101,248,000 B | 13,132,800 B |
| Historical 20,015-frame pilot | 245,944,320 B | 1,639,628,800 B | 10,247,680 B |

The historical LeWM uncompressed RGB64 cache alone exceeds the measured free
space. The compact CCHI cache fits: the selected adapter streams canonical
uint8 RGB64 into one `frames.npy` file, exposing read-only memory-mapped views.
Small per-episode NPZ files contain labels/actions and source/global offsets,
without duplicating compressed pixels. The manifest records the flat image
file hash and its source-episode/frame ordering. This also avoids the measured
random-frame decompression bottleneck from the earlier paddle implementation.

## Essential checks before training/control

1. Reacquired HDF5 schema, exact dtypes, offsets, row counts, finite transition
   actions, and source checksums match a saved receipt; every selected episode
   has L observations and only L-1 used successor actions.
2. An index-coded mock episode verifies action `a_i` leads from frame i to i+1,
   U receives the initial marker then each actual action once, and no source or
   target window crosses an episode/reset. K-step windows require `i+K < L`.
3. The group split is a complete disjoint partition; no exact/near initial
   configuration appears across partitions; normalizers and statistics refuse
   held-out rows. Tampered split/source/cache fingerprints fail explicitly.
4. Relative and absolute command modes are never mixed; same-target equivalent
   commands agree in a controlled simulator case; a nonzero initial velocity
   distinguishes a zero target displacement from an instantaneous position hold.
5. Current-image H never receives motion labels; R's derived motion targets use
   only current/prior source states and retain their displacement definition.
   Circular orientation residuals handle the 0/2π boundary. Changing future
   labels must not change the current R target or observer input.
6. Source reset/render/prefix-replay checks quantify the 0.01-second reset step,
   goal-overlay differences, and omitted block velocities before reporting
   simulator-grounded control. Preserve failures rather than assume exact replay.
7. Existing E/U/P causality, frozen-U gradient, branch isolation, exact statistics,
   and checkpoint/dependency tests apply to the new action/readout schema; a
   PushT checkpoint must refuse paddle action or label schemas.

The compact CCHI data prerequisite is now verified; the coordinator is handling
the new model's adapter, frozen split, tests and staged training scope. The large
LeWM source/checkpoints remain unavailable in the searched locations. This audit
did not initiate either LeWM continuation or new-model training.
