# Real footage for observation memory

User prefers real webcam footage and authorized selecting existing Documents
images/data and adding what is missing. Prepare a development media pack now:
12 COCO photographs across cup/bottle/book/chair/laptop/bowl, selected from the
existing train partition with distinct duplicate groups and useful object area;
6 Charades training clips with household object interactions. Preserve original
media, source annotations and identities. Inspect contact sheets before acceptance.
These are development examples, not a final train/test population.

The user's own webcam recordings are the target data. Existing stills support
perception; existing activity clips support temporal pipeline development. Neither
supplies a fully labeled persistent-identity/current-versus-historical memory task.
No new external downloads are needed for this selection. Do not activate a camera
as part of inspection; provide a user-run recorder/import recipe, source metadata,
video timestamps and unfilled event/entity/query templates. Any missing annotations
stay explicitly pending. Recording real interactions requires a staged physical scene.

Implement a small ordinary capture/import recipe using local FFmpeg/ffprobe. Explicit
CLI invocation records bounded webcam footage or imports a supplied video. Preserve
session grouping, source hash, decoded frame timestamps and failure status; refuse
existing output directories. Test import metadata/annotation skeleton and overwrite
refusal using a temporary video fixture; invalid capture settings must fail before
opening a device. No model training, GPU-fit claim or task-performance screen here.

Suggested first recording collection: four separate sessions, seven short takes
per session (move, hide, similar objects, changed state, no-information view, and
two takes with different histories but a similar final view). This is a 28-take
development pilot, not an established sufficient training population. Keep related
takes together and balance answers within sessions.
All viewed examples are development material. Reserve fresh complete sessions later
for final evaluation; merely selecting random frames is not an acceptable split.

## Selected local material

The local pack is `data/memory_media_v1/`: 12 original COCO photographs (two each
for cup, bottle, book, chair, laptop and bowl) and six original Charades videos.
The stills come only from the existing COCO train partition, with distinct known
duplicate groups; the videos come only from the official Charades training set.
Original annotations, source hashes, subject grouping and source license metadata
are preserved. No external dataset download is needed for this pilot.

Both contact sheets were inspected. The photographs contain clutter, dark scenes
and varied object instances; they support perception development, not temporal
identity evaluation. In the sampled video frames, `0GFE8` (box/closet) and `0LDP7`
(book/kitchen) look useful to start with. `0JQ26` has multiple people; `12VVC` is
backlit with a small object; `1KKYX` has difficult lighting; `0EJAG` has small desk
objects. These are screening notes, not frame-verified event annotations.

**Real webcam recordings are the target.** Existing indoor clips help build the
temporal pipeline; stills help the encoder. Synthetic sequences remain useful for
controlled failures and implementation checks. Capability evidence must include
fresh real recordings with verified answers. This pack alone does not establish
whether the project's own encoder has enough training data or fits the GPU budget.

## Record or import

From the repository root, after placing two easily distinguished household objects
in view, explicitly start a 30-second, 640×480, 15-fps video-only recording:

```bash
python experiments/capture_webcam.py --record --session desk_01 \
  --output data/webcam_memory_v1/desk_01/take_01 --seconds 30
```

This command opens `/dev/video0`; use `--device` for another camera. Start with a
wide enough view to see the objects and hiding places. Defaults are proposed capture
settings, not a measured model input resolution or GPU-fit claim. The capture path
has not been exercised on the physical webcam yet; devices may need other supported
settings. Recording/import is a data-preparation recipe, not another trainer.

Import an existing recording without opening a camera:

```bash
python experiments/capture_webcam.py --input /path/to/recording.mp4 \
  --session desk_01 --output data/webcam_memory_v1/desk_01/take_02
```

Each episode contains the unchanged imported video (or newly recorded video),
`episode.json` with source/hash/frame times, and an empty `annotations.json`.
Existing episode directories are refused. A failed attempt keeps its files and
failure status; choose a new take directory after fixing the problem. Relative
video timestamps are authoritative for annotations; invocation UTC is not a camera
exposure timestamp. Imported videos retain their original format and full length.

For the first collection, use a cup and book, then visually similar objects:

| Take | Visible interaction | What a verified question could test |
| --- | --- | --- |
| 1 | Move one object between two visible places | Last observed location of that object |
| 2 | Put an object into a container and close it | Earlier placement versus current visibility |
| 3 | Move one of two similar objects continuously in view | Maintaining the selected object's identity |
| 4 | Open and close a book or container | Remembering an earlier observed state |
| 5 | Point the camera away while something may change | Distinguishing remembered evidence from unknown current state |
| 6–7 | Put the object in different containers; end with both closed | Using history when the final view does not reveal the answer |

Repeat across four sessions with lighting, arrangement and object appearance varied.
Vary which object moves and which location is correct *within* each session. Do not
make a session, actor, background or container predict the answer. Paired takes are
one split group; changing only filenames does not create independent examples.
Real final frames will not be pixel-identical: verify the last-view baseline cannot
solve the task through visible contents or incidental cues before claiming memory
is necessary. Reserve later untouched sessions for a frozen evaluation procedure.

## Ground truth before training

Claude's brief independent review identified the missing answer-key protocol.
Adopt it as follows; the provided activity scripts never substitute for these checks.

1. An annotator watches each whole clip and assigns episode-local entity IDs from
   continuous visible evidence. IDs are label references, not inputs supplied to a
   future visual recognizer. Log visibility/occlusion intervals, observable changes,
   location labels and supporting frame indices/times. Boxes are optional for memory
   labels; any detector supervision used later must be declared separately.
2. For each query record its cutoff time, referenced entity, requested historical
   time or `last_observed`/`current` semantics, answer, supporting evidence and any
   ambiguity. An answer must be supported by frames at or before the cutoff. Keep
   later outcome verification separate from what was knowable at that cutoff.
   “Last seen in the box” does not establish that it remains there after an unseen
   intervention. No observation and conflicting observations need distinct flags.
3. A second reviewer independently labels identity, events and query answers before
   reading the first labels, especially all hiding/similar-object/paired cases.
   Record reviewer IDs, disagreement counts and answer agreement; compare event
   boundaries in decoded-frame indices. Adjudicate disagreements with evidence.
   If only one reviewer is available, mark verification pending. Ambiguous identity
   stays unresolved; do not force a match because a script suggests one.
4. Only verified, adjudicated queries enter a scored benchmark. Report excluded
   query counts and reasons, so difficult examples cannot disappear silently. Seal
   the media, annotation versions and complete-session splits before scoring.
   Query wording, model-visible inputs, losses and numerical pass gates still need
   their experiment contract before training; collecting footage does not settle them.

Readable names are aliases. A useful annotation entry has `entity_id`, `display_name`
and evidence references; an event has `start_frame`, `end_frame`, involved entities,
observed state/location and uncertainty. A query has `cutoff_frame`, `entity_id`,
`query_type`, `answer`, `evidence_frames` and `verification_status`. These are proposed
annotation fields, not a committed internal model/graph representation.

## Completion record

Local curation and capture/import tooling are implemented. Source copies and selected
annotation links are checked in `data/memory_media_v1/verification.json`. All six
real clips also have imported episode scaffolds with decoded timestamps and empty
annotations. The source media remain unchanged. No camera has been activated and
no training has run. Own webcam footage and reviewed memory labels remain missing.
The three focused tests cover successful import/alignment, invalid settings before
device access, and preserving evidence/failure status for a broken video.
