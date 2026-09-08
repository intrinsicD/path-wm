"""Read-only metadata/label readiness audit; never trains or decodes full media.

Run: .venv/bin/python scripts/audit_perception_data.py
The companion notebook shows the resulting checks and their scope.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
import csv
import hashlib
import json

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATASETS = Path.home() / "Documents/datasets"
OUTPUT = ROOT / "runs/perception_data_cowork_2026-09-08/data_readiness.json"


def read_json(path):
    return json.loads(Path(path).read_text())


def overlaps(groups):
    return {f"{a}/{b}": len(set(groups[a]) & set(groups[b])) for a, b in combinations(groups, 2)}


def coco():
    path = ROOT / "data/curriculum/coco_v1/manifest.json"
    m = read_json(path)
    records = m["records"]
    files = {p.name for p in (DATASETS / "train2014").glob("*.jpg")}
    sets = {name: set(rows) for name, rows in m["splits"].items()}
    groups = {name: {records[i]["group"] for i in rows} for name, rows in sets.items()}
    result = {
        "source_images": len(files), "prepared_frames": len(records),
        "prepared_shape": list(np.load(path.parent / "frames.npy", mmap_mode="r").shape),
        "missing_source_filenames": len({r["file"] for r in records} - files),
        "split_frames": {k: len(v) for k, v in sets.items()},
        "split_row_overlap": overlaps(sets), "split_duplicate_group_overlap": overlaps(groups),
        "distinct_duplicate_groups": len({r["group"] for r in records}),
        "duplicate_rule": m["duplicate_rule"],
        "official_validation_image_directory_present": (DATASETS / "val2014").is_dir(),
        "source_manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "annotation_files": {}, "prepared_union_masks": {},
    }
    for name in ["instances_train2014.json", "person_keypoints_train2014.json", "captions_train2014.json"]:
        ann_path = DATASETS / "annotations_trainval2014/annotations" / name
        d = read_json(ann_path)
        annotation_ids = [a["id"] for a in d["annotations"]]
        image_ids = {i["id"] for i in d["images"]}
        result["annotation_files"][name] = {
            "images": len(image_ids), "annotations": len(annotation_ids),
            "categories": len(d.get("categories", [])),
            "duplicate_annotation_ids": len(annotation_ids) - len(set(annotation_ids)),
            "orphan_annotation_image_ids": len({a["image_id"] for a in d["annotations"]} - image_ids),
            "missing_image_filenames": len({i["file_name"] for i in d["images"]} - files),
        }
        del d
    for split in ["train", "validation", "test"]:
        with np.load(ROOT / "runs/encoder_study_2026-09-08/coco_masks" / f"{split}.npz") as a:
            rows = a["rows"]
            expected_ids = np.array([int(records[int(i)]["file"].rsplit("_", 1)[-1].split(".")[0]) for i in rows])
            result["prepared_union_masks"][split] = {
                "images": len(rows), "shape": list(a["masks"].shape),
                "outside_declared_split": len(set(rows.tolist()) - sets[split]),
                "image_id_mismatches": int(np.count_nonzero(a["image_ids"] != expected_ids)),
                "group_mismatches": int(np.count_nonzero(a["groups"] != np.array([records[int(i)]["group"] for i in rows]))),
                "nonbinary_mask_pixels": int(np.count_nonzero((a["masks"] != 0) & (a["masks"] != 1))),
                "nonbinary_valid_pixels": int(np.count_nonzero((a["valid"] != 0) & (a["valid"] != 1))),
                "images_without_valid_pixels": int(np.count_nonzero(a["valid"].reshape(len(rows), -1).sum(1) == 0)),
            }
    result["scope"] = "Manifest identities, array headers and prepared mask values checked; image decoding/hashes and geometric mask alignment are not repeated. Earlier alignment audit remains the source."
    return result


def prepared_trajectories(relative, paddle=False):
    base = ROOT / relative
    m = read_json(base / "manifest.json")
    counts, frames, transitions, events = Counter(), Counter(), Counter(), Counter()
    groups = {s: set() for s in ["train", "validation", "test"]}
    failures, present = Counter(), 0
    for ep in m["episodes"]:
        split = ep["split"]
        counts[split] += 1
        frames[split] += ep["frames"]
        transitions[split] += ep["transitions"]
        groups[split].add(ep["seed"] if paddle else ep["group_id"])
        file = base / ep["path"]
        if not file.is_file():
            failures["missing_file"] += 1
            continue
        present += 1
        with np.load(file, allow_pickle=False) as a:
            state = a["states"] if paddle else a["poses_world"]
            actions, timestamps = a["actions"], a["timestamps"]
            failures["row_count"] += int(len(state) != ep["frames"] or len(actions) != len(state)-1)
            failures["transition_count"] += int(len(actions) != ep["transitions"])
            failures["nonfinite_state_or_action"] += int(not np.isfinite(state).all() or not np.isfinite(actions).all())
            failures["nonfinite_timestamps"] += int(not np.isfinite(timestamps).all())
            failures["time_order"] += int(len(timestamps) != len(state) or np.any(np.diff(timestamps) <= 0))
            if paddle:
                events.update(e["type"] for e in json.loads(str(a["events_json"])))
            else:
                failures["motion_shape"] += int(a["motion_targets"].shape != a["motion_mask"].shape)
                failures["nonfinite_motion"] += int(not np.isfinite(a["motion_targets"]).all())
                failures["initial_motion_unmasked"] += int(a["motion_mask"][:2].any())
    return {
        "episodes": dict(counts), "frames": dict(frames), "transitions": dict(transitions),
        "files_present": present, "group_overlap": overlaps(groups),
        "group_type": "generation seed" if paddle else "initial configuration group",
        "failed_episode_checks": dict(failures), "event_counts": dict(events),
        "label_semantics": "Stored simulator XY/velocity state; executable discrete actions and event records." if paddle else m["label_schema_version"],
        "scope": "All prepared episode scalar/state/action/time arrays checked; RGB arrays not decoded, no rehash/replay or inferred coverage guarantee.",
    }


def hdf_inventory(relative):
    path = ROOT / relative
    with h5py.File(path, "r") as f:
        lengths, offsets = np.asarray(f["ep_len"]), np.asarray(f["ep_offset"])
        result = {
            "file_bytes": path.stat().st_size, "episodes": len(lengths), "frames": int(lengths.sum()),
            "valid_transitions": int(np.maximum(lengths-1, 0).sum()),
            "contiguous_offsets": bool(np.array_equal(offsets, np.r_[0, np.cumsum(lengths)[:-1]])),
            "fields": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in f.items() if isinstance(v, h5py.Dataset)},
            "scope": "Headers, lengths and offsets; initial state labels for PushT. No full media decode, source-file rehash or simulator replay.",
        }
        if "state" in f:
            start_states = f["state"][offsets, :5]
            result["exact_initial_pose_groups"] = len(np.unique(start_states, axis=0))
            result["grouping_limit"] = "Exact first five state values; not a near-configuration or trajectory overlap audit."
        return result


def csv_rows(path, delimiter=","):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def natural_videos():
    charades = ROOT / "data/charades"
    sets = {s: csv_rows(charades / f"metadata/Charades/Charades_v1_{s}.csv") for s in ["train", "test"]}
    vids = {p.stem for p in (charades / "raw/Charades_v1_480").glob("*.mp4")}
    c = {"video_files": len(vids), "split_rows": {s: len(r) for s, r in sets.items()},
         "missing_by_split": {s: len({r["id"] for r in rows} - vids) for s, rows in sets.items()},
         "id_overlap": overlaps({s: [r["id"] for r in rows] for s, rows in sets.items()}),
         "subject_overlap": overlaps({s: [r["subject"] for r in rows] for s, rows in sets.items()})}
    tau = ROOT / "data/tau_urban_av_2021"
    meta = csv_rows(tau / "metadata/meta.csv", "\t")
    locations = {r["filename_audio"]: r["identifier"] for r in meta}
    splits = {s: csv_rows(tau / f"metadata/evaluation_setup/fold1_{s}.csv", "\t") for s in ["train", "test"]}
    t = {"metadata_pairs": len(meta),
         "audio_files": len(list((tau / "raw/audio").glob("*.wav"))),
         "video_files": len(list((tau / "raw/video").glob("*.mp4"))),
         "missing_audio": sum(not (tau / "raw" / r["filename_audio"]).is_file() for r in meta),
         "missing_video": sum(not (tau / "raw" / r["filename_video"]).is_file() for r in meta),
         "duplicate_pair_rows": len(meta) - len({(r["filename_audio"], r["filename_video"]) for r in meta}),
         "split_rows": {s: len(rows) for s, rows in splits.items()},
         "split_location_counts": {s: len({locations[r["filename_audio"]] for r in rows}) for s, rows in splits.items()},
         "location_overlap": overlaps({s: [locations[r["filename_audio"]] for r in rows] for s, rows in splits.items()})}
    return {"charades": c, "tau": t, "scope": "CSV identifiers and media-file existence; no full decoding, temporal annotation/synchronization audit or executable-action inference."}


def main():
    result = {"generated_at": datetime.now(timezone.utc).isoformat(),
              "scope": "Read-only metadata and prepared numerical-label audit, not a model experiment.",
              "coco": coco(), "paddle": prepared_trajectories("data/paddle/baseline", True),
              "pusht_cchi": prepared_trajectories("data/pusht_world_model/cchi_v1"),
              "large_pusht": hdf_inventory("data/pusht/pusht_expert_train.h5"),
              "tworoom": hdf_inventory("data/tworoom/tworoom.h5"),
              "natural_videos": natural_videos(),
              "kodak_images": len(list((DATASETS / "kodak24").glob("*.png"))),
              "fabric_time_directories": len(list((DATASETS / "2025_03_07_stage_with_fabric").glob("frame_*"))),
              "limitations": ["Existing evaluation populations already inspected", "No architecture or data-sufficiency guarantee from counts", "No new dataset generated or downloaded", "No GPU/model inference or training"]}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(OUTPUT), "large_pusht_exact_initial_groups": result["large_pusht"]["exact_initial_pose_groups"],
                      "prepared_episode_failures": {k: result[k]["failed_episode_checks"] for k in ["paddle", "pusht_cchi"]}}, indent=2))


if __name__ == "__main__":
    main()
