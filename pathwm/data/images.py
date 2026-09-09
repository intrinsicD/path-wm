"""Read existing prepared COCO and PushT without historical experiment imports."""

from pathlib import Path
from functools import lru_cache
import json
import numpy as np
import torch
from pathwm.io import file_hash, digest


def checked_array(path, expected):
    if file_hash(path) != expected:
        raise ValueError(f"Data hash mismatch: {path}")
    return np.load(path, mmap_mode="r", allow_pickle=False)


class Frames:
    """Small dataset interface: len, batch(indices, device), and a run identity."""

    def __init__(self, frames, rows, labels=None, identity=None):
        self.frames, self.rows = frames, np.asarray(rows, dtype=np.int64)
        self.labels = labels or {}
        if not len(self.rows) or self.rows.min() < 0 or self.rows.max() >= len(frames):
            raise ValueError("Frame population must be nonempty and in bounds")
        if any(len(v) != len(self.rows) for v in self.labels.values()):
            raise ValueError("Labels must align with the retained frame population")
        self.identity = dict(
            identity or {}, rows_sha256=digest(self.rows.tolist()), count=len(self.rows)
        )

    def __len__(self):
        return len(self.rows)

    def batch(self, indices, device="cpu"):
        ids = np.asarray(indices, dtype=np.int64)
        if ids.ndim != 1 or not len(ids) or ids.min() < 0 or ids.max() >= len(self):
            raise ValueError("Batch indices must be a nonempty in-bounds vector")
        rgb = (
            torch.from_numpy(np.asarray(self.frames[self.rows[ids]]).copy())
            .permute(0, 3, 1, 2)
            .float()
            / 255
        )
        return {
            "rgb": rgb.to(device),
            **{
                k: torch.from_numpy(np.asarray(v[ids]).copy()).float().to(device)
                for k, v in self.labels.items()
            },
        }

    def take(self, count):
        if count <= 0:
            raise ValueError("Subset count must be positive")
        return Frames(
            self.frames,
            self.rows[:count],
            {k: v[:count] for k, v in self.labels.items()},
            self.identity,
        )


@lru_cache(maxsize=8)
def _pusht(root, manifest_hash):
    root = Path(root)
    m = json.loads((root / "manifest.json").read_text())
    if m["schema_version"] != "pusht-cchi-eup-trajectories-v1" or not m["complete"]:
        raise ValueError("Unsupported or incomplete prepared PushT data")
    groups = {k: set(v["group_ids"]) for k, v in m["splits"].items()}
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        if groups[a] & groups[b]:
            raise ValueError("PushT group split overlap")
    frames = checked_array(root / m["frame_store"]["path"], m["frame_store"]["sha256"])
    if (
        tuple(frames.shape) != tuple(m["frame_store"]["shape"])
        or frames.dtype != np.uint8
    ):
        raise ValueError("PushT frame shape/dtype changed")
    episodes = []
    for entry in m["episodes"]:
        p = root / entry["path"]
        if file_hash(p) != entry["sha256"]:
            raise ValueError(f"PushT episode checksum mismatch: {p}")
        with np.load(p, allow_pickle=False) as z:
            ep = {k: z[k] for k in z.files}
        expected = np.arange(
            entry["frame_offset"], entry["frame_offset"] + entry["frames"]
        )
        if (
            not np.array_equal(ep["source_rows"], expected)
            or len(ep["actions"]) != len(expected) - 1
        ):
            raise ValueError("PushT frame/action alignment changed")
        if (
            ep["pose_targets"].shape != (len(expected), 6)
            or int(ep["group_id"]) not in groups[entry["split"]]
        ):
            raise ValueError("PushT pose/group alignment changed")
        episodes.append(dict(ep, split=entry["split"]))
    return (
        frames,
        episodes,
        {
            "kind": "prepared-pusht-v1",
            "root": str(root),
            "manifest_sha256": manifest_hash,
            "frames_sha256": m["frame_store"]["sha256"],
            "normalization": m["normalization"],
        },
    )


def pusht_records(root):
    root = Path(root).resolve()
    return _pusht(str(root), file_hash(root / "manifest.json"))


def PushTFrames(root, split="train"):
    frames, episodes, identity = pusht_records(root)
    chosen = [ep for ep in episodes if ep["split"] == split]
    if not chosen:
        raise ValueError(f"No PushT episodes for split {split!r}")
    return Frames(
        frames,
        np.concatenate([e["source_rows"] for e in chosen]),
        {"pose": np.concatenate([e["pose_targets"] for e in chosen])},
        dict(identity, split=split),
    )


def CocoFrames(root, split="train"):
    root = Path(root).resolve()
    m = json.loads((root / "manifest.json").read_text())
    if m["schema"] != "curriculum-coco-v1":
        raise ValueError("Unsupported prepared COCO data")
    frames = checked_array(root / "frames.npy", m["frames_sha256"])
    return Frames(
        frames,
        m["splits"][split],
        identity={
            "kind": "prepared-coco-rgb64",
            "root": str(root),
            "split": split,
            "manifest_sha256": file_hash(root / "manifest.json"),
            "frames_sha256": m["frames_sha256"],
            "transform": m["transform"],
        },
    )


def CocoMasks(root, masks, split="train"):
    data = CocoFrames(root, split)
    masks = Path(masks)
    manifest = json.loads((masks / "manifest.json").read_text())
    if manifest["coco_manifest_sha256"] != data.identity["manifest_sha256"]:
        raise ValueError("Masks were prepared for different COCO frames/splits")
    label_path = masks / f"{split}.npz"
    if file_hash(label_path) != manifest["splits"][split]["sha256"]:
        raise ValueError("COCO mask checksum mismatch")
    with np.load(label_path, allow_pickle=False) as z:
        rows, mask, valid = z["rows"], z["masks"][:, None], z["valid"][:, None]
    if not set(rows.tolist()) <= set(data.rows.tolist()):
        raise ValueError("Mask split crosses RGB split")
    return Frames(
        data.frames,
        rows,
        {"mask": mask, "valid": valid},
        dict(
            data.identity,
            mask_sha256=file_hash(label_path),
            mask_manifest_sha256=file_hash(masks / "manifest.json"),
        ),
    )
