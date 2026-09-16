"""Versioned understanding fixtures; annotations never enter model-input metadata.

Controlled contrast tasks plus real, coarse scene labels. Repeated use makes these
regression cohorts, not an untouched benchmark of unrestricted understanding.
"""

from collections import defaultdict
from itertools import combinations
from pathlib import Path
import csv
import json
import subprocess

import numpy as np
import torch

from pathwm.io import atomic_json, digest, file_hash
from pathwm.models.modalities import Observation, bytes_batch

KINDS = ("text", "image", "audio", "video")
SCENES = ("bus", "park", "street_traffic")
SCENE_WORDS = ("Bus", "Park", "Strassenverkehr")
CITIES = dict(calibration="barcelona", validation="helsinki", test="vienna")
VERSION = "understanding-fixtures-v1"
PROFILES = {"quick": (2, 1, 2), "full": (8, 4, 8)}
GAPS = [
    "Natural speech transcription, speaker identity and conversation",
    "Music, overlapping sound sources and long audio",
    "Natural object tracking, occlusion and measured action consequences",
    "Long-lived entity memory, retrieval, streaming and tool execution",
    "Fine visual detail, OCR, natural object counting and viewpoint identity",
    "Open-ended output quality in image/audio/video and dialogue",
    "Fresh independent confirmation and device/time-of-day confound controls",
]


def _record(
    case,
    split,
    group,
    member,
    question,
    choices,
    evidence,
    *,
    answer=None,
    required=None,
    groups=None,
    pair=True,
):
    return dict(
        id=f"{case}/{split}/{group}/{member}",
        case=case,
        split=split,
        pair=f"{case}/{split}/{group}" if pair else None,
        question=question,
        choices=list(choices),
        answer=member if answer is None else answer,
        evidence=evidence,
        required=list(required or evidence.keys()),
        groups=groups or [f"synthetic/{case}/{split}/{group}"],
    )


def synthetic_records(profile="quick"):
    if profile not in PROFILES:
        raise ValueError("Unknown fixture profile")
    specs = [
        ("TXT.roles", "text", "Wer hilft der anderen Person?", ("Nora", "Lina")),
        ("TXT.negation", "text", "Wo steht die Tasse?", ("links", "rechts")),
        ("TXT.correction", "text", "Wo liegt der Ball jetzt?", ("links", "rechts")),
        (
            "IMG.relation",
            "image",
            "Wo ist das rote Quadrat relativ zum blauen Kreis?",
            ("links", "rechts"),
        ),
        ("IMG.count", "image", "Wie viele rote Kreise siehst du?", ("zwei", "drei")),
        (
            "AUD.order",
            "audio",
            "Ist der erste Ton tiefer oder hoeher als der zweite?",
            ("tiefer", "hoeher"),
        ),
        (
            "AUD.duration",
            "audio",
            "Welcher der zwei Toene dauert laenger?",
            ("erster", "zweiter"),
        ),
        (
            "VID.motion",
            "video",
            "Wohin bewegt sich der Kreis im letzten Schritt?",
            ("links", "rechts"),
        ),
        ("VID.order", "video", "Welche Farbe erscheint zuerst?", ("rot", "blau")),
        (
            "VID.history",
            "video",
            "Wo war der rote Kreis zuletzt sichtbar?",
            ("links", "rechts"),
        ),
    ]
    arrays, records, cases = {}, [], []
    yy, xx = np.mgrid[:32, :32]
    red, blue = np.array([0.9, 0.1, 0.1]), np.array([0.1, 0.2, 0.9])
    for case, kind, question, choices in specs:
        cases.append(
            dict(id=case, modalities=[kind], domain="controlled contrast", paired=True)
        )
        for si, (split, count) in enumerate(zip(CITIES, PROFILES[profile])):
            for group in range(count):
                rng = np.random.default_rng(31000 + si * 100 + group)
                y = int(rng.integers(11, 21))
                radius = int(rng.integers(2, 4))
                context = (
                    ("Im Garten", "Im Zimmer", "Am Fenster")[si]
                    + " "
                    + (
                        "am Morgen",
                        "am Abend",
                        "am Montag",
                        "am Dienstag",
                        "am Mittwoch",
                        "am Donnerstag",
                        "am Freitag",
                        "am Wochenende",
                    )[group]
                )
                background = float(rng.uniform(0.02, 0.06))
                phase, amplitude, detune = (
                    rng.uniform(0, 2 * np.pi),
                    rng.uniform(0.3, 0.6),
                    rng.uniform(0.9, 1.1),
                )
                for member in (0, 1):
                    a, b = ("links", "rechts")[member], ("rechts", "links")[member]
                    value = None
                    if case == "TXT.roles":
                        value = ("Nora hilft Lina.", "Lina hilft Nora.")[member]
                    elif case == "TXT.negation":
                        value = f"Die Tasse steht nicht {b}, sondern {a}."
                    elif case == "TXT.correction":
                        value = (
                            f"Zuerst lag der Ball {b}. Korrektur: Jetzt liegt er {a}."
                        )
                    elif kind == "image":
                        value = np.full((32, 32, 3), background, dtype=np.float32)
                        if case == "IMG.relation":
                            x = (8, 24)[member]
                            bx = 32 - x
                            value[y - 3 : y + 3, x - 3 : x + 3] = red
                            value[(yy - y) ** 2 + (xx - bx) ** 2 <= 9] = blue
                        else:
                            for x in (8, 24) if member == 0 else (6, 16, 26):
                                value[(yy - y) ** 2 + (xx - x) ** 2 <= radius**2] = red
                        value = value.transpose(2, 0, 1)
                    elif kind == "audio":
                        if case == "AUD.order":
                            freqs = (330, 660) if member == 0 else (660, 330)
                            tones = [
                                amplitude
                                * np.sin(
                                    2 * np.pi * f * detune * np.arange(320) / 8000
                                    + phase
                                )
                                for f in freqs
                            ]
                        else:
                            lengths = (448, 192) if member == 0 else (192, 448)
                            tones = [
                                amplitude
                                * np.sin(
                                    2 * np.pi * 440 * detune * np.arange(n) / 8000
                                    + phase
                                )
                                for n in lengths
                            ]
                        value = (
                            np.concatenate(
                                [tones[0], np.zeros(128), tones[1], np.zeros(256)]
                            )
                            .astype(np.float32)
                            .reshape(-1, 64)
                        )
                    else:
                        value = np.full((3, 32, 32, 3), background, dtype=np.float32)
                        if case == "VID.motion":
                            for frame, x in zip(
                                value, ((8, 24, 16) if member == 0 else (24, 8, 16))
                            ):
                                frame[(yy - y) ** 2 + (xx - x) ** 2 <= 9] = red
                        elif case == "VID.order":
                            for frame, color in zip(
                                value[:2], ((red, blue) if member == 0 else (blue, red))
                            ):
                                frame[(yy - y) ** 2 + (xx - 16) ** 2 <= 9] = color
                        else:
                            for frame, x in zip(value[:2], (16, (8, 24)[member])):
                                frame[(yy - y) ** 2 + (xx - x) ** 2 <= 9] = red
                        value = value.transpose(0, 3, 1, 2)
                    key = f"{case}/{split}/{group}/{member}"
                    if kind != "text":
                        arrays[key] = value
                        value = key
                    else:
                        value = context + ": " + value
                    records.append(
                        _record(
                            case, split, group, member, question, choices, {kind: value}
                        )
                    )
    return records, arrays, cases


def validate_records(records, arrays):
    ids, groups, pairs = set(), {}, defaultdict(list)
    for r in records:
        if r["id"] in ids or r["split"] not in CITIES:
            raise ValueError("Duplicate ID or unknown split")
        ids.add(r["id"])
        if not 0 <= r["answer"] < len(r["choices"]) or len(set(r["choices"])) != len(
            r["choices"]
        ):
            raise ValueError("Invalid answer choices")
        if not r["evidence"] or not set(r["required"]) <= r["evidence"].keys():
            raise ValueError("Missing required evidence")
        for kind, value in r["evidence"].items():
            if kind not in KINDS:
                raise ValueError("Unsupported evidence modality")
            if kind != "text":
                a = arrays[value]
                if not np.isfinite(a).all():
                    raise ValueError("Nonfinite fixture")
                expected = {"image": 3, "video": 4, "audio": 2}[kind]
                if a.ndim != expected or (kind == "audio" and a.shape[-1] != 64):
                    raise ValueError("Invalid fixture shape")
        for group in r["groups"]:
            if groups.setdefault(group, r["split"]) != r["split"]:
                raise ValueError("Source group crosses cohorts")
        if r["pair"] is not None:
            pairs[r["pair"]].append(r)
    for rows in pairs.values():
        if len(rows) != 2:
            raise ValueError("Contrast pair needs exactly two members")
        a, b = rows
        if a["answer"] == b["answer"] or any(
            a[k] != b[k] for k in ("case", "split", "question", "choices", "required")
        ):
            raise ValueError("Invalid contrast pair contract")
        left = {k: v for k, v in a["evidence"].items() if k not in a["required"]}
        right = {k: v for k, v in b["evidence"].items() if k not in b["required"]}
        if left != right:
            raise ValueError("Contrast pair omission is not identical")
        if a["evidence"] == b["evidence"]:
            raise ValueError("Contrast pair has no changed evidence")


def _decode_recording(root, row):
    """One fixed interval, matching raw sources; image/video pairing is done later."""
    video, audio = (root / "raw" / row[k] for k in ("filename_video", "filename_audio"))
    common = [
        "ffmpeg",
        "-v",
        "error",
        "-threads",
        "1",
        "-filter_threads",
        "1",
        "-ss",
        "4",
        "-i",
    ]
    v = subprocess.run(
        common
        + [
            str(video),
            "-t",
            "1.024",
            "-vf",
            "fps=3/1.024,scale=64:64:force_original_aspect_ratio=decrease,pad=64:64:(ow-iw)/2:(oh-ih)/2",
            "-frames:v",
            "3",
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
        timeout=30,
    ).stdout
    a = subprocess.run(
        common
        + [
            str(audio),
            "-t",
            "1.024",
            "-ac",
            "1",
            "-ar",
            "8000",
            "-f",
            "f32le",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
        timeout=30,
    ).stdout
    frames = np.frombuffer(v, np.uint8)
    sound = np.frombuffer(a, "<f4")
    if frames.size != 3 * 64 * 64 * 3 or sound.size != 8192:
        raise ValueError(f"Incomplete fixed source interval: {row['identifier']}")
    return frames.reshape(3, 64, 64, 3).transpose(0, 3, 1, 2).astype(
        np.float32
    ) / 255, sound.copy().reshape(-1, 64)


def prepare_understanding(output, real_root, profile="quick"):
    output, real_root = Path(output), Path(real_root)
    if output.exists():
        raise FileExistsError("Never overwrite frozen fixtures")
    records, arrays, cases = synthetic_records(profile)
    metadata = real_root / "metadata/meta.csv"
    by_group = {}
    for row in csv.DictReader(metadata.open(), delimiter="\t"):
        if row["scene_label"] not in SCENES:
            continue
        if not all(
            (real_root / "raw" / row[k]).is_file()
            for k in ("filename_audio", "filename_video")
        ):
            continue
        by_group.setdefault(row["identifier"], row)
    sources, pools = [], {}
    locations = 3 if profile == "quick" else 4
    for split, city in CITIES.items():
        for scene in SCENES:
            candidates = sorted(
                (
                    r
                    for r in by_group.values()
                    if r["scene_label"] == scene
                    and r["identifier"].rsplit("-", 1)[0] == city
                ),
                key=lambda r: digest(r["identifier"]),
            )
            if len(candidates) < locations:
                raise ValueError(f"Not enough distinct source groups: {split}/{scene}")
            pools[split, scene] = []
            for row in candidates[:locations]:
                group = row["identifier"]
                frames, sound = _decode_recording(real_root, row)
                for k, v in [("video", frames), ("image", frames[1]), ("audio", sound)]:
                    arrays[f"tau/{group}/{k}"] = v
                pools[split, scene].append(group)
                sources.append(
                    dict(
                        group=group,
                        split=split,
                        city=city,
                        label=scene,
                        interval=[4, 5.024],
                        files={
                            k: dict(
                                path=str((real_root / "raw" / row[k]).resolve()),
                                sha256=file_hash(real_root / "raw" / row[k]),
                            )
                            for k in ("filename_audio", "filename_video")
                        },
                    )
                )
    subsets = [c for n in range(1, 5) for c in combinations(KINDS, n)]
    rotations = range(1 if profile == "quick" else 4)
    for subset in subsets:
        paired = len(subset) > 1
        case = ("MIX.agreement." if paired else "REAL.scene.") + "+".join(subset)
        cases.append(
            dict(
                id=case,
                modalities=list(subset),
                domain="real scene category",
                paired=paired,
            )
        )
        for split in CITIES:
            for ci, scene in enumerate(SCENES):
                for rotation in rotations:
                    evidence = {}
                    groups = []
                    for kind in subset:
                        # Three sensor modalities always use DISTINCT recordings.
                        idx = {"image": 0, "audio": 1, "video": 2, "text": 0}[kind]
                        group = pools[split, scene][(idx + rotation) % locations]
                        evidence[kind] = (
                            f"Die Umgebung ist {SCENE_WORDS[ci]}."
                            if kind == "text"
                            else f"tau/{group}/{kind}"
                        )
                        groups.append(group)
                    if not paired:
                        records.append(
                            _record(
                                case,
                                split,
                                f"{ci}-{rotation}",
                                0,
                                "Welche Umgebung beschreibt die Aufnahme?",
                                SCENE_WORDS,
                                evidence,
                                answer=ci,
                                groups=groups,
                                pair=False,
                            )
                        )
                        continue
                    required = [subset[(ci + rotation) % len(subset)]]
                    question = "Beschreiben alle Quellen dieselbe Art von Umgebung?"
                    pairgroup = f"{ci}-{rotation}"
                    for member in (0, 1):
                        ev = dict(evidence)
                        gs = list(groups)
                        if member:
                            kind = required[0]
                            other = (ci + 1) % len(SCENES)
                            idx = {"image": 0, "audio": 1, "video": 2, "text": 0}[kind]
                            group = pools[split, SCENES[other]][
                                (idx + rotation) % locations
                            ]
                            ev[kind] = (
                                f"Die Umgebung ist {SCENE_WORDS[other]}."
                                if kind == "text"
                                else f"tau/{group}/{kind}"
                            )
                            gs[subset.index(kind)] = group
                        records.append(
                            _record(
                                case,
                                split,
                                pairgroup,
                                member,
                                question,
                                ("ja", "nein"),
                                ev,
                                required=required,
                                groups=gs,
                            )
                        )
    validate_records(records, arrays)
    output.mkdir(parents=True)
    np.savez_compressed(output / "arrays.npz", **arrays)
    manifest = dict(
        schema=VERSION,
        profile=profile,
        records=records,
        cases=cases,
        gaps=GAPS,
        sources=sources,
        metadata_sha256=file_hash(metadata),
        arrays_sha256=file_hash(output / "arrays.npz"),
        preprocessing=dict(
            real_size=64,
            real_frames=3,
            audio_rate=8000,
            audio_interval_seconds=1.024,
            audio_chunk=64,
            visual_resize="aspect-preserving letterbox",
            city_split=CITIES,
            model_times="normalized evidence window 3..4 seconds, question at4; not original PTS",
        ),
        limits=[
            "Scene labels apply to recordings; short excerpts can be ambiguous.",
            "Device and time-of-day confounds are uncontrolled; no generalization certification.",
            "All modalities can include the same textual task question; evidence subsets exclude this query.",
            "Text scene cues are explicit labels, not natural captions.",
            "Source groups recur across tasks/conditions; counts and repeated draws are not independent samples.",
            "Controlled text uses fixed grammar with varied neutral context; no natural-language generalization claim.",
            "Fixture cohorts become development regression upon inspection; independent confirmation is separate.",
        ],
    )
    atomic_json(output / "manifest.json", manifest)
    return output


class UnderstandingData:
    def __init__(self, directory):
        self.path = Path(directory)
        self.manifest = json.loads((self.path / "manifest.json").read_text())
        if (
            self.manifest["schema"] != VERSION
            or file_hash(self.path / "arrays.npz") != self.manifest["arrays_sha256"]
        ):
            raise ValueError("Fixture version/hash mismatch")
        with np.load(self.path / "arrays.npz", allow_pickle=False) as z:
            self.arrays = {k: z[k] for k in z.files}
        self.records = self.manifest["records"]
        self.cases = self.manifest["cases"]
        validate_records(self.records, self.arrays)
        self.identity = digest(self.manifest)

    def inputs(self, record, *, omit=(), device="cpu", question_mode="full"):
        """Keep evidence intact; optional controls replace only the question bytes."""
        if question_mode not in ("full", "neutral", "masked"):
            raise ValueError("Unknown observation question mode")
        evidence = {k: v for k, v in record["evidence"].items() if k not in omit}
        question = record["question"]
        if question_mode != "full":
            question = "." * (
                len(question.encode("utf-8")) if question_mode == "masked" else 1
            )
        text = (evidence.get("text", "") + "\nFrage: " + question).strip()
        tokens, mask = bytes_batch([text], device=device)
        inputs = {
            "text": Observation(
                tokens, torch.full_like(tokens, 4, dtype=torch.float64), mask
            )
        }
        for kind, key in evidence.items():
            if kind == "text":
                continue
            x = torch.from_numpy(self.arrays[key].copy()).to(device)
            if kind == "image":
                x = x[None]
            count = len(x)
            # All content is available by t=4; never use future observations.
            times = torch.linspace(3, 4, count, device=device, dtype=torch.float64)[
                None
            ]
            inputs[kind] = Observation(x[None], times)
        return inputs
