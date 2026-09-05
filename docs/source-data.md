# Source data inventory

Status as of 2026-09-05. Everything under `data/` is downloaded source data;
generated checkpoints and logs live under `runs/`. Each entry records the
pinned origin, the acceptance check, what is on disk and the role it plays.
Passive datasets are retained for the deferred ideas in [ideas.md](ideas.md)
and are not used by the current LeWM baseline.

| Dataset | Role | On disk | Acceptance |
| --- | --- | --- | --- |
| LeWM PushT (official) | Baseline training and control | Incomplete: 89.3 percent of the archive received, no extracted HDF5 | Whole-file SHA256 pending; recovery handled by a concurrent Codex session |
| LeWM TwoRoom (official) | Second trajectory dataset | `data/tworoom/tworoom.h5`, 12,775,849,984 bytes, 10,000 episodes, 920,809 frames | Archive SHA256 matches pinned; extracted HDF5 matches archive stream |
| Diffusion Policy PushT (cchi) | Early development checks | `data/pusht_cchi/pusht_cchi.h5`, 206 episodes, 25,650 frames | Archive SHA256 recorded in `conversion.json` |
| Verified prefix subsets | Bounded diagnostics only | `data/pusht_prefix/` (8 episodes), `data/tworoom_prefix/` (32 episodes) | Chunk extents inside received prefixes; receipts beside each file |
| Released LeWM PushT weights | Evaluator parity reference | `data/reference/lewm-pusht/` | Hugging Face cache download |
| TAU Urban Audio-Visual Scenes 2021 | Passive audio/video, deferred | `data/tau_urban_av_2021/raw/`, 12,291 audio and 12,291 video clips, about 106 GB | All 24 Zenodo archives verified and extracted on 2026-09-05 |
| Charades v1 (480p) | Passive video, deferred | `data/charades/raw/Charades_v1_480/`, 9,848 mp4 files, about 16 GB, plus annotations | Zip integrity test passed; every CSV id has exactly one video; archive SHA256 in `data/charades/SHA256SUMS` |

## LeWM PushT (official)

Source: Hugging Face dataset `quentinll/lewm-pusht`, revision
`655cd446b9929369d7d406001da85c15d1457850`, file `pusht_expert_train.h5.zst`
(13,136,247,974 bytes, pinned SHA256 `7cfbd6d9…d212f318`, see
`configs/datasets/pusht.yaml`).

State on 2026-09-05 at 14:55: the Hugging Face download was interrupted by the
computer crash at 14:26 with 11,728,882,854 bytes received in 16 gaps. Two range
recovery attempts ran in parallel afterwards, one from this session and one
from a concurrent Codex session, and were stopped to avoid duplicate work; the
partially filled `pusht_expert_train.h5.zst.recovery` file was left in place.
The Codex session then resumed the official `hf download` into the original
`.incomplete` file at 14:54 and owns the rest of the recovery. Acceptance is
unchanged: the pinned SHA256 must match before `scripts/prepare_data.py`
decompresses the archive and writes `extraction.json`. The archive header
reports 18,685 episodes and 2,336,736 frames; nothing from the full archive has
been trained on or evaluated. Bounded checks used the 8-episode prefix subset.

## LeWM TwoRoom (official)

Source: Hugging Face dataset `quentinll/lewm-tworooms`, revision
`6903a2de048b13819d812da0b4dd661290bc01e4`, file `tworoom.tar.zst`
(3,425,937,909 bytes, pinned SHA256 `494b1a02…f8d080ca`, see
`configs/datasets/tworoom.yaml`).

The Hugging Face download was interrupted with gaps. An independent copy was
assembled from the received extents plus exact HTTP range requests for the
missing bytes, and accepted only because the whole-file SHA256 matched
(`data/tworoom/recovery.json`). `scripts/prepare_data.py` re-verified the hash
and extracted the single `tworoom.h5` member; `data/tworoom/extraction.json`
records the accepted revision and hash. The extracted HDF5 holds `pixels`
(920,809 × 224 × 224 × 3), `action` (920,809 × 2), `ep_len` and `ep_offset`
(10,000 each) plus agent and target positions. Its own SHA256 equals the hash of
the member streamed directly from the archive, so extraction is byte-exact.

The full TwoRoom archive has not yet been trained on or evaluated; the earlier
400-update check used the 32-episode prefix subset.

## Charades v1

Source: Allen Institute for AI, https://prior.allenai.org/projects/charades.
Files come from `https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/`:
`Charades.zip` (annotations, 3,519,822 bytes) and `Charades_v1_480.zip`
(480p videos, 16,339,546,533 bytes). `data/charades/download.sh` performs the
download, zip integrity test, extraction and checksum recording;
`data/charades/download.log` is its log.

Contents: 9,848 indoor activity videos (7,985 train, 1,863 test rows in
`Charades_v1_train.csv` and `Charades_v1_test.csv`) with free-text scripts,
scene labels, object lists and 157 temporally localised action classes.
Annotations are extracted to `data/charades/metadata/Charades/`; videos extract
to `data/charades/raw/`.

License: non-commercial research use only, no redistribution or public
modification of the data (`data/charades/metadata/Charades/license.txt`). Cite
Sigurdsson et al., "Hollywood in Homes", arXiv:1604.01753.

Role: passive third-person video with language and action annotations for the
passive-observation hypothesis (H3) in [ideas.md](ideas.md). It is not part of
the LeWM baseline and no training or preprocessing has been run on it.

Acquisition state: the annotation zip was fetched on 2026-09-05 at 12:40. The
video zip download was interrupted by the computer crash at 14:26 with
13,891,344,780 bytes received, resumed with `wget -c` at 14:41, and completed at
15:01. `unzip -t` passed at 15:03 and extraction finished at 15:06 with 9,848
mp4 files. The 9,848 ids in the train and test CSVs map one to one onto the
extracted videos with none missing or extra. The archive SHA256 values recorded
in `data/charades/SHA256SUMS` are local receipts; the source site publishes no
reference checksum, so acceptance rests on the zip integrity test and the
complete id coverage.

## TAU Urban Audio-Visual Scenes 2021

Source: Zenodo record 4477542, development set, 24 audio and video archives.
`data/tau_urban_av_2021/metadata/meta.csv` lists 12,291 paired 10-second audio
and video clips with scene labels. All archives were verified and extracted on
2026-09-05 (`download-full.log`). Passive data for H3; not used by the baseline.

## Diffusion Policy PushT (cchi)

Source: `https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip`,
SHA256 `63d52a11…d147bdf9`, converted by `scripts/convert_pusht_cchi.py` to the
shared HDF5 schema (206 episodes, 25,650 frames, absolute target actions in
pixels, 96-pixel frames). See [baseline.md](baseline.md) for why it is only an
early development dataset and not the LeWM protocol.
