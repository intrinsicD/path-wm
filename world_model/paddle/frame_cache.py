"""Optional derived raw RGB/state/action cache; source NPZ episodes stay immutable.

Files contain the exact uint8 pixels, float64 simulator labels, and int64 action
IDs. No feature encoding, normalization, sampling, or random stream is changed.
Read-only memory maps fault in requested frame pages rather than constructing an
all-population RAM array. Builders decode one episode at a time and publish the
complete directory atomically, after writing its checksummed manifest.
"""

from __future__ import annotations

import hashlib
import errno
import json
import mmap
import os
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np


SCHEMA_VERSION = "paddle-raw-frame-cache-v1-rgb-u8-state-f64-action-i64"


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _file_hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _identity(dataset):
    return {"schema_version": SCHEMA_VERSION, "dataset_fingerprint": dataset.fingerprint,
            "split": dataset.split,
            "episode_order": [{key: entry[key] for key in ("path", "sha256", "seed", "frames", "transitions")}
                              for entry in dataset.entries]}


def _flush(arrays, release_pages=False):
    for array in arrays.values():
        array.flush()
        if release_pages and hasattr(array._mmap, "madvise"):
            array._mmap.madvise(mmap.MADV_DONTNEED)


def _close(arrays):
    for array in arrays.values():
        array._mmap.close()


def build_frame_cache(dataset, directory):
    """Stream the exact source population, or verify and reuse an existing cache.

    Return its immutable manifest. The caller retains its original EpisodeDataset
    and uses FrameCache only as a storage optimization for those exact rows.
    """
    directory = Path(directory)
    if directory.exists():
        existing = FrameCache(dataset, directory)
        manifest = existing.manifest
        existing.close()
        return manifest
    if not len(dataset):
        raise ValueError("Cannot build a raw frame cache for an empty split")
    directory.parent.mkdir(parents=True, exist_ok=True)
    identity = _identity(dataset)
    lengths = [int(entry["frames"]) for entry in dataset.entries]
    actions = [int(entry["transitions"]) for entry in dataset.entries]
    if any(length != count+1 for length, count in zip(lengths, actions)):
        raise ValueError("Source manifest violates T actions/T+1 frames ordering")
    frame_offsets = np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))
    action_offsets = np.concatenate(([0], np.cumsum(actions, dtype=np.int64)))
    specifications = {
        "frames": ((int(frame_offsets[-1]), 64, 64, 3), np.dtype("uint8")),
        "states": ((int(frame_offsets[-1]), 5), np.dtype("float64")),
        "actions": ((int(action_offsets[-1]),), np.dtype("int64")),
    }
    payload_bytes = sum(int(np.prod(shape))*dtype.itemsize for shape, dtype in specifications.values())
    if shutil.disk_usage(directory.parent).free < payload_bytes + 64*2**20:
        raise OSError(f"Raw frame cache needs {payload_bytes:,} bytes plus a 64 MiB build margin")
    temporary = Path(tempfile.mkdtemp(prefix=f".{directory.name}.building-", dir=directory.parent))
    arrays, source_files = {}, []
    begin = time.perf_counter()
    try:
        arrays = {name: np.lib.format.open_memmap(temporary/f"{name}.npy", mode="w+", dtype=dtype, shape=shape)
                  for name, (shape, dtype) in specifications.items()}
        for index, entry in enumerate(dataset.entries):
            source = dataset.path/entry["path"]
            before = source.stat()
            if _file_hash(source) != entry["sha256"]:
                raise ValueError(f"Source episode fingerprint mismatch while building cache: {source}")
            episode = dataset[index]
            after = source.stat()
            if (before.st_size,before.st_mtime_ns) != (after.st_size,after.st_mtime_ns):
                raise ValueError(f"Source episode changed during cache build: {source}")
            for name, (shape, dtype) in specifications.items():
                count = actions[index] if name == "actions" else lengths[index]
                if episode[name].shape != (count,*shape[1:]) or episode[name].dtype != dtype:
                    raise ValueError(f"Source {name} dtype/shape differs from raw-cache schema in {source}")
                offsets = action_offsets if name == "actions" else frame_offsets
                arrays[name][offsets[index]:offsets[index+1]] = episode[name]
            source_files.append({"path": entry["path"], "bytes": after.st_size, "mtime_ns": after.st_mtime_ns})
            # Bound resident dirty mappings while the OS retains its normal page cache.
            if (index+1) % 128 == 0:
                _flush(arrays, release_pages=True)
            if (index+1) % 1000 == 0:
                print(f"raw frame cache {dataset.split}: {index+1}/{len(dataset)} episodes", flush=True)
        _flush(arrays)
        _close(arrays)
        arrays = {}
        files = {}
        for name, (shape, dtype) in specifications.items():
            path = temporary/f"{name}.npy"
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
            files[name] = {"path": path.name, "shape": list(shape), "dtype": dtype.str,
                           "bytes": path.stat().st_size, "sha256": _file_hash(path)}
        manifest = {**identity, "identity_fingerprint": _fingerprint(identity),
                    "frame_offsets": frame_offsets.tolist(), "action_offsets": action_offsets.tolist(),
                    "files": files, "source_files": source_files,
                    "payload_bytes": payload_bytes, "build_seconds": time.perf_counter()-begin,
                    "normalization": "none; immutable original pixel/state/action dtypes and values"}
        manifest["manifest_fingerprint"] = _fingerprint(manifest)
        with (temporary/"manifest.json").open("w") as handle:
            json.dump(manifest,handle,indent=2,sort_keys=True,allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            temporary.rename(directory)
        except OSError as error:
            # Another builder may finish first; never replace its completed output.
            if error.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                raise
            existing = FrameCache(dataset,directory)
            manifest = existing.manifest
            existing.close()
        return manifest
    finally:
        _close(arrays)
        if temporary.exists():
            shutil.rmtree(temporary)


class FrameCache:
    """Read-only maps for one exact ordered split of an immutable source dataset."""

    def __init__(self, dataset, directory, *, verify_hashes=True):
        self.directory = Path(directory)
        path = self.directory/"manifest.json"
        if not path.exists():
            raise ValueError(f"Raw frame cache has no complete manifest: {self.directory}")
        self.manifest = json.loads(path.read_text())
        identity = _identity(dataset)
        if (self.manifest.get("schema_version") != SCHEMA_VERSION
                or self.manifest.get("identity_fingerprint") != _fingerprint(identity)
                or any(self.manifest.get(key) != value for key,value in identity.items())):
            raise ValueError("Raw frame cache schema/split/episode-order identity fingerprint mismatch")
        unsigned = {key:value for key,value in self.manifest.items() if key != "manifest_fingerprint"}
        if self.manifest.get("manifest_fingerprint") != _fingerprint(unsigned):
            raise ValueError("Raw frame cache manifest fingerprint mismatch")
        self.fingerprint = self.manifest["identity_fingerprint"]
        self.lengths = [entry["frames"] for entry in dataset.entries]
        self.frame_offsets = np.asarray(self.manifest["frame_offsets"],dtype=np.int64)
        self.action_offsets = np.asarray(self.manifest["action_offsets"],dtype=np.int64)
        expected_frame_offsets = np.concatenate(([0],np.cumsum(self.lengths,dtype=np.int64)))
        expected_action_offsets = np.concatenate(([0],np.cumsum([entry["transitions"] for entry in dataset.entries],dtype=np.int64)))
        if not np.array_equal(self.frame_offsets,expected_frame_offsets) or not np.array_equal(self.action_offsets,expected_action_offsets):
            raise ValueError("Raw frame/action offsets differ from source ordering")
        for entry,facts in zip(dataset.entries,self.manifest["source_files"]):
            source = dataset.path/entry["path"]
            current = source.stat()
            if (facts["path"] != entry["path"]
                    or (current.st_size,current.st_mtime_ns) != (facts["bytes"],facts["mtime_ns"])):
                if _file_hash(source) != entry["sha256"]:
                    raise ValueError(f"Raw cache source episode fingerprint mismatch: {source}")
        self._arrays = {}
        try:
            for name,metadata in self.manifest["files"].items():
                path = self.directory/metadata["path"]
                if path.stat().st_size != metadata["bytes"]:
                    raise ValueError(f"Raw frame cache array size differs: {path}")
                if verify_hashes and _file_hash(path) != metadata["sha256"]:
                    raise ValueError(f"Raw frame cache array fingerprint differs: {path}")
                array = np.load(path,mmap_mode="r",allow_pickle=False)
                if list(array.shape) != metadata["shape"] or array.dtype.str != metadata["dtype"]:
                    array._mmap.close()
                    raise ValueError(f"Raw frame cache array schema differs: {path}")
                self._arrays[name] = array
        except BaseException:
            self.close()
            raise
        self.frames, self.states, self.actions = (self._arrays[key] for key in ("frames","states","actions"))

    def close(self):
        """Release mappings after all views obtained from this cache are no longer used."""
        _close(self._arrays)
        self._arrays = {}

    def frame_batch(self, indices):
        """Flat indices use episode order then within-episode observation order."""
        indices = np.asarray(indices)
        if indices.ndim != 1 or not np.issubdtype(indices.dtype,np.integer):
            raise ValueError("Frame-cache indices must be a one-dimensional integer array")
        if np.any(indices < 0) or np.any(indices >= len(self.frames)):
            raise IndexError("Frame-cache index outside this split")
        return self.frames[indices], self.states[indices]

    def episode(self, index):
        """Training inputs only: raw frames, aligned states, and T executable actions."""
        index = int(index)
        if not 0 <= index < len(self.lengths):
            raise IndexError("Episode-cache index outside this split")
        start,stop = self.frame_offsets[index:index+2]
        action_start,action_stop = self.action_offsets[index:index+2]
        return {"frames":self.frames[start:stop],"states":self.states[start:stop],
                "actions":self.actions[action_start:action_stop]}
