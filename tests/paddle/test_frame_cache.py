"""A raw disk cache must preserve source pixels, labels, ordering, and identity."""

import copy
import hashlib
import json

import numpy as np
import pytest
import torch

from world_model.paddle.data import EpisodeDataset, generate_dataset
from world_model.paddle.frame_cache import FrameCache, build_frame_cache
from world_model.paddle.training import images, normalized_states


@pytest.fixture
def population(tmp_path):
    source = tmp_path/"source"
    generate_dataset({"dataset":{"train":3,"validation":2,"test":1}}, source)
    return source


def source_hashes(dataset):
    return {entry["path"]: hashlib.sha256((dataset.path/entry["path"]).read_bytes()).hexdigest()
            for entry in dataset.entries}


def test_streamed_cache_preserves_bitwise_frames_states_actions_and_source(population,tmp_path,monkeypatch):
    dataset = EpisodeDataset(population,"train")
    before = source_hashes(dataset)
    source_episodes = [dataset[i] for i in range(len(dataset))]
    reads = []
    original = EpisodeDataset.__getitem__
    def observed(self,index):
        reads.append(index)
        return original(self,index)
    monkeypatch.setattr(EpisodeDataset,"__getitem__",observed)
    directory = tmp_path/"cache"
    manifest = build_frame_cache(dataset,directory)
    assert reads == [0,1,2]
    assert manifest["dataset_fingerprint"] == dataset.fingerprint
    assert manifest["split"] == "train"
    assert source_hashes(dataset) == before
    cache = FrameCache(dataset,directory)
    assert isinstance(cache.frames,np.memmap) and not cache.frames.flags.writeable
    assert isinstance(cache.states,np.memmap) and not cache.states.flags.writeable
    for index,source in enumerate(source_episodes):
        for field in ("frames","states","actions"):
            actual = cache.episode(index)[field]
            assert actual.dtype == source[field].dtype
            np.testing.assert_array_equal(actual,source[field])
        assert len(cache.episode(index)["frames"]) == len(cache.episode(index)["actions"])+1
    assert cache.frame_offsets[-1] == sum(len(e["frames"]) for e in source_episodes)
    assert cache.action_offsets[-1] == sum(len(e["actions"]) for e in source_episodes)
    frozen_bytes = (directory/"manifest.json").read_bytes()
    build_frame_cache(dataset,directory)
    assert (directory/"manifest.json").read_bytes() == frozen_bytes


def test_random_frame_batch_is_bitwise_identical_after_training_normalization(population,tmp_path,monkeypatch):
    dataset = EpisodeDataset(population,"train")
    episodes = [dataset[i] for i in range(len(dataset))]
    all_frames = np.concatenate([e["frames"] for e in episodes])
    all_states = np.concatenate([e["states"] for e in episodes])
    directory = tmp_path/"cache"
    build_frame_cache(dataset,directory)
    cache = FrameCache(dataset,directory)
    monkeypatch.setattr(EpisodeDataset,"__getitem__",lambda *args:pytest.fail("cached reads must not decompress source episodes"))
    rng = np.random.default_rng(128)
    indices = rng.integers(len(all_frames),size=100)
    frames,states = cache.frame_batch(indices)
    np.testing.assert_array_equal(frames,all_frames[indices])
    np.testing.assert_array_equal(states,all_states[indices])
    torch.testing.assert_close(images(frames,"cpu"),images(all_frames[indices],"cpu"),rtol=0,atol=0)
    torch.testing.assert_close(normalized_states(states,"cpu"),normalized_states(all_states[indices],"cpu"),rtol=0,atol=0)
    for invalid in ([-1], [len(all_frames)], [1.5]):
        with pytest.raises((ValueError,IndexError)):
            cache.frame_batch(invalid)


@pytest.mark.parametrize("change",["split","episode_order","fingerprint","schema"])
def test_cache_refuses_incompatible_population_or_order(population,tmp_path,change):
    dataset = EpisodeDataset(population,"train")
    directory = tmp_path/"cache"
    build_frame_cache(dataset,directory)
    changed = copy.copy(dataset)
    if change == "split":
        changed = EpisodeDataset(population,"validation")
    elif change == "episode_order":
        changed.entries = list(reversed(dataset.entries))
    elif change == "fingerprint":
        changed.fingerprint = "changed-population"
    else:
        path = directory/"manifest.json"
        manifest = json.loads(path.read_text())
        manifest["schema_version"] = "coarse-feature-cache-incompatible"
        path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match="identity|schema|fingerprint|split|ordering"):
        FrameCache(changed,directory)


def test_failed_stream_build_never_publishes_partial_cache(population,tmp_path,monkeypatch):
    dataset = EpisodeDataset(population,"train")
    original = EpisodeDataset.__getitem__
    def fail(self,index):
        if index == 1:
            raise RuntimeError("intentional interrupted source decode")
        return original(self,index)
    monkeypatch.setattr(EpisodeDataset,"__getitem__",fail)
    directory = tmp_path/"cache"
    with pytest.raises(RuntimeError,match="interrupted source"):
        build_frame_cache(dataset,directory)
    assert not directory.exists()
    assert source_hashes(dataset) == {entry["path"]:entry["sha256"] for entry in dataset.entries}
    monkeypatch.setattr(EpisodeDataset,"__getitem__",original)
    build_frame_cache(dataset,directory)
    assert FrameCache(dataset,directory).lengths == [entry["frames"] for entry in dataset.entries]


def test_cache_rejects_truncated_backing_array(population,tmp_path):
    dataset = EpisodeDataset(population,"train")
    directory = tmp_path/"cache"
    build_frame_cache(dataset,directory)
    path = directory/"frames.npy"
    with path.open("r+b") as handle:
        handle.truncate(path.stat().st_size-1)
    with pytest.raises(ValueError,match="array|size|fingerprint|cache"):
        FrameCache(dataset,directory)
