"""Portable continuation preserves exact bytes and refuses unsafe restoration."""
import json
from pathlib import Path

import pytest

from scripts import session_handoff as handoff


def source_tree(tmp_path):
    root = tmp_path / 'source'
    paths = ['data/paddle/baseline/example.npz', 'runs/paddle/example/last.pt', 'runs/paddle/example/best.pt']
    for path, data in zip(paths, (b'pixels-actions-state' * 31, b'checkpoint-state' * 47, b'checkpoint-state' * 47)):
        target = root / path; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(data)
    return root, paths


def test_chunked_package_roundtrip_deduplicates_and_preserves_exact_paths(tmp_path):
    root, paths = source_tree(tmp_path)
    package = tmp_path / 'package'
    manifest = handoff.build_package(root, paths, package, part_bytes=101, metadata={'example': 'fixture'})
    assert len(manifest['files']) == 3 and len({r['sha256'] for r in manifest['files']}) == 2
    assert all((package / p['path']).stat().st_size <= 101 for p in manifest['parts'])
    destination = tmp_path / 'home'
    result = handoff.restore_package(package, destination)
    assert result['restored'] == 3
    for path in paths: assert (destination / path).read_bytes() == (root / path).read_bytes()
    assert handoff.restore_package(package, destination)['restored'] == 0


def test_corrupt_chunk_is_rejected_before_any_destination_file_is_written(tmp_path):
    root, paths = source_tree(tmp_path); package = tmp_path / 'package'
    manifest = handoff.build_package(root, paths, package, part_bytes=101)
    first = package / manifest['parts'][0]['path']
    first.write_bytes(first.read_bytes()[:-1] + b'!')
    destination = tmp_path / 'home'
    with pytest.raises(ValueError, match='hash|checksum|digest'):
        handoff.restore_package(package, destination)
    assert not list(destination.rglob('*.pt'))


def test_restore_refuses_traversal_symlinks_and_conflicting_existing_files(tmp_path):
    root, paths = source_tree(tmp_path); package = tmp_path / 'package'
    handoff.build_package(root, paths, package)
    destination = tmp_path / 'home'; (destination / paths[0]).parent.mkdir(parents=True)
    (destination / paths[0]).write_bytes(b'local experiment')
    with pytest.raises(ValueError, match='conflict|different|overwrite'):
        handoff.restore_package(package, destination)
    assert (destination / paths[0]).read_bytes() == b'local experiment'
    other = tmp_path / 'other'; other.mkdir()
    (destination / paths[0]).unlink(); (destination / 'runs').symlink_to(other, target_is_directory=True)
    with pytest.raises(ValueError, match='symlink|outside|unsafe'):
        handoff.restore_package(package, destination)
    manifest_path = package / 'manifest.json'; manifest = json.loads(manifest_path.read_text())
    manifest['files'][0]['path'] = '../escape.npz'; manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='path|outside|unsafe'):
        handoff.restore_package(package, tmp_path / 'another-home')
    assert not (tmp_path / 'escape.npz').exists()


def test_packaging_rejects_unrelated_private_or_cache_paths(tmp_path):
    root, paths = source_tree(tmp_path)
    for invalid in ['.codex/auth.json', '.idea/workspace.xml', 'data/old_private.txt',
                    'data/paddle/frame_cache/pixels.npy', 'runs/paddle/observer_cache_train/0.npz']:
        path = root / invalid; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b'not transferable')
        with pytest.raises(ValueError, match='path|scope|cache|private'):
            handoff.build_package(root, [invalid], tmp_path / invalid.replace('/', '_'))
    link = root / 'data/paddle/baseline/link.npz'; link.symlink_to(root / paths[0])
    with pytest.raises(ValueError, match='symlink'):
        handoff.build_package(root, ['data/paddle/baseline/link.npz'], tmp_path / 'link-package')


def test_inventory_resolves_selected_snapshot_relative_to_stage_not_checkpoint_directory(tmp_path, monkeypatch):
    import torch
    root = tmp_path / 'repository'; stage = root / 'runs/paddle/example'
    (stage / 'checkpoints').mkdir(parents=True)
    value = {'best_checkpoint': 'checkpoints/best_00000004.pt', 'models': {'U': torch.zeros(1)}}
    torch.save(value, stage / 'last.pt'); torch.save(value, stage / value['best_checkpoint'])
    torch.save({'models': {'U': torch.ones(1)}}, stage / 'checkpoints/best_00000000.pt')
    monkeypatch.setattr(handoff, 'environment_manifest', lambda root: {})
    rows = handoff.inventory(root)['files']
    assert {row['path'] for row in rows} == {
        'runs/paddle/example/last.pt', 'runs/paddle/example/checkpoints/best_00000004.pt',
        'runs/paddle/example/checkpoints/best_00000000.pt'}


def test_build_refuses_active_trainers_and_detects_source_mutation(tmp_path, monkeypatch):
    import fcntl
    root, paths = source_tree(tmp_path); package = tmp_path / 'package'
    with (root / 'runs/paddle/example/.stage.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match='Active trainer'):
            handoff.build_package(root, paths, package)
    original = handoff.tarfile.TarFile.addfile
    def mutate_after_read(tar, member, source):
        original(tar, member, source)
        Path(source.name).write_bytes(b'changed during packaging')
    monkeypatch.setattr(handoff.tarfile.TarFile, 'addfile', mutate_after_read)
    with pytest.raises(ValueError, match='source changed'):
        handoff.build_package(root, paths, package)
    assert not (package / 'manifest.json').exists()
