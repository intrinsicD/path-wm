"""Pack/restore the explicit current world-model evidence without changing sources.

Restoration uses only the standard library. Inventory additionally reads trusted
local Torch checkpoints to retain their immutable selected/parent dependencies.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile

SCHEMA = 'path-wm-session-handoff-v1'
PART_BYTES = 45_000_000
_ROOT_FILES = {'runs/experiment_dashboard.html', 'runs/experiment_dashboard.artifact.json',
               'runs/experiment_dashboard.receipt.json', 'runs/pusht_cchi_data_audit_2026-09-07.json',
               'runs/pusht_claude_design_prompt.txt', 'runs/pusht_claude_design_review.json'}
_PREFIXES = ('data/paddle/', 'data/pusht_world_model/', 'data/pusht_cchi/',
             'runs/paddle/', 'runs/pusht_world_model/')
_DIGEST = re.compile(r'[0-9a-f]{64}')


def _allowed(relative):
    if not isinstance(relative, str) or '\\' in relative:
        raise ValueError('unsafe package path')
    path = PurePosixPath(relative)
    if path.is_absolute() or str(path) != relative or any(p in ('.', '..') or p.startswith('.') for p in path.parts):
        raise ValueError(f'unsafe package path: {relative}')
    if relative not in _ROOT_FILES and not relative.startswith(_PREFIXES):
        raise ValueError(f'path outside current-task scope: {relative}')
    if any(p in ('__pycache__', 'frame_cache', 'prior_dashboard_2026-09-07') or p.startswith('observer_cache')
           for p in path.parts):
        raise ValueError(f'private/cache path excluded: {relative}')
    return relative


def _local(root, relative):
    _allowed(relative)
    root = Path(root).absolute()
    current = root
    if root.is_symlink():
        raise ValueError('unsafe symlink destination/source root')
    for part in PurePosixPath(relative).parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f'unsafe symlink path: {relative}')
    if not current.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'path outside root: {relative}')
    return current


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _facts(path):
    value = path.stat()
    if not stat.S_ISREG(value.st_mode):
        raise ValueError(f'package source must be a regular file: {path}')
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def _lock_trainers(root, paths, stack):
    try:
        import fcntl
    except ImportError:
        return
    locks = set()
    # A stage created after the inventory must still block premature closure.
    for relative in ('runs/paddle', 'runs/pusht_world_model'):
        if (root / relative).exists():
            locks.update((root / relative).rglob('.stage.lock'))
    for relative in paths:
        for parent in (root / relative).parents:
            if parent == root:
                break
            lock = parent / '.stage.lock'
            if lock.is_file():
                locks.add(lock)
    for path in sorted(locks):
        stream = stack.enter_context(path.open('rb'))
        try:
            fcntl.flock(stream, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f'Active trainer owns {path.parent}; finish/pause writers before packaging') from error


def build_package(root, paths, output, part_bytes=PART_BYTES, metadata=None):
    """Build exact object archive from a reviewed relative-file selection."""
    if isinstance(part_bytes, bool) or not isinstance(part_bytes, int) or not 0 < part_bytes <= PART_BYTES:
        raise ValueError('part_bytes must be a positive integer no greater than 45,000,000')
    root, output = Path(root).absolute(), Path(output)
    paths = sorted(set(_allowed(path) for path in paths))
    if not paths:
        raise ValueError('empty package selection')
    if output.exists() and any(output.iterdir()):
        raise ValueError('package output already contains files; use a new directory')
    output.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack, tempfile.TemporaryDirectory(dir=output.parent, prefix='.handoff-build-') as temporary:
        _lock_trainers(root, paths, stack)
        facts = {name: _facts(_local(root, name)) for name in paths}
        rows, objects = [], {}
        for name in paths:
            source = _local(root, name); digest = _sha(source)
            if _facts(source) != facts[name]:
                raise ValueError(f'source changed while hashing: {name}')
            rows.append({'path': name, 'bytes': facts[name][2], 'sha256': digest,
                         'mtime_ns': facts[name][3], 'mode': stat.S_IMODE(source.stat().st_mode) & 0o777})
            objects.setdefault(digest, name)
        temporary = Path(temporary); archive = temporary / 'objects.tar.gz'
        with archive.open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0, filename='', compresslevel=6) as compressed:
            with tarfile.open(fileobj=compressed, mode='w|') as tar:
                for digest, name in sorted(objects.items()):
                    info = tarfile.TarInfo(f'objects/{digest}')
                    info.size = facts[name][2]; info.mode = 0o600; info.mtime = 0
                    with _local(root, name).open('rb') as source:
                        tar.addfile(info, source)
                    # Include content checking, not just timestamps, for the actual archived object.
                    if _sha(_local(root, name)) != digest:
                        raise ValueError(f'source changed while packaging: {name}')
        if any(_facts(_local(root, name)) != facts[name] for name in paths):
            raise ValueError('source changed during package construction; no package committed')
        parts = []
        with archive.open('rb') as stream:
            while block := stream.read(part_bytes):
                name = f'part-{len(parts):04d}.bin'; target = temporary / name
                target.write_bytes(block)
                parts.append({'path': name, 'bytes': len(block), 'sha256': hashlib.sha256(block).hexdigest()})
        manifest = {'schema': SCHEMA, 'compression': 'gzip6/tar SHA256 objects',
                    'files': rows, 'parts': parts, 'metadata': metadata or {},
                    'logical_bytes': sum(r['bytes'] for r in rows),
                    'unique_object_bytes': sum(facts[n][2] for n in objects.values()),
                    'compressed_bytes': sum(p['bytes'] for p in parts), 'part_limit': part_bytes}
        _write_json(temporary / 'manifest.json', manifest)
        output.mkdir(parents=True, exist_ok=True)
        for part in parts:
            (temporary / part['path']).replace(output / part['path'])
        # Manifest publication is the package completion boundary.
        (temporary / 'manifest.json').replace(output / 'manifest.json')
    return manifest


def restore_package(package, destination=None):
    """Verify all archived bytes before restoring; existing different files survive."""
    package = Path(package); manifest = json.loads((package / 'manifest.json').read_text())
    if manifest.get('schema') != SCHEMA:
        raise ValueError('incompatible handoff manifest schema')
    rows, objects, paths = manifest['files'], {}, set()
    for row in rows:
        name = _allowed(row['path'])
        if name in paths or not _DIGEST.fullmatch(row['sha256']) or not isinstance(row['bytes'], int) or row['bytes'] < 0:
            raise ValueError('invalid or duplicate file manifest entry')
        paths.add(name)
        if row['sha256'] in objects and objects[row['sha256']] != row['bytes']:
            raise ValueError('object hash has inconsistent sizes')
        objects[row['sha256']] = row['bytes']
        if destination is not None:
            _local(destination, name)
    with tempfile.TemporaryDirectory(prefix='path-wm-restore-') as temporary:
        temporary = Path(temporary); archive = temporary / 'objects.tar.gz'
        with archive.open('wb') as joined:
            for index, part in enumerate(manifest['parts']):
                if part['path'] != f'part-{index:04d}.bin' or not 0 < part['bytes'] <= PART_BYTES:
                    raise ValueError('unsafe or invalid archive part path/size')
                source = package / part['path']
                if source.is_symlink() or not source.is_file():
                    raise ValueError(f'missing/unsafe archive part: {part["path"]}')
                if source.stat().st_size != part['bytes'] or _sha(source) != part['sha256']:
                    raise ValueError(f'archive part checksum mismatch: {part["path"]}')
                with source.open('rb') as stream:
                    shutil.copyfileobj(stream, joined)
        verified = set()
        with tarfile.open(archive, mode='r|gz') as tar:
            for member in tar:
                digest = member.name.removeprefix('objects/')
                if (not member.isfile() or not member.name.startswith('objects/') or not _DIGEST.fullmatch(digest)
                        or digest not in objects or digest in verified or member.size != objects[digest]):
                    raise ValueError('unexpected or unsafe archive object')
                with tar.extractfile(member) as source, (temporary / digest).open('wb') as target:
                    shutil.copyfileobj(source, target)
                if _sha(temporary / digest) != digest:
                    raise ValueError('archive object checksum mismatch')
                verified.add(digest)
        if verified != set(objects):
            raise ValueError('archive is missing expected objects')
        targets = []
        if destination is not None:
            # Preflight every path and every conflict before installing any file.
            for row in rows:
                target = _local(destination, row['path'])
                if target.exists():
                    if not target.is_file() or target.stat().st_size != row['bytes'] or _sha(target) != row['sha256']:
                        raise ValueError(f'conflicting existing file; refusing overwrite: {row["path"]}')
                else:
                    targets.append((row, target))
            for row, target in targets:
                _local(destination, row['path']); target.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
                    staged = Path(stream.name)
                    with (temporary / row['sha256']).open('rb') as source:
                        shutil.copyfileobj(source, stream)
                    stream.flush(); os.fsync(stream.fileno())
                staged.chmod(int(row.get('mode', 0o644)) & 0o777)
                if 'mtime_ns' in row:
                    os.utime(staged, ns=(row['mtime_ns'], row['mtime_ns']))
                staged.replace(target)
    return {'verified_files': len(rows), 'verified_objects': len(objects),
            'restored': len(targets), 'logical_bytes': manifest['logical_bytes']}


def environment_manifest(root):
    """Capture versions only; never environment variables, index URLs or auth files."""
    result = {'python': sys.version, 'platform': platform.platform(), 'machine': platform.machine(),
              'distributions': sorted([{'name': d.metadata.get('Name'), 'version': d.version}
                                      for d in importlib.metadata.distributions()], key=lambda d: (d['name'] or '').lower())}
    for key, args in (('git_revision', ['git', 'rev-parse', 'HEAD']),
                      ('git_branch', ['git', 'branch', '--show-current']), ('remote_names', ['git', 'remote'])):
        result[key] = subprocess.check_output(args, cwd=root, text=True).strip()
    for command in ('node', 'chromium', 'chromium-browser', 'google-chrome', 'nvidia-smi'):
        if shutil.which(command):
            args = [command, '--version'] if command != 'nvidia-smi' else [command, '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader']
            checked = subprocess.run(args, capture_output=True, text=True)
            if checked.returncode == 0: result[command] = checked.stdout.strip()
    try:
        import torch
        result['torch'] = torch.__version__; result['cuda_runtime'] = torch.version.cuda
    except ImportError:
        pass
    return result


def inventory(root):
    """Generate an explicit reviewable allowlist; no archive or source mutations."""
    root = Path(root).absolute(); selected, checkpoints = set(), set()
    data_roots = [root / name for name in ('data/paddle/baseline', 'data/paddle/smoke', 'data/paddle/smoke_v2',
                                           'data/pusht_world_model')]
    run_roots = [root / name for name in ('runs/paddle', 'runs/pusht_world_model')]
    for folder in data_roots + run_roots:
        if not folder.exists(): continue
        for path in folder.rglob('*'):
            if not path.is_file() or path.is_symlink(): continue
            name = path.relative_to(root).as_posix()
            try: _allowed(name)
            except ValueError: continue
            if '.tmp' in path.name or path.suffix in ('.pyc', '.lock'): continue
            if path.suffix == '.pt':
                if path.name in ('best.pt', 'last.pt', 'inference.pt') or path.stem.endswith('00000000'):
                    checkpoints.add(name)
                continue
            selected.add(name)
    selected.update(name for name in _ROOT_FILES if (root / name).is_file())
    for name in ('pusht_cchi.h5', 'conversion.json', 'acquisition.json', 'conversion.log'):
        relative = f'data/pusht_cchi/{name}'
        if (root / relative).is_file(): selected.add(relative)
    # Read current raw citations to retain diagnosable immutable checkpoints, too.
    pattern = re.compile(rb'runs/(?:paddle|pusht_world_model)/[^\s"\'<>\\]+\.pt')
    for name in sorted(selected):
        path = root / name
        if path.suffix not in ('.json', '.jsonl', '.md', '.py', '.txt', '.log'): continue
        with path.open('rb') as stream:
            overlap = b''
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                block = overlap + block
                for match in pattern.finditer(block):
                    candidate = match.group().decode('utf-8', errors='strict')
                    try: source = _local(root, candidate)
                    except ValueError: continue
                    if source.is_file(): checkpoints.add(candidate)
                overlap = block[-8192:]
    # Only locally generated trusted checkpoints are unpickled during inventory.
    if checkpoints:
        import torch
        pending = list(checkpoints); inspected = set()
        while pending:
            name = pending.pop()
            if name in inspected: continue
            inspected.add(name); value = torch.load(root / name, map_location='cpu', weights_only=False)
            relative = value.get('best_checkpoint') or value.get('best_snapshot')
            if relative:
                stage = Path(name).parent
                if stage.name == 'checkpoints':
                    stage = stage.parent
                candidate = (stage / relative).as_posix()
                if not _local(root, candidate).is_file():
                    raise ValueError(f'checkpoint dependency is missing: {candidate}')
                if candidate not in checkpoints: checkpoints.add(candidate); pending.append(candidate)
    selected.update(checkpoints)
    rows = [{'path': name, 'bytes': (root / name).stat().st_size,
             'reason': 'selected/resume/initial/referenced checkpoint' if name in checkpoints else 'exact current-task data or raw evidence'}
            for name in sorted(selected)]
    return {'schema': SCHEMA, 'status': 'review selection before packing; writers may still be active',
            'files': rows, 'logical_bytes': sum(r['bytes'] for r in rows), 'environment': environment_manifest(root)}


def verify_restored(root):
    """Offline source/data verification and CPU forward smoke from restored bytes."""
    import time
    import numpy as np
    import torch
    from world_model.paddle import checkpoints as paddle_checkpoints, data as paddle_data
    from world_model.pusht import checkpoints as pusht_checkpoints, data as pusht_data
    from world_model.pusht.perception_data import verify_supplement

    root = Path(root).absolute(); started = time.monotonic(); torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    print('Verifying all paddle source episodes by exact simulator replay...', flush=True)
    paddle = paddle_data.verify_dataset(root / 'data/paddle/baseline')
    print('Verifying all PushT pixels/actions/labels against the relocated exact HDF5...', flush=True)
    pusht = pusht_data.verify_dataset(root / 'data/pusht_world_model/cchi_v1',
                                     source=root / 'data/pusht_cchi/pusht_cchi.h5')
    supplement = verify_supplement(root / 'data/pusht_world_model/pose_supplement_v1')
    expected = ('c6d255dc9919b1bb9ce38182f9d180047f964754a32c616db072bc965a3be54f',
                'e6ff0cad101c15ba3e2b839df28bc0bd75dca4063a7d6293141976ab44b4fa9c',
                '34e6e5a352a1355fd6ac7a1a6471bb77f5e5a466e18bf9882a8d9215afb9b7f3')
    if tuple(x['fingerprint'] for x in (paddle, pusht, supplement)) != expected:
        raise ValueError('restored data fingerprints differ from this session handoff')
    source_frames = {
        'paddle': paddle_data.EpisodeDataset(root / 'data/paddle/baseline', 'train')[0]['frames'][0],
        'pusht': pusht_data.EpisodeDataset(root / 'data/pusht_world_model/cchi_v1', 'train')[0]['frames'][0]}
    bundles = []
    for task, relative, purpose in (
        ('paddle', 'runs/paddle/continuation_v1/inference.pt', 'trained paddle continuation; control limitations retained'),
        ('paddle', 'runs/paddle/smoke_v2/inference.pt', 'software smoke only'),
        ('pusht', 'runs/pusht_world_model/smoke/inference.pt', 'original software smoke; failed evaluation receipt retained'),
        ('pusht', 'runs/pusht_world_model/smoke_v2/inference.pt', 'corrected software smoke only; no passing trained PushT bundle')):
        print(f'Loading and exercising {relative} on CPU...', flush=True)
        loader = paddle_checkpoints if task == 'paddle' else pusht_checkpoints
        system = loader.load_bundle(root / relative, device='cpu')
        frame = torch.from_numpy(np.array(source_frames[task], copy=True)).permute(2, 0, 1)[None].float() / 255
        previous = torch.zeros(1, 3) if task == 'paddle' else torch.full((1, 2), -1.)
        action = torch.tensor([[0., 1., 0.]]) if task == 'paddle' else torch.tensor([[.5, .5]])
        with torch.no_grad():
            observation = system['E'](frame)
            memory = system['U'](torch.zeros(1, 128), observation, previous)
            predicted = system['P'](observation, memory, action)
            imagined_memory = system['U'](memory.clone(), predicted, action)
            outputs = {'fine': predicted.fine, 'coarse': predicted.coarse,
                       'memory': imagined_memory, 'decoded': system['D'](predicted),
                       'H': system['H'](predicted), 'R': system['R'](imagined_memory)}
        if any(not torch.isfinite(value).all() for value in outputs.values()):
            raise ValueError(f'nonfinite restored inference output: {relative}')
        bundles.append({'path': relative, 'sha256': _sha(root / relative), 'purpose': purpose,
                        'finite': True, 'output_shapes': {k: list(v.shape) for k, v in outputs.items()}})
    return {'schema': SCHEMA, 'passed': True, 'verification': 'offline exact source checks and CPU inference smoke',
            'root': str(root), 'paddle': paddle, 'pusht': pusht, 'supplement': supplement,
            'bundles': bundles, 'seconds': time.monotonic() - started,
            'limitations': 'Software/data continuity only; does not establish world-model control success or cross-hardware bitwise training.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    inv = commands.add_parser('inventory'); inv.add_argument('--root', type=Path, default=Path('.')); inv.add_argument('--output', type=Path, required=True)
    build = commands.add_parser('build'); build.add_argument('--root', type=Path, default=Path('.'))
    build.add_argument('--selection', type=Path, required=True); build.add_argument('--output', type=Path, required=True)
    offline = commands.add_parser('verify-restored'); offline.add_argument('--root', type=Path, default=Path('.'))
    offline.add_argument('--output', type=Path, required=True)
    for command in ('restore', 'verify', 'requirements'):
        action = commands.add_parser(command); action.add_argument('--package', type=Path, required=True)
        if command == 'restore': action.add_argument('--destination', type=Path, default=Path('.'))
    args = parser.parse_args(argv)
    if args.command == 'inventory':
        result = inventory(args.root); args.output.parent.mkdir(parents=True, exist_ok=True); _write_json(args.output, result)
        print(json.dumps({'selection': str(args.output), 'files': len(result['files']), 'bytes': result['logical_bytes']}))
    elif args.command == 'build':
        selection = json.loads(args.selection.read_text()); rows = selection['files']
        result = build_package(args.root, [r['path'] if isinstance(r, dict) else r for r in rows], args.output,
            metadata={'environment': environment_manifest(args.root), 'selection_sha256': _sha(args.selection)})
        print(json.dumps({k: result[k] for k in ('logical_bytes', 'unique_object_bytes', 'compressed_bytes', 'parts')}, indent=2))
    elif args.command == 'verify-restored':
        result = verify_restored(args.root); args.output.parent.mkdir(parents=True, exist_ok=True); _write_json(args.output, result)
        print(json.dumps({'receipt': str(args.output), 'passed': result['passed'], 'seconds': result['seconds']}))
    elif args.command == 'requirements':
        manifest = json.loads((args.package / 'manifest.json').read_text())
        for distribution in manifest['metadata']['environment']['distributions']:
            name, version = distribution['name'], distribution['version']
            if name.lower().replace('_', '-') == 'path-wm': continue
            if not re.fullmatch(r'[A-Za-z0-9_.-]+', name) or not re.fullmatch(r'[A-Za-z0-9_.+-]+', version):
                raise ValueError('invalid recorded distribution name/version')
            print(f'{name}=={version}')
    else:
        print(json.dumps(restore_package(args.package, getattr(args, 'destination', None)), indent=2))


if __name__ == '__main__':
    main()
