"""Download pinned source archives, verify content hashes, and extract safely."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import tarfile
import yaml
import zstandard


def prepare(config):
    cfg=yaml.safe_load(Path(config).read_text())
    if cfg['kind'] != 'action_trajectory':
        raise ValueError('Passive data requires a separate future learning protocol')
    directory=Path(cfg['path']).parent
    directory.mkdir(parents=True,exist_ok=True)
    archive=directory/cfg['archive']
    if not archive.exists():
        subprocess.run(['hf','download',cfg['repo'],cfg['archive'],'--repo-type','dataset',
            '--revision',cfg['revision'],'--local-dir',str(directory)],check=True)
    digest=hashlib.file_digest(archive.open('rb'),'sha256').hexdigest()
    if digest != cfg['sha256']: raise ValueError('Source archive checksum mismatch')
    done=directory/'extraction.json'
    if done.exists():
        print('Already extracted',done,flush=True)
        return
    with archive.open('rb') as source, zstandard.ZstdDecompressor().stream_reader(source) as reader:
        if '.tar.' in archive.name:
            with tarfile.open(fileobj=reader,mode='r|') as tf:
                tf.extractall(directory,filter='data')
        else:
            target=directory/archive.name.removesuffix('.zst')
            partial=target.with_suffix(target.suffix+'.partial')
            with partial.open('wb') as dest:
                import shutil
                shutil.copyfileobj(reader,dest,length=16*1024**2)
            partial.replace(target)
    import json
    done.write_text(json.dumps(dict(repo=cfg['repo'],revision=cfg['revision'],sha256=digest),indent=2)+'\n')
    print('Extracted',directory,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('config');prepare(p.parse_args().config)
