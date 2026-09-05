"""Convert the original Diffusion Policy PushT release to the shared H5 schema.

Preserves absolute target actions and 96px observations. This is a separate
validation dataset, not the LeWM release's relative-action training population.
"""
import hashlib,json
from pathlib import Path
import h5py
import numpy as np
import zarr

root=Path('data/pusht_cchi')
source=next(root.rglob('*.zarr'))
z=zarr.open(str(source),mode='r')
ends=z['meta/episode_ends'][:]
offsets=np.r_[0,ends[:-1]]
out=root/'pusht_cchi.h5'
with h5py.File(out.with_suffix('.partial'),'w') as f:
    f['ep_len']=ends-offsets;f['ep_offset']=offsets
    for src,dst in [('img','pixels'),('action','action'),('state','state')]:
        x=z['data'][src]
        d=f.create_dataset(dst,shape=x.shape,dtype=x.dtype,chunks=(1,*x.shape[1:]) if dst=='pixels' else True,
                           compression='lzf')
        for start in range(0,len(x),512):d[start:start+512]=x[start:start+512]
out.with_suffix('.partial').replace(out)
(root/'conversion.json').write_text(json.dumps(dict(
    source='https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip',
    archive_sha256=hashlib.file_digest((root/'pusht.zip').open('rb'),'sha256').hexdigest(),
    episodes=len(ends),frames=int(ends[-1]),action_semantics='absolute target XY in pixels',
    resolution=96),indent=2)+'\n')
print(out,flush=True)
