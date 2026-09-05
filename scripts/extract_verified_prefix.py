"""Extract complete early episodes from a downloaded HDF5 byte prefix.

Only chunks wholly inside the downloaded prefix are accepted. The sparse HDF5
view is used to read its original metadata, never as training data. Full archive
verification remains pending; this subset is only for bounded alignment checks.
"""
from pathlib import Path
import hashlib,json
import h5py,hdf5plugin
import numpy as np

valid_bytes=30670848
source=Path('/tmp/pusht-prefix.h5')
output=Path('data/pusht_prefix/episodes.h5')

def verified(d,start,stop):
    chunk=d.chunks[0]
    for row in range(start//chunk*chunk,stop,chunk):
        ci=d.id.get_chunk_info_by_coord((row,)+(0,)*(d.ndim-1))
        if ci.byte_offset is None or ci.byte_offset+ci.size>valid_bytes:
            raise ValueError(f'{d.name} row {row} lies outside the verified prefix')
    return d[start:stop]

with h5py.File(source,'r') as f:
    lengths=verified(f['ep_len'],0,8)
    offsets=verified(f['ep_offset'],0,8)
    end=int(offsets[-1]+lengths[-1])
    data={k:verified(f[k],0,end) for k in ['pixels','action','state','proprio','episode_idx','step_idx']}
with h5py.File(output,'w') as f:
    f['ep_len']=lengths;f['ep_offset']=offsets
    for k,v in data.items():f.create_dataset(k,data=v,compression='lzf')
receipt=dict(source='https://huggingface.co/datasets/quentinll/lewm-pusht/resolve/655cd446b9929369d7d406001da85c15d1457850/pusht_expert_train.h5.zst',
    archive_byte_range=[0,8388607],archive_prefix_sha256=hashlib.file_digest(open('/tmp/pusht-prefix.zst','rb'),'sha256').hexdigest(),
    valid_decoded_bytes=valid_bytes,episodes=len(lengths),frames=end,
    checks='Every extracted HDF5 data chunk wholly inside downloaded byte prefix',
    scope='Early complete episodes for alignment; not full archive or representative training population')
output.with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
