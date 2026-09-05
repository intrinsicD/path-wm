"""Create a bounded TwoRoom source subset with checked physical chunk extents."""
import hashlib,json
from pathlib import Path
import h5py,hdf5plugin,numpy as np
valid=126090752
out=Path('data/tworoom_prefix');out.mkdir(exist_ok=True)
with h5py.File('/tmp/tworoom-prefix.h5','r') as f:
    def read(name,start,stop):
        d=f[name]
        for row in range(start//d.chunks[0]*d.chunks[0],stop,d.chunks[0]):
            ci=d.id.get_chunk_info_by_coord((row,)+(0,)*(d.ndim-1))
            if ci.byte_offset is None or ci.byte_offset+ci.size>valid:
                raise ValueError(f'Unavailable chunk {name}/{row}')
        return d[start:stop]
    lengths,offsets=read('ep_len',0,32),read('ep_offset',0,32)
    end=int(offsets[-1]+lengths[-1])
    data={k:read(k,0,end) for k in ['pixels','action','proprio','pos_agent','pos_target','observation','step_idx','ep_idx']}
    with h5py.File(out/'episodes.h5','w') as dest:
        dest['ep_len']=lengths;dest['ep_offset']=offsets
        for k,v in data.items():dest.create_dataset(k,data=v,compression='lzf')
    errors=[];counts=[]
    for offset,length in zip(offsets,lengths):
        sl=slice(offset,offset+length-1)
        delta=data['pos_agent'][offset+1:offset+length]-data['pos_agent'][sl]
        proposed=np.clip(data['action'][sl],-1,1)*5
        err=np.linalg.norm(delta-proposed,axis=1)
        errors.extend(err.tolist());counts.append(int(np.sum(err<.001)))
receipt=dict(repo='quentinll/lewm-tworooms',revision='6903a2de048b13819d812da0b4dd661290bc01e4',
    archive_byte_range=[0,33554431],archive_prefix_sha256=hashlib.file_digest(open('/tmp/tworoom-prefix.zst','rb'),'sha256').hexdigest(),
    valid_hdf5_prefix_bytes=valid,episodes=32,frames=end,
    action_semantics='outgoing 2D direction, clipped to [-1,1], speed 5 simulator pixels per step, with wall collisions',
    fraction_matching_unobstructed_motion=float(np.mean(np.asarray(errors)<.001)),
    scope='First 32 complete source episodes; chunk extents verified; full archive checksum pending')
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
