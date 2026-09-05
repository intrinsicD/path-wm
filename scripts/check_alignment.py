"""Compare recorded PushT states/images with reset and replayed simulator output."""
import argparse,os,json
os.environ.setdefault('SDL_VIDEODRIVER','dummy')
import h5py,hdf5plugin
import numpy as np
from pathlib import Path
from PIL import Image
from third_party.swm.pusht import PushT


def check(path,output,relative=True):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    records=[]
    with h5py.File(path,'r') as f:
        lengths,offsets=f['ep_len'][:],f['ep_offset'][:]
        for ep in range(min(4,len(lengths))):
            start=int(offsets[ep])+min(25,int(lengths[ep])-12)
            state=f['state'][start].astype(float)
            if len(state)==5:state=np.r_[state,[0,0]]
            frame=f['pixels'][start].astype(np.uint8)
            env=PushT(relative=relative,resolution=frame.shape[0])
            obs,info=env.reset(seed=42+ep,options={'state':state,'goal_state':state})
            render=env.render()
            initial_state=obs['state']
            differences=[]
            for step in range(10):
                obs,_,_,_,_=env.step(f['action'][start+step])
                actual=f['state'][start+step+1]
                differences.append({'step':step+1,'position_rmse':float(np.mean((obs['state'][:4]-actual[:4])**2)**.5),
                                    'angle_error':float(abs(np.arctan2(np.sin(obs['state'][4]-actual[4]),np.cos(obs['state'][4]-actual[4]))))})
            record={'episode':ep,'start':start-int(offsets[ep]),'reset_position_rmse':float(np.mean((initial_state[:4]-state[:4])**2)**.5),
                    'image_mae':float(np.abs(render.astype(float)-frame).mean()),
                    'pixel_fraction_different':float(np.any(render!=frame,axis=-1).mean()),'replay':differences}
            records.append(record)
            Image.fromarray(np.concatenate([frame,render,np.abs(render.astype(int)-frame.astype(int)).astype(np.uint8)],axis=1)).save(out/f'episode_{ep}.png')
            env.close()
    result={'dataset':str(path),'relative':relative,'records':records,
            'note':'Raw state distance excludes unrecorded block velocity; image panel is data, reset render, absolute difference.'}
    (out/'alignment.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('path');p.add_argument('output');p.add_argument('--absolute',action='store_true')
    a=p.parse_args();check(a.path,a.output,not a.absolute)
