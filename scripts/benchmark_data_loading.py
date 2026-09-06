"""Compare original and optimized complete dataset items on fixed real windows."""
import argparse
import json
import time
from pathlib import Path
import h5py
import numpy as np
import torch
import world_model.data as data_module
from world_model.train import write_json


def benchmark(output, windows=48):
    output = Path(output)
    if output.exists(): raise FileExistsError(output)
    torch.set_num_threads(1)
    optimized = data_module.read_pixel_window
    original = lambda pixels,start,stop,skip: pixels[start:stop:skip]
    result = dict(scope='Single-process paired complete dataset reads; alternating order, random fixed source windows; no model or optimizer',windows_per_dataset=windows,datasets={})
    try:
        for name,path in [('pusht','data/pusht/pusht_expert_train.h5'),('tworoom','data/tworoom/tworoom.h5')]:
            with h5py.File(path,'r') as source: episodes = list(range(len(source['ep_len'])))
            dataset = data_module.TrajectoryDataset(path,episodes)
            indices = torch.randperm(len(dataset),generator=torch.Generator().manual_seed(8319))[:windows].tolist()
            durations = dict(original=[],optimized=[]); exact = True
            for i,index in enumerate(indices):
                items = {}
                order = [('original',original),('optimized',optimized)]
                if i%2: order.reverse()
                for label,reader in order:
                    data_module.read_pixel_window = reader
                    started=time.perf_counter();items[label]=dataset[index]
                    durations[label].append(time.perf_counter()-started)
                left,right=items['original'],items['optimized']
                exact &= (left['episode']==right['episode'] and left['start']==right['start'] and
                    torch.equal(left['pixels'],right['pixels']) and torch.allclose(left['action'],right['action'],rtol=0,atol=0,equal_nan=True))
            medians = {k:float(np.median(v)) for k,v in durations.items()}
            result['datasets'][name] = dict(path=path,window_indices=indices,all_items_exact=exact,
                median_seconds=medians,total_seconds={k:sum(v) for k,v in durations.items()},
                median_speedup=medians['original']/medians['optimized'])
            print(json.dumps({k:v for k,v in result['datasets'][name].items() if k!='window_indices'}),flush=True)
    finally:
        data_module.read_pixel_window=optimized
    write_json(output,result)
    if not all(x['all_items_exact'] for x in result['datasets'].values()):
        raise RuntimeError('Reader changed source samples; optimization is invalid')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output');p.add_argument('--windows',type=int,default=48)
    a=p.parse_args();benchmark(a.output,a.windows)
