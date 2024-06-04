import os, numpy as np, pandas as pd
from glob import glob 

root = '/share/data/2pals/jim/data/'
def check_sequence_lenght(fl, min_lenght=19):
    k = 0
    starts = []
    ends = []
    for i in range(len(fl)-1):
        if (pd.to_datetime(fl[i+1].split('_')[1])-pd.to_datetime(fl[i].split('_')[1])).seconds == 1800: 
            pass
        else: 
            if i-k > min_lenght: 
                starts.append(k)
                ends.append(i)
            else: 
                pass
            k = i
    return starts, ends

band = 10
fl = glob(f'{root}processed/*B{band:02}.npy')
fl = np.sort(fl)

starts, ends = check_sequence_lenght(fl, 19)
for i, start in enumerate(starts):
    print(start)
    path = f'{root}openstl/stacks/{start:05}.npy'
    if os.path.exists(path) == False:
        try: 
            size = ends[i]-start
            out = np.empty((size,3,285,285))
            for k in range(size):
                out[k,0,:,:] = np.load( fl[start+k].replace('B10','B04') )
                out[k,1,:,:] = np.load( fl[start+k].replace('B10','B07') )
                out[k,2,:,:] = np.load( fl[start+k] )
            
            np.save(path, out.astype(np.float32))
        except: print('no file')