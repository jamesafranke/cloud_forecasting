import os, fnmatch, random, datetime, numpy as np, pandas as pd, pylab as plt
import datashader as das, dask.array as da
from glob import glob

datelist = pd.date_range('2020-01-01T00:00:00', '2020-06-01T00:00:00', freq='30min').tolist()
random.shuffle(datelist)

root = '/share/data/2pals/jim/data/'
goesbands = [2, 3, 5, 7, 8, 10, 11, 12, 13, 15, 16]
goesmin = [0.0,    0.0,   0.0, 209.0, 193.0, 195.0, 199.0, 199.0, 192.0, 183.0, 185.0]
goesmax = [84.0, 127.0, 105.0, 336.0, 260.0, 276.0, 309.0, 284.0, 315.0, 313.0, 292.0]

def scale(temp, minar, maxar):
    temp = (temp - minar) / (maxar - minar)
    temp = np.nan_to_num(temp)
    temp[temp>1] = 1
    temp[temp<0] = 0
    return temp 

for t in datelist:
     for i in [4,7,10]:
        path = f'{root}processed/goes_{t.year}{t.month:02}{t.day:02}{t.hour:02}{t.minute:02}_B{i:02}.npy'
        if os.path.exists(path) == False:
            try: 
                temp1 = np.load(glob(f'{root}geostat/goes17/{t.year}/*{goesbands[i]:02}_G17_s{t.year}{t.dayofyear:03}{t.hour:02}{t.minute:02}.npy')[0])
                temp2 = np.load(glob(f'{root}geostat/goes16/{t.year}/*{goesbands[i]:02}_G16_s{t.year}{t.dayofyear:03}{t.hour:02}{t.minute:02}.npy')[0])
                out = np.hstack( (temp1[:,:-11], temp2[:,10:] ))
                
                out = scale(out, goesmin[i], goesmax[i])
                out = da.coarsen(np.mean, da.from_array(out, chunks='auto'), {0:5, 1:5},  trim_excess=True).compute()
                np.save(path, out[17:-18,:].astype(np.float32))
                
            except: print('no file')
        else: print('already done')
