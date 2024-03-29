import os, random, datetime, numpy as np, pandas as pd, pylab as plt
from glob import glob
from zipfile import ZipFile
from satpy.scene import Scene
import datashader as das
import dask.array as da

root = '/share/data/2pals/jim/data/'

# tokenize
tokens = np.array([0,0,0,0,0,0,0,0,0,0,0,0]) #,0,0,0,0,0,0,0])
for i in range(40000): tokens = np.vstack((tokens,np.random.choice([0, 1], size=12)))
tokens = np.unique(tokens, axis=0) 

y = np.array([1,1,1,1,1,1,1,1,1,1,1,1])
token = np.argmin(np.abs(y-tokens).sum(axis=1))

fl = glob('/share/data/2pals/jim/data/tokenized/*')
i=9 #band

for fn in fl:
    t = pd.to_datetime(fn.split('/')[-1].strip('.npy')) + pd.Timedelta('30min')
    path = f'{root}tokenized/{t.year}{t.month:02}{t.day:02}{t.hour:02}{t.minute:02}.npy'
    if os.path.exists(path) == False:
        try: 
            for f in glob(f'{root}temp1/*'): os.remove(f)
            fl = glob(f'{root}geostat/zerodegree/{t.year}/*{t.year}{t.month:02}{t.day:02}{t.hour:02}{t.minute+12}*-NA.zip')
            with ZipFile(fl[0], 'r') as zObject: zObject.extractall( path=f'{root}temp1/' ) 
            scn1 = Scene( reader="seviri_l1b_native", filenames=glob(f'{root}temp1/*nat') )
        
            for f in glob(f'{root}temp2/*'): os.remove(f)
            fl = glob(f'{root}geostat/indian/{t.year}/*{t.year}{t.month:02}{t.day:02}{t.hour:02}{t.minute+12}*-NA.zip')
            with ZipFile(fl[0], 'r') as zObject: zObject.extractall( path=f'{root}temp2/' ) 
            scn2 = Scene( reader="seviri_l1b_native", filenames=glob(f'{root}temp2/*nat') )
        
            temp = aggmet(scn1, metbands[i], LAT0, LON0, np.arange(-37.7,22.8,0.1))
            out  = scale(temp, metmin[i], metmax[i])

            temp = aggmet(scn2, metbands[i], LATi, LONi, np.arange(22.8,92.0,0.1))
            temp = scale(temp, met2min[i], met2max[i])
            out  = np.hstack( (out, temp) )  
            
            temp = np.load(glob(f'{root}geostat/himawari/{t.year}/HS_{t.year}{t.month:02}{t.day:02}{t.hour:02}{t.minute:02}_B{himbands[i]:02}.npy')[0])
            out  = np.hstack( (out, scale(temp[:,:-12], himmin[i], himmax[i])) ) 
        
            temp = np.load(glob(f'{root}geostat/goes17/{t.year}/*{goesbands[i]:02}_G17_s{t.year}{t.dayofyear:03}{t.hour:02}{t.minute:02}.npy')[0])
            out  = np.hstack( (out, scale(temp[:,7:-11], goesmin[i], goesmax[i])) )  
            
            temp = np.load(glob(f'{root}geostat/goes16/{t.year}/*{goesbands[i]:02}_G16_s{t.year}{t.dayofyear:03}{t.hour:02}{t.minute:02}.npy')[0])
            out  = np.hstack( (out, scale(temp[:,10:-15], goesmin[i], goesmax[i])) ) 

            out = da.from_array(out)
            out = da.coarsen(np.mean, out, {0:4,1:4}, trim_excess=True)
            
            np.save(path, out.compute().astype(np.float32))
        except: print('not downloaded')
    else: print('done')




fl = glob('/share/data/2pals/jim/data/tokenized/*')
X = np.empty((450))
Y = np.empty((450))
for fn in fl:
    try:
        x = np.load(fn)
        time = int(fn.split('/')[-1].strip('.npy'))
        y = np.load(f'/share/data/2pals/jim/data/tokenized/{time+30}.npy')

        x = da.coarsen(np.mean, da.from_array(x), {0:7,1:30}, trim_excess=True).compute()
        x = x[6:51,:]
        x[x<=0.55] = 1
        x[x<1] = 0
        temp = []
        for i in range(15):
            for j in range(30):
                temp.append(np.argmin(np.abs(x[i*3:i*3+3,j*4:j*4+4].ravel()-tokens).sum(axis=1)))
        
        X = np.vstack((X,np.array(temp).flatten()))
    
        y = da.coarsen(np.mean, da.from_array(y), {0:7,1:30}, trim_excess=True).compute()
        y = y[6:51,:]
        y[y<=0.55] = 1
        y[y<1] = 0
        temp = []
        for i in range(15):
            for j in range(30):
                temp.append(np.argmin(np.abs(y[i*3:i*3+3,j*4:j*4+4].ravel()-tokens).sum(axis=1)))
    
        Y = np.vstack((Y,np.array(temp).flatten()))
    except: pass

np.save('/share/data/2pals/jim/data/X_450.npy', X)
np.save('/share/data/2pals/jim/data/Y_450.npy', Y)