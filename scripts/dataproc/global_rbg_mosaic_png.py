import os, torch, random, numpy as np, pandas as pd, datashader as das
from glob import glob
from torchvision.io import write_jpeg

datelist = pd.date_range('2019-07-01T00:00:00', '2019-07-10T00:00:00', freq='30min').tolist()
random.shuffle(datelist)

res = 0.1
lons = np.arange( -180, 180, res)
lats = np.arange( -70, 70, res)
cvs  = das.Canvas( plot_width=lons.shape[0], plot_height=lats.shape[0], x_range=(lons[0]-res/2, lons[-1]+res/2), y_range=(lats[0]-res/2, lats[-1]+res/2) )

for t in datelist:
    path = f'/share/data/2pals/jim/code/jupyter/notebooks/gif/{t.year}{t.month:02}{t.day:02}{t.hour:02}{t.minute:02}.jpeg'
    if os.path.exists(path) == False:
        try:
            fl = glob(f'/share/data/2pals/jim/data/geostat/temp/*{t.year}{t.month:02}{t.day:02}{t.hour:02}{t.minute:02}.arrow')
            df = pd.DataFrame({})
            for file in fl: df = pd.concat( (df, pd.read_feather(file)) )

            x = np.empty((3, 1400,3600))
            w = cvs.points( df, 'lon', 'lat', das.sum('view_el')).values
            
            x[0,:,:] = cvs.points( df, 'lon', 'lat', das.sum('red')).values / w
            x[1,:,:] = cvs.points( df, 'lon', 'lat', das.sum('green')).values / w
            x[2,:,:] = cvs.points( df, 'lon', 'lat', das.sum('blue')).values / w
            
            x *= 255
            x = np.flip(x, axis=1)
            x = np.nan_to_num(x)
            x = np.clip(x, 0, 255).astype(np.uint8)
            
            write_jpeg(torch.from_numpy(x), path)
    
        except: pass