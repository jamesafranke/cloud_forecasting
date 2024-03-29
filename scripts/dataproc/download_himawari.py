import os, fnmatch, random, s3fs, bz2, numpy as np, pandas as pd
from satpy.scene import Scene
import datashader as das

num = 8
rez  = 20
fs = s3fs.S3FileSystem( anon = True )
root = f'noaa-himawari{num}/AHI-L1b-FLDK/'

LAT = np.load(f'/share/data/2pals/jim/data/geostat/latlon/him{num}rez20lat.npy')
LON = np.load(f'/share/data/2pals/jim/data/geostat/latlon/him{num}rez20lon.npy')

datelist = pd.date_range('2019-01-01T00:00:00', '2022-11-30T23:30:00', freq='30min').tolist()
random.shuffle(datelist)

os.chdir(f'/share/data/2pals/jim/data/geostat/')

for t in datelist:
    for band in [3,4,5,7,8,10,11,12,13,15,16]:
        path = f'/share/data/2pals/jim/data/geostat/himawari/{t.year}/HS_{t.year}{t.month:02}{t.day:02}{t.hour:02}{t.minute:02}_B{band:02}.npy'
        if os.path.exists(path) == False:  
            try:
                files = np.array( fs.ls(f'{root}{t.year}/{t.month:02}/{t.day:02}/{t.hour:02}{t.minute:02}/') )
                bf = fnmatch.filter(files, f'*B{band:02}_FLDK_R{rez:02}*')        
                bf = bf[0].replace(f'noaa-himawari{num}',f'https://noaa-himawari{num}.s3.amazonaws.com')
                os.system(f"wget {bf}")
                
                filepath = bf.split('/')[-1]
                zipfile  = bz2.BZ2File(filepath)
                data = zipfile.read()
                open(filepath[:-4], 'wb').write(data) 
                os.remove(filepath)
                
                scn = Scene([filepath[:-4]], reader='ahi_hsd', reader_kwargs={'mask_space': False})
                scn.load([f'B{band:02}'])
                #lons, lats = scn[f'B{band:02}'].attrs['area'].get_lonlats()
                
                df = pd.DataFrame({'lat':LAT, 'lon':LON, 'rad':scn[f'B{band:02}'].values.ravel()})
                df = df[df.lat != np.inf]
                df = df[df.lon != np.inf]
                df.lon[df.lon<0] += 360
                df = df.dropna()       
                
                lats = np.arange(-80,80.1,0.1)
                lons = np.arange(92.0,182.8,0.1)
                
                cvs = das.Canvas( plot_width=lons.shape[0], plot_height=lats.shape[0], x_range=(lons[0]-0.05, lons[-1]+0.05), y_range=(lats[0]-0.05, lats[-1]+0.05) )
                agg = cvs.points( df, 'lon', 'lat', das.mean('rad') )
                np.save(path, agg.astype(np.float32))
                os.remove( filepath[:-4] )
                del scn, df
                
            except: pass
        else: print('done') 
