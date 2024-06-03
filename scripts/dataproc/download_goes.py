import os, fnmatch, random, s3fs, numpy as np, pandas as pd
from satpy.scene import Scene
import datashader as das

num = 17
fs = s3fs.S3FileSystem( anon = True )
root = f'noaa-goes{num}/ABI-L1b-RadF/'

if num==16: datelist = pd.date_range('2018-09-01T00:00:00', '2024-02-01T00:00:00', freq='30min').tolist()
else: datelist = pd.date_range('2019-01-01T00:00:00', '2024-02-01T00:00:00', freq='30min').tolist()
random.shuffle(datelist)

datelist = pd.date_range('2019-01-01T00:00:00', '2020-01-01T00:00:00', freq='30min').tolist()
random.shuffle(datelist)

biglat   = np.load(f'/share/data/2pals/jim/data/geostat/latlon/goes{num}biglat.npy')
biglon   = np.load(f'/share/data/2pals/jim/data/geostat/latlon/goes{num}biglon.npy')
medlat   = np.load(f'/share/data/2pals/jim/data/geostat/latlon/goes{num}medlat.npy')
medlon   = np.load(f'/share/data/2pals/jim/data/geostat/latlon/goes{num}medlon.npy')
smalllat = np.load(f'/share/data/2pals/jim/data/geostat/latlon/goes{num}smalllat.npy')
smalllon = np.load(f'/share/data/2pals/jim/data/geostat/latlon/goes{num}smalllon.npy')

os.chdir(f'/share/data/2pals/jim/data/geostat/')

for t in datelist:
    for band in [8,12,16]: #[2,3,5,7,8,10,11,12,13,15,16]:
        path = f'/share/data/2pals/jim/data/geostat/goes{num}/{t.year}/OR_ABI-L1b-RadF-M3C{band:02}_G{num}_s{t.year}{t.dayofyear:03}{t.hour:02}{t.minute:02}.npy'
        if os.path.exists(path) == False:  
            try:
                files = np.array( fs.ls(f'{root}{t.year}/{t.dayofyear:03}/{t.hour:02}/') )
                bf = fnmatch.filter(files, f'*C{band:02}_G{num}_s{t.year}{t.dayofyear:03}{t.hour:02}{t.minute:02}*')        
                bf = bf[0].replace(f'noaa-goes{num}',f'https://noaa-goes{num}.s3.amazonaws.com')
                os.system(f"wget {bf}")
    
                filepath = bf.split('/')[-1]
                scn = Scene([filepath], reader='abi_l1b')
                scn.load([f'C{band:02}'])
    
                if   band == 2: df = pd.DataFrame({'lat':biglat, 'lon':biglon, 'rad':scn[f'C{band:02}'].values[::4,::4].ravel()})
                elif band == 3: df = pd.DataFrame({'lat':medlat, 'lon':medlon, 'rad':scn[f'C{band:02}'].values[::2,::2].ravel()})
                elif band == 5: df = pd.DataFrame({'lat':medlat, 'lon':medlon, 'rad':scn[f'C{band:02}'].values[::2,::2].ravel()})
                else: df = pd.DataFrame({'lat':smalllat, 'lon':smalllon, 'rad':scn[f'C{band:02}'].values.ravel()})
                del scn
                
                df = df[df.lat != np.inf]
                df = df[df.lon != np.inf]
                df.lon[df.lon<0] += 360
                df = df.dropna()
                
                lats = np.arange(-80, 80.1,0.1)
                if num ==16: lons = np.arange(253, 324, 0.1)
                else: lons = np.arange(181, 255, 0.1)
                
                cvs = das.Canvas( plot_width=lons.shape[0], plot_height=lats.shape[0], x_range=(lons[0]-0.05, lons[-1]+0.05), y_range=(lats[0]-0.05, lats[-1]+0.05) )
                agg = cvs.points( df, 'lon', 'lat', das.mean('rad') )
                np.save( path, agg.astype( np.float32 ) )
                os.remove( filepath )
                del agg, df
            except: print(t, band, 'some error')
        else: print('done') 
