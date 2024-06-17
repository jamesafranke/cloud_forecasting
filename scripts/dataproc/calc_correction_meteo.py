import os, fnmatch, random, datetime, numpy as np, pandas as pd, pylab as plt
import datashader as das, dask.array as da
from glob import glob
from zipfile import ZipFile
from satpy.scene import Scene

root = '/share/data/2pals/jim/data/'

met0_central   = 0
india_central  = 45.5 
him_central    = 140.7
goes17_central = 222.8
goes16_central = 284.82

lon1 = np.arange(-37.7, 22.8, 0.1)
lon2 = np.arange(22.8, 92.0, 0.1)
lon3 = np.arange(92.0, 182.8, 0.1)[:-12]
lon4 = np.arange(253, 324, 0.1)[10:-15]
lon5 = np.arange(181, 255, 0.1)[7:-11]

metwavelenghts  = [0.635, 0.810, 1.64, 3.92, 6.25, 7.35, 8.7, 9.66, 10.8,  12.0, 13.4] #um
goeswavelenghts = [0.640, 0.865, 1.61, 3.90, 6.19, 7.34, 8.5, 9.61, 10.35, 12.3, 13.3] #um
himwavelenghts  = []

metbands  = ['VIS006','VIS008','IR_016','IR_039','WV_062','WV_073','IR_087','IR_097','IR_108','IR_120','IR_134']
goesbands = [2, 3, 5, 7, 8, 10, 11, 12, 13, 15, 16]
himbands  = [3, 4, 5, 7, 8, 10, 11, 12, 13, 15, 16]

datelist = pd.date_range('2019-01-01T00:00:00', '2020-12-31T23:30:00', freq='30min').tolist()
datelist = random.sample(datelist, 1000)

LAT0, LON0 = np.load( f'{root}geostat/latlon/metzerolat.npy'   ), np.load( f'{root}geostat/latlon/metzerolon.npy' )
LATi, LONi = np.load( f'{root}geostat/latlon/metindianlat.npy' ), np.load( f'{root}geostat/latlon/metindianlon.npy' )

def aggmet(scn, band, LAT, LON, lons):
    scn.load([band])
    df = pd.DataFrame({'lat':LAT, 'lon':LON, 'rad':scn[band].values.ravel()})
    df = df[df.lat != np.inf]
    df = df[df.lon != np.inf]
    df = df.dropna()       
    lats = np.arange(-80,80.1,0.1)
    cvs  = das.Canvas( plot_width=lons.shape[0], plot_height=lats.shape[0], x_range=(lons[0]-0.05, lons[-1]+0.05), y_range=(lats[0]-0.05, lats[-1]+0.05) )
    return cvs.points( df, 'lon', 'lat', das.mean('rad') )

sat = 'indian' #zerodegree
i = 0
for hour in range(0,23): 
    for minute in [0,30]:
        path = f'/share/data/2pals/jim/data/correction/{sat}_B{i:02}_h{hour:02}_{minute:02}_mean.npy'
        if os.path.exists(path) == False: 
            out = np.empty((20,1601,692))
            fl = glob(f'{root}geostat/{sat}/????/*????01??{hour:02}{minute+12}*-NA.zip')
            for j, file in enumerate(fl[:20]):
                print(file)
                for f in glob(f'{root}temp2/*'): os.remove(f)
                with ZipFile(file, 'r') as zObject: zObject.extractall( path=f'{root}temp2/' ) 
                scn2 = Scene( reader="seviri_l1b_native", filenames=glob(f'{root}temp2/*nat') )
                out[j,:,:] = aggmet(scn2, metbands[i], LATi, LONi, np.arange(22.8,92.0,0.1))

            mean = np.nanmean(out, axis=0)
            std  = np.nanstd(out, axis=0)
            np.save(path, mean)  
            np.save(f'/share/data/2pals/jim/data/correction/{sat}_B{i:02}_h{hour:02}_{minute:02}_std.npy', std)   