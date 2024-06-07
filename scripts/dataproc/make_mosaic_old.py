import os, random, datetime, numpy as np, pandas as pd, pylab as plt
from glob import glob
from zipfile import ZipFile
from satpy.scene import Scene
import datashader as das, dask.array as da

root = '/share/data/2pals/jim/data/'

metbands  = ['VIS006','VIS008','IR_016','IR_039','WV_062','WV_073','IR_087','IR_097','IR_108','IR_120','IR_134']
goesbands = [2, 3, 5, 7, 8, 10, 11, 12, 13, 15, 16]
himbands  = [3, 4, 5, 7, 8, 10, 11, 12, 13, 15, 16]

himmin  = [0.0,    0.0,   0.0, 207.0, 196.0, 196.0, 199.0, 198.0, 194.0, 196.0, 192.0]
goesmin = [0.0,    0.0,   0.0, 209.0, 193.0, 195.0, 199.0, 199.0, 192.0, 183.0, 185.0]
metmin  = [0.0,    0.0,   0.0, 213.0, 199.0, 199.0, 199.0, 210.0, 198.0, 197.0, 200.0]
met2min = [0.0,    0.0,   0.0, 213.0, 199.0, 199.0, 199.0, 206.0, 197.0, 196.0, 199.0]

himmax  = [86.0, 122.0, 120.0, 336.0, 261.0, 278.0, 321.0, 292.0, 327.0, 325.0, 293.0]
goesmax = [84.0, 127.0, 105.0, 336.0, 260.0, 276.0, 309.0, 284.0, 315.0, 313.0, 292.0]
metmax  = [71.0,  83.0,  84.0, 331.0, 262.0, 276.0, 318.0, 290.0, 327.0, 326.0, 285.0]
met2max = [76.0,  78.0,  97.0, 331.0, 261.0, 277.0, 321.0, 295.0, 327.0, 326.0, 290.0]

LAT0, LON0 = np.load( f'{root}geostat/latlon/metzerolat.npy'   ), np.load( f'{root}geostat/latlon/metzerolon.npy' )
LATi, LONi = np.load( f'{root}geostat/latlon/metindianlat.npy' ), np.load( f'{root}geostat/latlon/metindianlon.npy' )

def aggmet(scn, band, LAT, LON, lons):
    scn.load([band])
    df = pd.DataFrame({'lat':LAT, 'lon':LON, 'rad':scn[band].values.ravel()})
    df = df[df.lat != np.inf]
    df = df[df.lon != np.inf]
    df = df.dropna()       
    lats = np.arange(-80,80.1,0.1)
    cvs = das.Canvas( plot_width=lons.shape[0], plot_height=lats.shape[0], x_range=(lons[0]-0.05, lons[-1]+0.05), y_range=(lats[0]-0.05, lats[-1]+0.05) )
    return cvs.points( df, 'lon', 'lat', das.mean('rad') )
    
def scale(temp, minar, maxar):
    temp = (temp - minar) / (maxar - minar)
    temp = np.nan_to_num(temp)
    temp[temp>1] = 1
    temp[temp<0] = 0
    return temp 

datelist = pd.date_range('2019-01-01T00:00:00', '2019-01-03T00:00:00', freq='30min').tolist()

for t in datelist:
    if len( glob(f'{root}stitch_corrected/{t.year}{t.month:02}{t.day:02}{t.hour:02}{t.minute:02}_B*.npy') ) < 11:
        for f in glob(f'{root}temp1/*'): os.remove(f)
        fl = glob(f'{root}geostat/zerodegree/{t.year}/*{t.year}{t.month:02}{t.day:02}{t.hour:02}{t.minute+12}*-NA.zip')
        with ZipFile(fl[0], 'r') as zObject: zObject.extractall( path=f'{root}temp1/' ) 
        scn1 = Scene( reader="seviri_l1b_native", filenames=glob(f'{root}temp1/*nat') )
    
        for f in glob(f'{root}temp2/*'): os.remove(f)
        fl = glob(f'{root}geostat/indian/{t.year}/*{t.year}{t.month:02}{t.day:02}{t.hour:02}{t.minute+12}*-NA.zip')
        with ZipFile(fl[0], 'r') as zObject: zObject.extractall( path=f'{root}temp2/' ) 
        scn2 = Scene( reader="seviri_l1b_native", filenames=glob(f'{root}temp2/*nat') )
        
        for i in [0]: #range(1):
            path = f'{root}stitch_corrected/{t.year}{t.month:02}{t.day:02}{t.hour:02}{t.minute:02}_B{i:02}.npy'
            if os.path.exists(path) == False:
                temp = aggmet(scn1, metbands[i], LAT0, LON0, np.arange(-37.7,22.8,0.1))
                out  = scale(temp, metmin[i], metmax[i])
                out = (temp - mean) / std

                temp = aggmet(scn2, metbands[i], LATi, LONi, np.arange(22.8,92.0,0.1))
                temp = scale(temp, met2min[i], met2max[i])
                out  = np.hstack( (out, (temp - mean) / std ) )
                
                temp = np.load(glob(f'{root}geostat/himawari/{t.year}/HS_{t.year}{t.month:02}{t.day:02}{t.hour:02}{t.minute:02}_B{himbands[i]:02}.npy')[0])
                temp = scale(temp[:,:-12], himmin[i], himmax[i])
                out  = np.hstack( (out, (temp - mean) / std )[:,:-12] )
            
                temp = np.load(glob(f'{root}geostat/goes17/{t.year}/*{goesbands[i]:02}_G17_s{t.year}{t.dayofyear:03}{t.hour:02}{t.minute:02}.npy')[0])
                temp = scale(temp[:,7:-11], goesmin[i], goesmax[i])
                out  = np.hstack( (out, (temp - mean) / std )[:,7:-11] )
                
                temp = np.load(glob(f'{root}geostat/goes16/{t.year}/*{goesbands[i]:02}_G16_s{t.year}{t.dayofyear:03}{t.hour:02}{t.minute:02}.npy')[0])
                temp = scale(temp[:,10:-15], goesmin[i], goesmax[i])
                out  = np.hstack( (out, (temp - mean) / std )[:,10:-15] ) 
                
                np.save(path, out.astype(np.float32))
            else: print('done')
    else: print('done')

