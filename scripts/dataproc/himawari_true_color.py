import os, torch, fnmatch, random, s3fs, bz2, numpy as np, pandas as pd
from glob import glob
from satpy.scene import Scene
from satpy.composites import DayNightCompositor
from torchvision.io import write_jpeg

num = 8
rez = 20
fs  = s3fs.S3FileSystem( anon = True )
root = f'noaa-himawari{num}/AHI-L1b-FLDK/'

if num == 8: datelist = pd.date_range('2015-07-07T02:00:00', '2022-12-01T00:30:00', freq='30min').tolist()
else: datelist = pd.date_range('2022-01-01T00:00:00', '2024-06-10T23:30:00', freq='30min').tolist()
datelist = pd.date_range('2015-07-07T02:00:00', '2019-12-30T23:30:00', freq='30min').tolist()
random.shuffle(datelist)

os.chdir(f'/share/data/2pals/jim/data/geostat/')

def get_true_color(t):
    try:  
        for band in [1,2,3,4,13]: 
            files = np.array( fs.ls(f'{root}{t.year}/{t.month:02}/{t.day:02}/{t.hour:02}{t.minute:02}/') )
            bf = fnmatch.filter(files, f'*B{band:02}_FLDK_R{rez:02}*')        
            bf = bf[0].replace(f'noaa-himawari{num}',f'https://noaa-himawari{num}.s3.amazonaws.com')
            os.system(f"wget {bf}")
            filepath = bf.split('/')[-1]
            zipfile  = bz2.BZ2File(filepath)
            data = zipfile.read()
            open(filepath[:-4], 'wb').write(data) 
            os.remove(filepath)

        files = glob(f'HS_H0{num}_{t.year}{t.month:02}{t.day:02}_{t.hour:02}{t.minute:02}_B??_FLDK_R20_S0101.DAT')
        scn = Scene(files, reader='ahi_hsd', reader_kwargs={'mask_space': False})
        scn.load(['colorized_ir_clouds'])
        scn.load(['true_color'])

        compositor = DayNightCompositor("dnc", lim_low=85., lim_high=88., day_night="day_night")
        composite  = compositor([scn['true_color'], scn['colorized_ir_clouds']])
        
        x = torch.tensor( composite.values * 255 ).type(torch.uint8)
        write_jpeg(x, path)
        for file in files: os.remove( file )
        del x, composite         
    except: pass

for t in datelist:
    for dummy in range(5):
        path = f'/share/data/2pals/jim/data/geostat/himawari_truecolor/{t.year}/HS{num}_{t.year}{t.month:02}{t.day:02}{t.hour:02}{t.minute:02}.jpeg'
        if os.path.exists(path) == False: get_true_color(t)
        else: print('done') 
        t += pd.Timedelta('30min')

