import os, torch, fnmatch, random, s3fs, bz2, numpy as np, pandas as pd
from satpy.scene import Scene
from satpy.composites import DayNightCompositor

num = 8
rez = 20
fs  = s3fs.S3FileSystem( anon = True )
root = f'noaa-himawari{num}/AHI-L1b-FLDK/'

if num == 8: datelist = pd.date_range('2015-07-07T02:00:00', '2022-12-01T00:30:00', freq='30min').tolist()
else: datelist = pd.date_range('2022-01-01T00:00:00', '2024-06-10T23:30:00', freq='30min').tolist()
datelist = pd.date_range('2015-07-07T02:00:00', '2019-12-30T23:30:00', freq='30min').tolist()
random.shuffle(datelist)

os.chdir(f'/share/data/2pals/jim/data/geostat/')

for t in datelist:
    for band in [1,2,3,7,13]: #[1,2,3,7,13]
        path = f'/share/data/2pals/jim/data/geostat/himawari/{t.year}/HS{num}_{t.year}{t.month:02}{t.day:02}{t.hour:02}{t.minute:02}_B{band:02}.pt'
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
                x = torch.tensor( scn[f'B{band:02}'].values )
                
                torch.save(x.type(torch.float16), path)
                os.remove( filepath[:-4] )
                del scn, x
                
            except: pass
        else: print('done') 







