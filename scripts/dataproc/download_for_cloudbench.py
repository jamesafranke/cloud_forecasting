import os, fnmatch, random, s3fs, bz2, numpy as np, pandas as pd
from zipfile import ZipFile
from glob import glob

root = f'/share/data/2pals/jim/data/geostat/temp'
os.chdir(root)

def get_raw_geostaionary(t): # t: datetime object
    fs  = s3fs.S3FileSystem( anon = True )
    #### Himawari ####
    num = 8
    rez = 20
    root = f'noaa-himawari{num}/AHI-L1b-FLDK/'
    for band in range(1,17): 
        try: 
            files = np.array( fs.ls(f'{root}{t.year}/{t.month:02}/{t.day:02}/{t.hour:02}{t.minute:02}/') )
            bf = fnmatch.filter(files, f'*B{band:02}_FLDK_R{rez:02}*')  
            bf = bf[0].replace(f'noaa-himawari{num}',f'https://noaa-himawari{num}.s3.amazonaws.com')
            filepath = bf.split('/')[-1]
            if os.path.exists(filepath[:-4]) == False: 
                os.system(f"wget {bf}")
                zipfile  = bz2.BZ2File(filepath)
                data = zipfile.read()
                open(filepath[:-4], 'wb').write(data) 
                os.remove(filepath)
        except: pass
    
    #### GOES ####
    for num in [16,17]:
        root = f'noaa-goes{num}/ABI-L1b-RadF/'
        for band in range(1,17): 
            try: 
                files = np.array( fs.ls(f'{root}{t.year}/{t.dayofyear:03}/{t.hour:02}/') )
                bf = fnmatch.filter(files, f'*C{band:02}_G{num}_s{t.year}{t.dayofyear:03}{t.hour:02}{t.minute:02}*')        
                bf = bf[0].replace(f'noaa-goes{num}',f'https://noaa-goes{num}.s3.amazonaws.com')
                if os.path.exists(bf.split('/')[-1]) == False: os.system(f"wget {bf}")
            except: pass
    
    #### Meteostat ####
    t0 = t - pd.Timedelta('15min') # to account for meteostats weird delayed retrivial
    start = f'{t0.year}-{t0.month:02}-{t0.day:02}T{t0.hour:02}:{t0.minute:02}'
    end   = f'{t.year}-{t.month:02}-{t.day:02}T{t.hour:02}:{t.minute:02}'
    if len(glob(f'MSG4*{t0.year}{t0.month:02}{t0.day:02}{t0.hour:02}{t0.minute+12}*.nat')) == 0: 
        try: os.system(f"eumdac download --yes -c EO:EUM:DAT:MSG:HRSEVIRI -s {start} -e {end}")
        except: pass
    
    if len(glob(f'MSG1*{t0.year}{t0.month:02}{t0.day:02}{t0.hour:02}{t0.minute+12}*.nat')) == 0: 
        try: os.system(f"eumdac download --yes -c EO:EUM:DAT:MSG:HRSEVIRI-IODC -s {start} -e {end}")
        except: pass 

    files = glob(f'MSG*{t0.year}{t0.month:02}{t0.day:02}{t0.hour:02}{t0.minute+12}*.zip')
    for file in files:
        with ZipFile(file,'r') as zObject: zObject.extractall( ) 
        os.remove(file)



datelist = pd.date_range('2019-07-05T00:00:00', '2019-07-10T00:00:00', freq='30min').tolist()
random.shuffle(datelist)

for t in datelist: get_raw_geostaionary(t)