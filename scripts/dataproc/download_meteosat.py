import os, eumdac, random, pandas as pd
from glob import glob

datelist = pd.date_range('2019-01-01T00:00:00', '2024-02-01T00:00:00', freq='30min').tolist()
random.shuffle(datelist)

loc = 'HRSEVIRI-IODC' #'HRSEVIRI-IODC'
if loc == 'HRSEVIRI': folder = 'zerodegree'
else: folder = 'indian'

for t in datelist:     
    path = f'/share/data/2pals/jim/data/geostat/{folder}/{t.year}/'
    if len(glob(f'{path}*{t.year}{t.month:02}{t.day:02}{t.hour:02}{t.minute+12}*.zip'))==0:
        os.chdir(path)
        try:
            start = f'{t.year}-{t.month:02}-{t.day:02}T{t.hour:02}:{t.minute:02}'
            end   = f'{t.year}-{t.month:02}-{t.day:02}T{t.hour:02}:{t.minute+10:02}'
            os.system(f"eumdac download --yes -c EO:EUM:DAT:MSG:{loc} -s {start} -e {end}")
        except: pass
    else: pass
