import os, random
import numpy as np, xarray as xr, pandas as pd, datashader as das, pylab as plt
from glob import glob
from satpy.scene import Scene
from satpy.composites import DayNightCompositor
from pyorbital.orbital import get_observer_look
from torchvision.io import write_jpeg

root = f'/share/data/2pals/jim/data/geostat/temp'
os.chdir(root)

comps = ['natural_color','colorized_ir_clouds']
sats = ['H08', 'G16', 'G17', 'MSG1', 'MSG4']

datelist = pd.date_range('2019-07-01T00:00:00', '2019-07-10T00:00:00', freq='30min').tolist()
random.shuffle(datelist)

for t in datelist:
    for sat in sats:
        path = f'{sat}_{t.year}{t.month:02}{t.day:02}{t.hour:02}{t.minute:02}.arrow'
        if os.path.exists(path) == False:
            try:
                t0 = t - pd.Timedelta('15min')
                if sat == 'H08':    scn = Scene(glob(f'*{sat}_{t.year}{t.month:02}{t.day:02}_{t.hour:02}{t.minute:02}*.DAT'), reader='ahi_hsd', reader_kwargs={'mask_space': False})
                elif sat == 'G16':  scn = Scene(glob(f'*_G16_s{t.year}{t.dayofyear:03}{t.hour:02}{t.minute:02}*.nc'), reader='abi_l1b')
                elif sat == 'G17':  scn = Scene(glob(f'*_G17_s{t.year}{t.dayofyear:03}{t.hour:02}{t.minute:02}*.nc'), reader='abi_l1b')
                elif sat == 'MSG1': scn = Scene(glob(f'MSG1*{t0.year}{t0.month:02}{t0.day:02}{t0.hour:02}{t0.minute+12}*.nat'), reader='seviri_l1b_native')
                elif sat == 'MSG4': scn = Scene(glob(f'MSG4*{t0.year}{t0.month:02}{t0.day:02}{t0.hour:02}{t0.minute+12}*.nat'), reader='seviri_l1b_native')

                scn.load( comps, generate=False ) # set generate = False to resample in next, some bands have diffent res
                scn = scn.resample(scn.coarsest_area(), resampler='native')
                compositor = DayNightCompositor("dnc", lim_low=85., lim_high=87., day_night="day_night") #lim_low=85., lim_high=88.,
                composite  = compositor([scn[comps[0]], scn[comps[1]]])
                
                lons, lats = composite.area.get_lonlats()
                if sat == 'H08':
                    sat_lon = 140.7
                    sat_lat = 0
                    sat_alt = 35786 
                else: 
                    sat_lon = composite.attrs['orbital_parameters']['projection_longitude']
                    sat_lat = composite.attrs['orbital_parameters']['projection_latitude']
                    sat_alt = composite.attrs['orbital_parameters']['projection_altitude']/1000

                sata, satel = get_observer_look(sat_lon, sat_lat, sat_alt, composite.attrs['start_time'], lons, lats, alt=0)
                
                df = pd.DataFrame({'lat':lats.ravel(), 'lon':lons.ravel(), 'view_el': satel.ravel(), 'red':composite.values[0,:,:].ravel(), 'green':composite.values[1,:,:].ravel(), 'blue':composite.values[2,:,:].ravel()})
                df = df[df.lat != np.inf]
                df = df[df.lon != np.inf]
                df = df.dropna()       
                df.red *= df.view_el
                df.green *= df.view_el
                df.blue *= df.view_el
                df.to_feather(path)
            except: pass
