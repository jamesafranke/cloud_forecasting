import math, torch
import numpy as np
root = '/share/data/2pals/jim/data/'

lat = np.load(f'{root}latlon/himbiglat.npy')
lon = np.load(f'{root}latlon/himbiglat.npy')

#DAY
def contrast_correction(color, contrast): #hacked rayleigh scattering correction - just boost the colors
    F = ( 259 * ( contrast + 255 ) ) / ( 255. * 259 - contrast )
    COLOR = F * (color-.5) + .5
    return np.clip( COLOR, 0, 1 )

def scale(x, low, high):
    x = (x-low)/(high-low)
    return np.clip(x,0, 1)

RGB = np.empty( ( 5424, 5424, 3 ) )
kappa 
for i, band in [1,2,3]:
    RGB[:,:,i] =  torch.load(glob(f'{root}geostat/himawari/t.year/')) * kappa0

#hybrid green
Gh = (1- 0.07)* B2 + 0.07 * B4

gamma = 2.2
RGB = np.power(RGB, 1/gamma)
RGB = np.nan_to_num(RGB, 0)
RGB[:,:,1] = 0.45 * RGB[:,:,0] + 0.1 * RGB[:,:,1] + 0.45 * RGB[:,:,2]
RGB = contrast_correction(RGB, 105)
RGB *= 255 
RGB = RGB.astype(int)
RGB = np.clip(RGB,0, 255)



#NIGHT

ir = 

ir[np.abs(lat) < 30] = 200
ir[np.abs(lat).between(30,60)] = 200+20*(np.abs(lat)-30)/30
ir[np.abs(lat) > 60] = 220

ir[ir>280] =  280


def BT(flx, band_wl):
    # calcualte brightness temperature from satelite measurment based on Planck's law
    #band_wl = #central wavelenght in micrometers
    #flx: flux meausred in w m-2
    h =  6.626e-34     # Plank's const in Js
    k =  1.38e-23      # Boltzmanns const in J/K
    c =  299792000     # speed of light in m/s 

    freq = c / (band_wl * 1e-6)

    T = (h*freq/k) / math.log( 1 + 2*h*math.pow( freq, 3.0 )/( flx * math.pow( c, 2.0 ) ) )

    return T


#normalize BTD over land and water separately
1->4.5 #over land
0->4 #over water


NLi=R,G,B=NIR*1.0+(1.0-NIR)*[Ai*LC+(1.0-LC)*Si]


R, G, B = (0.55, 0.75, 0.98) #low cloud triplet lightish blue
R, G, B = (229, 115, 100) #low cloud triplet lightish red #E57364
R, G, B = (0.06, 0.03, 0.13)  #background purple ground at night

# day night blending

#night lights 
!wget https://eogdata.mines.edu/nighttime_light/annual/v22/2023/VNL_npp_2023_global_vcmslcfg_v2_c202402081600.median.dat.tif.gz