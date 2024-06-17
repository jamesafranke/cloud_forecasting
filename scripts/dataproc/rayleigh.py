from Py6S import SixS, Geometry, AeroProfile, AtmosProfile, Wavelength
import numpy as np, pandas as pd

s = SixS()

s.altitudes.set_sensor_satellite_level()
s.altitudes.set_target_sea_level()

s.geometry = Geometry.User.from_time_and_location(lat, lon, view_z, view_a)


s.aero_profile = AeroProfile.PredefinedType(AeroProfile.NoAerosols)
s.atmos_profile = AtmosProfile.PredefinedType(AtmosProfile.NoGaseousAbsorption)

wavelengths = np.arange(0.47, 0.51, 0.64)
results = []

for wv in wavelengths:
    s.wavelength = Wavelength(wv)
    s.run()
    results.append({'wavelength': wv, 'rayleigh_refl': s.outputs.atmospheric_intrinsic_reflectance})

results = pd.DataFrame(results)


