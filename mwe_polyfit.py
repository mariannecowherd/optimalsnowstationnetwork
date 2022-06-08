"""
weird trend and slope estimates from xr.polyfit, and unclear order of degrees
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

air = xr.tutorial.open_dataset("air_temperature").air
air = air.sel(lat=42.5,lon=210, time=slice('2013-01','2013-07'))
air = air - air.mean()

p = air.polyfit(dim='time', deg=1)
slope = p.polyfit_coefficients.sel(degree=1).values
intercept = p.polyfit_coefficients.sel(degree=0).values
print(intercept, slope)

x = xr.core.missing.get_clean_interp_index(air.time, "time")
import IPython; IPython.embed()
plt.plot(air.values)
plt.plot(intercept + np.arange(len(air.values))*slope)
plt.plot(intercept + x*slope)
plt.show()
