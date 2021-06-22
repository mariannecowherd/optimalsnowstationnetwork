import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# open data map
largefilepath = '/net/so4/landclim/bverena/large_files/'
data = xr.open_dataarray(f'{largefilepath}df_gaps.nc')

# calculate anomalies
mean = data.mean(dim='time')
std = data.std(dim='time')
data = (data - mean) / std

# plot
data.plot(vmin=-2, vmax=2, cmap='coolwarm_r')
plt.show()
