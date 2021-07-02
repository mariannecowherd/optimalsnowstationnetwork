import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# open data map
largefilepath = '/net/so4/landclim/bverena/large_files/'
data = xr.open_dataarray(f'{largefilepath}df_gaps.nc')
data = data.sortby('country')

# calculate anomalies
mean = data.mean(dim='time')
std = data.std(dim='time')
data = (data - mean) / std

# plot
fig = plt.figure()
ax = fig.add_subplot(111)
fig.suptitle('monthly aggregated soil moisture standardised anomalies for all ISMN stations')
ax.imshow(np.flipud(data.values), vmin=-2, vmax=2, cmap='coolwarm_r')

# set ticklabels
xticks = [0] + np.cumsum(np.unique(data.country.values, return_counts=True)[1])[:-1].tolist()
xticklabels = np.unique(data.country.values)
yticks = np.arange(0,504,12*5)
yticklabels = np.arange(1979,2021,5)[::-1]
ax.set_xticks(xticks)
ax.set_yticks(yticks)
ax.set_xticklabels(xticklabels, rotation=90)
ax.set_yticklabels(yticklabels)
#data.plot(ax=ax, vmin=-2, vmax=2, cmap='coolwarm_r')
plt.show()
