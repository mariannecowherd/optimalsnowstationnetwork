import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# open data map
largefilepath = '/net/so4/landclim/bverena/large_files/'
case = 'latlontime'
predmap = f'{largefilepath}RFpred_{case}.nc'
datamap = f'{largefilepath}ERA5_{case}.nc'
predmap = xr.open_dataarray(predmap)
datamap = xr.open_dataarray(datamap)
uncmap = f'{largefilepath}UncPred_{case}.nc'
uncmap = xr.open_dataarray(uncmap)
stations = pd.read_csv(f'{largefilepath}station_info_grid.csv')
rmse = np.sqrt(((predmap - datamap)**2).mean(dim='time'))

# plot prediction and uncertainty
proj = ccrs.PlateCarree()
fig = plt.figure(figsize=(50,8))
ax1 = fig.add_subplot(131, projection=proj)
ax2 = fig.add_subplot(132, projection=proj)
ax3 = fig.add_subplot(133, projection=proj)
datamap.mean(dim='time').plot(ax=ax1, transform=proj, cmap='Blues', vmin=0, vmax=1)
rmse.plot(ax=ax2, transform=proj, cmap='pink_r', vmin=0, vmax=0.5)
uncmap.mean(dim='time').plot(ax=ax3, transform=proj, cmap='pink_r', vmin=0, vmax=0.5)
ax1.scatter(stations.lon_grid, stations.lat_grid, marker='x', s=5, c='indianred')
ax2.scatter(stations.lon_grid, stations.lat_grid, marker='x', s=5, c='indianred')
ax3.scatter(stations.lon_grid, stations.lat_grid, marker='x', s=5, c='indianred')
ax1.coastlines()
ax2.coastlines()
ax3.coastlines()
ax1.set_title('ERA5 mean soil moisture (1979 - 2015)')
ax2.set_title('mean prediction RMSE between ERA5 and RF prediction')
ax3.set_title('mean uncertainty of RF prediction (tree quantiles)')
plt.savefig(f'rf_{case}.png')
