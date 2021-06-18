import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

sltfile = '/net/exo/landclim/data/dataset/ERA5_deterministic/recent/0.25deg_lat-lon_time-invariant/processed/regrid/era5_deterministic_recent.slt.025deg.time-invariant.nc'
slt = xr.open_dataset(sltfile)['slt']
slt = np.round(slt)

largefilepath = '/net/so4/landclim/bverena/large_files/'
case = 'latlontime'
predmap = f'{largefilepath}RFpred_{case}.nc'
datamap = f'{largefilepath}ERA5_{case}.nc'
predmap = xr.open_dataarray(predmap)
datamap = xr.open_dataarray(datamap)
uncmap = f'{largefilepath}UncPred_{case}.nc'
uncmap = xr.open_dataarray(uncmap)

rmse = np.sqrt(((predmap - datamap)**2).mean(dim='time'))
unc = uncmap.mean(dim='time')

proj = ccrs.PlateCarree()
fig = plt.figure(figsize=(10,18))
ax1 = fig.add_subplot(311, projection=proj)
ax2 = fig.add_subplot(312, projection=proj)
ax3 = fig.add_subplot(313, projection=proj)
rmse.plot(ax=ax1)
unc.plot(ax=ax2)
slt.plot(ax=ax3)
plt.show()
plt.close()

# mean error per soil type
import IPython; IPython.embed()
