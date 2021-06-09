"""
TEST
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from sklearn.ensemble import RandomForestRegressor

largefilepath = '/net/so4/landclim/bverena/large_files/'
case = 'yearly'

def to_latlon(data):
    t = data.time.shape[0]
    lsmfile = '/net/exo/landclim/data/dataset/ERA5_deterministic/recent/0.25deg_lat-lon_time-invariant/processed/regrid/era5_deterministic_recent.lsm.025deg.time-invariant.nc'
    lsm = xr.open_mfdataset(lsmfile, combine='by_coords')
    shape = lsm['lsm'].squeeze().shape
    landlat, landlon = np.where((lsm['lsm'].squeeze() > 0.8).load()) # land is 1, ocean is 0
    tmp = xr.DataArray(np.full((t,shape[0],shape[1]),np.nan), coords=[data.coords['time'],lsm.coords['lat'], lsm.coords['lon']], dims=['time','lat','lon'])
    tmp.values[:,landlat,landlon] = data
    return tmp

# load data
print('load data')
X_train = xr.open_dataarray(f'{largefilepath}X_train_{case}.nc')
y_train = xr.open_dataarray(f'{largefilepath}y_train_{case}.nc')
X_test = xr.open_dataarray(f'{largefilepath}X_test_{case}.nc')
y_test = xr.open_dataarray(f'{largefilepath}y_test_{case}.nc')
station_grid_lat = np.load(f'{largefilepath}station_grid_lat.npy', allow_pickle=True)
station_grid_lon = np.load(f'{largefilepath}station_grid_lon.npy', allow_pickle=True)

# normalise values
datamean = y_train.mean().values.copy()
datastd = y_train.std().values.copy()
y_train = (y_train - datamean) / datastd
y_test = (y_test - datamean) / datastd

# train RF on observed points 
n_trees = 100
kwargs = {'n_estimators': n_trees,
          'min_samples_leaf': 2,
          'max_features': 0.5, 
          'max_samples': 0.5, 
          'bootstrap': True,
          'warm_start': True,
          'n_jobs': None, # set to number of trees
          'verbose': 0}
rf = RandomForestRegressor(**kwargs)
# TODO idea: add penalty for negative soil moisture?
rf.fit(X_train, y_train)

res = np.zeros((n_trees, X_test.datapoints.shape[0]))
for t, tree in enumerate(rf.estimators_):
    print(t)
    res[t,:] = tree.predict(X_test)
mean = np.mean(res, axis=0)
upper, lower = np.percentile(res, [95 ,5], axis=0)

# predict GP on all other points of grid
y_train_empty = xr.full_like(y_train, np.nan)
y_predict = xr.full_like(y_test, np.nan)
y_unc = xr.full_like(y_test, np.nan)
y_predict[:] = mean
y_unc[:] = (upper - lower)
datamap = to_latlon(xr.concat([y_test, y_train_empty], dim='datapoints').set_index(datapoints=('time', 'landpoints')).unstack('datapoints'))
predmap = to_latlon(xr.concat([y_predict, y_train_empty], dim='datapoints').set_index(datapoints=('time', 'landpoints')).unstack('datapoints'))
uncmap = to_latlon(xr.concat([y_unc, y_train_empty], dim='datapoints').set_index(datapoints=('time', 'landpoints')).unstack('datapoints'))
datamean = xr.open_dataarray(f'{largefilepath}datamean_{case}.nc')
datastd = xr.open_dataarray(f'{largefilepath}datastd_{case}.nc')
datamap = datamap * datastd + datamean
predmap = predmap * datastd + datamean
uncmap = uncmap * datastd 

# plot prediction and uncertainty
proj = ccrs.PlateCarree()
fig = plt.figure(figsize=(20,3))
ax1 = fig.add_subplot(131, projection=proj)
ax2 = fig.add_subplot(132, projection=proj)
ax3 = fig.add_subplot(133, projection=proj)
datamap.mean(dim='time').plot(ax=ax1, transform=proj, cmap='Blues', vmin=0, vmax=1)
(datamap - predmap).mean(dim='time').plot(ax=ax2, transform=proj, cmap='coolwarm', vmin=-0.5, vmax=0.5)
uncmap.mean(dim='time').plot(ax=ax3, transform=proj, cmap='pink_r')
ax1.scatter(station_grid_lon, station_grid_lat, marker='x', s=5, c='indianred')
ax2.scatter(station_grid_lon, station_grid_lat, marker='x', s=5, c='indianred')
ax3.scatter(station_grid_lon, station_grid_lat, marker='x', s=5, c='indianred')
ax1.coastlines()
ax2.coastlines()
ax3.coastlines()
plt.savefig(f'rf_{case}.png')
