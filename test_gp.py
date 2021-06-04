"""
TEST
"""

import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor

largefilepath = '/net/so4/landclim/bverena/large_files/'
case = 'latlon'

def to_latlon(data):
    lsmfile = '/net/exo/landclim/data/dataset/ERA5_deterministic/recent/0.25deg_lat-lon_time-invariant/processed/regrid/era5_deterministic_recent.lsm.025deg.time-invariant.nc'
    lsm = xr.open_mfdataset(lsmfile, combine='by_coords')
    shape = lsm['lsm'].squeeze().shape
    landlat, landlon = np.where((lsm['lsm'].squeeze() > 0.8).load()) # land is 1, ocean is 0
    tmp = xr.DataArray(np.full((shape[0],shape[1]),np.nan), coords=[lsm.coords['lat'], lsm.coords['lon']], dims=['lat','lon'])
    tmp.values[landlat,landlon] = data
    return tmp

# load data
print('load data')
X_train = xr.open_dataarray(f'{largefilepath}X_train.nc')
y_train = xr.open_dataarray(f'{largefilepath}y_train.nc')
X_test = xr.open_dataarray(f'{largefilepath}X_test.nc')
y_test = xr.open_dataarray(f'{largefilepath}y_test.nc')
station_grid_lat = np.load(f'{largefilepath}station_grid_lat.npy', allow_pickle=True)
station_grid_lon = np.load(f'{largefilepath}station_grid_lon.npy', allow_pickle=True)

# normalise values
datamean = y_train.mean().values.copy()
datastd = y_train.std().values.copy()
y_train = (y_train - datamean) / datastd
y_test = (y_test - datamean) / datastd

# train GP on observed points 
print('train GP')
std_prior = 1**2 # std deviation of values
length_scale = 30.0 # std deviation in space
kernel = std_prior * Matern(length_scale, length_scale_bounds='fixed')
gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
gp.fit(X_train.T, y_train)

# predict GP on all other points of grid
print('predict GP')
predict, unc = gp.predict(X_test.T, return_std=True)

# convert back to worldmap, renormalise
print('back to worldmap')
y_train_empty = xr.full_like(y_train, np.nan)
y_predict = xr.full_like(y_test, np.nan)
y_unc = xr.full_like(y_test, np.nan)
y_predict[:] = predict
y_unc[:] = unc
datamap = to_latlon(xr.concat([y_test, y_train_empty], dim='landpoints'))
predmap = to_latlon(xr.concat([y_predict, y_train_empty], dim='landpoints'))
uncmap = to_latlon(xr.concat([y_unc, y_train_empty], dim='landpoints'))
datamap = datamap * datastd + datamean
predmap = predmap * datastd + datamean
uncmap = uncmap * datastd 

# plot prediction and uncertainty
print('plot')
proj = ccrs.PlateCarree()
fig = plt.figure(figsize=(20,3))
ax1 = fig.add_subplot(131, projection=proj)
ax2 = fig.add_subplot(132, projection=proj)
ax3 = fig.add_subplot(133, projection=proj)
datamap.plot(ax=ax1, transform=proj, cmap='coolwarm', vmin=230, vmax=310)
predmap.plot(ax=ax2, transform=proj, cmap='coolwarm', vmin=230, vmax=310)
uncmap.plot(ax=ax3, transform=proj, cmap='pink_r')
ax1.scatter(station_grid_lon, station_grid_lat, marker='x', s=5, c='indianred')
ax2.scatter(station_grid_lon, station_grid_lat, marker='x', s=5, c='indianred')
ax3.scatter(station_grid_lon, station_grid_lat, marker='x', s=5, c='indianred')
ax1.coastlines()
ax2.coastlines()
ax3.coastlines()
plt.savefig(f'gp_{case}.png')
