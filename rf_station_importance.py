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
case = 'latlontime'

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
X = xr.open_dataarray(f'{largefilepath}X_train_{case}.nc')
y = xr.open_dataarray(f'{largefilepath}y_train_{case}.nc')
y_other = xr.open_dataarray(f'{largefilepath}y_test_{case}.nc')
y_other[:] = np.nan

# normalise values
print('normalise values')
datamean = y.mean().values.copy()
datastd = y.std().values.copy()
y = (y - datamean) / datastd

# rf settings
n_trees = 100
kwargs = {'n_estimators': n_trees,
          'min_samples_leaf': 2,
          'max_features': 'auto', 
          'max_samples': None, 
          'bootstrap': True,
          'warm_start': True,
          'n_jobs': None, # set to number of trees
          'verbose': 0}

# drop one station and predict this station
y_predict = xr.full_like(y, np.nan)
y_unc = xr.full_like(y, np.nan)

print('fit and predict each station')
all_stations = np.unique(y.landpoints.values)
i = 0
lat = []
lon = []
for landpoint in all_stations:
    print(f'landpoint {landpoint}')
    y_train = y.where(y.landpoints != landpoint, drop=True)
    y_test = y.where(y.landpoints == landpoint, drop=True)

    X_train = X.where(y.landpoints != landpoint, drop=True)
    X_test = X.where(y.landpoints == landpoint, drop=True)

    # train RF on observed points 
    rf = RandomForestRegressor(**kwargs)
    rf.fit(X_train, y_train)

    res = np.zeros((n_trees, X_test.datapoints.shape[0]))
    for t, tree in enumerate(rf.estimators_):
        res[t,:] = tree.predict(X_test)
    mean = np.mean(res, axis=0)
    upper, lower = np.percentile(res, [95 ,5], axis=0)

    # save results
    y_predict[y.landpoints == landpoint] = mean
    y_unc[y.landpoints == landpoint] = (upper - lower)
    lat.append(y[y.landpoints == landpoint].lat.values[0])
    lon.append(y[y.landpoints == landpoint].lon.values[0])
    i += 1
    #if i > 10:
    #    break

# stack to landpoints
y_predict = y_predict.set_index(datapoints=('time', 'landpoints')).unstack('datapoints') 
y_unc = y_unc.set_index(datapoints=('time', 'landpoints')).unstack('datapoints') 
y = y.set_index(datapoints=('time', 'landpoints')).unstack('datapoints') 
rmse = np.sqrt(((y - y_predict)**2).mean(dim='time'))
munc = y_unc.mean(dim='time') 
lat, lon = y.lat.mean(dim='time'), y.lon.mean(dim='time')

# plot
lsmfile = '/net/exo/landclim/data/dataset/ERA5_deterministic/recent/0.25deg_lat-lon_time-invariant/processed/regrid/era5_deterministic_recent.lsm.025deg.time-invariant.nc'
lsm = xr.open_mfdataset(lsmfile, combine='by_coords')['lsm']
proj = ccrs.PlateCarree()
fig = plt.figure(figsize=(50,8))
ax1 = fig.add_subplot(121, projection=proj)
ax2 = fig.add_subplot(122, projection=proj)
ax1.coastlines()
ax2.coastlines()
(lsm>0.8).plot(ax=ax1, cmap='Greys', vmin=0, vmax=10, add_colorbar=False)
(lsm>0.8).plot(ax=ax2, cmap='Greys', vmin=0, vmax=10, add_colorbar=False)
ax1.scatter(lon, lat, c=rmse, cmap='Reds')
ax2.scatter(lon, lat, c=munc, cmap='Reds')
ax1.set_title('mean prediction RMSE between ERA5 and RF prediction')
ax2.set_title('mean uncertainty of RF prediction (tree quantiles)')
plt.show()
import IPython; IPython.embed()




quit()

# predict GP on all other points of grid
pred = to_latlon(xr.concat([y_other, y_predict], dim='datapoints').set_index(datapoints=('time', 'landpoints')).unstack('datapoints'))
unc = to_latlon(xr.concat([y_other, y_unc], dim='datapoints').set_index(datapoints=('time', 'landpoints')).unstack('datapoints'))
orig = to_latlon(xr.concat([y_other, y_train], dim='datapoints').set_index(datapoints=('time', 'landpoints')).unstack('datapoints'))
orig = orig * datastd + datamean
pred = pred * datastd + datamean
unc = unc * datastd 

# save results
datamap.to_netcdf(f'{largefilepath}ERA5_{case}.nc')
predmap.to_netcdf(f'{largefilepath}RFpred_{case}.nc')
uncmap.to_netcdf(f'{largefilepath}UncPred_{case}.nc')
