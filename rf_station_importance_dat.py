import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime

# open data map
largefilepath = '/net/so4/landclim/bverena/large_files/'
data = xr.open_dataarray(f'{largefilepath}df_gaps.nc')

icalc = False
if icalc:
    # load feature space X
    print('load data')
    largefilepath = '/net/so4/landclim/bverena/large_files/'
    era5path_invariant = '/net/exo/landclim/data/dataset/ERA5_deterministic/recent/0.25deg_lat-lon_time-invariant/processed/regrid/'
    invarnames = ['lsm','z','slor','cvl','cvh', 'tvl', 'tvh']
    freq = '1m'
    filenames_constant = [f'{era5path_invariant}era5_deterministic_recent.{varname}.025deg.time-invariant.nc' for varname in invarnames]
    filenames_variable = [f'{largefilepath}era5_deterministic_recent.temp.025deg.{freq}.max.nc', 
                         f'{largefilepath}era5_deterministic_recent.temp.025deg.{freq}.min.nc',
                         f'{largefilepath}era5_deterministic_recent.var.025deg.{freq}.mean.nc',
                         f'{largefilepath}era5_deterministic_recent.var.025deg.{freq}.roll.nc',
                         f'{largefilepath}era5_deterministic_recent.precip.025deg.{freq}.sum.nc']
    constant = xr.open_mfdataset(filenames_constant, combine='by_coords').load()
    variable = xr.open_mfdataset(filenames_variable, combine='by_coords').load()
    landlat, landlon = np.where((constant['lsm'].squeeze() > 0.8).load()) # land is 1, ocean is 0
    datalat, datalon = constant.lat.values, constant.lon.values
    #constant = constant.isel(lon=xr.DataArray(landlon, dims='landpoints'),
    #                         lat=xr.DataArray(landlat, dims='landpoints')).squeeze()
    #variable = variable.isel(lon=xr.DataArray(landlon, dims='landpoints'),
    #                         lat=xr.DataArray(landlat, dims='landpoints')).squeeze()
    #constant['latdat'] = ('landpoints', constant.lat.values)
    #constant['londat'] = ('landpoints', constant.lon.values)
    constant = constant.squeeze()

    # normalise data # for now omitted because expensive
    #print('normalise data')
    #mean = data.mean(dim='time')
    #std = data.std(dim='time')
    #data = (data - mean) / std
    #
    #mean = variable.mean(dim='time')
    #std = variable.std(dim='time')
    #variable = (variable - mean) / std

    # stack constant maps and merge with variables
    print('stack data')
    ntimesteps = variable.coords['time'].size # TODO use climfill package
    constant = constant.expand_dims({'time': ntimesteps}, axis=1)
    constant['time'] = variable['time']
    variable = variable.merge(constant)

    # select only station points in X (variable) data
    print('select stations')
    variable = variable.sel(lat=data.lat_grid, lon=data.lon_grid)

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

    pred = xr.full_like(data, np.nan)
    for s, station in enumerate(variable.stations):
        now = datetime.now()
        print(now, s)

        # define test and train data
        y_train = data.where(data.stations != station, drop=True)
        y_test = data.sel(stations=station)
        X_train = variable.where(data.stations != station, drop=True)
        X_test = variable.sel(stations=station)

        # remove timesteps w/o measurement or outside ERA5
        y_train = y_train.where(~np.isnan(y_test) & y_test.time.isin(X_test.time), drop=True)
        y_test = y_test.where(~np.isnan(y_test) & y_test.time.isin(X_test.time), drop=True)
        X_train = X_train.where(~np.isnan(y_test) & y_test.time.isin(X_test.time), drop=True)
        X_test = X_test.where(~np.isnan(y_test) & y_test.time.isin(X_test.time), drop=True)

        # stack along axis
        X_test = X_test.to_array().T
        y_train = y_train.stack(datapoints=('time','stations')).reset_index('datapoints')
        X_train = X_train.stack(datapoints=('time','stations')).reset_index('datapoints').to_array().T
        X_train = X_train.where(~np.isnan(y_train), drop=True)
        y_train = y_train.where(~np.isnan(y_train), drop=True)
        if X_train.size == 0 or y_train.size == 0:
            print(s, 'skipped')
            continue

        # train QRF
        rf = RandomForestRegressor(**kwargs)
        rf.fit(X_train, y_train)

        y_predict = rf.predict(X_test)
        pred.loc[X_test.time,s] = y_predict
        #res = np.zeros((n_trees, X_test.shape[0]))
        #for t, tree in enumerate(rf.estimators_):
        #    res[t,:] = tree.predict(X_test)
        #mean = np.mean(res, axis=0)
        #upper, lower = np.percentile(res, [95 ,5], axis=0)

    pred.to_netcdf(f'{largefilepath}pred_dat.nc')
else:
    pred = xr.open_dataset(f'{largefilepath}pred_dat.nc')
    pred = pred['__xarray_dataarray_variable__']

# calculate RMSE of CV
rmse = np.sqrt(((data - pred)**2).mean(dim='time'))

# plot
proj = ccrs.PlateCarree()
fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(111, projection=proj)
ax.scatter(pred.lon, pred.lat, c=rmse.values, marker='x', s=1, cmap='autumn_r', vmin=0, vmax=3)
ax.coastlines()
plt.show()

# plot per koeppen climate
koeppen_rmse = []
for i in range(14):
    koeppen_rmse.append(rmse.where(data.koeppen_simple == i, drop=True).mean().item())
reduced_names = ['Ocean','Af','Am','Aw','BW','BS','Cs','Cw','Cf','Ds','Dw','Df','ET','EF']
plt.bar(reduced_names, koeppen_rmse)
plt.show()
