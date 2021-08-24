"""
TEST
"""

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.ensemble import RandomForestRegressor

# check if gridpoint is already computed
largefilepath = '/cluster/work/climate/bverena/'
largefilepath = '/net/so4/landclim/bverena/large_files/'
modelname = 'CanESM5'
experimentname = 'historical'
ensemblename = 'r1i1p1f1'

# load feature tables
print('load feature tables')
mrso_hist = xr.open_dataset(f'{largefilepath}mrso_land_historical.nc').load()
pred_hist = xr.open_dataset(f'{largefilepath}pred_land_historical.nc').load()

mrso_ssp585 = xr.open_dataset(f'{largefilepath}mrso_land_ssp585.nc').load()
pred_ssp585 = xr.open_dataset(f'{largefilepath}pred_land_ssp585.nc').load()

#mrso_hist = mrso_hist.set_index(datapoints=('landpoints','time'))
#pred_hist = pred_hist.set_index(datapoints=('landpoints','time'))
#
#mrso_ssp585 = mrso_ssp585.set_index(datapoints=('landpoints','time'))
#pred_ssp585 = pred_ssp585.set_index(datapoints=('landpoints','time'))

# concat historical and future data
mrso_land = xr.concat([mrso_hist, mrso_ssp585], dim='time')
pred_land = xr.concat([pred_hist, pred_ssp585], dim='time')

# if upscale: delete non-observed datapoints
stations_cmip = xr.open_dataset(f'{largefilepath}stations_cmip.nc')
import IPython; IPython.embed()

# stack
mrso_land = mrso_land.stack(datapoints=('landpoints','time')).reset_index('datapoints')
pred_land = pred_land.stack(datapoints=('landpoints','time')).to_array().T.reset_index('datapoints')

# rf settings TODO later use GP
kwargs = {'n_estimators': 100, 
          'min_samples_leaf': 1, # those are all default values anyways
          'max_features': 'auto', 
          'max_samples': None, 
          'bootstrap': True,
          'warm_start': False,
          'n_jobs': 100, # set to number of trees
          'verbose': 0}

# divide into test and train data
years = np.arange(2025,2036)
mrso_res = xr.full_like(mrso_land.where(mrso_land['time.year'].isin(years), drop=True), np.nan)
mrso_res = mrso_res.unstack('datapoints')

for year in years:

    y_test = mrso_land.where(mrso_land['time.year'] == year, drop=True)
    y_train = mrso_land.where(mrso_land['time.year'] < year, drop=True)

    X_test = pred_land.where(pred_land['time.year'] == year, drop=True)
    X_train = pred_land.where(pred_land['time.year'] < year, drop=True)

    print('train')
    rf = RandomForestRegressor(**kwargs)
    rf.fit(X_train, y_train)

    print('predict')
    y_predict = xr.full_like(y_test, np.nan)
    y_predict[:] = rf.predict(X_test)

    corr = xr.corr(y_test, y_predict).item()
    print(year, corr**2)

    y_predict = y_predict.unstack('datapoints')
    mrso_res.loc[:,str(year)] = y_predict

import IPython; IPython.embed()
print('save')
mrso_res.to_netcdf(f'{largefilepath}mrso_benchmark_{modelname}_future_{ensemblename}.nc')
