"""
TEST
"""

import cftime
#import concurrent
import numpy as np
import pandas as pd
import xarray as xr
import regionmask
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# load feature tables
largefilepath = '/cluster/work/climate/bverena/'
largefilepath = '/net/so4/landclim/bverena/large_files/'
mrso_land = xr.open_dataset(f'{largefilepath}mrso_land.nc')['mrso'].load()
pred_land = xr.open_dataset(f'{largefilepath}pred_land.nc')['__xarray_dataarray_variable__'].load()

mrso_land = mrso_land.set_index(datapoints=('landpoints','time'))
pred_land = pred_land.set_index(datapoints=('landpoints','time'))

# rf settings TODO later use GP
kwargs = {'n_estimators': 100, # TODO 100 this is debug
          'min_samples_leaf': 1, # those are all default values anyways
          'max_features': 'auto', 
          'max_samples': None, 
          'bootstrap': True,
          'warm_start': False,
          'n_jobs': 50, # set to number of trees
          'verbose': 0}

#mrso_pred = xr.full_like(mrso, np.nan)
largefilepath = '/net/so4/landclim/bverena/large_files/'
from os.path import exists
#mrso_pred = xr.open_dataset(f'{largefilepath}mrso_benchmark_{modelname}_{experimentname}_{ensemblename}.nc')['mrso']
landpoints = np.unique(mrso_land.landpoints)
modelname = 'CanESM5'
experimentname = 'historical'
ensemblename = 'r1i1p1f1'

for g, gridpoint in enumerate(landpoints): # random folds of observed gridpoints # LG says doesnot matter if random or regionally grouped, both has advantages and disadvantages, just do something and reason why

    # check if gridpoint is already computed
    #lat, lon = mrso_land.sel(landpoints=gridpoint).lat[0].item(), mrso_land.sel(landpoints=gridpoint).lon[0].item()
    #if ~np.isnan(mrso_pred.loc[:,lat,lon][0].item()):
    #    #print(mrso_pred.loc[:,lat,lon][0].item())
    #    print(f'gridpoint {g} is already computed, skip')
    #    continue # this point is already computed, skip
    #else:
    #    #print(mrso_pred.loc[:,lat,lon][0].item())
    #    print(f'gridpoint {g} is not yet computed, continue')
    if exists(f'{largefilepath}mrso_benchmark_{g}_{modelname}_{experimentname}_{ensemblename}.nc'):
        print(f'gridpoint {g} is already computed, skip')
        continue
    else:
        print(f'gridpoint {g} is not yet computed, continue')
        
    X_test = pred_land.sel(landpoints=gridpoint)
    y_test = mrso_land.sel(landpoints=gridpoint)
    X_train = pred_land.where(pred_land.landpoints != gridpoint, drop=True)
    y_train = mrso_land.where(pred_land.landpoints != gridpoint, drop=True)

    rf = RandomForestRegressor(**kwargs)
    rf.fit(X_train, y_train)

    y_predict = xr.full_like(y_test, np.nan)
    y_predict[:] = rf.predict(X_test)

    corr = xr.corr(y_test, y_predict).item()
    print(g, corr**2)

    y_predict.to_netcdf(f'{largefilepath}mrso_benchmark_{g}_{modelname}_{experimentname}_{ensemblename}.nc')
