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

import argparse

# read gridpoint info
parser = argparse.ArgumentParser()
parser.add_argument('--gridpoint', '-g', dest='g', type=int)
args = parser.parse_args()
gridpoint = args.g

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
          'n_jobs': 100, # set to number of trees
          'verbose': 0}

from os.path import exists
landpoints = np.unique(mrso_land.landpoints)
modelname = 'CanESM5'
experimentname = 'historical'
ensemblename = 'r1i1p1f1'

if exists(f'{largefilepath}mrso_benchmark_{gridpoint}_{modelname}_{experimentname}_{ensemblename}.nc'):
    print(f'gridpoint {gridpoint} is already computed, skip')
    quit()
else:
    print(f'gridpoint {gridpoint} is not yet computed, continue')
    
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

y_predict.to_netcdf(f'{largefilepath}mrso_benchmark_{gridpoint}_{modelname}_{experimentname}_{ensemblename}.nc')
