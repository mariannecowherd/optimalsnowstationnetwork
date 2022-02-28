"""
TEST
"""

import numpy as np
import xarray as xr
from sklearn.ensemble import RandomForestRegressor
import argparse
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', 
                    datefmt='%d-%b-%y %H:%M:%S', level=logging.DEBUG)

# read gridpoint info
parser = argparse.ArgumentParser()
parser.add_argument('--gridpoint', '-g', dest='g', type=int)
args = parser.parse_args()
gridpoint = args.g

# check if gridpoint is already computed
upscalepath = '/net/so4/landclim/bverena/large_files/opscaling/'

#largefilepath = '/net/so4/landclim/bverena/large_files/'
#from os.path import exists
#if exists(f'{largefilepath}mrso_benchmark_{gridpoint}_{modelname}_{experimentname}_{ensemblename}.nc'):
#    print(f'gridpoint {gridpoint} is already computed, skip')
#    quit()
#else:
#    print(f'gridpoint {gridpoint} is not yet computed, continue')

# load feature tables
logging.info('load feature tables')
mrso = xr.open_dataset(f'{upscalepath}mrso_ssp585.nc')['mrso'].load()
pred = xr.open_dataset(f'{upscalepath}pred_ssp585.nc').load()

# select land points
logging.info('select land points')
landmask = ~np.isnan(mrso.sel(time='2021-08'))
landmask.to_dataset(name='landmask').to_netcdf(f'{upscalepath}landmask_cmip6-ng.nc')
landmask = xr.open_dataset(f'{upscalepath}landmask_cmip6-ng.nc')['landmask'].squeeze()
landlat, landlon = np.where(landmask)
landpoints = np.arange(len(landlat))
landlat = xr.DataArray(landlat, dims='landpoints', coords=[landpoints]) 
landlon = xr.DataArray(landlon, dims='landpoints', coords=[landpoints])

mrso = mrso.isel(lat=landlat, lon=landlon)
pred = pred.isel(lat=landlat, lon=landlon)

# stack landpoints and time
logging.info('stack')
mrso = mrso.stack(datapoints=('landpoints','time'))
pred = pred.stack(datapoints=('landpoints','time')).to_array().T

# rf settings TODO later use GP
kwargs = {'n_estimators': 100, # TODO 100 this is debug
          'min_samples_leaf': 1, # those are all default values anyways
          'max_features': 'auto', 
          'max_samples': None, 
          'bootstrap': True,
          'warm_start': False,
          'n_jobs': 30, # set to number of trees
          'verbose': 2}

landpoints = np.unique(mrso.landpoints)
X_test = pred.sel(landpoints=gridpoint)
y_test = mrso.sel(landpoints=gridpoint)
X_train = pred.where(pred.landpoints != gridpoint, drop=True)
y_train = mrso.where(pred.landpoints != gridpoint, drop=True)

logging.info('train')
rf = RandomForestRegressor(**kwargs)
rf.fit(X_train, y_train)

logging.info('predict')
y_predict = xr.full_like(y_test, np.nan)
y_predict[:] = rf.predict(X_test)

corr = xr.corr(y_test, y_predict).item()
logging.info(f'{gridpoint}, {corr**2}')

logging.info('save')
#y_predict.to_netcdf(f'{largefilepath}mrso_benchmark_{gridpoint}_{modelname}_{experimentname}_{ensemblename}.nc')
