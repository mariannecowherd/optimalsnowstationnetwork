"""
TEST
"""

import cftime
import concurrent
import cartopy.crs as ccrs
import numpy as np
import pandas as pd
import xarray as xr
import regionmask
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from sklearn.ensemble import RandomForestRegressor

# read CMIP6 files
def rename_vars(data):
    varname = list(data.keys())[0]
    _, _, modelname, _, ensemblename, _ = data.encoding['source'].split('_')
    data = data.rename({varname: f'{modelname}_{ensemblename}'})
    try:
        data = data.drop_vars('file_qf')
    except ValueError:
        pass
    #data = data.drop_dims('height')
    if isinstance(data.time[0].item(), cftime._cftime.DatetimeNoLeap): # TODO add in again once several models used
        data['time'] = data.indexes['time'].to_datetimeindex()
    if isinstance(data.time[0].item(), cftime._cftime.Datetime360Day):
        data['time'] = data.indexes['time'].to_datetimeindex()
    return data

def open_cmip_suite(varname, modelname, experimentname, ensemblename):
    cmip6_path = f'/net/atmos/data/cmip6-ng/{varname}/mon/g025/'
    data = xr.open_mfdataset(f'{cmip6_path}{varname}*_{modelname}_{experimentname}_{ensemblename}_*.nc', 
                             combine='nested', # timestamp problem
                             concat_dim=None,  # timestamp problem
                             preprocess=rename_vars)  # rename vars to model name
    data = data.to_array().rename(varname)
    data.coords['lon'] = (data.coords['lon'] + 180) % 360 - 180
    data = data.sortby('lon')
    data = data.mean(dim='variable').load() # here for now

    return data

modelname = 'CanESM5'
experimentname = 'historical'
ensemblename = 'r1i1p1f1'
mrso = open_cmip_suite('mrso', modelname, experimentname, ensemblename)
tas = open_cmip_suite('tas', modelname, experimentname, ensemblename)
pr = open_cmip_suite('pr', modelname, experimentname, ensemblename)

# cut out Greenland and Antarctica
n_greenland = regionmask.defined_regions.natural_earth.countries_110.map_keys('Greenland')
n_antarctica = regionmask.defined_regions.natural_earth.countries_110.map_keys('Antarctica')
mask = regionmask.defined_regions.natural_earth.countries_110.mask(mrso)
mrso = mrso.where((mask != n_greenland) & (mask != n_antarctica) & (~np.isnan(mask)))

# create lagged features
tas_1month = tas.copy(deep=True).shift(time=1, fill_value=0).rename('tas_1m')
tas_2month = tas.copy(deep=True).shift(time=2, fill_value=0).rename('tas_2m')
tas_3month = tas.copy(deep=True).shift(time=3, fill_value=0).rename('tas_3m')
tas_4month = tas.copy(deep=True).shift(time=4, fill_value=0).rename('tas_4m')
tas_5month = tas.copy(deep=True).shift(time=5, fill_value=0).rename('tas_5m')
tas_6month = tas.copy(deep=True).shift(time=6, fill_value=0).rename('tas_6m')
tas_7month = tas.copy(deep=True).shift(time=7, fill_value=0).rename('tas_7m')
tas_8month = tas.copy(deep=True).shift(time=8, fill_value=0).rename('tas_8m')
tas_9month = tas.copy(deep=True).shift(time=9, fill_value=0).rename('tas_9m')
tas_10month = tas.copy(deep=True).shift(time=10, fill_value=0).rename('tas_10m')
tas_11month = tas.copy(deep=True).shift(time=11, fill_value=0).rename('tas_11m')
tas_12month = tas.copy(deep=True).shift(time=12, fill_value=0).rename('tas_12m')

pr_1month = pr.copy(deep=True).shift(time=1, fill_value=0).rename('pr_1m') 
pr_2month = pr.copy(deep=True).shift(time=2, fill_value=0).rename('pr_2m')
pr_3month = pr.copy(deep=True).shift(time=3, fill_value=0).rename('pr_3m')
pr_4month = pr.copy(deep=True).shift(time=4, fill_value=0).rename('pr_4m')
pr_5month = pr.copy(deep=True).shift(time=5, fill_value=0).rename('pr_5m')
pr_6month = pr.copy(deep=True).shift(time=6, fill_value=0).rename('pr_6m')
pr_7month = pr.copy(deep=True).shift(time=7, fill_value=0).rename('pr_7m')
pr_8month = pr.copy(deep=True).shift(time=8, fill_value=0).rename('pr_8m')
pr_9month = pr.copy(deep=True).shift(time=9, fill_value=0).rename('pr_9m')
pr_10month = pr.copy(deep=True).shift(time=10, fill_value=0).rename('pr_10m')
pr_11month = pr.copy(deep=True).shift(time=11, fill_value=0).rename('pr_11m')
pr_12month = pr.copy(deep=True).shift(time=12, fill_value=0).rename('pr_12m')

# merge predictors into one dataset 
pred = xr.merge([tas, tas_1month, tas_2month, tas_3month, tas_4month, tas_5month, tas_6month,
                 tas_7month, tas_8month, tas_9month, tas_10month, tas_11month, tas_12month,
                 pr, pr_1month, pr_2month, pr_3month, pr_4month, pr_5month, pr_6month,
                 pr_7month, pr_8month, pr_9month, pr_10month, pr_11month, pr_12month])

# select timerange of ismn
mrso = mrso.sel(time=slice('1960','2014'))
pred = pred.sel(time=slice('1960','2014'))

# not sure why this is necessary # TODO
#mrso = mrso.resample(time='1M').mean()
#pred = pred.resample(time='1M').mean()

# calculate deseasonalised anomaly 
mrso_seasonal_mean = mrso.groupby('time.month').mean()
mrso_seasonal_std = mrso.groupby('time.month').std()
mrso = (mrso.groupby('time.month') - mrso_seasonal_mean) 
mrso = mrso.groupby('time.month') / mrso_seasonal_std

seasonal_mean = pred.groupby('time.month').mean() # for reasoning see crossval file
#seasonal_std = pred.groupby('time.month').std()
pred = (pred.groupby('time.month') - seasonal_mean) 
#pred = pred.groupby('time.month') / seasonal_std

# remove oceans
landmask = ~np.isnan(mrso[0,:,:]).copy(deep=True)
landlat, landlon = np.where(landmask)
landpoints = np.arange(len(landlat))
landlat, landlon = xr.DataArray(landlat, dims='landpoints', coords=[landpoints]), xr.DataArray(landlon, dims='landpoints', coords=[landpoints])

mrso_land = mrso.isel(lat=landlat, lon=landlon)
pred_land = pred.isel(lat=landlat, lon=landlon)

# stack
mrso_land = mrso_land.stack(datapoints=('landpoints','time'))
pred_land = pred_land.stack(datapoints=('landpoints','time')).to_array().T

# save feature tables
largefilepath = '/net/so4/landclim/bverena/large_files/'
mrso_land.reset_index('datapoints').to_netcdf(f'{largefilepath}mrso_land.nc')
pred_land.reset_index('datapoints').to_netcdf(f'{largefilepath}pred_land.nc')
mrso.to_netcdf(f'{largefilepath}mrso_test.nc')

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

icalc = False
if icalc:
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

else:
    mrso_pred = xr.full_like(mrso, np.nan)
    for g, gridpoint in enumerate(landpoints): 
        try:
            y_predict = xr.open_dataset(f'{largefilepath}mrso_benchmark_{g}_{modelname}_{experimentname}_{ensemblename}.nc')
            mrso_pred.loc[:,y_predict.lat,y_predict.lon] = y_predict['mrso']
        except FileNotFoundError:
            continue
        else:
            print(f'gridpoint {g} processed')

    mrso_pred = (mrso_pred.groupby('time.month') * mrso_seasonal_std)
    mrso_pred = (mrso_pred.groupby('time.month') + mrso_seasonal_mean) 

    mrso_pred.to_netcdf(f'{largefilepath}mrso_benchmark_{modelname}_{experimentname}_{ensemblename}.nc')
