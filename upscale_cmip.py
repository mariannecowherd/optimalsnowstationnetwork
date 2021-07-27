"""
TEST
"""

import cftime
import numpy as np
import pandas as pd
import xarray as xr
import regionmask
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from sklearn.ensemble import RandomForestRegressor

# read CMIP6 files
varnames_predictors = ['tas','tasmax','pr','hfls','rsds'] # TODO add lai, topo, lagged features, treeFrac?
varname_predictand = 'mrso'

def rename_vars(data):
    varname = list(data.keys())[0]
    _, _, modelname, _, ensemblename, _ = data.encoding['source'].split('_')
    data = data.rename({varname: f'{modelname}_{ensemblename}'})
    try:
        data = data.drop_vars('file_qf')
    except ValueError:
        pass
    #data = data.drop_dims('height')
    if isinstance(data.time[0].item(), cftime._cftime.DatetimeNoLeap):
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
tasmax = open_cmip_suite('tasmax', modelname, experimentname, ensemblename)
tas = open_cmip_suite('tas', modelname, experimentname, ensemblename)
pr = open_cmip_suite('pr', modelname, experimentname, ensemblename)
hfls = open_cmip_suite('hfls', modelname, experimentname, ensemblename)
rsds = open_cmip_suite('rsds', modelname, experimentname, ensemblename)

# cut out Greenland and Antarctica
n_greenland = regionmask.defined_regions.natural_earth.countries_110.map_keys('Greenland')
n_antarctica = regionmask.defined_regions.natural_earth.countries_110.map_keys('Antarctica')
mask = regionmask.defined_regions.natural_earth.countries_110.mask(mrso)
mrso = mrso.where((mask != n_greenland) & (mask != n_antarctica) & (~np.isnan(mask)))

# merge predictors into one dataset 
pred = xr.merge([tasmax,tas,pr,hfls,rsds])

# select timerange of ismn
mrso = mrso.sel(time=slice('1960','2014'))
pred = pred.sel(time=slice('1960','2014'))

# not sure why this is necessary # TODO
mrso = mrso.resample(time='1M').mean()
pred = pred.resample(time='1M').mean()

# read station data
largefilepath = '/net/so4/landclim/bverena/large_files/'
stations = xr.open_dataset(f'{largefilepath}df_gaps.nc')
stations = stations['__xarray_dataarray_variable__']
stations = stations.sel(time=slice('1960','2014'))

# calculate deseasonalised anomaly 
seasonal_mean = mrso.groupby('time.month').mean()
seasonal_std = mrso.groupby('time.month').std()
mrso = (mrso.groupby('time.month') - seasonal_mean) 
mrso = mrso.groupby('time.month') / seasonal_std

seasonal_mean = pred.groupby('time.month').mean()
seasonal_std = pred.groupby('time.month').std()
seasonal_std['rsds'] = seasonal_std['rsds'].where(seasonal_std['rsds'] != 0,1) # zero sh in in high lat winter leads to nan values
pred = (pred.groupby('time.month') - seasonal_mean) 
pred = pred.groupby('time.month') / seasonal_std

seasonal_mean = stations.groupby('time.month').mean()
seasonal_std = stations.groupby('time.month').std()
stations = (stations.groupby('time.month') - seasonal_mean) 
stations = stations.groupby('time.month') / seasonal_std

# regrid station data to CMIP6 grid
landmask = ~np.isnan(mrso[0,:,:]).copy(deep=True)
obsmask = xr.full_like(landmask, False)
unobsmask = landmask.copy(deep=True)
lat_cmip = []
lon_cmip = []
for lat, lon in zip(stations.lat, stations.lon):
    point = landmask.sel(lat=lat, lon=lon, method='nearest')
    
    if landmask.loc[point.lat,point.lon].item(): # obs gridpoint if station contained and on CMIP land
        obsmask.loc[point.lat, point.lon] = True
    unobsmask.loc[point.lat, point.lon] = False


    lat_cmip.append(point.lat.item())
    lon_cmip.append(point.lon.item())
stations = stations.assign_coords(lat_cmip=('stations',lat_cmip))
stations = stations.assign_coords(lon_cmip=('stations',lon_cmip))

# divide into obs and unobs data
obslat, obslon = np.where(obsmask)
obslat, obslon = xr.DataArray(obslat, dims='obspoints'), xr.DataArray(obslon, dims='obspoints')

mrso_obs = mrso.isel(lat=obslat, lon=obslon)
pred_obs = pred.isel(lat=obslat, lon=obslon)

unobslat, unobslon = np.where(unobsmask)
unobslat, unobslon = xr.DataArray(unobslat, dims='unobspoints'), xr.DataArray(unobslon, dims='unobspoints')

mrso_unobs = mrso.isel(lat=unobslat, lon=unobslon)
pred_unobs = pred.isel(lat=unobslat, lon=unobslon)

# flatten to skikit-learn digestable table
X_train = pred_obs.stack(datapoints=('obspoints','time')).to_array().T
y_train = mrso_obs.stack(datapoints=('obspoints','time'))

X_test = pred_unobs.stack(datapoints=('unobspoints','time')).to_array().T
y_predict = mrso_unobs.stack(datapoints=('unobspoints','time'))

# rf settings TODO later use GP
n_trees = 100
kwargs = {'n_estimators': n_trees,
          'min_samples_leaf': 2,
          'max_features': 'auto', 
          'max_samples': None, 
          'bootstrap': True,
          'warm_start': True,
          'n_jobs': None, # set to number of trees
          'verbose': 0}

#res = xr.full_like(mrso_unobs.sel(time=slice('1979','2015')), np.nan)
rf = RandomForestRegressor(**kwargs)
rf.fit(X_train, y_train)
y_predict[:] = rf.predict(X_test)

# back to worldmap
y_predict = y_predict.unstack('datapoints').T
mrso_pred = xr.full_like(mrso, np.nan)
mrso_pred.values[:,unobslat,unobslon] = y_predict

# save as netcdf
mrso_pred.to_netcdf(f'{largefilepath}mrso_pred_{modelname}_{experimentname}_{ensemblename}.nc') # TODO add orig values from mrso_obs
mrso.to_netcdf(f'{largefilepath}mrso_orig_{modelname}_{experimentname}_{ensemblename}.nc')

# loop over years
#for year in np.arange(1976,2015):
#    next_year = str(year + 1)
#    year = str(year)
#    
#    rf = RandomForestRegressor(**kwargs)
#    rf.fit(X_train.sel(time=slice('1960',year)), 
#           y_train.sel(time=slice('1960',year)))
#    
#    res = xr.full_like(X_test.sel(variable='pr',time=slice(year, next_year)).squeeze(), np.nan)
#    res[:] = rf.predict(X_test.sel(time=slice(year, next_year))) 
#    import IPython; IPython.embed()
