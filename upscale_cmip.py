"""
TEST
"""

import cftime
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from sklearn.ensemble import RandomForestRegressor

# read CMIP6 files
varnames_predictors = ['tas','tasmax','pr','hfls','rsds']
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

def open_cmip_suite(varname, experimentname, ensemblename):
    cmip6_path = f'/net/atmos/data/cmip6-ng/{varname}/mon/g025/'
    data = xr.open_mfdataset(f'{cmip6_path}{varname}*_{experimentname}_{ensemblename}_*.nc', 
                             combine='nested', # timestamp problem
                             concat_dim=None,  # timestamp problem
                             preprocess=rename_vars)  # rename vars to model name
    data = data.to_array().rename(varname)
    data.coords['lon'] = (data.coords['lon'] + 180) % 360 - 180
    data = data.sortby('lon')
    data = data.mean(dim='variable').load() # here for now

    return data

experimentname = 'historical'
ensemblename = 'r1i1p1f1'
mrso = open_cmip_suite('mrso', experimentname, ensemblename)
tasmax = open_cmip_suite('tasmax', experimentname, ensemblename)
tas = open_cmip_suite('tas', experimentname, ensemblename)
pr = open_cmip_suite('pr', experimentname, ensemblename)
hfls = open_cmip_suite('hfls', experimentname, ensemblename)
rsds = open_cmip_suite('rsds', experimentname, ensemblename)

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
pred = (pred.groupby('time.month') - seasonal_mean) 
pred = pred.groupby('time.month') / seasonal_std

seasonal_mean = stations.groupby('time.month').mean()
seasonal_std = stations.groupby('time.month').std()
stations = (stations.groupby('time.month') - seasonal_mean) 
stations = stations.groupby('time.month') / seasonal_std

# regrid station data to CMIP6 grid
lat_cmip = []
lon_cmip = []
for lat, lon in zip(stations.lat, stations.lon):
    point = mrso.sel(lat=lat, lon=lon, method='nearest')
    lat_cmip.append(point.lat.item())
    lon_cmip.append(point.lon.item())
stations = stations.assign_coords(lat_cmip=('stations',lat_cmip))
stations = stations.assign_coords(lon_cmip=('stations',lon_cmip))
latlon_unique = np.unique(np.array([lat_cmip, lon_cmip]), axis=1)

gridpoints = xr.DataArray(np.zeros((stations.shape[0],latlon_unique.shape[1])),
                          dims=['time','latlon'],
                          coords=[stations.time, np.arange(508)])
lat_mesh, lon_mesh = np.meshgrid(mrso.lat, mrso.lon) # coords need to be in mesh format
for i in range(latlon_unique.shape[1]):
    lat, lon = latlon_unique[:,i]
    gridpoint_mean = stations.where((stations.lat_cmip == lat) & (stations.lon_cmip == lon), drop=True).mean(dim='stations')
    gridpoints.loc[:,i] = gridpoint_mean
    mask = ((lat_mesh == lat) & (lon_mesh == lon))
    lat_mesh[mask] = np.nan
    lon_mesh[mask] = np.nan

gridpoints = gridpoints.assign_coords(lat_cmip=('latlon',latlon_unique[0,:]))
gridpoints = gridpoints.assign_coords(lon_cmip=('latlon',latlon_unique[1,:]))

lat_mesh = lat_mesh[~np.isnan(lat_mesh)]
lon_mesh = lon_mesh[~np.isnan(lon_mesh)]
latlon_unobs = np.array([lat_mesh, lon_mesh])
arr_unobs = xr.DataArray(np.zeros((latlon_unobs.shape[1])),
                         dims = ['latlon'],
                         coords = [np.arange(9860)])
arr_unobs = arr_unobs.assign_coords(lat_cmip=('latlon',latlon_unobs[0,:]))
arr_unobs = arr_unobs.assign_coords(lon_cmip=('latlon',latlon_unobs[1,:]))


# use xoak to select gridpoints from trajectory
import xoak
lat_mesh, lon_mesh = np.meshgrid(mrso.lat, mrso.lon) # coords need to be in mesh format
mrso['lat_mesh'] = (('lon','lat'), lat_mesh)
mrso['lon_mesh'] = (('lon','lat'), lon_mesh)
mrso.xoak.set_index(['lat_mesh', 'lon_mesh'], 'sklearn_geo_balltree')
mrso_obs = mrso.xoak.sel(lat_mesh=gridpoints.lat_cmip, lon_mesh=gridpoints.lon_cmip)

pred = pred.assign_coords(lat_mesh=(('lon','lat'), lat_mesh))
pred = pred.assign_coords(lon_mesh=(('lon','lat'), lon_mesh))
pred.xoak.set_index(['lat_mesh', 'lon_mesh'], 'sklearn_geo_balltree')
pred_obs = pred.xoak.sel(lat_mesh=gridpoints.lat_cmip, lon_mesh=gridpoints.lon_cmip)

pred_unobs = pred.xoak.sel(lat_mesh=arr_unobs.lat_cmip, lon_mesh=arr_unobs.lon_cmip)

# flatten to skikit-learn digestable table
X_train = pred_obs.stack(obspoints=('latlon','time')).to_array().T
y_train = mrso_obs.stack(obspoints=('latlon','time'))

X_test = pred_unobs.stack(obspoints=('latlon','time')).to_array().T

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

import IPython; IPython.embed()
# loop over years
for year in np.arange(1976,2015):
    next_year = str(year + 1)
    year = str(year)
    
    rf = RandomForestRegressor(**kwargs)
    rf.fit(X_train.sel(time=slice('1960',year)), 
           y_train.sel(time=slice('1960',year)))

    rf.predict(X_test.sel(time=slice(year, next_year))) 
