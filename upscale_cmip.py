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
for i in range(latlon_unique.shape[1]):
    lat, lon = latlon_unique[:,i]
    gridpoint_mean = stations.where((stations.lat_cmip == lat) & (stations.lon_cmip == lon), drop=True).mean(dim='stations')
    gridpoints.loc[:,i] = gridpoint_mean
import IPython; IPython.embed()
gridpoints = gridpoints.assign_coords(lat_cmip=('latlon',latlon_unique[0,:]))
gridpoints = gridpoints.assign_coords(lon_cmip=('latlon',latlon_unique[1,:]))

import IPython; IPython.embed()
# use xoak to select gridpoints from trajectory
import xoak
lat_mesh, lon_mesh = np.meshgrid(mrso.lat, mrso.lon) # coords need to be in mesh format
mrso['lat_mesh'] = (('lon','lat'), lat_mesh)
mrso['lon_mesh'] = (('lon','lat'), lon_mesh)
mrso.xoak.set_index(['lat_mesh', 'lon_mesh'], 'sklearn_geo_balltree')
mrso_obs = mrso.xoak.sel(lat_mesh=gridpoints.lat_cmip, lon_mesh=gridpoints.lon_cmip)
