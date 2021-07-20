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
    _, _, modelname, _, ensemblename, _ = data.encoding['source'].split('_')
    data = data.rename({varname: f'{modelname}_{ensemblename}'})
    try:
        data = data.drop_vars('file_qf')
    except ValueError:
        pass
    #data = data.drop_dims('height')
    if isinstance(data.time[0].item(), cftime._cftime.DatetimeNoLeap):
        data['time'] = data.indexes['time'].to_datetimeindex()
    return data

sm_path = '/net/atmos/data/cmip6-ng/mrso/mon/g025/'
largefilepath = '/net/so4/landclim/bverena/large_files/'
varname = 'mrso'
mrso = xr.open_mfdataset(f'{sm_path}{varname}*_historical_r1i1p1f1_*.nc', # shoyer says better use instead of glob
                         combine='nested', # timestamp problem
                         concat_dim=None,  # timestamp problem
                         preprocess=rename_vars)  # rename vars to model name
mrso = mrso.to_array()
mrso.coords['lon'] = (mrso.coords['lon'] + 180) % 360 - 180
mrso = mrso.sortby('lon')

# read station data
df_gaps = xr.open_dataset(f'{largefilepath}df_gaps.nc')
df_gaps = df_gaps['__xarray_dataarray_variable__']

# calculate sm anomaly 
seasonal_mean = mrso.groupby('time.month').mean()
seasonal_std = mrso.groupby('time.month').std()
mrso = (mrso.groupby('time.month') - seasonal_mean) 
mrso = mrso.groupby('time.month') / seasonal_std

seasonal_mean = df_gaps.groupby('time.month').mean()
seasonal_std = df_gaps.groupby('time.month').std()
df_gaps = (df_gaps.groupby('time.month') - seasonal_mean) 
df_gaps = df_gaps.groupby('time.month') / seasonal_std

# select timerange of ismn
mrso = mrso.sel(time=slice('1960','2014'))

# regrid station data to CMIP6 grid
lat_cmip = []
lon_cmip = []
for lat, lon in zip(df_gaps.lat, df_gaps.lon):
    point = mrso.sel(lat=lat, lon=lon, method='nearest')
    lat_cmip.append(point.lat.item())
    lon_cmip.append(point.lon.item())
#df_gaps['lat_cmip'] = lat_cmip
#df_gaps['lon_cmip'] = lon_cmip
latlon_unique = np.unique(np.array([lat_cmip, lon_cmip]), axis=1)

# calculate MMM here for now
mrso = mrso.mean(dim='variable')

# use xoak to select gridpoints from trajectory
import xoak
lat_mesh, lon_mesh = np.meshgrid(mrso.lat, mrso.lon) # coords need to be in mesh format
mrso['lat_mesh'] = (('lon','lat'), lat_mesh)
mrso['lon_mesh'] = (('lon','lat'), lon_mesh)
mrso.xoak.set_index(['lat_mesh', 'lon_mesh'], 'sklearn_geo_balltree')
mrso_obs = mrso.xoak.sel(lat=df_gaps.lat, lon=df_gaps.lon)
