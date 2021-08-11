"""
TEST
"""

import cftime
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
mrso_hist = open_cmip_suite('mrso', modelname, experimentname, ensemblename)
tas_hist = open_cmip_suite('tas', modelname, experimentname, ensemblename)
pr_hist = open_cmip_suite('pr', modelname, experimentname, ensemblename)
modelname = 'CanESM5'
experimentname = 'ssp585'
ensemblename = 'r1i1p1f1'
mrso_ssp = open_cmip_suite('mrso', modelname, experimentname, ensemblename)
tas_ssp = open_cmip_suite('tas', modelname, experimentname, ensemblename)
pr_ssp = open_cmip_suite('pr', modelname, experimentname, ensemblename)

# merge historical and future scenarios
mrso = xr.concat([mrso_hist, mrso_ssp ], dim='time')
tas = xr.concat([tas_hist, tas_ssp ], dim='time')
pr = xr.concat([pr_hist, pr_ssp ], dim='time')

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
mrso = mrso.sel(time=slice('1960','2030'))
pred = pred.sel(time=slice('1960','2030'))

# not sure why this is necessary # TODO
#mrso = mrso.resample(time='1M').mean()
#pred = pred.resample(time='1M').mean()

# read station data
largefilepath = '/net/so4/landclim/bverena/large_files/'
stations = xr.open_dataset(f'{largefilepath}df_gaps.nc')
stations = stations['__xarray_dataarray_variable__']

# select only "still existing" stations in future
iold = np.isnan(stations.sel(time=slice('2015','2020'))).all(axis=0)
stations.loc[slice('2015','2020'),~iold] = 1
tmp = xr.full_like(stations.sel(time=slice('2010','2019')), np.nan)
timerange = pd.date_range('2021-01-01','2031-01-01', freq='M')
tmp['time'] = timerange
tmp[:,~iold] = 1
stations = xr.concat([stations, tmp], dim='time')

# calculate deseasonalised anomaly 
mrso_seasonal_mean = mrso.groupby('time.month').mean()
mrso_seasonal_std = mrso.groupby('time.month').std()
mrso = (mrso.groupby('time.month') - mrso_seasonal_mean) 
mrso = mrso.groupby('time.month') / mrso_seasonal_std

seasonal_mean = pred.groupby('time.month').mean() # for reasoning see crossval file
#seasonal_std = pred.groupby('time.month').std()
pred = (pred.groupby('time.month') - seasonal_mean) 
#pred = pred.groupby('time.month') / seasonal_std

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
latlon_cmip = []
for lat, lon in zip(stations.lat, stations.lon):
    point = landmask.sel(lat=lat, lon=lon, method='nearest')

    unobsmask.loc[point.lat, point.lon] = False

    lat_cmip.append(point.lat.item())
    lon_cmip.append(point.lon.item())
    
    if landmask.loc[point.lat,point.lon].item(): # obs gridpoint if station contained and on CMIP land
        obsmask.loc[point.lat, point.lon] = True
        latlon_cmip.append(f'{point.lat.item()} {point.lon.item()}')
    else:
        latlon_cmip.append('ocean')

stations = stations.assign_coords(lat_cmip=('stations',lat_cmip))
stations = stations.assign_coords(lon_cmip=('stations',lon_cmip))
stations_copy = stations.copy(deep=True)
stations = stations.assign_coords(latlon_cmip=('stations',latlon_cmip))
stations = stations.groupby('latlon_cmip').mean()
stations = stations.drop_sel(latlon_cmip='ocean')

# add lat and lon again to grouped station data
lat_cmip = []
lon_cmip = []
for latlon in stations.latlon_cmip:
    lat, lon = latlon.item().split()
    lat, lon = float(lat), float(lon)
    lat_cmip.append(lat)
    lon_cmip.append(lon)
stations = stations.assign_coords(lat_cmip=('latlon_cmip',lat_cmip))
stations = stations.assign_coords(lon_cmip=('latlon_cmip',lon_cmip))

# mask for unobserved future points given future network
future_unobsmask = landmask.copy(deep=True)
future_stations = stations.loc['2030-12-31',~np.isnan(stations.sel(time='2030-12-31'))]
for lat, lon in zip(future_stations.lat_cmip, future_stations.lon_cmip):
    future_unobsmask.loc[lat, lon] = False

# divide into obs and unobs gridpoints
obslat, obslon = np.where(obsmask)
obslat, obslon = xr.DataArray(obslat, dims='obspoints'), xr.DataArray(obslon, dims='obspoints')

mrso_obs = mrso.isel(lat=obslat, lon=obslon)
pred_obs = pred.isel(lat=obslat, lon=obslon)

unobslat, unobslon = np.where(future_unobsmask)
unobslat, unobslon = xr.DataArray(unobslat, dims='unobspoints'), xr.DataArray(unobslon, dims='unobspoints')
#
mrso_unobs = mrso.isel(lat=unobslat, lon=unobslon)
pred_unobs = pred.isel(lat=unobslat, lon=unobslon)

# remove unobs points from gridpoints with stations
mrso_obstime = mrso_obs.copy(deep=True)
mrso_obstime = mrso_obstime.where(~np.isnan(stations.values))

pred_obstime = pred_obs.copy(deep=True)
pred_obstime = pred_obstime.where(~np.isnan(stations.values))

# flatten to skikit-learn digestable table 
mrso_obstime = mrso_obstime.stack(datapoints=('obspoints','time'))
pred_obstime = pred_obstime.stack(datapoints=('obspoints','time')).to_array().T
pred_unobs = pred_unobs.to_array()

# remove nans from dataset
mrso_obstime = mrso_obstime.where(~np.isnan(mrso_obstime), drop=True)
pred_obstime = pred_obstime.where(~np.isnan(mrso_obstime), drop=True)

# rf settings TODO later use GP
kwargs = {'n_estimators': 100,
          'min_samples_leaf': 1, # those are all default values anyways
          'max_features': 'auto', 
          'max_samples': None, 
          'bootstrap': True,
          'warm_start': False,
          'n_jobs': 100, # set to number of trees
          'verbose': 0}

# initialise array for saving results
mrso_pred = xr.full_like(mrso.sel(time=slice('2020','2031')), np.nan)

for year in range(2020,2031):

    # define test and training dataset
    y_train = mrso_obstime.where(mrso_obstime['time.year'] <= year, drop=True)
    X_train = pred_obstime.where(mrso_obstime['time.year'] <= year, drop=True)

    y_test = mrso_unobs.where(mrso_unobs['time.year'] == year, drop=True)
    X_test = pred_unobs.where(pred_unobs['time.year'] == year, drop=True)

    y_test = y_test.stack(datapoints=('unobspoints','time'))
    X_test = X_test.stack(datapoints=('unobspoints','time')).T

    # train and upscale
    rf = RandomForestRegressor(**kwargs)
    rf.fit(X_train, y_train)

    y_predict = xr.full_like(y_test, np.nan)
    y_predict[:] = rf.predict(X_test)

    print(f'upscale R2 in year {year}', xr.corr(y_predict, y_test).item()**2)

    # format back to worldmap
    mrso_pred.loc[str(year),:,:].values[:,unobslat,unobslon] = y_predict.unstack('datapoints').T

# renormalise 
mrso = (mrso.groupby('time.month') * mrso_seasonal_std)
mrso = (mrso.groupby('time.month') + mrso_seasonal_mean) 

mrso_pred = (mrso_pred.groupby('time.month') * mrso_seasonal_std)
mrso_pred = (mrso_pred.groupby('time.month') + mrso_seasonal_mean) 

# save as netcdf
mrso_pred.to_netcdf(f'{largefilepath}mrso_fut_{modelname}_{experimentname}_{ensemblename}.nc') 
mrso.to_netcdf(f'{largefilepath}mrso_orig_{modelname}_{experimentname}_{ensemblename}.nc')
