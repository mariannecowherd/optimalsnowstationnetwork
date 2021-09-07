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

# read station data
largefilepath = '/net/so4/landclim/bverena/large_files/'
stations = xr.open_dataset(f'{largefilepath}df_gaps.nc')
stations = stations['__xarray_dataarray_variable__']
stations = stations.sel(time=slice('1960','2014'))

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

# add some grid points for poster DEBUG TODO
tmp = xr.full_like(stations[:,:5], np.nan)
tmp = tmp.drop(('network','country','month','koeppen','koeppen_simple'))
stations = stations.drop(('network','country','month','koeppen','koeppen_simple'))
tmp = tmp.assign_coords(stations=range(2816,2821))
tmp = tmp.assign_coords(lat=('stations',[1.25,1.25,3.75,66.25,63.75]))
tmp = tmp.assign_coords(lat_grid=('stations',[1.25,1.25,3.75,66.25,63.75]))
tmp = tmp.assign_coords(lon=('stations',[-58.75,-61.25,-58.75,138.8,153.8]))
tmp = tmp.assign_coords(lon_grid=('stations',[-58.75,-61.25,-58.75,138.8,153.8]))
tmp[1:,:] = 1
stations = xr.concat([stations, tmp], dim='stations')
# add new proposed station locations
# Af: (177, 242) latlon: 1.25, -58.75 latloncmip: 1.25 -58.75
# Am : (177, 243) latlon: 1.25, -58.25 latloncmip:  1.25 -58.75 REJECTED
# Am: (third largest value) latlon: 1.25 60.35 latloncmip: 1.25 -61.25
# Aw:  (174, 240) latlon: 2.75, -59.75 latloncmip: 3.75 -58.75
# Dw: (47, 637) latlon: 66.25, 138.8 latloncmip: 66.25 138.8
# Df: (52, 668) latlon: 63.75, 154.2 latloncmip: 63.75 153.8

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
    #else:
    #    stations = stations.where((stations.lat != lat) & (stations.lon != lon), drop=True) # station is in ocean, remove

stations = stations.assign_coords(lat_cmip=('stations',lat_cmip))
stations = stations.assign_coords(lon_cmip=('stations',lon_cmip))
stations_copy = stations.copy(deep=True)
stations = stations.assign_coords(latlon_cmip=('stations',latlon_cmip))
stations = stations.groupby('latlon_cmip').mean()
stations = stations.drop_sel(latlon_cmip='ocean')

# divide into obs and unobs gridpoints
obslat, obslon = np.where(obsmask)
obslat, obslon = xr.DataArray(obslat, dims='obspoints'), xr.DataArray(obslon, dims='obspoints')

mrso_obs = mrso.isel(lat=obslat, lon=obslon)
pred_obs = pred.isel(lat=obslat, lon=obslon)

unobslat, unobslon = np.where(unobsmask)
unobslat, unobslon = xr.DataArray(unobslat, dims='unobspoints'), xr.DataArray(unobslon, dims='unobspoints')

mrso_unobs = mrso.isel(lat=unobslat, lon=unobslon)
pred_unobs = pred.isel(lat=unobslat, lon=unobslon)

# move unobs times of obs gridpoints into unobs
mrso_obstime = mrso_obs.copy(deep=True)
mrso_obstime = mrso_obstime.where(~np.isnan(stations.values))
mrso_unobstime = mrso_obs.copy(deep=True)
mrso_unobstime = mrso_unobstime.where(np.isnan(stations.values)) 

pred_obstime = pred_obs.copy(deep=True)
pred_obstime = pred_obstime.where(~np.isnan(stations.values))
pred_unobstime = pred_obs.copy(deep=True)
pred_unobstime = pred_unobstime.where(np.isnan(stations.values)) 

# flatten to skikit-learn digestable table 
mrso_obstime = mrso_obstime.stack(datapoints=('obspoints','time'))
pred_obstime = pred_obstime.stack(datapoints=('obspoints','time')).to_array().T

mrso_unobstime = mrso_unobstime.stack(datapoints=('obspoints','time'))
pred_unobstime = pred_unobstime.stack(datapoints=('obspoints','time')).to_array().T

pred_unobs = pred_unobs.stack(datapoints=('unobspoints','time')).to_array().T
mrso_unobs = mrso_unobs.stack(datapoints=('unobspoints','time'))

# remove nans from dataset
mrso_obstime = mrso_obstime.where(~np.isnan(mrso_obstime), drop=True)
pred_obstime = pred_obstime.where(~np.isnan(mrso_obstime), drop=True)

mrso_unobstime = mrso_unobstime.where(~np.isnan(mrso_unobstime), drop=True)
pred_unobstime = pred_unobstime.where(~np.isnan(mrso_unobstime), drop=True)

# define test and training dataset
# TODO check that concat mixes the datapoints correctly
# solution currently without stack/unstack because xr.concat seems to mix up the order of datapoints and indexing them is non-trivial
#y_test = xr.DataArray(np.full((mrso_unobs.size + mrso_unobstime.size), np.nan))
#X_test = xr.DataArray(np.full((mrso_unobs.size + mrso_unobstime.size, pred_unobs.shape[1]), np.nan))
#y_test[:mrso_unobs.size] = mrso_unobs
#y_test[mrso_unobs.size:] = mrso_unobstime
#X_test[:mrso_unobs.size,:] = pred_unobs
#X_test[mrso_unobs.size:,:] = pred_unobstime
#y_test = xr.concat([mrso_unobs, mrso_unobstime], dim='datapoints', coords='all')
#X_test = xr.concat([pred_unobs, pred_unobstime], dim='datapoints', coords='all')
y_test_unobs = mrso_unobs
X_test_unobs = pred_unobs
y_test_unobstime = mrso_unobstime
X_test_unobstime = pred_unobstime

y_train = mrso_obstime
X_train = pred_obstime

#X_train = pred_obs.stack(datapoints=('obspoints','time')).to_array().T
#y_train = mrso_obs.stack(datapoints=('obspoints','time'))
#
#X_test = pred_unobs.stack(datapoints=('unobspoints','time')).to_array().T
#y_test = mrso_unobs.stack(datapoints=('unobspoints','time'))

# rf settings TODO later use GP
kwargs = {'n_estimators': 100,
          'min_samples_leaf': 1, # those are all default values anyways
          'max_features': 'auto', 
          'max_samples': None, 
          'bootstrap': True,
          'warm_start': False,
          'n_jobs': 100, # set to number of trees
          'verbose': 0}

#res = xr.full_like(mrso_unobs.sel(time=slice('1979','2015')), np.nan)
rf = RandomForestRegressor(**kwargs)
rf.fit(X_train, y_train)

y_predict_unobs = xr.full_like(y_test_unobs, np.nan)
y_predict_unobs[:] = rf.predict(X_test_unobs)

y_predict_unobstime = xr.full_like(y_test_unobstime, np.nan)
y_predict_unobstime[:] = rf.predict(X_test_unobstime)
print('upscale R2 in unobserved gridpoints', xr.corr(y_predict_unobs, y_test_unobs).item()**2)
print('upscale R2 in unobserved timepoints', xr.corr(y_predict_unobstime, y_test_unobstime).item()**2)

# back to worldmap
#y_unobs = y_predict[:mrso_unobs.size]
#y_unobstime = y_predict[mrso_unobs.size:]

#y_unobs = y_unobs.unstack('datapoints').T
#y_obs = xr.concat([mrso_obstime, y_unobstime], dim='datapoints', coords='all')
#y_obs = y_obs.unstack('datapoints').T
#y_obs = xr.DataArray(np.full((mrso_obstime.size + y_unobstime.size), np.nan))
#y_obs[:mrso_obstime.size] = mrso_obstime
#y_obs[mrso_obstime.size:] = y_unobstime

mrso_pred = xr.full_like(mrso, np.nan)
mrso_pred.values[:,unobslat,unobslon] = y_predict_unobs.unstack('datapoints').T
#mrso_pred.values[:,unobslat,unobslon] = y_unobs.values.reshape((660,1864)) # or other way round? # no because zero correlation if other way around # this way around is correct in space but not yet in time
mrso_pred.values[:,obslat,obslon] = y_predict_unobstime.unstack('datapoints').T
mrso_pred = mrso_pred.fillna(mrso) # observed points fill in
print('upscale R2 in whole dataset', xr.corr(mrso, mrso_pred).item()**2)

#proj = ccrs.PlateCarree()
#fig = plt.figure()
#ax = fig.add_subplot(111, projection=proj)

# renormalise 
mrso = (mrso.groupby('time.month') * mrso_seasonal_std)
mrso = (mrso.groupby('time.month') + mrso_seasonal_mean) 

mrso_pred = (mrso_pred.groupby('time.month') * mrso_seasonal_std)
mrso_pred = (mrso_pred.groupby('time.month') + mrso_seasonal_mean) 

# save as netcdf
import IPython; IPython.embed()
mrso_pred.to_netcdf(f'{largefilepath}mrso_histnew_{modelname}_{experimentname}_{ensemblename}.nc') # TODO add orig values from mrso_obs
#mrso.to_netcdf(f'{largefilepath}mrso_orig_{modelname}_{experimentname}_{ensemblename}.nc') # save orig in future run because all years are in ther # save orig in future run because all years are in there
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
