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
import argparse
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', 
                    datefmt='%d-%b-%y %H:%M:%S', level=logging.DEBUG)

# define constants
largefilepath = '/net/so4/landclim/bverena/large_files/'
upscalepath = '/net/so4/landclim/bverena/large_files/opscaling/'

# read gridpoint info
#parser = argparse.ArgumentParser()
#parser.add_argument('--lat', dest='lat', type=int)
#parser.add_argument('--lon', dest='lon', type=int)
#args = parser.parse_args()
#gridpoint = args.g

# read CMIP6 files
mrso = xr.open_dataset(f'{upscalepath}mrso_ssp585.nc')['mrso'].load()
pred = xr.open_dataset(f'{upscalepath}pred_ssp585.nc').load()

# read station data
stations = xr.open_dataset(f'{largefilepath}df_gaps.nc')['mrso']

# select the stations still active
inactive_networks = ['HOBE','PBO_H20','IMA_CAN1','SWEX_POLAND','CAMPANIA',
                     'HiWATER_EHWSN', 'METEROBS', 'UDC_SMOS', 'KHOREZM',
                     'ICN','AACES','HSC_CEOLMACHEON','MONGOLIA','RUSWET-AGRO',
                     'CHINA','IOWA','RUSWET-VALDAI','RUSWET-GRASS']
stations = stations.where(~stations.network.isin(inactive_networks), drop=True)

# calculate deseasonalised anomaly 
#mrso_seasonal_mean = mrso.groupby('time.month').mean()
#mrso_seasonal_std = mrso.groupby('time.month').std()
#mrso = (mrso.groupby('time.month') - mrso_seasonal_mean) 
#mrso = mrso.groupby('time.month') / mrso_seasonal_std
#
#seasonal_mean = pred.groupby('time.month').mean() # for reasoning see crossval file
##seasonal_std = pred.groupby('time.month').std()
#pred = (pred.groupby('time.month') - seasonal_mean) 
##pred = pred.groupby('time.month') / seasonal_std
#
#seasonal_mean = stations.groupby('time.month').mean()
#seasonal_std = stations.groupby('time.month').std()
#stations = (stations.groupby('time.month') - seasonal_mean) 
#stations = stations.groupby('time.month') / seasonal_std

## regrid station data to CMIP6 grid
landmask = xr.open_dataset(f'{largefilepath}landmask_cmip6-ng.nc')['mrso'].squeeze()
obsmask = xr.full_like(landmask, False)
unobsmask = landmask.copy(deep=True)
for lat, lon in zip(stations.lat_cmip, stations.lon_cmip):
    unobsmask.loc[lat,lon] = False
    if landmask.loc[lat,lon].item(): # obs gridpoint if station contained and on CMIP land 
        obsmask.loc[lat,lon] = True

# add another "theoretical" future stations TODO
#latlist = [18.75]
#lonlist = [48.75]
#for lat, lon in zip(latlist, lonlist):
#    station = xr.full_like(stations[:,0], np.nan)
#    station['lat_cmip'] = lat 
#    station['lon_cmip'] = lon
#    station.loc[slice('2022','2030')] = 1
#    station['latlon_cmip'] = f'{lat} {lon}'
#    stations = xr.concat([stations,station], dim='latlon_cmip')
#    obsmask.loc[lat, lon] = True
##df_gaps = df_gaps.swap_dims({'latlon_cmip': 'lat_cmip'}).reset_index('latlon_cmip', drop=True) # remove latlon_cmip coordinate
#
## mask for unobserved future points given future network # TODO don't know why this was necessary
#future_unobsmask = landmask.copy(deep=True)
#future_stations = stations.loc['2030-12-31',~np.isnan(stations.sel(time='2030-12-31'))]
#for lat, lon in zip(future_stations.lat_cmip, future_stations.lon_cmip):
#    future_unobsmask.loc[lat, lon] = False
#import IPython; IPython.embed()

# divide into obs and unobs gridpoints
obslat, obslon = np.where(obsmask)
obslat, obslon = xr.DataArray(obslat, dims='obspoints'), xr.DataArray(obslon, dims='obspoints')

mrso_obs = mrso.isel(lat=obslat, lon=obslon)
pred_obs = pred.isel(lat=obslat, lon=obslon)

#unobslat, unobslon = np.where(future_unobsmask)
unobslat, unobslon = np.where(unobsmask)
unobslat, unobslon = xr.DataArray(unobslat, dims='unobspoints'), xr.DataArray(unobslon, dims='unobspoints')

mrso_unobs = mrso.isel(lat=unobslat, lon=unobslon)
pred_unobs = pred.isel(lat=unobslat, lon=unobslon)

# remove unobs points from gridpoints with stations
#mrso_obstime = mrso_obs.copy(deep=True)
#mrso_obstime = mrso_obstime.where(~np.isnan(stations.values))
#
#pred_obstime = pred_obs.copy(deep=True)
#pred_obstime = pred_obstime.where(~np.isnan(stations.values))

# flatten to skikit-learn digestable table 
#mrso_obstime = mrso_obstime.stack(datapoints=('obspoints','time'))
#pred_obstime = pred_obstime.stack(datapoints=('obspoints','time')).to_array().T
#pred_unobs = pred_unobs.to_array()
#
## remove nans from dataset
#mrso_obstime = mrso_obstime.where(~np.isnan(mrso_obstime), drop=True)
#pred_obstime = pred_obstime.where(~np.isnan(mrso_obstime), drop=True)

# stack landpoints and time
logging.info('stack')
y_train = mrso_obs.stack(datapoints=('obspoints','time'))
y_test = mrso_unobs.stack(datapoints=('obspoints','time'))

X_train = pred_obs.stack(datapoints=('unobspoints','time')).to_array().T
X_test = pred_unobs.stack(datapoints=('unobspoints','time')).to_array().T

# rf settings TODO later use GP
kwargs = {'n_estimators': 100,
          'min_samples_leaf': 1, # those are all default values anyways
          'max_features': 'auto', 
          'max_samples': None, 
          'bootstrap': True,
          'warm_start': False,
          'n_jobs': 30, # set to number of trees
          'verbose': 0}

# initialise array for saving results
#mrso_pred = xr.full_like(mrso.sel(time=slice('2020','2051')), np.nan)['mrso']
#
#for year in range(2020,2051):
#
#    # define test and training dataset
#    y_train = mrso_obstime.where(mrso_obstime['time.year'] <= year, drop=True)['mrso']
#    X_train = pred_obstime.where(mrso_obstime['time.year'] <= year, drop=True)['mrso']
#
#    y_test = mrso_unobs.where(mrso_unobs['time.year'] == year, drop=True)
#    X_test = pred_unobs.where(pred_unobs['time.year'] == year, drop=True)
#
#    y_test = y_test.stack(datapoints=('unobspoints','time'))['mrso']
#    X_test = X_test.stack(datapoints=('unobspoints','time')).T
#
#    # train and upscale
#    rf = RandomForestRegressor(**kwargs)
#    rf.fit(X_train, y_train)
#
#    y_predict = xr.full_like(y_test, np.nan)
#    y_predict[:] = rf.predict(X_test)
#
#    print(f'upscale R2 in year {year}', xr.corr(y_predict, y_test).item()**2)
#
#    # format back to worldmap
#    mrso_pred.loc[str(year),:,:].values[:,unobslat,unobslon] = y_predict.unstack('datapoints').T

# renormalise 
#mrso = (mrso.groupby('time.month') * mrso_seasonal_std)
#mrso = (mrso.groupby('time.month') + mrso_seasonal_mean) 
#
#mrso_pred = (mrso_pred.groupby('time.month') * mrso_seasonal_std)
#mrso_pred = (mrso_pred.groupby('time.month') + mrso_seasonal_mean) 
#
#mrso = mrso['mrso']
#mrso_pred = mrso_pred['mrso']

rf = RandomForestRegressor(**kwargs)
rf.fit(X_train, y_train)

logging.info('predict')
y_predict = xr.full_like(y_test, np.nan)
y_predict[:] = rf.predict(X_test)

corr = xr.corr(y_test, y_predict).item()
logging.info(f'{gridpoint}, {corr**2}')

# find gridpoint with minimum performance
corr = xr.corr(mrso, mrso_pred, dim='time')
latidx, lonidx = np.where(corr.min() == corr)
lat, lon = corr[latidx,lonidx].lat.item() , corr[latidx,lonidx].lon.item()
print(lat, lon)

# save as netcdf
mrso_pred.to_netcdf(f'{largefilepath}mrso_fut_{modelname}_{experimentname}_{ensemblename}.nc') 
mrso.to_netcdf(f'{largefilepath}mrso_orig_{modelname}_{experimentname}_{ensemblename}.nc')
