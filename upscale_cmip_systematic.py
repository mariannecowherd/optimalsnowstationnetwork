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

# read CMIP6 files
logging.info('read cmip ...')
mrso = xr.open_dataset(f'{upscalepath}mrso_ssp585.nc')['mrso'].load()
pred = xr.open_dataset(f'{upscalepath}pred_ssp585.nc').load()

# read station data
logging.info('read station data ...')
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

latlist = stations.lat_cmip.values.tolist()
lonlist = stations.lon_cmip.values.tolist()
corrlist = []
numberlist = []
for i in range(100):
    # create obsmask and unobsmask
    #logging.info('divide into obs and unobs ...')
    landmask = xr.open_dataset(f'{largefilepath}landmask_cmip6-ng.nc')['mrso'].squeeze()
    obsmask = xr.full_like(landmask, False)
    unobsmask = landmask.copy(deep=True)
    for lat, lon in zip(latlist, lonlist):
        unobsmask.loc[lat,lon] = False
        if landmask.loc[lat,lon].item(): # obs gridpoint if station contained and on CMIP land 
            obsmask.loc[lat,lon] = True

    # divide into obs and unobs gridpoints
    obslat, obslon = np.where(obsmask)
    obslat, obslon = xr.DataArray(obslat, dims='landpoints'), xr.DataArray(obslon, dims='landpoints')

    mrso_obs = mrso.isel(lat=obslat, lon=obslon)
    pred_obs = pred.isel(lat=obslat, lon=obslon)

    unobslat, unobslon = np.where(unobsmask)
    unobslat, unobslon = xr.DataArray(unobslat, dims='landpoints'), xr.DataArray(unobslon, dims='landpoints')

    mrso_unobs = mrso.isel(lat=unobslat, lon=unobslon)
    pred_unobs = pred.isel(lat=unobslat, lon=unobslon)

    # stack landpoints and time
    #logging.info('stack')
    y_train = mrso_obs.stack(datapoints=('landpoints','time'))
    y_test = mrso_unobs.stack(datapoints=('landpoints','time'))

    X_train = pred_obs.stack(datapoints=('landpoints','time')).to_array().T
    X_test = pred_unobs.stack(datapoints=('landpoints','time')).to_array().T

    # rf TODO later use GP
    kwargs = {'n_estimators': 100,
              'min_samples_leaf': 1, # those are all default values anyways
              'max_features': 'auto', 
              'max_samples': None, 
              'bootstrap': True,
              'warm_start': False,
              'n_jobs': 30, # set to number of trees
              'verbose': 0}

    #logging.info('train ...')
    rf = RandomForestRegressor(**kwargs)
    rf.fit(X_train, y_train)

    #logging.info('predict ...')
    y_predict = xr.full_like(y_test, np.nan)
    y_predict[:] = rf.predict(X_test)

    y_train_predict = xr.full_like(y_train, np.nan)
    y_train_predict[:] = rf.predict(X_train)

    # unstack
    #logging.info('unstack ...')
    y_predict = y_predict.unstack('datapoints').load()
    y_test = y_test.unstack('datapoints').load()
    y_train = y_train.unstack('datapoints').load()
    y_train_predict = y_train_predict.unstack('datapoints').load()

    #logging.info('corr ...')
    corr = xr.corr(y_test, y_predict, dim='time')

    # find gridpoint with minimum performance
    #landpt = np.where(corr.min() == corr)[0].item()
    #lat = y_test.where(y_test.landpoints == landpt, drop=True).coords["lat"].values[0][0]
    #lon = y_test.where(y_test.landpoints == landpt, drop=True).coords["lon"].values[0][0]
    #logging.info(f'{lat}, {lon}')
    #latlist.append(lat)
    #lonlist.append(lon)

    # find x gridpoints with minimum performance
    numberlist.append(len(latlist))
    landpts = np.argsort(corr)[:3]
    lats = y_test.where(y_test.landpoints.isin(landpts), drop=True).coords["lat"].values[:,0]
    lons = y_test.where(y_test.landpoints.isin(landpts), drop=True).coords["lon"].values[:,0]
    logging.info(f'{lats}, {lons}')
    latlist = latlist + lats.tolist()
    lonlist = lonlist + lons.tolist()

    # corr to worldmap
    corr_train = xr.corr(y_train, y_train_predict, dim='time')
    corrmap = xr.full_like(landmask.astype(float), np.nan)
    corrmap[unobslat, unobslon] = corr
    corrmap[obslat, obslon] = corr_train
    mean_corr = corrmap.mean().item()**2
    corrlist.append(mean_corr)
    logging.info(f'iteration {i} mean corr {mean_corr}')

    # plot
    proj = ccrs.Robinson()
    transf = ccrs.PlateCarree()
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111, projection=proj)
    corrmap.plot(ax=ax, cmap='coolwarm', transform=transf, vmin=-1, vmax=1)
    ax.coastlines()
    ax.set_title(f'iter {i} mean corr {np.round(mean_corr,2)}')
    im = ax.scatter(lons, lats, c='black', transform=transf, marker='x', s=5)
    plt.savefig(f'corr_{i:03}.png')
    plt.close()

# save as netcdf
#mrso_pred.to_netcdf(f'{largefilepath}mrso_fut_{modelname}_{experimentname}_{ensemblename}.nc') 
#mrso.to_netcdf(f'{largefilepath}mrso_orig_{modelname}_{experimentname}_{ensemblename}.nc')

# plot
import IPython; IPython.embed()
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
ax.scatter(numberlist,corrlist)
