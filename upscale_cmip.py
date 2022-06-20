"""
TEST
"""

import cftime
import cartopy.crs as ccrs
import numpy as np
import pandas as pd
import xarray as xr
import regionmask
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import argparse
from calc_geodist import calc_geodist_exact as calc_geodist
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', 
                    datefmt='%d-%b-%y %H:%M:%S', level=logging.DEBUG)
logging.getLogger('cftime').setLevel(logging.WARNING) 
logging.getLogger('matplotlib').setLevel(logging.WARNING) 
logging.getLogger('cartopy').setLevel(logging.WARNING) 
logging.getLogger('fiona').setLevel(logging.WARNING) 
logging.getLogger('GDAL').setLevel(logging.WARNING) 

# define options
parser = argparse.ArgumentParser()
parser.add_argument('--method', '-m', dest='method', type=str)
parser.add_argument('--metric', '-p', dest='metric', type=str)
parser.add_argument('--model', '-c', dest='model', type=str)
args = parser.parse_args()

method = args.method
metric = args.metric
modelname = args.model

# define constants
largefilepath = '/net/so4/landclim/bverena/large_files/'
upscalepath = '/net/so4/landclim/bverena/large_files/opscaling/'
logging.info(f'method {method} metric {metric} modelname {modelname}...')

# read CMIP6 files
logging.info('read cmip ...')
mrso = xr.open_dataset(f'{upscalepath}mrso_{modelname}.nc')['mrso'].load().squeeze()
pred = xr.open_dataset(f'{upscalepath}pred_{modelname}.nc').load().squeeze()

# detrend
p = mrso.polyfit('time', deg=1)
fit = xr.polyval(mrso.time, p.polyfit_coefficients)
mrso_detr = mrso - fit

p = pred.polyfit('time', deg=1)
fit = xr.polyval(pred.time, p) #mhauser verified this. bug not feature
fit = fit.rename_vars({key: key.replace("_polyfit_coefficients", "") for key in fit.data_vars})
pred_detr = pred - fit

# calc standardised anomalies
# tree-based methods do not need standardisation see https://datascience.stack
#exchange.com/questions/5277/do-you-have-to-normalize-data-when-building-decis
#ion-trees-using-r
#mrso_mean = mrso.groupby('time.month').mean()
#mrso_std = mrso.groupby('time.month').std()
#
#pred_mean = pred.groupby('time.month').mean()
#pred_std = pred.groupby('time.month').std() # precip problem TODO

if metric == 'corr':
    mrso = mrso_detr.groupby('time.month') - mrso_mean
    mrso = mrso_detr.groupby('time.month') / mrso_std
    pred = pred_detr.groupby('time.month') - pred_mean

elif metric == 'seasonality':
    mrso = mrso_mean / mrso_std
    pred = pred_mean# / pred_std

    mrso = mrso.rename(month='time')
    pred = pred.rename(month='time')

elif metric == 'trend':
    mrso = mrso.groupby('time.month') - mrso_mean
    mrso = mrso.groupby('time.month') / mrso_std
    pred = pred.groupby('time.month') - pred_mean

    mrso = mrso.resample(time='1y').mean()
    pred = pred.resample(time='1y').mean()
else:
    raise AttributeError('metric not known')
logging.info(f'metric {metric} data shape {mrso.shape}')

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

# calc min distance to closest station

latlist = stations.lat_cmip.values.tolist() # only needed for first iteration
lonlist = stations.lon_cmip.values.tolist()
corrlist = []
numberlist = []
n = 100
#for i in range(101):
i = 0
lats_added = []
lons_added = []
logging.info('start loop ...')
while True:
    # create obsmask and unobsmask
    #logging.info('divide into obs and unobs ...')
    #landmask = xr.open_dataset(f'{largefilepath}landmask_cmip6-ng.nc')['mrso'].squeeze()
    landmask = ~np.isnan(mrso.mean(dim='time')) # models need individual landmasks
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
    logging.info(f'{mrso_obs.shape[1]} gridpoints observed, {mrso_unobs.shape[1]} gridpoints unobserved')

    # initially observed values
    if i == 0:
        latlist = mrso_obs.lat.values.tolist()
        lonlist = mrso_obs.lon.values.tolist()

    # stack landpoints and time
    #logging.info('stack')
    y_train = mrso_obs.stack(datapoints=('landpoints','time'))
    y_test = mrso_unobs.stack(datapoints=('landpoints','time'))

    X_train = pred_obs.stack(datapoints=('landpoints','time')).to_array().T
    X_test = pred_unobs.stack(datapoints=('landpoints','time')).to_array().T

    if y_test.size == 0:
        logging.info('all points are observed. stop process...')
        break
    if mrso_unobs.shape[1] < n:
        n = mrso_unobs.shape[1] # rest of points
        logging.info(f'last iteration with {n} points ...')

    # rf TODO later use GP
    kwargs = {'n_estimators': 100,
              'min_samples_leaf': 1, # those are all default values anyways
              'max_features': 'auto', 
              'max_samples': None, 
              'bootstrap': True,
              'warm_start': False,
              'n_jobs': 10, # set to number of trees
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
    corrmap = xr.full_like(landmask.astype(float), np.nan)
    if metric == 'corr':
        corr = xr.corr(y_test, y_predict, dim='time')
        corr_train = xr.corr(y_train, y_train_predict, dim='time')
    elif metric == 'seasonality':
        corr = xr.corr(y_test, y_predict, dim='time')
        corr_train = xr.corr(y_train, y_train_predict, dim='time')
        #test_seas = y_test.groupby("time.month").mean() 
        #predict_seas = y_predict.groupby("time.month").mean() 
        #train_seas = y_train.groupby("time.month").mean() 
        #train_predict_seas = y_train_predict.groupby("time.month").mean() 
        #corr = xr.corr(test_seas, predict_seas, dim='month')
        #corr_train = xr.corr(train_seas, train_predict_seas, dim='month')
    elif metric == 'trend': # TODO until like in Cook 2020: anomalies with reference to baseline period (1851-1880)
        ms_to_year = 365*24*3600*10**9
        test_trends = y_test.polyfit(dim="time", deg=1).polyfit_coefficients.sel(degree=1)*ms_to_year
        predict_trends = y_predict.polyfit(dim="time", deg=1).polyfit_coefficients.sel(degree=1)*ms_to_year
        train_trends = y_train.polyfit(dim="time", deg=1).polyfit_coefficients.sel(degree=1)*ms_to_year
        train_predict_trends = y_train_predict.polyfit(dim="time", deg=1).polyfit_coefficients.sel(degree=1)*ms_to_year
        corr = -np.abs(test_trends - predict_trends) # corr is neg abs diff
        corr_train = -np.abs(train_trends - train_predict_trends) 
    else:
        raise AttributeError('metric not known')
    corrmap[unobslat, unobslon] = corr
    corrmap[obslat, obslon] = corr_train

    # find gridpoint with minimum performance
    #landpt = np.where(corr.min() == corr)[0].item()
    #lat = y_test.where(y_test.landpoints == landpt, drop=True).coords["lat"].values[0][0]
    #lon = y_test.where(y_test.landpoints == landpt, drop=True).coords["lon"].values[0][0]
    #logging.info(f'{lat}, {lon}')
    #latlist.append(lat)
    #lonlist.append(lon)

    # find x gridpoints with minimum performance
    numberlist.append(len(latlist))
    if method == 'systematic':
        landpts = np.argsort(corr)[:n]
    elif method == 'random':
        landpts = np.random.choice(np.arange(corr.size), size=n, replace=False)
    elif method == 'interp':
        # landpts = [] # actually needs to recompute for every point
        landlat = mrso_obs.lat.values.tolist() + mrso_unobs.lat.values.tolist()
        landlon = mrso_obs.lon.values.tolist() + mrso_unobs.lon.values.tolist()
        nobs = mrso_obs.shape[-1]
        dist = calc_geodist(landlon, landlat)
        np.fill_diagonal(dist, np.nan) # inplace
        dist = dist[nobs:, :nobs] # dist of obs to unobs
        mindist = np.nanmin(dist, axis=1)
        mindist[np.isnan(mindist)] = 0 # nans are weird in argsort
        landpts = np.argsort(mindist)[-n:] 
    else:
        raise AttributeError('method not known')
    lats = y_test.where(y_test.landpoints.isin(landpts), drop=True).coords["lat"].values[:,0]
    lons = y_test.where(y_test.landpoints.isin(landpts), drop=True).coords["lon"].values[:,0]
    #logging.info(f'{lats}, {lons}')
    latlist = latlist + lats.tolist()
    lonlist = lonlist + lons.tolist()
    lats_added.append(lats.tolist())
    lons_added.append(lons.tolist())

    # calc mean corr for log and res
    if metric == 'trend':
        mean_corr = -corrmap.mean().item()
    else:
        mean_corr = corrmap.mean().item()**2
    corrlist.append(mean_corr)
    logging.info(f'iteration {i} obs landpoints {len(latlist)} mean metric {mean_corr}')

    # save intermediate results
    with open(f'corr_{method}_{modelname}_{metric}.pkl','wb') as f:
        pickle.dump(corrlist, f)

    with open(f'nobs_{method}_{modelname}_{metric}.pkl','wb') as f:
        pickle.dump(numberlist, f)

    with open(f'lats_{method}_{modelname}_{metric}.pkl','wb') as f:
        pickle.dump(lats_added, f)

    with open(f'lons_{method}_{modelname}_{metric}.pkl','wb') as f:
        pickle.dump(lons_added, f)

    # plot
    proj = ccrs.Robinson()
    transf = ccrs.PlateCarree()
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111, projection=proj)
    #if metric in ['corr','seasonality']:
    #    corrmap.plot(ax=ax, cmap='coolwarm', transform=transf, vmin=-1, vmax=1)
    #elif metric == 'trend':
    #    (-corrmap).plot(ax=ax, cmap='coolwarm', transform=transf, vmin=0, vmax=0.1)
    #else:
    #    raise AttributeError('method not known')
    ax.coastlines()
    ax.set_global()
    ax.set_title(f'iter {i} mean corr {np.round(mean_corr,2)}')
    im = ax.scatter(lonlist, latlist, c='grey', transform=transf, marker='x', s=5)
    im = ax.scatter(lons, lats, c='black', transform=transf, marker='x', s=5)
    plt.savefig(f'corr_{i:03}_{method}_{modelname}_{metric}.png')
    plt.close()
    i += 1

# save as netcdf
#mrso_pred.to_netcdf(f'{largefilepath}mrso_fut_{modelname}_{experimentname}_{ensemblename}.nc') 
#mrso.to_netcdf(f'{largefilepath}mrso_orig_{modelname}_{experimentname}_{ensemblename}.nc')

# plot
#import IPython; IPython.embed()
#fig = plt.figure(figsize=(10,5))
#ax = fig.add_subplot(111)
#ax.scatter(numberlist,corrlist)
