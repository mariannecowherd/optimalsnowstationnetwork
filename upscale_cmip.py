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
from sklearn.metrics import r2_score
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
parser.add_argument('--method', '-m', dest='method', type=str) # eg systematic
parser.add_argument('--metric', '-p', dest='metric', type=str) # eg corr
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
#landmask = ~np.isnan(mrso.mean(dim='time')) # models need individual landmasks
landmask = xr.open_dataset(f'{upscalepath}landmask.nc').to_array().squeeze()
landmask = landmask.drop_vars('variable')

# calc standardised anomalies
# tree-based methods do not need standardisation see https://datascience.stack
#exchange.com/questions/5277/do-you-have-to-normalize-data-when-building-decis
#ion-trees-using-r
# therefore these are only used for metric computation
mrso_mean = mrso.groupby('time.month').mean()
mrso_std = mrso.groupby('time.month').std()

pred_mean = pred.groupby('time.month').mean()
pred_std = pred.groupby('time.month').std() 

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

# create iter objects
latlist = stations.lat_cmip.values.tolist() # only needed for first iteration
lonlist = stations.lon_cmip.values.tolist()
niter = xr.full_like(landmask.astype(float), np.nan)
n = 100
i = 0
testcase = '_new'
corrmaps = []

# some models (MPI) need individual land mask bec don't have e.g. Cuba
#landpts_before = landmask.sum()
#model_no_land = np.isnan(mrso)[0,:,:] & landmask
#landmask = landmask.where(~model_no_land, False)
#landpts_after = landmask.sum()
#if landpts_before != landpts_after:
#    logging.info(f'model {modelname} is missing {(landpts_before - landpts_after).item()} landpoints. Those are defined as ocean now')
#else:
#    logging.info(f'model {modelname} no landpoints removed')
#landmask = landmask.drop_vars('time')

logging.info('start loop ...')
while True:

    # create obsmask and unobsmask
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
    mrso_mean_obs = mrso_mean.isel(lat=obslat, lon=obslon)
    mrso_std_obs = mrso_std.isel(lat=obslat, lon=obslon)

    unobslat, unobslon = np.where(unobsmask)
    unobslat, unobslon = xr.DataArray(unobslat, dims='landpoints'), xr.DataArray(unobslon, dims='landpoints')

    mrso_unobs = mrso.isel(lat=unobslat, lon=unobslon)
    pred_unobs = pred.isel(lat=unobslat, lon=unobslon)
    mrso_mean_unobs = mrso_mean.isel(lat=unobslat, lon=unobslon)
    mrso_std_unobs = mrso_std.isel(lat=unobslat, lon=unobslon)

    logging.info(f'{mrso_obs.shape[1]} gridpoints observed, {mrso_unobs.shape[1]} gridpoints unobserved')

    # initially observed values
    if i == 0:
        latlist = mrso_obs.lat.values.tolist()
        lonlist = mrso_obs.lon.values.tolist()

    # drop missing landpoints if there are any (e.g. MPI)
    # dropping nan landpoints does not work because unstack fails bec missing points
    #if np.isnan(mrso_obs).sum() != 0:
    #    pred_obs = pred_obs.where(~np.isnan(mrso_obs)).dropna(dim='landpoints', how='all')
    #    mrso_obs = mrso_obs.dropna(dim='landpoints', how='all')
    #    pred_obs = pred_obs.dropna(dim='landpoints', how='all')

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
              'max_features': 'sqrt', 
              'max_samples': None, 
              'bootstrap': True,
              'warm_start': False,
              'n_jobs': 40, # set to number of trees
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
    #y_test.to_netcdf('y_test.nc') # DEBUG
    #y_predict.to_netcdf('y_predict.nc') #DEBUG
    y_train = y_train.unstack('datapoints').load()
    y_train_predict = y_train_predict.unstack('datapoints').load()
    y_latlon = y_test.copy(deep=True)

    #logging.info('corr ...')
    corrmap = xr.full_like(landmask.astype(float), np.nan)
    if metric == 'corr': # TODO detrend?
        y_test = y_test.groupby('time.month') - mrso_mean_unobs
        y_predict = y_predict.groupby('time.month') - mrso_mean_unobs
        y_train = y_train.groupby('time.month') - mrso_mean_obs
        y_train_predict = y_train_predict.groupby('time.month') - mrso_mean_obs

        corr = xr.corr(y_test, y_predict, dim='time')
        corr_train = xr.corr(y_train, y_train_predict, dim='time')
    elif metric == 'seasonality':
        y_test = y_test.groupby('time.month').mean().rename(month='time')
        y_predict = y_predict.groupby('time.month').mean().rename(month='time')
        y_train = y_train.groupby('time.month').mean().rename(month='time')
        y_train_predict = y_train_predict.groupby('time.month').mean().rename(month='time')

        corr = xr.corr(y_test, y_predict, dim='time')
        corr_train = xr.corr(y_train, y_train_predict, dim='time')
    elif metric == 'trend': # TODO until like in Cook 2020: anomalies with reference to baseline period (1851-1880)
        y_test = y_test.resample(time='1y').mean()
        y_predict = y_predict.resample(time='1y').mean()
        y_train = y_train.resample(time='1y').mean()
        y_train_predict = y_train_predict.resample(time='1y').mean()

        ms_to_year = 365*24*3600*10**9
        test_trends = y_test.polyfit(dim="time", deg=1).polyfit_coefficients.sel(degree=1)*ms_to_year
        predict_trends = y_predict.polyfit(dim="time", deg=1).polyfit_coefficients.sel(degree=1)*ms_to_year
        train_trends = y_train.polyfit(dim="time", deg=1).polyfit_coefficients.sel(degree=1)*ms_to_year
        train_predict_trends = y_train_predict.polyfit(dim="time", deg=1).polyfit_coefficients.sel(degree=1)*ms_to_year
        corr = -np.abs(test_trends - predict_trends) # corr is neg abs diff
        corr_train = -np.abs(train_trends - train_predict_trends) 
    elif metric == 'r2':
        # R2 should not be used for non-linear models, as it is only pearson**2
        # for linear models and can lead to incorrect conclusions in nonlinear
        # cases see: https://stackoverflow.com/questions/893657/how-do-i-calculate-r-squared-using-python-and-numpy
        corr = xr.corr(y_test, y_predict, dim='time')
        corr_train = xr.corr(y_train, y_train_predict, dim='time')
    else:
        raise AttributeError('metric not known')
    corrmap[unobslat, unobslon] = corr
    corrmap[obslat, obslon] = corr_train

    # find x gridpoints with minimum performance
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

    lats = y_latlon.where(y_latlon.landpoints.isin(landpts), drop=True).coords["lat"].values[:,0]
    lons = y_latlon.where(y_latlon.landpoints.isin(landpts), drop=True).coords["lon"].values[:,0]
    latlist = latlist + lats.tolist()
    lonlist = lonlist + lons.tolist()

    # calc mean corr for log and turn sign corrmap
    if metric == 'trend':
        mean_corr = -corrmap.mean().item()
        corrmap = -corrmap
    else:
        mean_corr = corrmap.mean().item()
    logging.info(f'iteration {i} obs landpoints {len(latlist)} mean metric {mean_corr}')

    # save results
    frac_obs = mrso_obs.shape[1] / (mrso_obs.shape[1] + mrso_unobs.shape[1])
    corrmap = corrmap.assign_coords(frac_observed=frac_obs)
    # assign scalar coords bug: https://stackoverflow.com/questions/58858083/how-to-create-scalar-dimension-in-xarray
    corrmap = corrmap.expand_dims({'model': 1, 'metric': 1, 'strategy': 1}).copy()
    corrmap = corrmap.assign_coords({'model': np.atleast_1d(modelname),
                                     'metric': np.atleast_1d(metric), 
                                     'strategy': np.atleast_1d(method)}) 
    corrmaps.append(corrmap)
    for lat, lon in zip(lats, lons):
        niter.loc[lat,lon] = i

    # set iter
    i += 1

niter = niter.expand_dims({'model': 1, 'metric': 1, 'strategy': 1}).copy()
niter = niter.assign_coords({'model': np.atleast_1d(modelname),
                             'metric': np.atleast_1d(metric), 
                             'strategy': np.atleast_1d(method)}) 

corrmaps = xr.concat(corrmaps, dim='frac_observed')
corrmaps = corrmaps.to_dataset(name="mrso")
corrmaps.to_netcdf(f'corrmap_{method}_{modelname}_{metric}{testcase}.nc')
niter = niter.to_dataset(name="mrso")
niter.to_netcdf(f'niter_{method}_{modelname}_{metric}{testcase}.nc')
