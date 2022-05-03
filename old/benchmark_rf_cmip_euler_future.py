"""
TEST
"""

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.ensemble import RandomForestRegressor

# check if gridpoint is already computed
largefilepath = '/cluster/work/climate/bverena/'
largefilepath = '/net/so4/landclim/bverena/large_files/'
modelname = 'CanESM5'
experimentname = 'historical'
ensemblename = 'r1i1p1f1'

# load feature tables
print('load feature tables')
mrso_hist = xr.open_dataset(f'{largefilepath}mrso_land_historical.nc').load()
pred_hist = xr.open_dataset(f'{largefilepath}pred_land_historical.nc').load()

mrso_ssp585 = xr.open_dataset(f'{largefilepath}mrso_land_ssp585.nc').load()
pred_ssp585 = xr.open_dataset(f'{largefilepath}pred_land_ssp585.nc').load()

# concat historical and future data
mrso_land = xr.concat([mrso_hist, mrso_ssp585], dim='time')['mrso']
pred_land = xr.concat([pred_hist, pred_ssp585], dim='time')
pred_land = pred_land.to_array()

# if upscale: delete non-observed datapoints
iupscale = True
inewstations = True
if iupscale:
    stations_cmip = xr.open_dataset(f'{largefilepath}stations_cmip.nc')['__xarray_dataarray_variable__']
    iold = np.isnan(stations_cmip.sel(time=slice('2015','2020'))).all(axis=0)
    tmp = xr.full_like(stations_cmip.sel(time=slice('1991','2010')), np.nan) # add 5 years until 2035
    timerange = pd.date_range('2031-01-01','2051-01-01', freq='M')
    tmp['time'] = timerange
    tmp[:,~iold] = 1
    stations_cmip = xr.concat([stations_cmip, tmp], dim='time')
    iobs = ~np.isnan(stations_cmip)
    mrso_obs = xr.full_like(mrso_land, np.nan)
    pred_obs = xr.full_like(pred_land, np.nan)
    for lat, lon in zip(iobs.lat_cmip, iobs.lon_cmip):
        itime = iobs.where((iobs.lat_cmip == lat) & (iobs.lon_cmip == lon), drop=True).astype(bool)
        timeidx = np.where(itime)[0]
        #if itime.sum().item() == 0:
        #    continue
        #print(itime.sum().item())
        landpt = mrso_land.where((mrso_land.lat == lat) & (mrso_land.lon == lon), drop=True).landpoints.item()
        mrso_obs.loc[:,landpt][timeidx] = mrso_land.loc[:,landpt][timeidx]
        pred_obs.loc[:,:,landpt][:,timeidx] = pred_land.loc[:,:,landpt][:,timeidx]
        #pred_obs.loc[dict(landpoints=landpt)][timeidx] = pred_land.loc[dict(landpoints=landpt)][timeidx]
    if inewstations:
        def return_landpt(lat, lon):
            return mrso_land.where((mrso_land.lat == 1.25) & (mrso_land.lon == -58.75), drop=True).landpoints.item()
        # add new proposed station locations
        # Af: (177, 242) latlon: 1.25, -58.75 latloncmip: 1.25 -58.75
        # Am : (177, 243) latlon: 1.25, -58.25 latloncmip:  1.25 -58.75 REJECTED
        # Am: (third largest value) latlon: 1.25 60.35 latloncmip: 1.25 -61.25
        # Aw:  (174, 240) latlon: 2.75, -59.75 latloncmip: 3.75 -58.75
        # Dw: (47, 637) latlon: 66.25, 138.8 latloncmip: 66.25 138.8
        # Df: (52, 668) latlon: 63.75, 154.2 latloncmip: 63.75 153.8
        mrso_obs.loc[slice('2020','2035'),return_landpt(1.25,-58.75)] = mrso_land.loc[slice('2020','2035'),return_landpt(1.25,-58.75)]
        mrso_obs.loc[slice('2020','2035'),return_landpt(1.25,-61.25)] = mrso_land.loc[slice('2020','2035'),return_landpt(1.25,-61.25)]
        mrso_obs.loc[slice('2020','2035'),return_landpt(3.75,-58.75)] = mrso_land.loc[slice('2020','2035'),return_landpt(3.75,-58.75)]
        mrso_obs.loc[slice('2020','2035'),return_landpt(66.25,138.8)] = mrso_land.loc[slice('2020','2035'),return_landpt(66.25,138.8)]
        mrso_obs.loc[slice('2020','2035'),return_landpt(63.75,153.8)] = mrso_land.loc[slice('2020','2035'),return_landpt(63.75,153.8)]
        pred_obs.loc[:,slice('2020','2035'),return_landpt(1.25,-58.75)] = pred_land.loc[:,slice('2020','2035'),return_landpt(1.25,-58.75)]
        pred_obs.loc[:,slice('2020','2035'),return_landpt(1.25,-61.25)] = pred_land.loc[:,slice('2020','2035'),return_landpt(1.25,-61.25)]
        pred_obs.loc[:,slice('2020','2035'),return_landpt(3.75,-58.75)] = pred_land.loc[:,slice('2020','2035'),return_landpt(3.75,-58.75)]
        pred_obs.loc[:,slice('2020','2035'),return_landpt(66.25,138.8)] = pred_land.loc[:,slice('2020','2035'),return_landpt(66.25,138.8)]
        pred_obs.loc[:,slice('2020','2035'),return_landpt(63.75,153.8)] = pred_land.loc[:,slice('2020','2035'),return_landpt(63.75,153.8)]
    import IPython; IPython.embed()
else:
    mrso_obs = mrso_land
    pred_obs = pred_land

# make result xr
years = np.arange(2025,2036)
mrso_res = xr.full_like(mrso_land.where(mrso_land['time.year'].isin(years), drop=True), np.nan)

# stack
mrso_obs = mrso_obs.stack(datapoints=('landpoints','time')).reset_index('datapoints')
pred_obs = pred_obs.stack(datapoints=('landpoints','time')).reset_index('datapoints').T

mrso_land = mrso_land.stack(datapoints=('landpoints','time')).reset_index('datapoints')
pred_land = pred_land.stack(datapoints=('landpoints','time')).reset_index('datapoints').T

# remove unobserved datapoints
pred_obs = pred_obs.where(~np.isnan(mrso_obs), drop=True)
mrso_obs = mrso_obs.where(~np.isnan(mrso_obs), drop=True)
import IPython; IPython.embed()

# rf settings TODO later use GP
kwargs = {'n_estimators': 100, 
          'min_samples_leaf': 1, # those are all default values anyways
          'max_features': 'auto', 
          'max_samples': None, 
          'bootstrap': True,
          'warm_start': False,
          'n_jobs': 100, # set to number of trees
          'verbose': 0}

# divide into test and train data
#mrso_res = mrso_res.unstack('datapoints')

for year in years:

    y_test = mrso_land.where(mrso_land['time.year'] == year, drop=True)
    y_train = mrso_obs.where(mrso_obs['time.year'] < year, drop=True)

    X_test = pred_land.where(pred_land['time.year'] == year, drop=True)
    X_train = pred_obs.where(pred_obs['time.year'] < year, drop=True)

    print('train')
    rf = RandomForestRegressor(**kwargs)
    rf.fit(X_train, y_train)

    print('predict')
    y_predict = xr.full_like(mrso_res.loc[str(year),:].stack(datapoints=('landpoints','time')), np.nan)
    y_predict[:] = rf.predict(X_test)

    corr = xr.corr(y_test, y_predict).item()
    print(year, corr**2)

    y_predict = y_predict.unstack('datapoints')
    mrso_res.loc[str(year),:] = y_predict.T

# renormalise 
#mrso_res = (mrso_res.groupby('time.month') * seasonal_std)
#mrso_res = (mrso_res.groupby('time.month') + seasonal_mean) 

import IPython; IPython.embed()
print('save')
if iupscale:
    if inewstations:
        mrso_res.to_netcdf(f'{largefilepath}mrso_newstations_{modelname}_future_{ensemblename}.nc')
    else:
        mrso_res.to_netcdf(f'{largefilepath}mrso_upscale_{modelname}_future_{ensemblename}.nc')
else:
    mrso_res.to_netcdf(f'{largefilepath}mrso_benchmark_{modelname}_future_{ensemblename}.nc')
