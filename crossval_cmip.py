"""
TEST
"""

import cftime
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
    if isinstance(data.time[0].item(), cftime._cftime.DatetimeNoLeap):
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

modelname = 'MPI-ESM1-2-HR'
experimentname = 'historical'
ensemblename = 'r1i1p1f1'
mrso = open_cmip_suite('mrso', modelname, experimentname, ensemblename)
tas = open_cmip_suite('tas', modelname, experimentname, ensemblename)
pr = open_cmip_suite('pr', modelname, experimentname, ensemblename)

# cut out Greenland and Antarctica for landmask
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

pr_1month = pr.copy(deep=True).shift(time=1, fill_value=0).rename('pr_1m') 
pr_2month = pr.copy(deep=True).shift(time=2, fill_value=0).rename('pr_2m')
pr_3month = pr.copy(deep=True).shift(time=3, fill_value=0).rename('pr_3m')
pr_4month = pr.copy(deep=True).shift(time=4, fill_value=0).rename('pr_4m')
pr_5month = pr.copy(deep=True).shift(time=5, fill_value=0).rename('pr_5m')
pr_6month = pr.copy(deep=True).shift(time=6, fill_value=0).rename('pr_6m')

# merge predictors into one dataset 
pred = xr.merge([tas, tas_1month, tas_2month, tas_3month, tas_4month, tas_5month, tas_6month,
                 pr, pr_1month, pr_2month, pr_3month, pr_4month, pr_5month, pr_6month])

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
#mrso = (mrso - mrso.mean()) / mrso.std()

#seasonal_mean = pred.groupby('time.month').mean() # not necessary for RF and difficult for pr
#seasonal_std = pred.groupby('time.month').std()
#pred = (pred.groupby('time.month') - seasonal_mean) 
#pred = pred.groupby('time.month') / seasonal_std

seasonal_mean = stations.groupby('time.month').mean()
seasonal_std = stations.groupby('time.month').std()
stations = (stations.groupby('time.month') - seasonal_mean) 
stations = stations.groupby('time.month') / seasonal_std

# regrid station data to CMIP6 grid
landmask = ~np.isnan(mrso[0,:,:]).copy(deep=True)
obsmask = xr.full_like(landmask, False)
#unobsmask = landmask.copy(deep=True)
lat_cmip = []
lon_cmip = []
latlon_cmip = [] # because groupby on two coords is not yet implemented in xarray
for lat, lon in zip(stations.lat, stations.lon):
    point = landmask.sel(lat=lat, lon=lon, method='nearest')
    
    if landmask.loc[point.lat,point.lon].item(): # obs gridpoint if station contained and on CMIP land
        obsmask.loc[point.lat, point.lon] = True
    #unobsmask.loc[point.lat, point.lon] = False


    lat_cmip.append(point.lat.item())
    lon_cmip.append(point.lon.item())
    latlon_cmip.append(f'{point.lat.item()} {point.lon.item()}')
stations = stations.assign_coords(lat_cmip=('stations',lat_cmip))
stations = stations.assign_coords(lon_cmip=('stations',lon_cmip))
stations = stations.assign_coords(latlon_cmip=('stations',latlon_cmip))

# add time of measurement begin and end
stations = stations.groupby('latlon_cmip').mean()

# divide into obs and unobs data
obslat, obslon = np.where(obsmask)
obspoints = np.arange(len(obslat))
obslat, obslon = xr.DataArray(obslat, dims='obspoints', coords=[obspoints]), xr.DataArray(obslon, dims='obspoints', coords=[obspoints])

mrso_obs = mrso.isel(lat=obslat, lon=obslon)
pred_obs = pred.isel(lat=obslat, lon=obslon)

# prepare cross-val 
cv_results = np.zeros((10,30))
ntrees_list = [10,50,100]

# stack
mrso_obs = mrso_obs.stack(datapoints=('obspoints','time'))
pred_obs = pred_obs.stack(datapoints=('obspoints','time')).to_array().T

for n, ntrees in enumerate(ntrees_list):

    # rf settings TODO later use GP
    kwargs = {'n_estimators': ntrees,
              'min_samples_leaf': 2,
              'max_features': 'auto', 
              'max_samples': None, 
              'bootstrap': True,
              'warm_start': False,
              'n_jobs': None, # set to number of trees
              'verbose': 0}

    rf = RandomForestRegressor(**kwargs)

    for o, gridpoints in enumerate(np.array_split(obspoints, 30)): # random folds of observed gridpoints # LG says doesnot matter if random or regionally grouped, both has advantages and disadvantages, just do something and reason why

        X_train = pred_obs.where(pred_obs.obspoints.isin(gridpoints), drop=True)
        y_train = mrso_obs.where(pred_obs.obspoints.isin(gridpoints), drop=True)
        X_test = pred_obs.where(~pred_obs.obspoints.isin(gridpoints), drop=True)
        y_test = mrso_obs.where(~pred_obs.obspoints.isin(gridpoints), drop=True)
        y_predict = xr.full_like(y_test, np.nan)

        rf.fit(X_train, y_train)
        y_predict[:] = rf.predict(X_test)

        corr = xr.corr(y_test, y_predict).item()
        cv_results[n,o] = corr
        print(ntrees, o, corr)
import IPython; IPython.embed()
