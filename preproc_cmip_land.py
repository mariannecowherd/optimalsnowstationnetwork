"""
TEST
"""

import random
import glob
import cftime
import numpy as np
import pandas as pd
import xarray as xr
import regionmask
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from sklearn.ensemble import RandomForestRegressor
upscalepath = '/net/so4/landclim/bverena/large_files/opscaling/'

# read CMIP6 files
def rename_vars(data):
    varname = list(data.keys())[0]
    _, _, modelname, _, ensemblename, _ = data.encoding['source'].split('_')
    data = data.rename({varname: f'{modelname}_{ensemblename}'})
    try:
        data = data.drop_vars('file_qf')
    except ValueError:
        pass
    try:
        data = data.drop_dims('height')
    except ValueError:
        pass
    if isinstance(data.time[0].item(), cftime._cftime.DatetimeNoLeap):
        data['time'] = data.indexes['time'].to_datetimeindex()
    if isinstance(data.time[0].item(), cftime._cftime.Datetime360Day):
        data['time'] = data.indexes['time'].to_datetimeindex()
    return data

def open_cmip_suite(varname, modelname, experimentname, ensemblename):
    cmip6_path = f'/net/atmos/data/cmip6-ng/{varname}/mon/g025/'
    filenames = glob.glob(f'{cmip6_path}{varname}_mon_{modelname}_{experimentname}_{ensemblename}_*.nc')
    data = xr.open_mfdataset(filenames, 
                             combine='nested', # timestamp problem
                             concat_dim=None,  # timestamp problem
                             preprocess=rename_vars)  # rename vars to model name
    data = data.to_array(dim='model').rename(varname)
    data.coords['lon'] = (data.coords['lon'] + 180) % 360 - 180
    data = data.sortby('lon')
    #data = data.mean(dim='variable').load() # here for now
    #data = data.load()

    return data

modelnames = ['HadGEM3-GC31-MM','MIROC6','MPI-ESM1-2-HR','IPSL-CM6A-LR']
ensemblenames = ['r1i1p1f3','r1i1p1f1','r1i1p1f1','r1i1p1f1']
for modelname, ensemblename in zip(modelnames, ensemblenames):
    #modelname = '*'
    experimentname = 'ssp585'
    #ensemblename = 'r1i1p1f1'
    mrso = open_cmip_suite('mrso', modelname, experimentname, ensemblename)
    tas = open_cmip_suite('tas', modelname, experimentname, ensemblename)
    pr = open_cmip_suite('pr', modelname, experimentname, ensemblename)

    # cut out 2014 to 2050
    mrso = mrso.sel(time=slice('2015','2050'))
    tas = tas.sel(time=slice('2015','2050'))
    pr = pr.sel(time=slice('2015','2050'))

    # cut out Greenland and Antarctica for landmask
    n_greenland = regionmask.defined_regions.natural_earth.countries_110.map_keys('Greenland')
    n_antarctica = regionmask.defined_regions.natural_earth.countries_110.map_keys('Antarctica')
    mask = regionmask.defined_regions.natural_earth.countries_110.mask(mrso)
    mrso = mrso.where((mask != n_greenland) & (mask != n_antarctica) & (~np.isnan(mask)))
    tas = tas.where((mask != n_greenland) & (mask != n_antarctica) & (~np.isnan(mask)))
    pr = pr.where((mask != n_greenland) & (mask != n_antarctica) & (~np.isnan(mask)))

    # cut out deserts
    nyears = len(np.unique(pr["time.year"]))
    isdesert = pr.sum(dim="time") / float(nyears) < 0.00003
    #isdesert.to_netcdf(savepath + f"isdesert.nc")
    mrso = mrso.where(~isdesert)   
    tas = tas.where(~isdesert)   
    pr = pr.where(~isdesert)   

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

    # calculate deseasonalised anomaly 
    #seasonal_mean = mrso.groupby('time.month').mean()
    #seasonal_std = mrso.groupby('time.month').std()
    #mrso = (mrso.groupby('time.month') - seasonal_mean) 
    #mrso = mrso.groupby('time.month') / seasonal_std

    #seasonal_mean = pred.groupby('time.month').mean() 
    #pred = (pred.groupby('time.month') - seasonal_mean) 

    # save
    # TODO save individual landmask
    mrso = mrso.to_dataset(name="mrso")
    mrso.to_netcdf(f'{upscalepath}mrso_{modelname}_seasonality.nc')
    pred.to_netcdf(f'{upscalepath}pred_{modelname}_seasonality.nc')
    #mrso_land.to_netcdf(f'{largefilepath}mrso_land_{experimentname}.nc')
