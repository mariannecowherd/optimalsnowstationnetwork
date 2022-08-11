"""
TEST
"""

import random
import glob
import cftime
import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe
import regionmask
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import warnings
from sklearn.ensemble import RandomForestRegressor
upscalepath = '/net/so4/landclim/bverena/large_files/opscaling/'
warnings.filterwarnings('ignore', message='Converting a CFTimeIndex with dates from a non-standard calendar, \'360_day\', to a pandas.DatetimeIndex, which uses dates from the standard calendar.  This may lead to subtle errors in operations that depend on the length of time between dates.')
warnings.filterwarnings('ignore', message='Converting a CFTimeIndex with dates from a non-standard calendar, \'noleap\', to a pandas.DatetimeIndex, which uses dates from the standard calendar.  This may lead to subtle errors in operations that depend on the length of time between dates.')

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

def open_cmip_suite(modelname, varname):
    cmip6_path = f'/net/atmos/data/cmip6-ng/{varname}/mon/g025/'
    filename = glob.glob(cmip6_path + f'{varname}_mon_{modelname}_ssp370_r1i1*_g025.nc')[0] # sometimes 2 enmseble members are chose. select first one
    data = xr.open_mfdataset(filename, 
                             combine='nested', # timestamp problem
                             concat_dim=None,  # timestamp problem
                             preprocess=rename_vars)  # rename vars to model name
    data = data.to_array(dim='model').rename(varname)
    data.coords['lon'] = (data.coords['lon'] + 180) % 360 - 180
    data = data.sortby('lon')

    return data

cmip6_path = f'/net/atmos/data/cmip6-ng/mrso/mon/g025/'
filepaths = glob.glob(f'{cmip6_path}mrso_mon_*_ssp370_*r1i1*_g025.nc')
modelnames = [filepath.split('_')[2] for filepath in filepaths]
modelnames.remove('MCM-UA-1-0') # does not have pr
modelnames = np.unique(modelnames) # double entries
for modelname in modelnames:
    #modelname = 'MPI-ESM1-2-LR' # DEBUG
    print(modelname)

    # load data
    mrso = open_cmip_suite(modelname, 'mrso')
    tas = open_cmip_suite(modelname, 'tas')
    pr = open_cmip_suite(modelname, 'pr')

    # cut out 2014 to 2050
    mrso = mrso.sel(time=slice('2015','2050'))
    tas = tas.sel(time=slice('2015','2050'))
    pr = pr.sel(time=slice('2015','2050'))

    # cut out Greenland and Antarctica and ocean for landmask
    n_greenland = regionmask.defined_regions.natural_earth_v5_0_0.countries_110.map_keys('Greenland')
    n_antarctica = regionmask.defined_regions.natural_earth_v5_0_0.countries_110.map_keys('Antarctica')
    mask = regionmask.defined_regions.natural_earth_v5_0_0.countries_110.mask(mrso)
    mrso = mrso.where((mask != n_greenland) & (mask != n_antarctica) & (~np.isnan(mask)))
    tas = tas.where((mask != n_greenland) & (mask != n_antarctica) & (~np.isnan(mask)))
    pr = pr.where((mask != n_greenland) & (mask != n_antarctica) & (~np.isnan(mask)))

    # cut out deserts
    largefilepath = '/net/so4/landclim/bverena/large_files/'
    koeppen = xr.open_dataset(f'{largefilepath}opscaling/koeppen_simple.nc').to_array().squeeze()
    koeppen = koeppen.drop('variable')
    regridder = xe.Regridder(koeppen, mrso, 'bilinear') # does not lead to fractional classes prob because downsampling not up
    koeppen = regridder(koeppen)
    isdesert = koeppen == 4
    mrso = mrso.where(~isdesert)   
    tas = tas.where(~isdesert)   
    pr = pr.where(~isdesert)   

    # create landmask of valid (not ice or desert) land mask
    landmask = mask.where((mask != n_greenland) & (mask != n_antarctica))
    landmask = ~np.isnan(landmask)
    landmask = landmask.where(~isdesert, False)

    # mask out land ice (as suggested by MH)
    # land ice mask sftgif not avail in cmip6-ng.

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

    # save
    import IPython; IPython.embed()
    mrso = mrso.to_dataset(name="mrso")
    mrso.to_netcdf(f'{upscalepath}mrso_{modelname}.nc')
    pred.to_netcdf(f'{upscalepath}pred_{modelname}.nc')
    landmask.to_netcdf(f'{upscalepath}landmask.nc')
