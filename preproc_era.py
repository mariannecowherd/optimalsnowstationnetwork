import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from sklearn.cluster import OPTICS

# define time range
years = list(np.arange(1979,2021))

# define paths
largefilepath = '/net/so4/landclim/bverena/large_files/'
era5path_variant = '/net/exo/landclim/data/dataset/ERA5_deterministic/recent/0.25deg_lat-lon_1m/processed/regrid/'
era5path_invariant = '/net/exo/landclim/data/dataset/ERA5_deterministic/recent/0.25deg_lat-lon_time-invariant/processed/regrid/'
era5path_max = '/net/exo/landclim/data/dataset/ERA5_deterministic/recent/0.25deg_lat-lon_1h/processed/regrid_tmax1d/'
era5path_min = '/net/exo/landclim/data/dataset/ERA5_deterministic/recent/0.25deg_lat-lon_1h/processed/regrid_tmin1d/'
era5path_sum = '/net/exo/landclim/data/dataset/ERA5_deterministic/recent/0.25deg_lat-lon_1h/processed/regrid_tsum1d/'
invarnames = ['lsm','z','slor','cvl','cvh', 'tvl', 'tvh']
varnames = ['skt','t2m','tp','e','ro','sshf','slhf','ssr','str']
varxnames = ['skt','t2m']
varsnames = ['e','tp']

# define files
filenames_var = [f'{era5path_variant}era5_deterministic_recent.{varname}.025deg.1m.{year}.nc' for year in years for varname in varnames]
filenames_max = [f'{era5path_max}era5_deterministic_recent.{varname}.025deg.1h.{year}.tmax1d.nc' for year in years for varname in varxnames]
filenames_min = [f'{era5path_min}era5_deterministic_recent.{varname}.025deg.1h.{year}.tmin1d.nc' for year in years for varname in varxnames]
filenames_sum = [f'{era5path_sum}era5_deterministic_recent.{varname}.025deg.1h.{year}.tsum1d.nc' for year in years for varname in varsnames]
filenames_invar = [f'{era5path_invariant}era5_deterministic_recent.{varname}.025deg.time-invariant.nc' for varname in invarnames]

# open files
data = xr.open_mfdataset(filenames_var, combine='by_coords')
constant_maps = xr.open_mfdataset(filenames_invar, combine='by_coords')
data_max = xr.open_mfdataset(filenames_max, combine='by_coords').drop('time_bnds')
data_min = xr.open_mfdataset(filenames_min, combine='by_coords').drop('time_bnds')
data_sum = xr.open_mfdataset(filenames_sum, combine='by_coords')

# 3-month and 6-month rolling means backwards
data_roll = data.rolling(time=6, center=False, min_periods=1).mean()

# rename variables
data_max = data_max.rename({'skt':'sktmax','t2m':'t2mmax'})
data_min = data_min.rename({'skt':'sktmin','t2m':'t2mmin'})
data_sum = data_sum.rename({'e':'esum','tp':'tpsum'})
data_roll = data_roll.to_array(dim='variable').assign_coords(variable=[f'{var}roll' for var in varnames]).to_dataset('variable')

# downsample to yearly resolution
freq = '1m'
data = data.resample(time=freq).mean()
data_roll = data_roll.resample(time=freq).mean()
data_max = data_max.resample(time=freq).max()
data_min = data_min.resample(time=freq).min()
data_sum = data_sum.resample(time=freq).sum()

# save to netcdf
data.to_netcdf(largefilepath + f'era5_deterministic_recent.var.025deg.{freq}.mean.nc')
data_roll.to_netcdf(largefilepath + f'era5_deterministic_recent.var.025deg.{freq}.roll.nc')
data_max.to_netcdf(largefilepath + f'era5_deterministic_recent.temp.025deg.{freq}.max.nc')
data_min.to_netcdf(largefilepath + f'era5_deterministic_recent.temp.025deg.{freq}.min.nc')
data_sum.to_netcdf(largefilepath + f'era5_deterministic_recent.precip.025deg.{freq}.sum.nc')
