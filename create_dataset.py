"""
create dataset for investigating optimal station networks

    @author: verena bessenbacher
    @date: 26 05 2020
"""

# decide which variables to use
# temperature, precipitation, evapotranspiration, runoff, soil moisture, sensible heat, carbon fluxes?
# mean, extremes and trends
# constant maps: altitude, topographic complexity, vegetation cover

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# define time range
years = list(np.arange(2000,2020))
years = [2000]

# define paths
era5path_hourly = '/net/exo/landclim/data/dataset/ERA5_deterministic/recent/0.25deg_lat-lon_1h/processed/regrid/'
era5path_invariant = '/net/exo/landclim/data/dataset/ERA5_deterministic/recent/0.25deg_lat-lon_time-invariant/processed/regrid/'

# define files
filenames_skt = [f'{era5path_hourly}era5_deterministic_recent.skt.025deg.1h.{year}.nc' for year in years]

# open files
skt = xr.open_mfdataset(filenames_skt, combine='by_coords')['skt'] # [K]
sktmax = skt.groupby('time.date').max()
sktmin = skt.groupby('time.date').min()
sktmean = skt.groupby('time.date').mean()
import IPython; IPython.embed()
