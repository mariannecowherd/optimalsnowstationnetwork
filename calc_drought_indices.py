"""
TEST
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# load data
largefilepath = '/net/so4/landclim/bverena/large_files/'
modelname = 'CanESM5'
experimentname = 'historical'
ensemblename = 'r1i1p1f1'
orig = xr.open_dataset(f'{largefilepath}mrso_orig_{modelname}_{experimentname}_{ensemblename}.nc')['mrso']
pred = xr.open_dataset(f'{largefilepath}mrso_pred_{modelname}_{experimentname}_{ensemblename}.nc')['mrso']

# drought indices: duration, severity, frequency, extent 
perc90 = orig.quantile(0.1, dim='time')
orig_drought = (orig - perc90)
orig_drought = orig_drought.where(orig_drought < 0, 0)
perc90 = pred.quantile(0.1, dim='time') # two options, this one is harder, show that not only the maginutide is off (same percentiles) but also some droughts are missed outside of well-observed spaces (different percentiles)
pred_drought = (pred - perc90)
pred_drought = pred_drought.where(pred_drought < 0, 0)

# plot 
proj = ccrs.PlateCarree()
for t, time in enumerate(pred_drought.time):
    fig = plt.figure(figsize=(10,3))
    fig.suptitle(f'{str(time.values)[:7]}')
    ax1 = fig.add_subplot(121, projection=proj)
    ax2 = fig.add_subplot(122, projection=proj)
    orig_drought.sel(time=time).plot(ax=ax1, cmap='hot', add_colorbar=False, vmin=-2, vmax=0)
    pred_drought.sel(time=time).plot(ax=ax2, cmap='hot', add_colorbar=False, vmin=-2, vmax=0)
    ax1.set_title('orig')
    ax2.set_title('upscaled')
    ax1.coastlines()
    ax2.coastlines()
    plt.savefig(f'drought_gif/drought_evolution_{t:03d}.png')
