"""
TEST
"""

import regionmask
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
benchmark = xr.open_dataset(f'{largefilepath}mrso_benchmark_{modelname}_{experimentname}_{ensemblename}.nc')['mrso']

# define drought as driest 10% of values for each of the datasets individually
orig_perc = orig.groupby('time.month').quantile(0.1)
pred_perc = pred.groupby('time.month').quantile(0.1)

# mask non-drought values
orig_drought = orig.groupby('time.month').where(orig.groupby('time.month') < orig_perc) 
pred_drought = pred.groupby('time.month').where(pred.groupby('time.month') < pred_perc)

# set percentile to zero: soil moisture anomalies above drought threshold in mm
orig_drought = (orig_drought.groupby('time.month') - orig_perc)*(-1)
pred_drought = (pred_drought.groupby('time.month') - pred_perc)*(-1)
import IPython; IPython.embed()

#orig_perc = orig.quantile(0.1, dim='time')
#orig_drought = (orig - orig_perc)
#orig_drought = orig_drought.where(orig_drought < 0, 0)
##pred_perc = pred.quantile(0.1, dim='time') # two options, this one is harder, show that not only the maginutide is off (same percentiles) but also some droughts are missed outside of well-observed spaces (different percentiles)
#pred_drought = (pred - pred_perc)
#pred_drought = pred_drought.where(pred_drought < 0, 0)

# drought indices: duration, severity, frequency, extent 
# TODO also do severity and duration, but for now only frequency and extent
#pred_bool = (pred_drought < 0)*1 # True (1) is drought
#orig_bool = (orig_drought < 0)*1 # True (1) is drought
#
#proj = ccrs.PlateCarree()
#fig = plt.figure(figsize=(10,3))
#ax1 = fig.add_subplot(121, projection=proj)
#ax2 = fig.add_subplot(122, projection=proj)
#ax1.coastlines()
#ax2.coastlines()
#(orig_bool.diff(dim='time') == 1).sum(dim='time').plot(ax=ax1, vmin=0, vmax=60, cmap='hot_r')
#(pred_bool.diff(dim='time') == 1).sum(dim='time').plot(ax=ax2, vmin=0, vmax=60, cmap='hot_r')
#plt.show()


# look at drought pattern per region
#mask = regionmask.defined_regions.ar6.all.mask(orig)
#res = xr.full_like(mask, np.nan)
##orig_freq = orig_drought != 0
##pred_freq = pred_drought != 0
##drought_underestimation = (orig_freq.groupby('time.year').sum() - pred_freq.groupby('time.year').sum()) # underestimation is positive
#for i in range(58):
#    tmp = orig_drought.where(mask == i).mean(dim=('lat','lon')) - pred_drought.where(mask == i).mean(dim=('lat','lon'))
#    res = res.where(mask != i, tmp.sum().item())
#
#import IPython; IPython.embed()
#
## plot 
proj = ccrs.PlateCarree()
for t, time in enumerate(pred_drought.time):
    fig = plt.figure(figsize=(10,3))
    fig.suptitle(f'{str(time.values)[:7]}')
    ax1 = fig.add_subplot(121, projection=proj)
    ax2 = fig.add_subplot(122, projection=proj)
    orig_drought.sel(time=time).plot(ax=ax1, cmap='hot', add_colorbar=False, vmin=0, vmax=1000)
    pred_drought.sel(time=time).plot(ax=ax2, cmap='hot', add_colorbar=False, vmin=0, vmax=1000)
    ax1.set_title('orig')
    ax2.set_title('upscaled')
    ax1.coastlines()
    ax2.coastlines()
    #plt.show()
    plt.savefig(f'drought_gif/drought_evolution_{t:03d}.png')
    plt.close()
